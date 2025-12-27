import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
import altair as alt
import config


# 匯入你的本地模組
from src import model
from src.dataset import get_dataloaders # 用來抓類別名稱

# --- 1. 頁面設定 ---
st.set_page_config(
    page_title="名畫辨識系統",
    page_icon="🎨",
    layout="wide"
)

st.title("🎨 藝術名畫辨識系統 (Art Recognition AI)")
st.markdown("上傳一張畫作，AI 將會分析這是哪位大師的作品。")

# --- 2. 系統設定與工具函數 ---
DEVICE = torch.device(config.DEVICE)

# 定義預處理 (必須跟驗證集的一模一樣)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@st.cache_resource
def load_class_names():
    """載入類別名稱 (只執行一次)"""
    # 這裡我們稍微偷懶，利用 get_dataloaders 取得類別，但設 batch_size=1 加快速度
    # 如果你有存 class_names.txt 也可以直接讀檔
    try:
        _, _, classes = get_dataloaders(config.DATA_DIR, batch_size=1)
        return classes
    except Exception as e:
        st.error(f"無法讀取類別資訊: {e}")
        return []

@st.cache_resource
def load_single_model(num_classes):
    """載入單一 ResNet50 模型"""
    try:
        # 建立結構
        net = model.get_model(num_classes, model_name='resnet50', tune_backend=False)
        # 載入權重
        weight_path = './checkpoints/resnet50_best.pth'
        net.load_state_dict(torch.load(weight_path, map_location=DEVICE))
        net.to(DEVICE)
        net.eval()
        return net
    except FileNotFoundError:
        st.error("找不到 checkpoints/resnet50_best.pth，請確認檔案位置。")
        return None

@st.cache_resource
def load_ensemble_models(num_classes):
    """載入三合一集成模型"""
    models = []
    configs = [
        ('resnet50', './checkpoints/resnet50_best.pth'),
        ('densenet121', './checkpoints/densenet121_best.pth'),
        ('efficientnet_b0', './checkpoints/efficientnet_b0_best.pth')
    ]
    
    for name, path in configs:
        try:
            net = model.get_model(num_classes, model_name=name, tune_backend=False)
            net.load_state_dict(torch.load(path, map_location=DEVICE))
            net.to(DEVICE)
            net.eval()
            models.append(net)
        except FileNotFoundError:
            st.warning(f"⚠️ 警告: 找不到 {path}，集成模型將缺少此成員。")
            
    return models

def predict_single(net, img_tensor):
    """單一模型預測"""
    with torch.no_grad():
        outputs = net(img_tensor)
        probs = F.softmax(outputs, dim=1) # 轉成機率
    return probs[0]

def predict_ensemble(models, img_tensor):
    """集成模型預測 (平均法)"""
    total_probs = None
    with torch.no_grad():
        for net in models:
            outputs = net(img_tensor)
            probs = F.softmax(outputs, dim=1)
            
            if total_probs is None:
                total_probs = probs
            else:
                total_probs += probs
    
    # 取平均
    avg_probs = total_probs / len(models)
    return avg_probs[0]

# --- 3. 側邊欄設定 ---
st.sidebar.header("⚙️ 設定面板")

# 選擇模式
model_mode = st.sidebar.radio(
    "選擇模型模式：",
    ("單一模型 (ResNet50)", "三合一集成 (Ensemble)")
)

st.sidebar.info(
    """
    **模式說明：**
    - **單一模型**：速度快，使用 ResNet50 (Acc ~85%)。
    - **集成模型**：準確度最高，結合 DenseNet, EfficientNet (Acc ~87%)。
    """
)

# --- 4. 主程式邏輯 ---

# 1. 載入類別
class_names = load_class_names()
num_classes = len(class_names)

if num_classes > 0:
    # 2. 載入模型 (根據使用者選擇)
    if model_mode == "單一模型 (ResNet50)":
        active_model = load_single_model(num_classes)
        ensemble_models = None
    else:
        active_model = None
        ensemble_models = load_ensemble_models(num_classes)

    # 3. 上傳圖片
    uploaded_file = st.file_uploader("請上傳圖片 (jpg, png, jpeg)", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("原始圖片")
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, use_column_width=True)

        # 4. 預測按鈕
        if st.button("🔍 開始辨識", type="primary"):
            # 預處理
            img_tensor = transform(image).unsqueeze(0).to(DEVICE)

            # 執行預測
            with st.spinner(f"正在使用 {model_mode} 進行分析..."):
                if model_mode == "單一模型 (ResNet50)" and active_model:
                    probs = predict_single(active_model, img_tensor)
                elif model_mode == "三合一集成 (Ensemble)" and ensemble_models:
                    probs = predict_ensemble(ensemble_models, img_tensor)
                else:
                    st.error("模型載入失敗，無法預測。")
                    probs = None

            # 5. 顯示結果
            if probs is not None:
                # 取得前 5 名
                top5_prob, top5_idx = torch.topk(probs, 5)
                
                top5_data = []
                for i in range(5):
                    class_name = class_names[top5_idx[i].item()]
                    probability = top5_prob[i].item()
                    top5_data.append({"畫家": class_name, "信心度": probability})

                with col2:
                    st.subheader("辨識結果")
                    winner = top5_data[0]
                    st.success(f"🏆 最高機率：**{winner['畫家']}** ({winner['信心度']:.1%})")
                    
                    # 製作圖表
                    df = pd.DataFrame(top5_data)
                    chart = alt.Chart(df).mark_bar().encode(
                        x=alt.X('信心度', axis=alt.Axis(format='%'), scale=alt.Scale(domain=[0, 1])),
                        y=alt.Y('畫家', sort='-x'),
                        color=alt.Color('信心度', scale=alt.Scale(scheme='blues')),
                        tooltip=['畫家', alt.Tooltip('信心度', format='.1%')]
                    ).properties(height=300)
                    
                    st.altair_chart(chart, use_container_width=True)

else:
    st.warning("正在初始化系統，請稍候...")