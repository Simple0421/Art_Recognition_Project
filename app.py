import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
import altair as alt
import config
import numpy as np

# 匯入你的本地模組
from src import model
from src.dataset import get_dataloaders
# --- 新增模組 ---
from src.feature_extractor import FeatureExtractor
from src.image_search import ImageSearcher

# --- 1. 頁面設定 ---
st.set_page_config(
    page_title="藝術名畫辨識系統",
    page_icon="🎨",
    layout="wide"
)

st.title("🎨 藝術名畫辨識系統")
st.markdown("### AI 藝術鑑賞與靈感搜尋引擎")

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
        net = model.get_model(num_classes, model_name='resnet50', tune_backend=False)
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

# --- 新增：載入特徵提取器與搜尋引擎 ---
@st.cache_resource
def load_retrieval_system():
    """載入以圖搜圖系統 (Feature Extractor + FAISS Searcher)"""
    # 1. 特徵提取器 (使用 ResNet50)
    # 注意：這裡建議用跟訓練時一樣的權重，效果最好
    try:
        extractor = FeatureExtractor(
            model_name='resnet50', 
            weight_path='./checkpoints/resnet50_best.pth'
        )
    except Exception as e:
        st.error(f"無法載入特徵提取器: {e}")
        return None, None

    # 2. 搜尋引擎 (讀取 .npy)
    try:
        searcher = ImageSearcher(
            feature_path='data/processed/wikiart_features.npy',
            path_file_path='data/processed/wikiart_paths.npy'
        )
        return extractor, searcher
    except Exception as e:
        st.warning(f"⚠️ 無法載入搜尋資料庫 (若是第一次執行，請先跑 build_features.py): {e}")
        return extractor, None

def predict_single(net, img_tensor):
    """單一模型預測"""
    with torch.no_grad():
        outputs = net(img_tensor)
        probs = F.softmax(outputs, dim=1)
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
    
    avg_probs = total_probs / len(models)
    return avg_probs[0]

# --- 3. 側邊欄設定 ---
st.sidebar.header("⚙️ 設定面板")

# 選擇模式 (只影響 Tab 1)
model_mode = st.sidebar.radio(
    "選擇辨識模型：",
    ("單一模型 (ResNet50)", "三合一集成 (Ensemble)")
)

st.sidebar.info(
    """
    **功能說明：**
    1. **畫家辨識**：分辨這幅畫是誰畫的 (50位大師)。可選擇單一模型 (ResNet50), 三合一集成 (Ensemble)。
    2. **以圖搜圖**：在 8 萬張 WikiArt 資料庫中，尋找風格相似的畫作。使用單一模型 (ResNet50)。
    """
)

# --- 4. 主程式邏輯 ---

# A. 初始化所有系統
class_names = load_class_names()
num_classes = len(class_names)

# 載入辨識模型
if num_classes > 0:
    if model_mode == "單一模型 (ResNet50)":
        active_model = load_single_model(num_classes)
        ensemble_models = None
    else:
        active_model = None
        ensemble_models = load_ensemble_models(num_classes)
else:
    active_model = None
    ensemble_models = None

# 載入搜圖系統
feature_extractor, image_searcher = load_retrieval_system()

# B. 全域圖片上傳 (放在 Tab 之上)
st.markdown("---")
uploaded_file = st.file_uploader("請上傳圖片 (jpg, png, jpeg)", type=["jpg", "png", "jpeg"])

# 預備變數
image = None
img_tensor = None

if uploaded_file is not None:
    # 顯示原始圖片
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("🖼️ 原始圖片")
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, use_container_width=True)
        
        # 準備 Tensor 給模型用
        img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with col2:
        # C. 建立分頁
        tab1, tab2 = st.tabs(["🎨 畫家辨識 (Classifier)", "🔍 以圖搜圖 (Image Search)"])

        # === Tab 1: 畫家辨識 ===
        with tab1:
            st.markdown("#### 分析這幅畫的作者與流派")
            
            if st.button("🚀 開始辨識", type="primary", key="btn_classify"):
                with st.spinner(f"正在使用 {model_mode} 進行分析..."):
                    probs = None
                    if model_mode == "單一模型 (ResNet50)" and active_model:
                        probs = predict_single(active_model, img_tensor)
                    elif model_mode == "三合一集成 (Ensemble)" and ensemble_models:
                        probs = predict_ensemble(ensemble_models, img_tensor)
                    else:
                        st.error("模型載入失敗，無法預測。")

                    # 顯示結果
                    if probs is not None:
                        # 取得前 5 名
                        top5_prob, top5_idx = torch.topk(probs, 5)
                        
                        top5_data = []
                        for i in range(5):
                            class_name = class_names[top5_idx[i].item()]
                            probability = top5_prob[i].item()
                            top5_data.append({"畫家": class_name, "信心度": probability})

                        # 結果文字
                        winner = top5_data[0]
                        st.success(f"🏆 預測結果：**{winner['畫家']}** (信心度: {winner['信心度']:.1%})")
                        
                        # Altair 圖表
                        df = pd.DataFrame(top5_data)
                        chart = alt.Chart(df).mark_bar().encode(
                            x=alt.X('信心度', axis=alt.Axis(format='%'), scale=alt.Scale(domain=[0, 1])),
                            y=alt.Y('畫家', sort='-x'),
                            color=alt.Color('信心度', scale=alt.Scale(scheme='blues')),
                            tooltip=['畫家', alt.Tooltip('信心度', format='.1%')]
                        ).properties(height=300)
                        
                        st.altair_chart(chart, use_container_width=True)

        # === Tab 2: 以圖搜圖 ===
        with tab2:
            st.markdown("#### 從 WikiArt 資料庫 (80,000+) 尋找相似畫作")
            
            if image_searcher is None:
                st.warning("⚠️ 搜尋資料庫尚未建立。請確認 `data/processed/` 下是否有 `.npy` 檔案。")
            else:
                if st.button("🔍 尋找相似畫作", key="btn_search"):
                    with st.spinner("正在提取特徵並比對 8 萬張畫作..."):
                        # 1. 提取特徵 (使用 feature_extractor)
                        # 注意：extract 方法預期的是 PIL Image，不需要轉 Tensor
                        query_vec = feature_extractor.extract(image)
                        
                        # 2. 執行搜尋 (找 Top 6)
                        results = image_searcher.search(query_vec, k=6)
                    
                    st.success("搜尋完成！以下是風格最相近的畫作：")
                    
                    # 3. 顯示結果 (3欄 x 2列)
                    res_cols = st.columns(3)
                    for i, (path, score) in enumerate(results):
                        with res_cols[i % 3]:
                            try:
                                # 顯示圖片
                                res_img = Image.open(path)
                                st.image(res_img, use_container_width=True)
                                
                                # 解析檔名 (假設格式: Artist_Name_Title.jpg)
                                file_name = os.path.basename(path)
                                # 嘗試簡單分割，如果檔名很亂也沒關係，直接顯示檔名
                                caption_txt = f"**Top {i+1}**\n\n相似度: {score:.3f}\n📂 {file_name}"
                                st.caption(caption_txt)
                                
                            except Exception as e:
                                st.error(f"圖片讀取錯誤: {path}")

else:
    # 歡迎畫面
    st.info("👈 請從左側或上方上傳圖片以開始分析")