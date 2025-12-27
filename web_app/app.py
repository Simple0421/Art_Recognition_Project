import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import sys
import os

# --- 路徑設定 (解決 Python 找不到 src 的問題) ---
# 取得目前檔案 (app.py) 的上一層目錄 (專案根目錄)
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

# 匯入我們寫好的模組
import config
from src.model import get_model

# --- 1. 載入模型 (使用快取避免每次重新整理都要重載) ---
@st.cache_resource
def load_trained_model():
    # 建立空的模型架構
    # 注意：這裡的類別數量必須跟訓練時一樣 (看你的 dataset.py print 出來是多少)
    # 你之前說是 49 位畫家，所以這裡填 49
    model = get_model(num_classes=49) 
    
    # 載入權重
    try:
        # map_location=torch.device('cpu') 確保就算沒 GPU 的電腦也能跑介面
        checkpoint = torch.load(config.MODEL_SAVE_PATH, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint)
        model.eval() # 設定為評估模式
        return model
    except FileNotFoundError:
        st.error(f"找不到模型檔案：{config.MODEL_SAVE_PATH}，請先執行 src/train.py")
        return None

# --- 2. 圖片預處理 (跟驗證集一樣的邏輯) ---
def process_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0) # 增加一個 batch 維度 [1, 3, 224, 224]

# --- 3. 取得類別名稱 (Hardcode 或讀取資料夾) ---
# 為了方便，我們先從你的 dataset.py 執行結果複製過來，或者動態讀取
# 這裡示範動態讀取 data/raw/images 下的資料夾名稱
def get_class_names():
    try:
        class_names = sorted(os.listdir(config.DATA_DIR))
        return class_names
    except:
        return [f"Class {i}" for i in range(49)]

# --- 主程式 ---
def main():
    st.set_page_config(page_title="名畫辨識系統", page_icon="🎨")
    
    st.title("🎨 藝術名畫辨識系統")
    st.write("上傳一張畫作，AI 將告訴你這是哪位大師的風格！")
    
    # 側邊欄：顯示模型狀態
    st.sidebar.header("模型狀態")
    device = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"
    st.sidebar.text(f"運行裝置: {device}")
    
    # 載入模型
    model = load_trained_model()
    class_names = get_class_names()
    
    if model is None:
        return

    # 圖片上傳區
    uploaded_file = st.file_uploader("請選擇圖片...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # 顯示圖片
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='上傳的圖片', use_container_width=True)
        
        # 進行辨識
        if st.button('🔍 開始辨識'):
            with st.spinner('AI 正在鑑賞中...'):
                # 1. 處理圖片
                img_tensor = process_image(image)
                
                # 2. 推論
                with torch.no_grad():
                    outputs = model(img_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                
                # 3. 取得最高分的結果
                top_prob, top_catid = torch.topk(probabilities, 1)
                
                predicted_class = class_names[top_catid.item()]
                confidence = top_prob.item() * 100
                
                # 4. 顯示結果
                st.success(f"這幅畫最像是 **{predicted_class}** 的作品")
                st.info(f"信心指數: {confidence:.2f}%")
                
                # (進階) 顯示前三名可能性
                st.subheader("📊 其他可能性")
                top3_prob, top3_catid = torch.topk(probabilities, 3)
                for i in range(3):
                    cls = class_names[top3_catid[0][i].item()]
                    prob = top3_prob[0][i].item() * 100
                    st.write(f"{i+1}. **{cls}**: {prob:.2f}%")
                    st.progress(int(prob))

if __name__ == "__main__":
    main()