import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src import model

class FeatureExtractor:
    def __init__(self, model_name='resnet50', weight_path=None, device=None):
        self.device = device if device else torch.device(config.DEVICE)
        
        # 1. 建立模型架構 (這裡 num_classes 先隨便填，因為我們要把頭切掉)
        print(f"正在載入 {model_name} 用於特徵提取...")
        self.net = model.get_model(num_classes=49, model_name=model_name, tune_backend=False)
        
        # 2. 載入訓練好的權重 (遷移學習的核心！)
        if weight_path and os.path.exists(weight_path):
            print(f"載入權重: {weight_path}")
            # 注意：這裡要處理一下，因為我們存檔時可能包含 'fc' 層的權重
            # 但只要 key 對得起來，load_state_dict 還是可以運作
            self.net.load_state_dict(torch.load(weight_path, map_location=self.device))
        else:
            print("⚠️ 警告: 未指定權重檔或檔案不存在，將使用 ImageNet 預訓練權重 (或隨機權重)")

        # 3. 【關鍵步驟】切掉分類頭 (Head Surgery)
        # 我們不要最後的 49 類分類結果，我們要倒數第二層的 2048 維特徵
        if model_name == 'resnet50':
            # ResNet 的最後一層叫 fc，我們把它換成 Identity (直通車，不做任何事直接輸出輸入)
            self.net.fc = nn.Identity()
        elif 'densenet' in model_name:
            # DenseNet 的最後一層叫 classifier
            self.net.classifier = nn.Identity()
        elif 'efficientnet' in model_name:
            # EfficientNet 的最後一層叫 classifier
            self.net.classifier = nn.Sequential(
                nn.Flatten(), # 確保拉平
                nn.Identity()
            )

        self.net.to(self.device)
        self.net.eval() # 設定為評估模式

        # 4. 定義預處理 (必須跟訓練時一模一樣！)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def extract(self, img):
        """
        輸入: PIL Image
        輸出: numpy array (特徵向量), 形狀通常是 (2048,)
        """
        # 轉成 Tensor 並增加 Batch 維度 (3, 224, 224) -> (1, 3, 224, 224)
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # 取得特徵
            feature = self.net(img_tensor)
            
            # 如果輸出是多維的 (例如 EfficientNet 有時會保留空間維度)，要把它壓扁
            feature = torch.flatten(feature, start_dim=1)
        
        # 轉回 CPU 並變成 Numpy 格式，最後正規化 (L2 Normalize)
        # 正規化很重要！這樣之後算 Cosine Similarity 才會準
        feature = feature.cpu().numpy().flatten()
        norm = np.linalg.norm(feature)
        if norm > 0:
            feature = feature / norm
            
        return feature

# --- 測試區 (直接執行這個檔案可以測試) ---
if __name__ == "__main__":
    # 假設你有存好的 ResNet50 權重
    weight_file = './checkpoints/resnet50_best.pth'
    
    extractor = FeatureExtractor(model_name='resnet50', weight_path=weight_file)