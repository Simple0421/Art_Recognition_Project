import os
import sys
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import glob
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 引入我們剛剛寫好的特徵提取器
from src.feature_extractor import FeatureExtractor
import config

# 設定 WikiArt 資料夾路徑 (請確認這裡的路徑是對的)
# 假設你解壓縮在 data/wikiart 且裡面有很多子資料夾
WIKIART_ROOT = os.path.join('data', 'wikiart') 
SAVE_DIR = os.path.join('data', 'processed')

# 確保輸出資料夾存在
os.makedirs(SAVE_DIR, exist_ok=True)

class WikiArtDataset(Dataset):
    def __init__(self, file_paths, transform):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        try:
            # 開啟圖片並轉為 RGB (避免有些圖是 CMYK 或 Grayscale 報錯)
            img = Image.open(path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, path, True # True 代表讀取成功
        except Exception as e:
            # 如果圖片壞了，回傳一個全黑圖與 False 標記
            # 這是為了讓 DataLoader 不會崩潰
            # 我們會在之後過濾掉這些失敗的資料
            dummy_img = torch.zeros(3, 224, 224) 
            return dummy_img, path, False

def get_all_image_paths(root_dir):
    """遞迴搜尋所有圖片檔案"""
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    print(f"正在掃描資料夾: {root_dir} ...")
    
    # 使用 glob 遞迴搜尋 (這可能需要一點時間)
    for ext in extensions:
        # **/* 代表搜尋所有子資料夾
        image_paths.extend(glob.glob(os.path.join(root_dir, '**', ext), recursive=True))
    
    return sorted(image_paths)

def main():
    device = torch.device(config.DEVICE)
    print(f"使用裝置: {device}")

    # 1. 取得所有圖片路徑
    all_paths = get_all_image_paths(WIKIART_ROOT)
    if len(all_paths) == 0:
        print(f"❌ 錯誤: 在 {WIKIART_ROOT} 找不到任何圖片！請檢查路徑。")
        return
    print(f"共找到 {len(all_paths)} 張圖片。")

    # 2. 初始化特徵提取器
    # 這裡我們用 ResNet50 (不載入你的 50 人權重也沒關係，用 ImageNet 預訓練就很強了)
    # 如果你想用你訓練好的權重，把 weight_path 加回去
    extractor = FeatureExtractor(model_name='resnet50', weight_path='./checkpoints/resnet50_best.pth', device=device)
    
    # 3. 建立 DataLoader
    # 為了加速，我們直接用 extractor 裡面的 transform
    dataset = WikiArtDataset(all_paths, transform=extractor.transform)
    
    # Batch Size 設大一點可以跑比較快 (如果顯存爆了就調小，例如 32)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    # 4. 開始提取特徵
    feature_list = []
    path_list = []
    
    print("開始提取特徵 (這需要一段時間，去喝杯咖啡吧 ☕)...")
    
    with torch.no_grad():
        for images, paths, valids in tqdm(dataloader):
            # 只有那些讀取成功的圖片才丟進去算
            valid_mask = valids.bool()
            
            if valid_mask.sum() == 0:
                continue # 這批全部壞掉 (機率很低)

            # 過濾壞圖
            valid_images = images[valid_mask].to(device)
            valid_paths = np.array(paths)[valid_mask.numpy()] # 轉 numpy 方便 indexing

            # --- 核心運算 ---
            # 因為 extractor.extract 是設計給單張圖的，我們這裡稍微手動操作一下以支援 Batch
            # 直接使用 extractor.net 進行 Forward
            features = extractor.net(valid_images)
            
            # 如果是多維 (如 EfficientNet)，壓扁
            features = torch.flatten(features, start_dim=1)
            
            # 轉回 CPU Numpy
            features = features.cpu().numpy()
            
            # L2 Normalize (批次處理版)
            # axis=1 代表對每一個向量做正規化
            norms = np.linalg.norm(features, axis=1, keepdims=True)
            features = features / (norms + 1e-10) # 加上 epsilon 避免除以 0

            # 存起來
            feature_list.append(features)
            path_list.extend(valid_paths)

    # 5. 整合與存檔
    print("正在整合數據...")
    all_features = np.concatenate(feature_list, axis=0)
    all_paths = np.array(path_list)

    print(f"提取完成！")
    print(f"特徵形狀: {all_features.shape}") # 應該是 (N, 2048)
    print(f"路徑數量: {len(all_paths)}")

    # 存成 .npy 檔 (這就是我們的資料庫)
    feat_save_path = os.path.join(SAVE_DIR, 'wikiart_features.npy')
    path_save_path = os.path.join(SAVE_DIR, 'wikiart_paths.npy')
    
    np.save(feat_save_path, all_features)
    np.save(path_save_path, all_paths)
    
    print(f"✅ 檔案已儲存至:\n  - {feat_save_path}\n  - {path_save_path}")

if __name__ == "__main__":
    main()