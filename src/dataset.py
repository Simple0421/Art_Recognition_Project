import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_dataloaders(data_dir, batch_size=32, val_split=0.2, num_workers=2):
    """
    建立訓練與驗證的 DataLoaders
    
    Args:
        data_dir (str): 圖片資料夾路徑 (例如: 'data/raw/images')
        batch_size (int): 批次大小
        val_split (float): 驗證集比例 (預設 0.2，即 20% 驗證，80% 訓練)
        num_workers (int): 資料讀取執行緒數量 (Windows 建議設 0 或 2)
    """
    
    # 1. 定義影像轉換 (Transforms)
    # src/dataset.py 修改 transform
    train_transforms = transforms.Compose([
        transforms.Resize(256),             # 先縮放到短邊為 256 (保持比例)
        transforms.RandomCrop(224),         # 隨機裁切 224x224 (增加變化性)
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),      # 增加旋轉角度
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3), # 加強色彩擾動
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(256),             # 先縮放
        transforms.CenterCrop(224),         # 取中間最精華的部分
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 2. 讀取完整資料集
    # 注意：這裡我們先用 train_transforms 讀取，切分後再手動覆寫驗證集的 transform
    try:
        full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transforms)
    except FileNotFoundError:
        print(f"❌ 錯誤：找不到路徑 {data_dir}，請確認 Kaggle 資料是否已放入 data/raw/images")
        return None, None, None

    class_names = full_dataset.classes
    print(f"✅ 成功讀取資料集！共發現 {len(class_names)} 位畫家 (類別)。")
    print(f"   總圖片數: {len(full_dataset)} 張")

    # 3. 切分訓練集與驗證集
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # 重要技巧：修正驗證集的 Transform (避免驗證時也被「資料增強」干擾)
    # 這是 PyTorch subset 的一個小坑，標準解法是這樣：
    val_dataset.dataset.transform = val_transforms 

    print(f"   訓練集數量: {len(train_dataset)} 張")
    print(f"   驗證集數量: {len(val_dataset)} 張")

    # 4. 建立 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, class_names

# 簡易測試區 (當直接執行此檔案時會跑這段)
if __name__ == "__main__":
    # 假設你的路徑結構是 Art_Recognition_Project/data/raw/images/Van_Gogh...
    # Windows 路徑注意：請確保路徑分隔符號正確
    TEST_DATA_DIR = os.path.join("data", "raw", "images")
    
    if os.path.exists(TEST_DATA_DIR):
        get_dataloaders(TEST_DATA_DIR)
    else:
        print(f"⚠️ 測試路徑不存在: {TEST_DATA_DIR}")
        print("請建立資料夾並放入 Kaggle 圖片")