import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset

def get_dataloaders(data_dir, batch_size=32, val_split=0.2, num_workers=2):
    # 1. 定義影像轉換 (Transforms) - 維持不變
    train_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)) 
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 2. 為了取得類別名稱和長度，我們先讀一次 (用哪個 transform 沒差)
    try:
        # 這裡建立一個基礎 dataset，用來計算長度跟產生索引
        base_dataset = datasets.ImageFolder(root=data_dir, transform=None)
    except FileNotFoundError:
        print(f"❌ 錯誤：找不到路徑 {data_dir}")
        return None, None, None

    class_names = base_dataset.classes
    print(f"✅ 成功讀取資料集！共發現 {len(class_names)} 位畫家 (類別)。")
    
    # 3. 計算切分大小
    val_size = int(len(base_dataset) * val_split)
    train_size = len(base_dataset) - val_size
    
    # 4. 產生固定的隨機索引 (關鍵步驟！)
    generator = torch.Generator().manual_seed(42)
    # 我們這裡只在乎 split 產生的 "indices"，不在乎 dataset 本身
    train_subset_temp, val_subset_temp = random_split(
        base_dataset, 
        [train_size, val_size],
        generator=generator
    )
    
    # 取得切分後的索引清單
    train_indices = train_subset_temp.indices
    val_indices = val_subset_temp.indices

    # 5. 建立兩個獨立的 ImageFolder (重點在這裡！)
    # 一個專門給訓練用 (套用 train_transforms)
    train_dataset_full = datasets.ImageFolder(root=data_dir, transform=train_transforms)
    # 一個專門給驗證用 (套用 val_transforms)
    val_dataset_full = datasets.ImageFolder(root=data_dir, transform=val_transforms)

    # 6. 使用剛剛固定的索引，分別從兩個 dataset 中取資料
    # 這裡使用 Subset 來把索引套用到對應的 dataset 上
    train_dataset = Subset(train_dataset_full, train_indices)
    val_dataset = Subset(val_dataset_full, val_indices)

    print(f"   總圖片數: {len(base_dataset)} 張")
    print(f"   訓練集數量: {len(train_dataset)} 張 (已套用資料增強)")
    print(f"   驗證集數量: {len(val_dataset)} 張 (無資料增強)")

    # 7. 建立 DataLoader
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