import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset

# --- 全域設定 ---
# ImageNet 的標準化參數 (Mean & Std)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_dataloaders(data_dir, batch_size=32, val_split=0.2, num_workers=4): # 建議 worker 改 4
    """
    建立訓練與驗證的 DataLoaders，並確保資料增強只套用於訓練集。
    """
    
    # 1. 定義影像轉換 (Transforms)
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)), # 明確指定長寬，有些特殊的圖 Resize 單一數值可能會出錯
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1), 
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)) 
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    # 2. 為了取得類別名稱和長度，先讀一次
    if not os.path.exists(data_dir):
        print(f"❌ 錯誤：找不到路徑 {data_dir}")
        return None, None, None

    # 使用 transform=None 讀取最快，因為我們只想要 metadata
    base_dataset = datasets.ImageFolder(root=data_dir, transform=None)
    class_names = base_dataset.classes
    print(f"✅ 成功讀取資料集！共發現 {len(class_names)} 位畫家 (類別)。")
    
    # 3. 計算切分大小
    val_size = int(len(base_dataset) * val_split)
    train_size = len(base_dataset) - val_size
    
    # 4. 產生固定的隨機索引 (關鍵步驟：固定種子)
    generator = torch.Generator().manual_seed(42)
    
    # 我們只需要這裡產生的 "indices"
    train_subset_temp, val_subset_temp = random_split(
        base_dataset, 
        [train_size, val_size],
        generator=generator
    )
    
    train_indices = train_subset_temp.indices
    val_indices = val_subset_temp.indices

    # 5. 建立兩個獨立的 ImageFolder (平行宇宙策略)
    # 這裡雖然建立了兩個 Dataset 物件，但 ImageFolder 是 Lazy Loading (懶載入)，
    # 它只會存路徑字串，不會真的把幾萬張圖讀進 RAM，所以不會浪費兩倍記憶體。
    train_dataset_full = datasets.ImageFolder(root=data_dir, transform=train_transforms)
    val_dataset_full = datasets.ImageFolder(root=data_dir, transform=val_transforms)

    # 6. 使用 Subset 進行映射
    train_dataset = Subset(train_dataset_full, train_indices)
    val_dataset = Subset(val_dataset_full, val_indices)

    print(f"   總圖片數: {len(base_dataset)} 張")
    print(f"   訓練集數量: {len(train_dataset)} 張 (已套用資料增強)")
    print(f"   驗證集數量: {len(val_dataset)} 張 (無資料增強)")

    # 7. 建立 DataLoader (加入效能優化參數)
    # pin_memory=True: 加速 CPU 到 GPU 的傳輸
    # persistent_workers=True: 保持 worker 活躍，減少每個 epoch 的啟動開銷
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True, 
        persistent_workers=(num_workers > 0) 
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True, 
        persistent_workers=(num_workers > 0)
    )

    return train_loader, val_loader, class_names

# 簡易測試區
if __name__ == "__main__":
    # 根據你的專案結構，這裡的路徑通常是 data/raw 或者是 data/wikiart
    # 假設我們測 WikiArt
    TEST_DATA_DIR = os.path.join("data", "wikiart")
    
    # 如果 WikiArt 不在，測 raw
    if not os.path.exists(TEST_DATA_DIR):
        TEST_DATA_DIR = os.path.join("data", "raw")

    print(f"正在測試讀取路徑: {TEST_DATA_DIR}")
    
    if os.path.exists(TEST_DATA_DIR):
        train_loader, val_loader, classes = get_dataloaders(TEST_DATA_DIR, batch_size=4)
        if train_loader:
            # 測試取一個 batch 看看形狀對不對
            images, labels = next(iter(train_loader))
            print(f"測試成功！Batch 形狀: {images.shape}") # 應該是 [4, 3, 224, 224]
            print(f"類別範例: {classes[:5]}")
    else:
        print(f"⚠️ 測試路徑不存在，請確認 data 資料夾。")