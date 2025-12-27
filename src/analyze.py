import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm # <--- 記得加這行在最上面
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from dataset import get_dataloaders
from ensemble_test import load_trained_model, ensemble_predict

# 設定中文字型 (以免亂碼，如果在 Windows)
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False

def plot_confusion_matrix(models, val_loader, class_names, device):
    y_true = []
    y_pred = []
    
    print("正在計算集成模型的混淆矩陣 (這可能需要幾分鐘)...")
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            
            # 使用集成預測
            avg_probs = ensemble_predict(models, images)
            _, predicted = avg_probs.max(1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            
    # 計算矩陣
    cm = confusion_matrix(y_true, y_pred)
    
    # 繪圖
    plt.figure(figsize=(16, 14)) # 圖變大一點，因為有 49 類
    # 加入 norm=LogNorm()，這會讓數值小的格子也能發光
    # 加入 cbar_kws={'label': 'Log Scale Count'} 讓右邊的色條顯示這是對數
    sns.heatmap(cm, annot=False, cmap='Blues', 
                norm=LogNorm(),  # <--- 關鍵！開啟對數模式
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('預測結果 (Predicted)')
    plt.ylabel('真實標籤 (True)')
    plt.title('集成模型混淆矩陣')
    plt.xticks(rotation=90) # X軸文字轉直，避免重疊
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("✅ 混淆矩陣已儲存為 confusion_matrix.png")

def main():
    device = torch.device(config.DEVICE)
    
    # 1. 優先取得資料集資訊 (為了拿到正確的 num_classes)
    print("正在讀取資料集資訊...")
    # 注意：這裡呼叫 get_dataloaders 會有點慢，因為要掃描資料夾，但這是必須的
    _, val_loader, class_names = get_dataloaders(config.DATA_DIR, batch_size=config.BATCH_SIZE, val_split=config.VAL_SPLIT)
    
    num_classes = len(class_names)
    print(f"偵測到 {num_classes} 位畫家。")

    # 2. 定義模型設定
    model_configs = [
        ('resnet50', './checkpoints/resnet50_best.pth'),
        ('densenet121', './checkpoints/densenet121_best.pth'),
        ('efficientnet_b0', './checkpoints/efficientnet_b0_best.pth'),
    ]
    
    # 3. 正確載入模型 (傳入 num_classes)
    models = []
    for name, path in model_configs:
        # 這裡傳入的是 num_classes (49)，而不是 model_configs 的長度 (3)
        m = load_trained_model(name, path, num_classes, device)
        if m: 
            models.append(m)
        else:
            print(f"⚠️ 跳過 {name}，請確認權重檔是否存在。")

    if not models:
        print("❌ 沒有載入任何模型，程式結束。")
        return

    # 4. 開始畫圖
    plot_confusion_matrix(models, val_loader, class_names, device)

if __name__ == "__main__":
    main()