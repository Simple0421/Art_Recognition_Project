import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
from tqdm import tqdm  # 進度條顯示

# 將目前檔案所在的上一層目錄 (專案根目錄) 加入 Python 搜尋路徑
# 這樣 Python 才看得到外面的 config.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 匯入我們寫好的模組
import config
from dataset import get_dataloaders
from model import get_model

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train() # 設定為訓練模式
    running_loss = 0.0
    correct = 0
    total = 0
    
    # tqdm 只是為了讓 Terminal 有漂亮的進度條
    loop = tqdm(loader, desc="Training", leave=False)
    
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        
        # 1. 清空梯度
        optimizer.zero_grad()
        
        # 2. 前向傳播 (Forward)
        outputs = model(images)
        
        # 3. 計算 Loss
        loss = criterion(outputs, labels)
        
        # 4. 反向傳播 (Backward)
        loss.backward()
        
        # 5. 更新參數
        optimizer.step()
        
        # 統計數據
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # 更新進度條資訊
        loop.set_postfix(loss=loss.item())
        
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
    model.eval() # 設定為評估模式 (不更新參數、不Dropout)
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad(): # 驗證時不需要計算梯度，節省記憶體
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def main():
    # 1. 準備資料
    print("正在讀取資料...")
    train_loader, val_loader, class_names = get_dataloaders(
        config.DATA_DIR, 
        batch_size=config.BATCH_SIZE,
        val_split=config.VAL_SPLIT
    )
    
    num_classes = len(class_names)
    print(f"類別數量: {num_classes}")
    
    # 2. 準備模型 (加入 tune_backend=True 開啟微調模式)
    print(f"正在建立模型: {config.MODEL_NAME} (微調模式)...")
    device = torch.device(config.DEVICE)
    
    # 傳入模型名稱
    model = get_model(num_classes, model_name=config.MODEL_NAME, tune_backend=True).to(device)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # --- 自動化 Optimizer 設定 ---
    # 因為不同模型的層名稱不一樣 (layer3 vs features.7 vs denseblock4)
    # 我們改用 "過濾法"：只要是 requires_grad=True 的參數，就丟進去練
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    optimizer = optim.Adam(trainable_params, lr=config.LEARNING_RATE, weight_decay=1e-4)
    
    # (選用) 加入學習率排程器：讓 LR 隨著 Epoch 慢慢變小
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS)

    # 4. 開始訓練
    print(f"開始訓練，共 {config.NUM_EPOCHS} 個 Epochs...")
    best_acc = 0.0
    
    for epoch in range(config.NUM_EPOCHS):
        # 訓練階段
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # 驗證階段
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step() # <--- 每個 Epoch 結束後更新學習率
        
        print(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # 儲存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            if not os.path.exists(config.CHECKPOINT_DIR):
                os.makedirs(config.CHECKPOINT_DIR)
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"🚀 發現最佳模型 (Acc: {best_acc:.2f}%)，已儲存至 {config.MODEL_SAVE_PATH}")
            
    print("訓練結束！")

if __name__ == "__main__":
    main()