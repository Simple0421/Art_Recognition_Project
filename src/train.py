import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import random
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler  # [優化] 使用內建的 AMP 加速

# 將目前檔案所在的上一層目錄 (專案根目錄) 加入 Python 搜尋路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from dataset import get_dataloaders
from model import get_model

# [優化] 固定亂數種子，確保實驗結果一致
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 

def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    loop = tqdm(loader, desc="Training", leave=False)
    
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # [優化] 開啟混合精度 (減少記憶體佔用，加快速度)
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        # [優化] 使用 Scaler 進行反向傳播
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        loop.set_postfix(loss=loss.item())
        
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
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
    # 0. 固定種子
    seed_everything(42)
    
    # 1. 準備資料
    print("正在讀取資料...")
    train_loader, val_loader, class_names = get_dataloaders(
        config.DATA_DIR, 
        batch_size=config.BATCH_SIZE,
        val_split=config.VAL_SPLIT
    )
    
    num_classes = len(class_names)
    print(f"類別數量: {num_classes}")
    
    # 2. 準備模型
    print(f"正在建立模型: {config.MODEL_NAME} (微調模式)...")
    device = torch.device(config.DEVICE)
    model = get_model(num_classes, model_name=config.MODEL_NAME, tune_backend=True).to(device)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS)
    
    # [優化] 初始化 Scaler (給 AMP 用)
    scaler = GradScaler()

    # [優化] 設定 Early Stopping 參數
    patience = 5  # 容忍幾個 Epoch 沒有進步
    counter = 0   # 目前累積幾次沒進步
    best_acc = 0.0
    
    print(f"開始訓練，共 {config.NUM_EPOCHS} 個 Epochs...")
    
    for epoch in range(config.NUM_EPOCHS):
        # 訓練與驗證
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # 更新學習率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 顯示結果
        print(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}] "
              f"LR: {current_lr:.6f} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
        
        # [優化] 儲存最佳模型與 Early Stopping 判斷
        if val_acc > best_acc:
            best_acc = val_acc
            counter = 0 # 重置計數器
            if not os.path.exists(config.CHECKPOINT_DIR):
                os.makedirs(config.CHECKPOINT_DIR)
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"🚀 發現最佳模型 (Acc: {best_acc:.2f}%)，已儲存。")
        else:
            counter += 1
            print(f"⚠️ Validation Accuracy 未提升 ({counter}/{patience})")
            if counter >= patience:
                print("🛑 觸發 Early Stopping，提早結束訓練。")
                break
                
    print("訓練結束！")

if __name__ == "__main__":
    main()