import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 匯入我們寫好的模組
import config
from dataset import get_dataloaders
from model import get_model

def load_trained_model(model_name, weight_path, num_classes, device):
    """載入單一模型並讀取權重"""
    print(f"正在載入 {model_name}...")
    try:
        model = get_model(num_classes, model_name=model_name, tune_backend=False) # 測試時不需要 tune
        model.load_state_dict(torch.load(weight_path, map_location=device))
        model.to(device)
        model.eval()
        return model
    except FileNotFoundError:
        print(f"⚠️ 警告: 找不到 {weight_path}，將跳過此模型。")
        return None

def ensemble_predict(models, inputs):
    """
    集成預測核心邏輯：平均機率
    """
    total_probs = None
    
    with torch.no_grad():
        for model in models:
            outputs = model(inputs)
            # 使用 Softmax 轉成機率分布 (0~1)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            if total_probs is None:
                total_probs = probs
            else:
                total_probs += probs
    
    # 取平均
    avg_probs = total_probs / len(models)
    return avg_probs

def main():
    device = torch.device(config.DEVICE)
    
    # 1. 準備資料 (只需要驗證集)
    _, val_loader, class_names = get_dataloaders(
        config.DATA_DIR, 
        batch_size=config.BATCH_SIZE,
        val_split=config.VAL_SPLIT
    )
    num_classes = len(class_names)
    
    # 2. 定義要集成的模型清單
    # 格式: (模型名稱, 權重路徑)
    model_configs = [
        ('resnet50', './checkpoints/resnet50_best.pth'),
        ('densenet121', './checkpoints/densenet121_best.pth'),
        ('efficientnet_b0', './checkpoints/efficientnet_b0_best.pth'), 
    ]
    
    # 3. 載入所有模型
    models = []
    for name, path in model_configs:
        m = load_trained_model(name, path, num_classes, device)
        if m is not None:
            models.append(m)
            
    if not models:
        print("沒有成功載入任何模型！")
        return

    print(f"🔥 開始集成測試！共使用 {len(models)} 個模型進行投票...")
    
    # 4. 開始評估
    correct = 0
    total = 0
    
    loop = tqdm(val_loader, desc="Ensemble Evaluating")
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        
        # 取得集成後的機率
        avg_probs = ensemble_predict(models, images)
        
        # 取最高分
        _, predicted = avg_probs.max(1)
        
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # 更新進度條
        acc = 100. * correct / total
        loop.set_postfix(acc=f"{acc:.2f}%")
        
    print(f"\n🏆 集成模型最終準確率: {100. * correct / total:.2f}%")

if __name__ == "__main__":
    main()