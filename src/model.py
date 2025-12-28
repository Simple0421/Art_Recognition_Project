import torch
import torch.nn as nn
from torchvision import models

def create_classifier_head(num_ftrs, num_classes, dropout_prob=0.5):
    """
    建立統一的分類頭 (Head)
    結構: Linear -> ReLU -> Dropout -> Linear
    """
    return nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(dropout_prob),
        nn.Linear(512, num_classes)
    )

def get_model(num_classes, model_name='resnet50', tune_backend=True):
    print(f"正在建立模型: {model_name} (Fine-tuning: {tune_backend})...")
    
    model = None
    num_ftrs = 0

    # --- 1. 載入預訓練模型 ---
    if model_name == 'resnet50':
        model = models.resnet50(weights='DEFAULT')
        num_ftrs = model.fc.in_features
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights='DEFAULT')
        num_ftrs = model.classifier[1].in_features
    elif model_name == 'densenet121':
        model = models.densenet121(weights='DEFAULT')
        num_ftrs = model.classifier.in_features
    else:
        raise ValueError(f"不支援的模型名稱: {model_name}")

    # --- 2. 設定凍結/解凍邏輯 (Backbone Handling) ---
    # 先將所有參數凍結 (這是最保險的起手式)
    # 這樣如果 tune_backend=False，我們就什麼都不用做，預設就是凍結的
    for param in model.parameters():
        param.requires_grad = False

    if tune_backend:
        # 針對不同模型解凍特定的後層 (Fine-tuning)
        if model_name == 'resnet50':
            # 解凍 layer3, layer4
            for name, child in model.named_children():
                if name in ['layer3', 'layer4']:
                    for param in child.parameters():
                        param.requires_grad = True
                        
        elif model_name == 'efficientnet_b0':
            # 解凍最後幾個 features block
            for name, param in model.named_parameters():
                if "features.7" in name or "features.8" in name:
                    param.requires_grad = True
                    
        elif model_name == 'densenet121':
            # 解凍最後一個 Dense Block
            for name, param in model.named_parameters():
                if "denseblock4" in name or "norm5" in name:
                    param.requires_grad = True

    # --- 3. 替換分類頭 (Head Replacement) ---
    # 注意：新建立的層預設 requires_grad=True，所以不需要額外設定
    new_head = create_classifier_head(num_ftrs, num_classes)

    if model_name == 'resnet50':
        model.fc = new_head
    elif model_name == 'efficientnet_b0':
        model.classifier = new_head
    elif model_name == 'densenet121':
        model.classifier = new_head

    return model

# --- 簡易測試區 ---
if __name__ == "__main__":
    # 測試一下是否有參數被正確解凍
    net = get_model(50, 'resnet50', tune_backend=True)
    
    print("\n[檢查參數狀態]")
    frozen_count = 0
    trainable_count = 0
    for name, param in net.named_parameters():
        if param.requires_grad:
            trainable_count += 1
        else:
            frozen_count += 1
            
    print(f"凍結參數層數: {frozen_count}")
    print(f"可訓練參數層數: {trainable_count}")
    print("邏輯檢查: 應有部分凍結、部分可訓練。")