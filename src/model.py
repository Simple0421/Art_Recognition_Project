import torch.nn as nn
from torchvision import models

def get_model(num_classes, model_name='resnet50', tune_backend=True):
    print(f"正在建立模型: {model_name}...")
    
    # --- 1. ResNet50 ---
    if model_name == 'resnet50':
        model = models.resnet50(weights='DEFAULT')
        num_ftrs = model.fc.in_features
        # 凍結/解凍邏輯 (維持你 v4 的設定)
        if tune_backend:
            for name, child in model.named_children():
                if name in ['layer3', 'layer4', 'fc']:
                    for param in child.parameters():
                        param.requires_grad = True
                else:
                    for param in child.parameters():
                        param.requires_grad = False
        else:
            for param in model.parameters():
                param.requires_grad = False
        
        # 修改分類頭
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    # --- 2. EfficientNet B0 ---
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights='DEFAULT')
        num_ftrs = model.classifier[1].in_features # EfficientNet 的最後一層叫 classifier
        
        # 簡單策略：只解凍最後幾個區塊 (features.7, features.8)
        if tune_backend:
            for param in model.parameters():
                param.requires_grad = False # 先全鎖
            
            # 解凍最後幾層
            for name, param in model.named_parameters():
                if "features.7" in name or "features.8" in name or "classifier" in name:
                    param.requires_grad = True
        
        # 修改分類頭
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    # --- 3. DenseNet 121 ---
    elif model_name == 'densenet121':
        model = models.densenet121(weights='DEFAULT')
        num_ftrs = model.classifier.in_features
        
        if tune_backend:
            for param in model.parameters():
                param.requires_grad = False
            
            # 解凍最後一個 Dense Block (denseblock4)
            for name, param in model.named_parameters():
                if "denseblock4" in name or "norm5" in name or "classifier" in name:
                    param.requires_grad = True
                    
        # 修改分類頭
        model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    else:
        raise ValueError(f"不支援的模型名稱: {model_name}")

    return model