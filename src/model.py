# src/model.py
import torch.nn as nn
from torchvision import models

def get_model(num_classes, tune_backend=True): # 新增 tune_backend 參數
    model = models.resnet50(weights='DEFAULT')
    
    # 策略改變：
    # 如果 tune_backend=True，我們不凍結最後幾個區塊 (layer3, layer4)
    # 讓模型能學習更深層的藝術特徵
    for name, child in model.named_children():
        if name in ['layer3', 'layer4', 'fc']:
            # 這些層我們要訓練
            for param in child.parameters():
                param.requires_grad = True
        else:
            # 前面的層 (conv1, layer1, layer2) 維持凍結 (基礎特徵)
            for param in child.parameters():
                param.requires_grad = False
    
    # 全連接層維持不變
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.5), # 增加 Dropout 到 0.5 加強抗過擬合
        nn.Linear(512, num_classes)
    )
    
    return model