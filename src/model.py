# src/model.py
import torch.nn as nn
from torchvision import models

def get_model(num_classes, tune_backend=True):
    model = models.resnet50(weights='DEFAULT')
    
    if tune_backend:
        for name, child in model.named_children():
            # 改回：解凍 layer3, layer4 和 fc
            if name in ['layer3', 'layer4', 'fc']:
                for param in child.parameters():
                    param.requires_grad = True
            else:
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