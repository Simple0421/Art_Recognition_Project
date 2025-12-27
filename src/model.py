import torch.nn as nn
from torchvision import models

def get_model(num_classes):
    """
    建立基於 ResNet50 的遷移學習模型
    
    Args:
        num_classes (int): 分類數量 (例如 49)
    """
    # 1. 下載預訓練的 ResNet50 (使用 ImageNet 權重)
    # weights='DEFAULT' 代表使用最新的最佳權重
    model = models.resnet50(weights='DEFAULT')
    
    # 2. 凍結所有卷積層參數 (Feature Extractor)
    # 這樣訓練時只會更新我們自己加的最後一層，速度快且效果好
    for param in model.parameters():
        param.requires_grad = False
    
    # 3. 修改全連接層 (Classifier)
    # ResNet50 最後一層叫做 'fc'，輸入特徵數是 2048
    num_ftrs = model.fc.in_features
    
    # 換成我們自己的分類層，輸出為 num_classes (49)
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.3), # 防止過擬合
        nn.Linear(512, num_classes)
    )
    
    return model