import torch

# 路徑設定
DATA_DIR = "./data/raw/images"
CHECKPOINT_DIR = "./checkpoints"
MODEL_SAVE_PATH = "./checkpoints/resnet50_best.pth"

# 訓練參數
BATCH_SIZE = 32          # 一次讀幾張圖 (如果顯卡記憶體不夠，改小成 16 或 8)
LEARNING_RATE = 0.001    # 學習率
NUM_EPOCHS = 50          # 總共訓練幾輪
VAL_SPLIT = 0.2          # 驗證集比例

# 新增：要訓練的模型名稱 (可選: 'resnet50', 'efficientnet_b0', 'densenet121')
MODEL_NAME = "densenet121" 

# 修改：存檔路徑要跟著模型名稱變
MODEL_SAVE_PATH = f"./checkpoints/{MODEL_NAME}_best.pth"

# 硬體設定
# 自動偵測是否有 GPU，如果沒有就用 CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"