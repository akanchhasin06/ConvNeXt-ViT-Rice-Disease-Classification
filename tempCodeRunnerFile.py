import torch
from models.cnn import BaselineCNN
from configs.config import CHECKPOINT_NAME

model = BaselineCNN(num_classes=10)
