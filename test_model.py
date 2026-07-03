import torch
from models.cnn import BaselineCNN

model = BaselineCNN(num_classes=10)

x = torch.randn(2, 3, 224, 224)

output = model(x)

print(output.shape)