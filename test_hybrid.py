import torch

from models.hybrid import HybridModel

model = HybridModel()

x = torch.randn(2,3,224,224)

y = model(x)

print(y.shape)