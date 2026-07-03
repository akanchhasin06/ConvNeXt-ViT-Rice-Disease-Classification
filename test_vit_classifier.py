import torch

from models.vit_classifier import ViTClassifier

model = ViTClassifier()

x = torch.randn(2,3,224,224)

y = model(x)

print(y.shape)