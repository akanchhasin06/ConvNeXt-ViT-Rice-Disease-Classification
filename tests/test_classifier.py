import torch

from models.convnext_classifier import ConvNeXtClassifier

model = ConvNeXtClassifier(num_classes=10)

x = torch.randn(2,3,224,224)

y = model(x)

print(y.shape)