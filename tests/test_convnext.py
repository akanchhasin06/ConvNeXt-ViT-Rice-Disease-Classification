import torch

from models.convnext_encoder import ConvNeXtEncoder

model = ConvNeXtEncoder()

x = torch.randn(2,3,224,224)

y = model(x)

print(type(y))
print(y.shape)