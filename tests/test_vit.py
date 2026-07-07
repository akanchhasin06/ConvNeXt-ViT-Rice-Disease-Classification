import torch

from models.vit_encoder import ViTEncoder

model = ViTEncoder()

x = torch.randn(2,3,224,224)

y = model(x)

print(type(y))
print(y.shape)