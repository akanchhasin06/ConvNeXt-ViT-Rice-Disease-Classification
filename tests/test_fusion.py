import torch

from models.fusion import FeatureFusion

model = FeatureFusion()

conv = torch.randn(2,256)

vit = torch.randn(2,256)

out = model(conv,vit)

print(out.shape)