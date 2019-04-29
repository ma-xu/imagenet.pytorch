import numpy as np
import torch
import torch.nn as nn


A=torch.randn(1,224,224)
BB=nn.AdaptiveAvgPool2d(1)
CC = nn.AdaptiveAvgPool2d(2)
DD = nn.AdaptiveAvgPool2d(4)
print(BB(A))
print(CC(A))
print(CC(DD(A)))
