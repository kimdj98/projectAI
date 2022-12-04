import torch
from torchinfo import summary
from torch import nn
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

# change conv1 stride into (1,1)
temp = model.conv1.weight # store weight
model.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
model.conv1.weight = temp # restore weight

temp = model.layer2[0].conv1.weight
model.layer2[0].conv1 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
model.layer2[0].conv1.weight = temp

temp = model.layer2[0].downsample[0].weight
model.layer2[0].downsample[0] = nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
model.layer2[0].downsample[0].weight = temp

model.eval()
