import torch
from torchvision.models import resnet18, alexnet,vit_b_16,resnet50,vit_l_16
from brainscore_vision import load_dataset

#Download model weights and datasets from online before training


data='IMAGENET1K_V1'
resnet18(weights=data)
alexnet(weights=data)
vit_b_16(weights=data)
resnet50(weights=data)
vit_l_16(weights=data)

load_dataset('FreemanZiemba2013.public')
load_dataset('MajajHong2015.public')
