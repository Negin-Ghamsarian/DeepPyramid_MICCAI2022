

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.io import read_image
import os



def IoU(preds, targets, eps = 0.0001):


    inputs = F.sigmoid(preds)
        
    inputs = (inputs>0.5).int()
    #flatten label and prediction tensors

    intersection = torch.sum(inputs * targets, dim=(1,2,3))
    total = torch.sum(inputs + targets, dim=(1,2,3))
    union = total - intersection 

    IoU = (intersection + eps)/(union + eps)

        
    return torch.mean(IoU)  



prediction_path = ''
groundtruth_path = ''

imdir = os.listdir(prediction_path)

for i in range(len(imdir)):
  
    