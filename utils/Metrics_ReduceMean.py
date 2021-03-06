import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


# =============================================================================
# some good resources:
# https://smp.readthedocs.io/en/latest/losses.html  
# https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook  
# =============================================================================

# For FocalLoss:
ALPHA = 0.8
GAMMA = 2    

class Dice(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(Dice, self).__init__()

    def forward(self, inputs, targets, eps = 0.0001):

        # print(f'Target Min: {torch.min(targets)}')
        # print(f'Target Max: {torch.max(targets)}')
        # print(f'Input Min: {torch.min(inputs)}')
        # print(f'Input Max: {torch.max(inputs)}')

        inputs = F.sigmoid(inputs)


        # assert torch.max(inputs)>1 or torch.min(inputs)<0 or torch.max(targets)>1 or torch.min(targets)<0, \
        #     f'Inputs and targets should be in range of [0,1]'
        inputs = (inputs>0.5).int()  
        
        #flatten label and prediction tensors
        intersection = torch.sum(inputs * targets, dim=(1,2,3))
        total = torch.sum(inputs + targets, dim=(1,2,3))
        dice = (2*intersection + eps)/(total + eps)
        
        return torch.mean(dice)



    
    
    
class IoU(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoU, self).__init__()

    def forward(self, inputs, targets, eps = 0.0001):

        inputs = F.sigmoid(inputs)
        
        # print(f'Target Min: {torch.min(targets)}')
        # print(f'Target Max: {torch.max(targets)}')
        # print(f'Input Min: {torch.min(inputs)}')
        # print(f'Input Max: {torch.max(inputs)}')
        
        # assert torch.max(inputs)>1 or torch.min(inputs)<0 or torch.max(targets)>1 or torch.min(targets)<0, \
        #     f'Inputs and targets should be in range of [0,1]'  
        inputs = (inputs>0.5).int()
        #flatten label and prediction tensors
        
        
        intersection = torch.sum(inputs * targets, dim=(1,2,3))
        total = torch.sum(inputs + targets, dim=(1,2,3))
        union = total - intersection 
        
        IoU = (intersection + eps)/(union + eps)
        
        
                
        return torch.mean(IoU) 

    

    