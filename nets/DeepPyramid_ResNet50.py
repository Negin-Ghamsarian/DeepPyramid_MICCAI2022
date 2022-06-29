# Implementation of DeepPyramid, MICCAI 2022, with ResNet50 backbone
from __future__ import absolute_import 
import torchvision.models as models
from torchsummary import summary
import torch.nn as nn
import torch
from torchvision.ops import DeformConv2d

from .NetModules_utils import *
class DeepPyramid_ResNet50(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, Pyramid_Loss=True):
        super(DeepPyramid_ResNet50, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.Pyramid_Loss = Pyramid_Loss

        self.Backbone = ResNet50_Separate()
        
        self.firstUpSample = nn.Upsample(scale_factor=2)
        
        self.glob1 = PVF(input_channels=2048, pool_sizes=[32, 9, 7, 5], dim=32)
        
        self.glob2 = PVF(input_channels=512, pool_sizes=[64, 9, 7, 5], dim=64)
        self.glob3 = PVF(input_channels=256, pool_sizes=[128, 9, 7, 5], dim=128)
        self.glob4 = PVF(input_channels=64, pool_sizes=[256, 9, 7, 5], dim=256)

        self.up1 = Up(2048+1024, 512, [3, 6, 7], bilinear)
        self.up2 = Up(1024, 256, [3, 6, 7], bilinear)
        self.up3 = Up(512, 64, [3, 6, 7], bilinear)
        self.up4 = Up(128, 32, [3, 6, 7], bilinear)

        self.outc = OutConv(32, n_classes)

        if self.Pyramid_Loss:
            print("Pyramid Loss activated in the network")
            self.mask1 = nn.Conv2d(512, 1, kernel_size=3, padding=1)
            self.mask2 = nn.Conv2d(256, 1, kernel_size=3, padding=1)
            self.mask3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

        else:
            print("Pyramid Loss is deactivated in the network")
            print("You can activate it by setting Pyramid_Loss=True when initializing the network")


    def forward(self, x):
        
        x = self.firstUpSample(x)
        
        out5, out4, out3, out2, out1 = self.Backbone(x)

        out1 = self.glob1(out1)
        
        x1 = self.up1(out1, out2)
        x1 = self.glob2(x1)
        
        x2 = self.up2(x1, out3)
        x2 = self.glob3(x2)

        x3 = self.up3(x2, out4)
        x3 = self.glob4(x3)

        x4 = self.up4(x3, out5)
        
        logits = self.outc(x4)
      
        if self.Pyramid_Loss:
            mask1 = self.mask1(x1)
            mask2 = self.mask2(x2)
            mask3 = self.mask3(x3)

            return logits, mask1, mask2, mask3

        else:
            return logits    

    
    
    
if __name__ == '__main__':

    template = torch.ones((1, 3, 512, 512))
    
    model_without_PyramidLoss = DeepPyramid_ResNet50(n_channels=3, n_classes=1, bilinear=True, Pyramid_Loss=False)
    y = model_without_PyramidLoss(template)
    print(f'Output Shape: {y.shape}')

    print('Model summary without pyramid loss:')
    print(summary(model_without_PyramidLoss, (3,512,512))) 


    model_with_PyramidLoss = DeepPyramid_ResNet50(n_channels=3, n_classes=1, bilinear=True, Pyramid_Loss=True)
    y, y1, y2, y3 = model_with_PyramidLoss(template)
    print(f'Output Shape of the main branch: {y.shape}')
    print(f'Output Shape of the first branch of the pyramid loss module: {y1.shape}')
    print(f'Output Shape of the second branch of the pyramid loss module: {y2.shape}')
    print(f'Output Shape of the third branch of the pyramid loss module: {y3.shape}')

    print('Model summary with pyramid loss:')
    print(summary(model_with_PyramidLoss, (3,512,512))) 

    
