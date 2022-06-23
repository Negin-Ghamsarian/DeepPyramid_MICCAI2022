# -*- coding: utf-8 -*-
# Implementation of DeepPyram, MICCAI 2022, with VGG16 backbone

import torchvision.models as models
from torchsummary import summary
import torch.nn as nn
import torch
from torchvision.ops import DeformConv2d

class VGG_Separate(nn.Module):
    def __init__(self):
        super(VGG_Separate,self).__init__()

        vgg_model = models.vgg16(pretrained=True)
        self.Conv1 = nn.Sequential(*list(vgg_model.features.children())[0:4])
        self.Conv2 = nn.Sequential(*list(vgg_model.features.children())[4:9]) 
        self.Conv3 = nn.Sequential(*list(vgg_model.features.children())[9:16])
        self.Conv4 = nn.Sequential(*list(vgg_model.features.children())[16:23])
        self.Conv5 = nn.Sequential(*list(vgg_model.features.children())[23:30])

    def forward(self,x):

        out1 = self.Conv1(x)
        out2 = self.Conv2(out1)
        out3 = self.Conv3(out2)
        out4 = self.Conv4(out3)
        out5 = self.Conv5(out4)

        return out1, out2, out3, out4, out5
        
                


class PoolUp(nn.Module):
      def __init__(self, input_channels, pool_kernel_size, reduced_channels):
          super().__init__()

          self.pool = nn.AvgPool2d(kernel_size=pool_kernel_size, stride = pool_kernel_size)
          self.up = nn.Upsample(scale_factor=pool_kernel_size)
          
      def forward(self,x):

          y = self.pool(x)
          y = self.up(y)
          
          return y


class Pool_pixelWise(nn.Module):
      def __init__(self, input_channels, pool_kernel_size, reduced_channels):
          super().__init__()

          self.pool = nn.AvgPool2d(kernel_size=pool_kernel_size, stride = 1, padding = pool_kernel_size//2)
          self.conv = nn.Conv2d(input_channels, reduced_channels, kernel_size=1, padding=0)
          
          
      def forward(self,x):

          y = self.pool(x)
          y = self.conv(y)
          
          return y



class PVF(nn.Module):
    def __init__(self, input_channels, pool_sizes, dim):
        super().__init__()
        
        reduced_channels = input_channels//8
        output_channels = input_channels//2
        
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1, padding=0)

        self.PoolUp1 = PoolUp(output_channels, pool_sizes[0], reduced_channels)
        self.PoolUp2 = nn.AvgPool2d(kernel_size=pool_sizes[1], padding = (pool_sizes[1])//2, stride=1)
        self.PoolUp3 = nn.AvgPool2d(kernel_size=pool_sizes[2], padding = (pool_sizes[2])//2, stride=1)
        self.PoolUp4 = nn.AvgPool2d(kernel_size=pool_sizes[3], padding = (pool_sizes[3])//2, stride=1)

        self.conv1 = nn.Conv2d(output_channels, reduced_channels, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(output_channels, reduced_channels, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(output_channels, reduced_channels, kernel_size=1, padding=0)
        self.conv4 = nn.Conv2d(output_channels, reduced_channels, kernel_size=1, padding=0)
        
        self.fc = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, groups=input_channels)
        self.norm = nn.LayerNorm([dim,dim])

        
    def forward(self,x):


        y = self.conv(x)
        
        glob1 = self.conv1(self.PoolUp1(y))       
        glob2 = self.conv2(self.PoolUp2(y))       
        glob3 = self.conv3(self.PoolUp3(y))       
        glob4 = self.conv4(self.PoolUp4(y))
     
        concat = torch.cat([y,glob1,glob2,glob3,glob4],1)       
        fully_connected = self.fc(concat)       
        z = self.norm(fully_connected)
        
        return z




class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, dilations, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DPR(in_channels, out_channels, dilations)
            

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)            


    
class DeformableBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilate):
        super().__init__()
        
        self.offset = nn.Conv2d(in_channels, 2*kernel_size*kernel_size, kernel_size=3, padding = 1, dilation = 1)
        self.tan = nn.Hardtanh()
        self.deform = DeformConv2d(in_channels, out_channels, kernel_size = 3, stride = (1,1), 
                                    padding = dilate, dilation = dilate)
        
    def forward(self,x):
        
        off = self.offset(x)
        off1 = self.tan(off)
        out = self.deform(x, off1)

        return out


class DPR(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, dilations):
        super().__init__()
           
        self.conv0 = nn.Conv2d(in_channels, in_channels//4, kernel_size=3, padding=1)
        self.conv1 = DeformableBlock(in_channels, in_channels//8, kernel_size=3, dilate = dilations[0])
        self.conv2 = DeformableBlock(in_channels, in_channels//8, kernel_size=3, dilate = dilations[1])
        
        self.out = nn.Sequential(
                   nn.BatchNorm2d(in_channels//2),
                   nn.ReLU(inplace=True),
                   nn.Conv2d(in_channels//2, out_channels, kernel_size=3, padding=1),
                   nn.BatchNorm2d(out_channels),
                   nn.ReLU(inplace=True))
   

    def forward(self, x):
        
        x0 = self.conv0(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        
        y = torch.cat([x0, x1, x2], dim=1)
        y1 = self.out(y)
        
        return y1
        
        
        
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)        


class DeepPyram_VGG16(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, Pyramid_Loss=True):
        super(DeepPyram_VGG16, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.Pyramid_Loss = Pyramid_Loss

        self.Backbone = VGG_Separate()

        self.PVF1 = PVF(input_channels=512, pool_sizes=[32, 9, 7, 5], dim=32)
        self.PVF2 = PVF(input_channels=256, pool_sizes=[64, 9, 7, 5], dim=64)
        self.PVF3 = PVF(input_channels=128, pool_sizes=[128, 9, 7, 5], dim=128)
        self.PVF4 = PVF(input_channels=64, pool_sizes=[256, 9, 7, 5], dim=256)

        self.up1 = Up(1024, 256, [3, 6, 7], bilinear)
        self.up2 = Up(512, 128, [3, 6, 7], bilinear)
        self.up3 = Up(256, 64, [3, 6, 7], bilinear)
        self.up4 = Up(128, 32, [3, 6, 7], bilinear)

        self.outc = OutConv(32, n_classes)


        if self.Pyramid_Loss:
            print("Pyramid Loss activated in the network")
            self.mask1 = nn.Conv2d(256, 1, kernel_size=3, padding=1)
            self.mask2 = nn.Conv2d(128, 1, kernel_size=3, padding=1)
            self.mask3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

        else:
            print("Pyramid Loss is deactivated in the network")
            print("You can activate it by setting Pyramid_Loss=True when initializing the network")



    def forward(self, x):
        
        out5, out4, out3, out2, out1 = self.Backbone(x)        

        out1 = self.PVF1(out1)
        
        x1 = self.up1(out1, out2)
        x1 = self.PVF2(x1)

        x2 = self.up2(x1, out3)
        x2 = self.PVF3(x2)
 
        x3 = self.up3(x2, out4)
        x3 = self.PVF4(x3)

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
    
    model_without_PyramidLoss = DeepPyram_VGG16(n_channels=3, n_classes=1, bilinear=True, Pyramid_Loss=False)
    y = model_without_PyramidLoss(template)
    print(f'Output Shape: {y.shape}')

    print('Model summary without pyramid loss:')
    print(summary(model_without_PyramidLoss, (3,512,512))) 


    model_with_PyramidLoss = DeepPyram_VGG16(n_channels=3, n_classes=1, bilinear=True, Pyramid_Loss=True)
    y, y1, y2, y3 = model_with_PyramidLoss(template)
    print(f'Output Shape of the main branch: {y.shape}')
    print(f'Output Shape of the first branch of the pyramid loss module: {y1.shape}')
    print(f'Output Shape of the second branch of the pyramid loss module: {y2.shape}')
    print(f'Output Shape of the third branch of the pyramid loss module: {y3.shape}')

    print('Model summary with pyramid loss:')
    print(summary(model_with_PyramidLoss, (3,512,512))) 
