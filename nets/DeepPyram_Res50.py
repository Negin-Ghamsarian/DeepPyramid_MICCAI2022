# -*- coding: utf-8 -*-
"""
Created on Wed May 12 13:42:39 2021

@author: Negin
"""
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 20:40:50 2021

@author: Negin
"""

# -*- coding: utf-8 -*-



import torchvision.models as models
from torchsummary import summary
import torch.nn as nn
import torch
from torchvision.ops import DeformConv2d




class Res50(nn.Module):
    def __init__(self):
        super(Res50,self).__init__()
        
        resnet = models.resnet50(pretrained=True)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        
        
    def forward(self, x):
         
        x0 = self.firstrelu(self.firstbn(self.firstconv(x)))
        x0m = self.firstmaxpool(x0)
        x1 = self.encoder1(x0m)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        features = self.encoder4(x3)
        
        
        '''
        print(x0.shape)
        print(x0m.shape)
        print(x1.shape)
        print(x2.shape)
        print(x3.shape)
        print(features.shape)
        '''
        
        
        return x0, x1, x2, x3, features              




class PoolUp(nn.Module):
      def __init__(self, input_channels, pool_kernel_size, reduced_channels):
          super().__init__()

          self.pool = nn.AvgPool2d(kernel_size=pool_kernel_size, stride = pool_kernel_size)
          #self.conv = nn.Conv2d(input_channels, reduced_channels, kernel_size=1, padding=0)
          self.up = nn.Upsample(scale_factor=pool_kernel_size)
          
      def forward(self,x):
          y = self.pool(x)
          #y = self.conv(y)
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


'''
class PGA(nn.Module):
    def __init__(self, input_channels, pool_sizes, dim):
        super().__init__()
        
        reduced_channels = input_channels//8
        output_channels = input_channels-(3*reduced_channels)
        
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1, padding=0)
        #self.PoolUp1 = PoolUp(output_channels, pool_sizes[0], output_channels)
        self.PoolUp2 = PoolUp(output_channels, pool_sizes[1], reduced_channels)
        self.PoolUp3 = PoolUp(output_channels, pool_sizes[2], reduced_channels)
        self.PoolUp4 = PoolUp(output_channels, pool_sizes[3], reduced_channels)
        self.fc = nn.Conv2d(input_channels, input_channels, kernel_size=1, padding=0)
        
        self.norm = nn.LayerNorm([dim,dim])
        
    def forward(self,x):
        y = self.conv(x)
        #glob1 = self.PoolUp1(y)
        #z = torch.mul(y,glob1)
        z = self.norm(y)
        
        glob2 = self.PoolUp2(z)
        
        glob3 = self.PoolUp3(z)
        
        glob4 = self.PoolUp4(z)
        
        
        fully_connected = self.fc(torch.cat([z, glob2, glob3, glob4], dim=1))
        
        return fully_connected
'''
'''
class PGA(nn.Module):
    def __init__(self, input_channels, pool_sizes, dim):
        super().__init__()
        
        #reduced_channels = input_channels//8
        output_channels = input_channels
        
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1, padding=0)
        self.PoolUp1 = PoolUp(output_channels, pool_sizes[0], output_channels)
        #self.PoolUp2 = PoolUp(output_channels, pool_sizes[1], reduced_channels)
        #self.PoolUp3 = PoolUp(output_channels, pool_sizes[2], reduced_channels)
        #self.PoolUp4 = PoolUp(output_channels, pool_sizes[3], reduced_channels)
        self.fc = nn.Conv2d(input_channels, input_channels, kernel_size=1, padding=0)
        
        #self.norm = nn.LayerNorm([dim,dim])
        
    def forward(self,x):
        y = self.conv(x)
        glob1 = self.PoolUp1(y)
        y = self.norm(y)
        z = y+glob1
        #z = torch.mul(y,glob1)
        
        
        #glob2 = self.PoolUp2(y)
        
        #glob3 = self.PoolUp3(y)
        
        #glob4 = self.PoolUp4(y)
        
        
        fully_connected = self.fc(z)
        
        
        
        return fully_connected

'''

class PGA1(nn.Module):
    def __init__(self, input_channels, pool_sizes, dim):
        super().__init__()
        
        reduced_channels = input_channels//8
        output_channels = input_channels//2
        
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1, padding=0)
        self.PoolUp1 = PoolUp(output_channels, pool_sizes[0], reduced_channels)
        self.PoolUp2 = nn.AvgPool2d(kernel_size=pool_sizes[1], padding = (pool_sizes[1])//2, stride=1)
        self.PoolUp3 = nn.AvgPool2d(kernel_size=pool_sizes[2], padding = (pool_sizes[2])//2, stride=1)
        self.PoolUp4 = nn.AvgPool2d(kernel_size=pool_sizes[3], padding = (pool_sizes[3])//2, stride=1)
        #self.PoolUp3 = PoolUp(output_channels, pool_sizes[2], reduced_channels)
        #self.PoolUp4 = PoolUp(output_channels, pool_sizes[3], reduced_channels)
        self.conv1 = nn.Conv2d(output_channels, reduced_channels, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(output_channels, reduced_channels, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(output_channels, reduced_channels, kernel_size=1, padding=0)
        self.conv4 = nn.Conv2d(output_channels, reduced_channels, kernel_size=1, padding=0)

        
        self.fc = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, groups=input_channels)
        
        self.norm = nn.LayerNorm([dim,dim])
        
    def forward(self,x):
        
        
        
        y = self.conv(x)
        #print(y.shape)
        
        #z = x+glob1
        
        #z = torch.mul(y,glob1)
        
        glob1 = self.conv1(self.PoolUp1(y))
        
        
        glob2 = self.conv2(self.PoolUp2(y))
        
        glob3 = self.conv3(self.PoolUp3(y))
        
        glob4 = self.conv4(self.PoolUp4(y))
    
        
        #print(z.shape)
        
        
        concat = torch.cat([y,glob1,glob2,glob3,glob4],1)
        
        #print(concat.shape)
        
        fully_connected = self.fc(concat)
        
        z = self.norm(fully_connected)
        
        return z
'''
class PGA1(nn.Module):
    def __init__(self, input_channels, pool_sizes, dim):
        super().__init__()
        
        #reduced_channels = input_channels//8
        output_channels = input_channels
        
        #self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1, padding=0)
        self.PoolUp1 = PoolUp(output_channels, pool_sizes[0], output_channels)
        self.PoolUp2 = nn.AvgPool2d(kernel_size=pool_sizes[1], padding = (pool_sizes[1])//2, stride=1)
        self.PoolUp3 = nn.AvgPool2d(kernel_size=pool_sizes[2], padding = (pool_sizes[2])//2, stride=1)
        self.PoolUp4 = nn.AvgPool2d(kernel_size=pool_sizes[3], padding = (pool_sizes[3])//2, stride=1)
        #self.PoolUp3 = PoolUp(output_channels, pool_sizes[2], reduced_channels)
        #self.PoolUp4 = PoolUp(output_channels, pool_sizes[3], reduced_channels)
        self.fc = nn.Conv2d(input_channels*5, input_channels, kernel_size=3, padding=1, groups=input_channels)
        
        self.norm = nn.LayerNorm([dim,dim])
        
    def forward(self,x):
        #y = self.conv(x)
        glob1 = self.PoolUp1(x)
        z = self.norm(x)
        #z = x+glob1
        
        #z = torch.mul(y,glob1)
        
        
        
        glob2 = self.PoolUp2(z)
        
        glob3 = self.PoolUp3(z)
        
        glob4 = self.PoolUp4(z)
        
        z = torch.unsqueeze(z,2)
        glob1 = torch.unsqueeze(glob1,2)
        glob2 = torch.unsqueeze(glob2,2)
        glob3 = torch.unsqueeze(glob3,2)
        glob4 = torch.unsqueeze(glob4,2)
        
        #print(z.shape)
        
        
        concat = torch.cat([z,glob1,glob2,glob3,glob4],2)
        concat = torch.flatten(concat, start_dim=1, end_dim=2)
        
        #print(concat.shape)
        
        fully_connected = self.fc(concat)
        
        
        
        return fully_connected
'''




class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, dilations, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, dilations)
            

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)            


    
class Deform(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilate):
        super().__init__()
        
        self.offset = nn.Conv2d(in_channels, 2*kernel_size*kernel_size, kernel_size=3, padding = 1, dilation = 1)
        self.tan = nn.Hardtanh()
        self.deform = DeformConv2d(in_channels, out_channels, kernel_size = 3, stride = (1,1), 
                                    padding = dilate, dilation = dilate)
        
    def forward(self,x):
        
        off = self.offset(x)
        #print(off.shape)
        off1 = self.tan(off)
        out = self.deform(x, off1)
        #print(out.shape)
        return out


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, dilations):
        super().__init__()
           
        self.conv0 = nn.Conv2d(in_channels, in_channels//4, kernel_size=3, padding=1)
        self.conv1 = Deform(in_channels, in_channels//8, kernel_size=3, dilate = dilations[0])
        self.conv2 = Deform(in_channels, in_channels//8, kernel_size=3, dilate = dilations[1])
        #self.conv3 = Deform(in_channels, in_channels//8, kernel_size=3, dilate = dilations[2])
        
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
        #x3 = self.conv3(x)
        
        y = torch.cat([x0, x1, x2], dim=1)
        y1 = self.out(y)
        
        return y1
        
        
        
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)        


class DeepPyram_Res50(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(DeepPyram_Res50, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.Backbone = Res50()
        
        self.firstUpSample = nn.Upsample(scale_factor=2)
        

        self.glob1 = PGA1(input_channels=2048, pool_sizes=[32, 9, 7, 5], dim=32)
        
        self.glob2 = PGA1(input_channels=512, pool_sizes=[64, 9, 7, 5], dim=64)
        self.glob3 = PGA1(input_channels=256, pool_sizes=[128, 9, 7, 5], dim=128)#31, 15, 9
        self.glob4 = PGA1(input_channels=64, pool_sizes=[256, 9, 7, 5], dim=256)
        #self.glob5 = PGA(input_channels=32, pool_sizes=[512, 16, 8, 4], dim=512)

        self.up1 = Up(2048+1024, 512, [3, 6, 7], bilinear)
        self.up2 = Up(1024, 256, [3, 6, 7], bilinear)
        self.up3 = Up(512, 64, [3, 6, 7], bilinear)
        self.up4 = Up(128, 32, [3, 6, 7], bilinear)

        self.outc = OutConv(32, n_classes)
        #self.mask0 = nn.Conv2d(512, 1, kernel_size=3, padding=1)
        self.mask1 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.mask2 = nn.Conv2d(256, 1, kernel_size=3, padding=1)
        self.mask3 = nn.Conv2d(512, 1, kernel_size=3, padding=1)


    def forward(self, x):
        
        x = self.firstUpSample(x)
        
        out5, out4, out3, out2, out1 = self.Backbone(x)
        
        '''
        print('out1', out1.shape)
        print('out2', out2.shape)
        print('out3', out3.shape)
        print('out4', out4.shape)
        print('out5', out5.shape)
        '''

        out1 = self.glob1(out1)
        #out2 = self.glob2(out2)
        #out3 = self.glob3(out3)
        #out4 = self.glob4(out4)
        #out5 = self.glob5(out5)
        
        x1 = self.up1(out1, out2)
        x1 = self.glob2(x1)
        #print('Up1', x1.shape)
        
        #out4 = self.conv1(out4)
        x2 = self.up2(x1, out3)
        x2 = self.glob3(x2)
        #print('Up2', x2.shape)
        
        #out3 = self.conv2(out3)
        x3 = self.up3(x2, out4)
        x3 = self.glob4(x3)
        #print('Up3', x3.shape)
        
        #out2 = self.conv3(out2)
        x4 = self.up4(x3, out5)
        
        #print('Up4',x4.shape)
        

        
        logits = self.outc(x4)
        #print(logits.shape)
        
        #mask0 = self.mask0(out5)        
        mask3 = self.mask3(x1)
        mask2 = self.mask2(x2)
        mask1 = self.mask1(x3)

        return logits, mask3, mask2, mask1

    
    
    
if __name__ == '__main__':
    #model = Res50().cuda()
    model = DeepPyram_Res50(n_channels=3, n_classes=1)
    model = model.cuda()
    #template = torch.ones((1, 3, 512, 512))
    #detection= torch.ones((1, 1, 512, 512))
    print(summary(model, (3,512,512)))
    #y1 = model(template)
    #print(y1.shape)
    
 #[1, 10, 17, 17]
    #print(y2.shape) #[1, 20, 17, 17]15    
