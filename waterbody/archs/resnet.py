# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 17:02:58 2020

@author: huijianpzh
"""

import math
import copy

import torch
import torch.nn as nn

# basic module
# they may not be used by resnet.
def conv1x1(in_chs,out_chs,
            stride=1,pad=0,
            bias=False):
    "1x1 convolution"
    return nn.Conv2d(in_channels=in_chs,out_channels=out_chs,kernel_size=1,
                     stride=stride,padding=pad,bias=bias)

def upsample(scale_factor,mode="bilinear",align_corners=False):
    return nn.Upsample(scale_factor=scale_factor,mode=mode,align_corners=align_corners)

def conv3x3(in_chs,out_chs,stride=1,pad=1,dilation=1,bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_channels=in_chs,out_channels=out_chs, 
                     kernel_size=3,stride=stride,padding=pad,dilation=dilation,bias=bias)

def conv_norm(in_chs,out_chs,
              kernel=3,stride=1,
              dilation=1,pad=1,
              bias=False):
    op = nn.Sequential(nn.Conv2d(in_chhannels=in_chs,out_channels=out_chs,
                                 kernel_size=kernel,stride=stride,
                                 dilation=dilation,padding=pad,bias=bias),
                       nn.BatchNorm2d(out_chs),
                       )
    return op
    
def conv_norm_act(in_chs,out_chs,
                  kernel=3,stride=1,
                  dilation=1,pad=1,
                  bias=False):
    op = nn.Sequential(nn.Conv2d(in_channels=in_chs,out_channels=out_chs,
                                 kernel_size=kernel,stride=stride,
                                 dilation=dilation,padding=pad,bias=bias),
                       nn.BatchNorm2d(out_chs),
                       nn.ReLU(),
                       )
    return op

def upconv_norm_act(in_chs,out_chs,
                    kernel=2,stride=2,
                    dilation=1,
                    pad=0,output_pad=0,
                    bias=False):
    op = nn.Sequential(nn.ConvTranspose2d(in_channels=in_chs,out_channels=out_chs,
                                          kernel_size=kernel,stride=stride,
                                          padding=pad,output_padding=output_pad,
                                          dilation=dilation,bias=bias),
                       nn.BatchNorm2d(out_chs),
                       nn.ReLU()
        )
    return op

"""
basic block: successive 3x3 block for ResNet34 and ResNet18 
"""
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self,inplanes,planes,downsample=None,stride=1,dilation=1):
        """
        inplane: the dimension for the block, like 64,128,256...
        For BasicBlock, the dialtion is not modified.
        """
        super(BasicBlock,self).__init__()

        self.conv1 = conv3x3(in_chs=inplanes,out_chs=planes,
                             stride=stride,pad=dilation,dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = conv3x3(in_chs=planes,out_chs=planes)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.downsample = downsample
        self.stride = stride
    
    def forward(self,input_tensor):
        
        identify = input_tensor
        
        x = self.conv1(input_tensor)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        if self.downsample is not None:
            identify = self.downsample(input_tensor)
    
        output_tensor = self.relu( x + identify )
        return output_tensor
"""
Bottleneck is the basci module for ResNet50, ResNet101 and ResNet152
"""
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self,inplanes,planes,downsample=None,bias=False,stride=1,dilation=1):
                
        super(Bottleneck,self).__init__()
        
        # 1x1 conv
        self.conv1 = conv1x1(inplanes,planes)
        self.bn1 = nn.BatchNorm2d(planes)
        
        # 3x3 conv
        self.conv2 = conv3x3(planes,planes,stride=stride,pad=dilation,dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        
        # 1x1 conv
        self.conv3 = conv1x1(planes,planes*4)
        self.bn3 = nn.BatchNorm2d(planes*4)
        
        self.relu = nn.ReLU(inplace = True)
        
        self.downsample = downsample
        self.stride = stride
    def forward(self,input_tensor):
        
        identify = input_tensor
        
        x = self.conv1(input_tensor)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        
        if self.downsample is not None:
            identify = self.downsample(input_tensor)
          
        output_tensor = self.relu(x + identify)
        return output_tensor
"""
------ ResNet34 ------
"""

class ResNet(nn.Module):
    def __init__(self,in_chs,out_chs,block=BasicBlock,layers=[3,4,6,3]):
        """
        block: the type of the basic module the model used
        layers: the nums for the blocks
        """
        self.inplanes = 64
        super(ResNet,self).__init__()
        
        self.conv1 = nn.Conv2d(in_chs, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False) # change
        
        self.layer1 = self._make_layer(block,  64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2) 
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2) 
        
        # AdaptiveAvg
        self.fc_pool = nn.AdaptiveAvgPool2d((1,1))
        # AdaptiveMax
        # self.fc_pool = nn.AdaptiveMaxPool2d((1,1))

        self.fc = nn.Linear(512*block.expansion,out_chs)
    
    def _initialize_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()    
    
    def _make_layer(self,block,planes,blocks,
                    stride=1,bias=False):
        """
        block: the basic module we need to build the resnet
        planes: the dimension for the module
        blocks: the list of the nums for the block
        bias: wether we decide to use the bias
        stride
        """
        downsample = None
        if stride != 1 or self.inplanes != planes*block.expansion:
            # for downsample, dilation makes no difference.
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes,planes*block.expansion,kernel_size=1,stride=stride,bias=bias),
                nn.BatchNorm2d(planes*block.expansion))
        layers = []
        
        layers.append(block(inplanes=self.inplanes,planes=planes,stride=stride,downsample=downsample))
        # update the self.inplanes
        self.inplanes = planes * block.expansion
        for i in range(1,blocks):
            layers.append(block(self.inplanes,planes))
        return nn.Sequential(*layers)
    
    def forward(self,input_tensor):
        
        x = self.conv1(input_tensor)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.fc_pool(x)
        x = x.view(x.size(0),-1)
        
        output_tensor = self.fc(x)
        
        return output_tensor

if __name__=="__main__":
    print("Testing ResNet")
    # resent
    sample_input = torch.rand(1,5,256,256)
    resnet34 = ResNet(in_chs=5, out_chs=10)
    with torch.no_grad():
        sample_output = resnet34(sample_input)
    print(sample_output.shape)
    