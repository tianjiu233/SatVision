# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 11:08:24 2022

@author: bayes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# https://github.com/hualin95/Deeplab-v3plus/blob/d809d83c4183eb900a6a5248e516c76723d18e24/graphs/models/encoder.py
def _AsppConv(in_chs, out_chs, 
              kernel_size, stride=1, pad=0, dilation=1, bn_momentum=0.1):
    asppconv = nn.Sequential(
            nn.Conv2d(in_chs, out_chs, kernel_size, stride, pad, dilation, bias=False),
            #SynchronizedBatchNorm2d(out_chs, momentum=bn_momentum),
            nn.BatchNorm2d(out_chs,momentum=bn_momentum),
            nn.ReLU()
        )
    return asppconv

class AsppModule(nn.Module):
    def __init__(self,in_chs,out_chs,bn_momentum=0.1,output_stride=16):
        super(AsppModule,self).__init__()
        # the ratio of the input image spatial resolution to the final output
        # the memory is required much, when the output_stride is the small.
        if output_stride==16:
            astrous_rates = [6,12,18]
        elif output_stride==8:
            astrous_rates = [12,24,36] 
        
        # astrous_spatial_pyramind_pooling part
        self._astrous_conv1 = _AsppConv(in_chs=in_chs, out_chs=out_chs, kernel_size=1, stride=1)
        self._astrous_conv2 = _AsppConv(in_chs=in_chs, out_chs=out_chs, kernel_size=3, stride=1,
                                        pad=astrous_rates[0],dilation=astrous_rates[0])
        self._astrous_conv3 = _AsppConv(in_chs=in_chs, out_chs=out_chs, kernel_size=3, stride=1,
                                        pad=astrous_rates[1],dilation=astrous_rates[1])
        self._astrous_conv4 = _AsppConv(in_chs=in_chs, out_chs=out_chs, kernel_size=3, stride=1,
                                        pad=astrous_rates[2],dilation=astrous_rates[2])
        
        # image pooling part
        self._image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(in_chs, out_chs, kernel_size=1,bias=False),
            #SynchronizedBatchNorm2d(out_chs, momentum=bn_momentum),
            nn.BatchNorm2d(out_chs,momentum=bn_momentum),
            nn.ReLU()
            )
            
        return
    def forward(self,input_tensor):
        
        input1_ = self._astrous_conv1(input_tensor)
        input2_ = self._astrous_conv2(input_tensor)
        input3_ = self._astrous_conv3(input_tensor)
        input4_ = self._astrous_conv4(input_tensor)
        input5_ = self._image_pool(input_tensor)
        input5_ = F.interpolate(input=input5_, size=input4_.size()[2:4], mode='bilinear', align_corners=True)
        
        return torch.cat((input1_,input2_,input3_,input4_,input5_),dim=1) # 256-->1280

# the basic module to build the ResNetBackBone and DeepLabV3+
from resnet import Bottleneck,upconv_norm_act

class ResNetBackbone(nn.Module):
    """ When the resnet is used as the backbone for the deeplabv3 and deeplabv3+ model,
    only ResNet50, ResNet101 and ResNet152 is adopted.
    Meanwhile, the dilations and strides of the final
    """
    def __init__(self,in_chs,block=Bottleneck,layers=[3,4,6,3],output_stride=16):
        super(ResNetBackbone,self).__init__()
        # fixed parameters
        self.inplanes = 64
        self.dilation = 1
        
        self.in_chs = in_chs
        self.output_stride = output_stride
        
        self.conv1 = nn.Conv2d(in_chs, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False) # change
        
        self.layer1 = self._make_layer(block,  64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        
        if self.output_stride == 16:        
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2) 
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=2)
        elif self.output_stride == 8:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2) 
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        
        return
    
    def _make_layer(self,block,planes,blocks,stride=1,dilation=1,bias=False):
        """
        block: the basic module we need to build the resnet
        planes: the dimension for the module
        blocks: the list of the nums for the block
        bias: wether we decide to use the bias
        stride
        """
        downsample = None
        previous_dilation = self.dilation
        
        # build the downsample path
        if stride != 1 or self.inplanes != planes*block.expansion:
            # for downsample, dilation makes no difference.
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes,planes*block.expansion,kernel_size=1,stride=stride,bias=bias),
                nn.BatchNorm2d(planes*block.expansion))
        layers = []
        
        if dilation!=1:
            self.dilation = dilation
        
        layers.append(block(inplanes=self.inplanes,planes=planes,stride=stride,dilation=previous_dilation,downsample=downsample))
        
        # update the self.inplanes
        self.inplanes = planes * block.expansion
        for i in range(1,blocks):
            layers.append(block(self.inplanes,planes,dilation=self.dilation))
        return nn.Sequential(*layers)
    
    def forward(self,input_tensor):
        
        _features = []
        
        x = self.conv1(input_tensor)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        _features.append(x) # 1/4
        
        x = self.layer2(x)
        _features.append(x) # 1/8
        x = self.layer3(x)
        _features.append(x) # 1/16 or 1/8
        x = self.layer4(x)
        _features.append(x) # 1/16 or 1/8
        
        return _features

class DeepLabV3Plus(nn.Module):
    """
    Encoder-Decoder with Atrous Conv
    """
    
    def __init__(self,in_chs,out_chs,backbone="resnet50",output_stride=16):
        
        super(DeepLabV3Plus,self).__init__()
        # backbone
        if backbone == "resnet50":   
            self.backbone = ResNetBackbone(in_chs=in_chs,block=Bottleneck,layers=[3,4,6,3],output_stride=output_stride)
            low_level_chs = 256
            aspp_in_chs = 2048
            aspp_out_chs = 256 
            conv1x1_chs = 48
        
        # aspp
        aspp_module = AsppModule(in_chs=aspp_in_chs, out_chs=aspp_out_chs, output_stride=output_stride)
        # project to reduce the dimension
        project = nn.Sequential(
            nn.Conv2d(5*aspp_out_chs,aspp_out_chs,kernel_size=1,bias=False),
            nn.BatchNorm2d(aspp_out_chs),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),)
        
        # conv1x1 (for low-level features)
        conv1x1 = nn.Sequential(
            nn.Conv2d(low_level_chs,conv1x1_chs,kernel_size=1,bias=False),
            nn.BatchNorm2d(conv1x1_chs),
            nn.ReLU(inplace=True))
        
        # conv3x3 (after concat at 1/4)
        conv3x3 = nn.Sequential(
            nn.Conv2d(conv1x1_chs+aspp_out_chs,aspp_out_chs,3,padding=1,bias=False),
            nn.BatchNorm2d(aspp_out_chs),
            nn.ReLU(inplace=True))
        
        # upsamplex4
        # this op can also be replaced with F.interpolate
        # fcn upconv_norm_act(in_chs,out_chs,kernel=2,stride=2,dilation=1,pad=0,output_pad=0,bias=False)
        upsamplex4 = nn.Sequential(
            upconv_norm_act(in_chs = aspp_out_chs, out_chs = aspp_out_chs//2),
            upconv_norm_act(in_chs = aspp_out_chs//2, out_chs = aspp_out_chs//2))
        
        # classifier 
        classifier = nn.Conv2d(aspp_out_chs//2,out_chs,1)
        
        # decoder (aspp+conv1x1+conv3x3+upsamplex4+classifier)
        self.decoder = nn.ModuleDict({
            "aspp":aspp_module,
            "project":project,
            "conv1x1":conv1x1,
            "conv3x3":conv3x3,
            "upsamplex4":upsamplex4,
            "classifier":classifier})
        
        return
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _encode(self,x):
        
        features = self.backbone(x)
        
        # get aspp feature
        backbone_feature = features[-1]
        x = self.decoder["aspp"](backbone_feature) # concat mutli-branch features
        x = self.decoder["project"](x) # reduce the dimension
        return x, features
    
    def _decode(self,x,features):
    
        # get low-level feature
        low_level_feature = features[0]
        low_level_feature = self.decoder["conv1x1"](low_level_feature) 
        
        # upsamplex4 the aspp feature (using the interpolate)
        x = F.interpolate(x, size=low_level_feature.shape[2:], mode='bilinear', align_corners=False)
        # and then concat with the low-level features
        concat_feature = torch.cat([low_level_feature,x],dim=1)
        # conv with conv3x3
        concat_feature = self.decoder["conv3x3"](concat_feature)
        
        # upsamplex4 (using conv)
        concat_feature = self.decoder["upsamplex4"](concat_feature)
        # and then classify
        output_tensor = self.decoder["classifier"](concat_feature)
        
        return output_tensor
    
    def forward_w_features(self,x):
        """
        return the classifier output and hierachical features from different levels
        """
        # encode w aspp module
        x, features = self._encode(x) 
        # decode
        output_tensor = self._decode(x,features)
        
        return output_tensor, features
    
    def forward(self,x):
        # encode w aspp module
        x, features = self._encode(x)
        # decode
        output_tensor = self._decode(x,features)
        
        return output_tensor
    
if __name__=="__main__":
    # hyperparameters
    output_stride = 16 # 8, 16
    sample_input = torch.rand(2,5,256,256)
    print("Testing resent-backbone")
    backbone = ResNetBackbone(in_chs=5,block=Bottleneck,layers=[3,4,6,3],output_stride=16)
    with torch.no_grad():
        _features = backbone(sample_input)
    print("the feature map")
    for i in range(0,4):
        print("_features[{}]".format(i),_features[i].shape)
    print("Testing deeplabv3+")
    model = DeepLabV3Plus(in_chs=5, out_chs=7)
    with torch.no_grad():
        output_tensor = model(sample_input)
    print("output_tensor size is:",output_tensor.shape)