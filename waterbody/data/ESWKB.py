#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 22 10:07:13 2025

@author: no42pc
"""
import torch
import torch.nn as nn
import torchvision

from torch.utils.data.dataset import Dataset

import numpy as np
import skimage
import os
import random
import copy

import matplotlib.pyplot as plt

"""
Xin Luo, Xiaohua Tong, Zhongwen Hu. 
An applicable and automatic method for earth surface water mapping based on multispectral images. 
International Journal of Applied Earth Observation and Geoinformation, 2021, 103, 102472. 
"""

# Earth surface water knowledge base
"""
    four visible-near infrared 10-m bands and two 20-m shortwave infrared bands, 
    which corresponding to the band number of 2, 3, 4, 8, 11, and 12 of sentinel-2 image.
    2:blue
    3:green
    4:red
    8:nir 10m
    11:swir 20m
    12:swir 20m
    
    get the rgb image using 4,3,2.
    
    According to the band setting, some water body index could be built.
    
    ndwi: (band3-band8)/(band3+band8) 
    
    mndwi: (band3-band11)/(band3+band11) 

    AWEIsh: band2 + 2.5xband3 - 1.5x(band8+band11) - 0.25xband12
    
    AWEInsh: 4x(band3-band11)-(0.25xband8+2.75xband12)
        
    mbwi: (band3+band8-band11)/(band3+band8+band11)
    
    
    """
class ESWKBDataset(Dataset):
    def __init__(self,root_path,
                 data_type="Train",
                 ratio=1, spec="rgbnir",
                 data_transform=None):
        self.root_path = root_path
        self.data_type = data_type
        self.ratio = ratio
        # spectral
        self.spec = spec
        self.data_transform = data_transform
        
        if data_type == "Train":
            image_dir = os.path.join(root_path, "tra_scene")
            label_dir = os.path.join(root_path, "tra_truth")
        elif data_type == "Val":
            image_dir = os.path.join(root_path, "val_scene")
            label_dir = os.path.join(root_path, "val_truth")
        
        # meta info
        # raw data
        # "S2A_L2A_20190125_N0211_R034_6Bands_S1.tif"
        # "S2A_L2A_20190125_N0211_R034_S1_Truth.tif"
        # new data
        # "S2A_L2A_20190125_N0211_R034_6Bands_S1_tile_row2col1.tif"
        # "S2A_L2A_20190125_N0211_R034_S1_Truth_tile_row0col1.tif"
        items = []
        label_files = os.listdir(label_dir)
        for _file in label_files:
            item,suffix = _file.split("_Truth")[0:2] # suffix: .tif or _p_row0col1.tif
            words = item.split("_")
            _s = words[-1] # for example S1
            words = words[0:-1] 
            item = "_".join(words) # like S2A_L2A_20190125_R034
            
            image_file = item+"_6Bands_" + _s + suffix
            
            image_file = os.path.join(image_dir,image_file)
            label_file = os.path.join(label_dir,_file)
            
            item = {"image":image_file,
                    "label":label_file}
            items.append(item)
        
        random.shuffle(items)
        if self.ratio<1:
            items = items[0:int(len(items)*ratio)]
        
        self.items = items
        return
    def __getitem__(self, index):
        
        item = self.items[index]
        image_file = item["image"]
        label_file = item["label"]
        
        image = skimage.io.imread(image_file) # [c,w,h] [0,10000]
        image = image.transpose(1,2,0)
        image = image.astype(np.float32)
        image = np.clip(image/10000.,0,1)
        
        label = skimage.io.imread(label_file) # [h,w], {0,1}
        label = label[:,:,np.newaxis] # [h,w,1]
        label = label.astype(np.float32)
        
        if self.spec == "rgbnir":
            image = image[:,:,[2,1,0,4]] # bgrnir->rgbnir
        elif self.spec == "rgb":
            image = image[:,:,[2,1,0]] # bgrnit->rgb
        
        sample = {"image":image,
                  "label":label}
        
        if self.data_transform is not None:
            return self.data_transform(sample)
        else:        
            return sample
    
    def __len__(self):
        return len(self.items)
    def vis_data(self,index,alpha=0.5,fontsize=8,cache=False):
        
        if self.data_transform is None:
            sample = self.__getitem__(index)
        image = sample["image"] # (h,w,c) [0,1]
        label = sample["label"] # (h,w,1)
        
        # for visualization
        image = image[:,:,0:3] # rgb
        
        _low,_high = np.percentile(image, (2,98))
        image = skimage.exposure.rescale_intensity(image,in_range=(_low,_high))
        image = image*255.
        
        label = label[:,:,0].astype(np.uint8)
        colors = np.array([[0,0,0],[0,0,255]])
        rgb_label = colors[label]
        
        #label = label[:,:,0].astype(np.uint8)
        #rgb_label = skimage.color.label2rgb(label=label,bg_label=0) # [0,1] float
        
        vis_image = copy.deepcopy(image)
        vis_image[label==1] = vis_image[label==1]*(1-alpha)
        vis_image =vis_image + rgb_label*alpha
        
        vis_image = vis_image.astype(np.uint8)
        image = image.astype(np.uint8)
        
        flgs,axes = plt.subplots(nrows=1,ncols=3)
        
        axes[0].set_title("image",fontsize=fontsize)
        axes[0].axis("off")
        axes[0].imshow(image)
        
        axes[1].set_title("label",fontsize=fontsize)
        axes[1].axis("off")
        axes[1].imshow(rgb_label)
        
        axes[2].set_title("mask",fontsize=fontsize)
        axes[2].axis("off")
        axes[2].imshow(vis_image)
        
        if cache:
            os.makedirs("./cache",exist_ok=True)
            plt.savefig(os.path.join("./cache", "combined_image.png"), bbox_inches='tight')

        return


if __name__ == "__main__":
    print("testing ESKWBDataset")
    data_root_path = r"C:\Users\no42pc\Documents\mydata\dset-s2_tile"
    # data_root_path = r"/home/no42pc/Documents/mydata/dset-s2_tile"
    dataset = ESWKBDataset(root_path= data_root_path,
                           data_type="Train",
                           ratio=1,
                           spec="rgb")
    dataset.vis_data(index=1,cache=True)
    data_loader = torch.utils.data.DataLoader(dataset,batch_size=16,shuffle=True)
    _iter = iter(data_loader)
    _batch = next(_iter)