# -*- coding: utf-8 -*-
"""
Created on Sun Jun  8 16:22:20 2025

@author: no42pc
"""

from data.ESWKB import ESWKBDataset

from augment import NpTranspose
from augment import RandomCrop,VMirror,HMirror,Rotation
from augment import GrayScale,Solarization # only for RGB
from augment import GaussianBlurImage
from augment import RandomizedQuantizationImg

import torchvision.transforms as transforms

from dataclasses import dataclass

import matplotlib.pyplot as plt
from skimage import io
import skimage
import numpy as np

@dataclass
class args(object):
    # data
    raw_data_dir:str = r"C:\Users\no42pc\Documents\mydata\dset-s2_tile"
    new_data_dir:str = r"C:\Users\no42pc\Documents\data\dset-s2"
    
    spec:str = "rgb"
    # spec:str = "rgbnir"
    
    image_size:int = 256

p, mode = 0.5, "seg"
sl_t = transforms.Compose([
    RandomCrop(output_size=args.image_size,
               mode=mode),
    HMirror(p=p,mode=mode),
    VMirror(p=p,mode=mode),
    Rotation(mode=mode),
    GaussianBlurImage(p=p,mode=mode),
    #Solarization(p=p,mode=mode),
    #RandomizedQuantizationImg(region_num=12,collapse_to_val="inside_random",spacing="random",p=p,mode=mode),
    #GrayScale(p=p,mode=mode),
    NpTranspose(mode=mode)])


if __name__=="__main__":
    print("Test file running.")
    train_t = None
    train_t = sl_t
    dataset = ESWKBDataset(root_path=args.new_data_dir,data_type="Train",spec=args.spec,data_transform=train_t)
    """
    def vis_sample(dataset,index):
        sample = dataset[index]
        return 
    
    vis_sample(dataset,17)
    """
    index = 19
    sample = dataset[index]
    image = sample["image"].transpose(1,2,0)
    _low,_high = np.percentile(image, (2,98))
    image = skimage.exposure.rescale_intensity(image,in_range=(_low,_high))
    label = sample["label"]
    io.imshow(image)
