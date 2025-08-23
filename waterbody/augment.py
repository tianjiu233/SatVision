#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 22 09:40:47 2025

@author: no42pc
"""

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

import skimage
from skimage import transform
from skimage.color import rgb2gray

import numpy as np
import random
import copy
import math

import matplotlib.pyplot as plt

from PIL import Image,ImageOps,ImageFilter


"""
The funcs below are the basic data augment methods with skimage and numpy
"""

class NpTranspose(object):
    def __init__(self,mode="seg"):
        self.mode = mode
        
    def __call__(self,sample):
        
        if self.mode == "cls":
            image = sample
        elif self.mode == "seg":
            image = sample["image"]
            label = sample["label"]
            # [h,w,1]->[1,h,w]->[h,w] / loss_fcn requires the label to be Long.
            # label = label.transpose(2,0,1).squeeze().astype(np.int64) 
            label = label.squeeze().astype(np.int64) # [h,w]
            
        # assuming that the image are of the format (h,w,c), 
        # and we make it from (h,w,c) to (c,h,w)
        image = image.transpose(2,0,1)
        
        if self.mode == "cls":
            sample = image
        elif self.mode == "seg":
            sample["image"] = image
            sample["label"] = label
            
            # --- add for mask --- 
            if "mask" in sample.keys():
                mask = sample["mask"]
                mask = mask.squeeze().astype(np.int32)
                sample["mask"] = mask
        
        return sample


# ------ geometric or spatial augmentation ------  
# randomcrop,stdcrop,rotation,h_mirror,v_mirror
# to be checked!
class RandomResizedCropImage(object):
    """
    Crop a random portion of image and resize it to a given size.
    """
    def __init__(self,
                 size = (256,256),
                 p=0.5, mode ="seg",
                 scale = (0.08, 1.0),
                 ratio=(3/4,4/3),
                 interpolation = "nearest"):
        
        """
        scale(tuple of python:float): specifies the lower and the upper bounds of 
        the random area of the crop before resizing. The scale is defined with respect
        to the area of the original image.
        
        ratio(tuple of python:float): lower and the upper bounds for the random aspect ratio of the crop,
        before resizing.
        ratio = width/height
        """
        self.size = size
        
        self.p = p
        self.mode = mode
        
        self.scale = scale
        self.ratio = ratio
        
        if interpolation == "nearest":    
            self.interpolation = 0
        elif interpolation == "bilinear":
            self.interpolation = 1
        elif interpolation == "biquadratic":
            self.interpolation = 2
        elif interpolation == "bicubic":
            self.interpolation = 3
        elif interpolation == "biquartic":
            self.interpolation = 4
        elif interpolation == "biquintic":
            self.interpolation = 5
    
    def __call__(self,sample):
        """
        image, label, mask have a shape of (h,w,c), and for mask and label, the c is 1.
        """
        if self.mode == "cls":
            image = sample
        elif self.mode =="seg":
            image = sample["image"]
            label = sample["label"]
        
        h,w,c = image.shape
        while True:
            _scale = np.random.uniform(self.scale[0],self.scale[1])
            _ratio = np.random.uniform(self.ratio[0],self.ratio[1])
            
            _h = math.sqrt(_scale*h*w/_ratio)
            _w = _ratio*h
            
            _h = math.floor(_h)
            _w = math.floor(_w)
            
            if (_h <= h) and (_w <= w):
                break
        
        top = np.random.randint(0, h-_h)
        left = np.random.randint(0, w-_w)
        
        cropimage = image[top:top+_h, left:left+_w,:]
        resizedcropimage = skimage.transform.resize(image = cropimage,
                                                    output_shape = self.size,
                                                    anti_aliasing=True,order=self.interpolation)
        
        if self.mode == "cls":
            sample = resizedcropimage
        elif self.mode == "seg":
            croplabel = label[top:top+_h, left:left+_w,:]
            resizedcroplabel = skimage.transform.resize(image = croplabel,
                                                        output_shape = self.size,
                                                        anti_aliasing=True,order=self.interpolation)
            sample["image"] = resizedcropimage
            sample["label"] = resizedcroplabel
            
            # --- add for mask --- 
            if "mask" in sample.keys():
                mask = sample["mask"]
                
                cropmask = mask[top:top+_h, left:left+_w,:]
                resizedcropmask = skimage.transform.resize(image=cropmask,
                                                           output_shape=self.size,
                                                           anti_aliasing=True,order=self.interpolation)
                
                sample["mask"] = resizedcropmask
        
        
        return sample 


class RandomCrop(object):
    def __init__(self,output_size,mode="seg"):
        
        self.mode = mode
        
        assert isinstance(output_size,(int,tuple))
        if isinstance(output_size,int):
            self.output_size=(output_size,output_size)
        else:
            assert len(output_size)==2
            self.output_size = output_size
        
    def __call__(self,sample):
        
        if self.mode == "cls":
            image = sample
        elif self.mode == "seg":
            image,label = sample["image"], sample["label"]
        
        h,w = image.shape[0:2]
        new_h, new_w = self.output_size
        
        # top = np.random.randint(0, h-new_h)
        # left = np.random.randint(0, w-new_w)
        
        if (new_h==h) and (new_w==w):
            top = 0
            left = 0
        else:
            top = np.random.randint(0,h-new_h)
            left = np.random.randint(0,w-new_w)
        
        image = image[top:top+new_h, left:left+new_w,:]
        
        if self.mode == "cls":
            sample = image
        elif self.mode == "seg":
            label = label[top:top+new_h, left:left+new_w,:]
            sample["image"] = image
            sample["label"] = label
            
            # --- add for mask --- 
            if "mask" in sample.keys():
                mask = sample["mask"]
                mask = mask[top:top+new_h, left:left+new_w,:]
                sample["mask"] = mask
            
        return sample
    
class StdCrop(object):
    """Crop randomly the image in a sample
    Args:
        output_size (tuple or int): Desired ouput size, if int, square crop 
        is made.
    """
    def __init__(self,output_size,top=0,left=0,mode="seg"):
        
        self.mode = mode
        
        assert isinstance(output_size,(int,tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size)==2
            self.output_size = output_size
        self.top = top
        self.left = left
    def __call__(self,sample):
        
        if self.mode == "cls":
            image = sample
        elif self.mode == "seg":
            image, label = sample['image'], sample['label']
        
        h,w = image.shape[0:2]
        new_h, new_w = self.output_size
        
        top = self.top
        left = self.left
        
        image = image[top:top+new_h, left:left+new_w,:]
        if self.mode == "cls":
            sample = image
        elif self.mode == "seg":
            label = label[top:top+new_h,left:left+new_w,:]
            sample["image"] = image
            sample["label"] = label

            # --- add for mask --- 
            if "mask" in sample.keys():
                mask = sample["mask"]
                mask = mask[top:top+new_h, left:left+new_w,:]
                sample["mask"] = mask
            
        return sample

class Rotation(object):
    def __init__(self,angle=90,mode="seg"):
        
        self.mode = mode
        self.angle = angle
        
    def __call__(self,sample):
        
        if self.mode == "cls":
            image = sample
        elif self.mode == "seg":
            image,label= sample["image"],sample["label"]
        
        ids = np.around(360/self.angle)
        multi = np.random.randint(0,ids)
        
        if multi>0.001:
            """
            notice:
            skimage.transform.rotate may change the dtype and the val of the np.array.
            It is better to make sure that both image and label are in the format of np.float32.
            """
            image = transform.rotate(image,self.angle*multi)
            if self.mode == "cls":
                sample = image
            elif self.mode == "seg":
                label = transform.rotate(label,self.angle*multi)
                sample["image"] = image
                sample["label"] = label
                
                # --- add for mask --- 
                if "mask" in sample.keys():
                    mask = sample["mask"]
                    mask = transform.rotate(mask,self.angle*multi)
                    sample["mask"] = mask
            
        return sample

class HMirror(object):
    def __init__(self,p=0.5,mode="seg"):
        
        self.mode = mode
        self.p = p
        
    def __call__(self,sample):
        
        if self.mode == "cls":
            image = sample
        elif self.mode == "seg":
            image, label = sample["image"], sample["label"]
        
        if np.random.random()<self.p:
            
            image = np.flip(image,0).copy()
            
            if self.mode == "cls":
                return image
            
            elif self.mode == "seg":
                
                label = np.flip(label,0).copy()
                
                sample["image"] = image
                sample["label"] = label
                
                # --- add for mask --- 
                if "mask" in sample.keys():
                    mask = sample["mask"]
                    mask = np.flip(mask,0).copy()
                    sample["mask"] = mask
        
        return sample

class VMirror(object):
    def __init__(self,p = 0.5,mode="seg"):
        self.mode = mode
        self.p = p
    def __call__(self,sample):
        if self.mode == "cls":
            image = sample
        elif self.mode == "seg":
            image, label= sample["image"], sample["label"]
        
        if np.random.random()<self.p:
            image = np.flip(image,1).copy()
            
            if self.mode == "cls":
                
                return image
            
            elif self.mode == "seg":
                
                label = np.flip(label,1).copy()
                
                sample["image"] = image
                sample["label"] = label
                
                # --- add for mask --- 
                if "mask" in sample.keys():
                    mask = sample["mask"]
                    mask = np.flip(mask,1).copy()
                    sample["mask"] = mask
                
        return sample
    
# ------ spectral augmentation ------
    
# GaussianBlurImage can be applied to the multispectral images,
# while the GaussianBlur is only fit to rgb images.
class GaussianBlurImage(object):
    def __init__(self,p=0.5,mode="seg"):
        self.p = p
        self.mode = mode
    def __call__(self,sample,sigma=3):
        if self.mode == "cls":
            image = sample
        elif self.mode == "seg":
            image = sample["image"]
        
        if np.random.random()<self.p:
            # channel_axis=-1,means that the dimension for color is at the last.
            image = skimage.filters.gaussian(image,sigma=sigma,channel_axis=-1)
            
            if self.mode == "cls":
                return image
            elif self.mode == "seg":
                sample["image"] = image
        
        return sample
    
# the grayscale method is only for rgb image.
class GrayScale(object):
    def __init__(self,p=0.5,mode="seg"):
        self.p = p
        self.mode = mode
    def __call__(self,sample):
        if self.mode == "cls":
            image = sample
        elif self.mode == "seg":
            image = sample["image"]
        
        if np.random.random()<self.p:
            
            image = rgb2gray(image,channel_axis=-1) # (h,w)
            image = np.expand_dims(image,axis=-1) # expand from (h,w) to (h,w,1)
            image = np.repeat(image, repeats=3,axis=-1) # repeats from (h,w,1) to (h,w,3)
            
            if self.mode == "cls":
                sample = image
            elif self.mode == "seg":
                sample["image"] = image
        
        return sample    

# the solarization is only for rgb image
class Solarization(object):
    def __init__(self, p=0.5, mode="seg", threshold=128):
        self.p = p
        self.mode = mode
        self.threshold = threshold
    def __call__(self, sample):
        
        if random.random()>=self.p:
            return sample
        
        # random.random()<self.p
        if self.mode == "cls":
            image = sample
        elif self.mode == "seg":
            image= sample["image"]
        
        if isinstance(image,Image.Image):
            image = ImageOps.solarize(image,threshold=self.threshold)
        else:
            image = ImageOps.solarize(image = Image.fromarray(np.uint8(image*255)),
                                      threshold = self.threshold)
            image = np.array(image,dtype=np.float32)/255.
        
        if self.mode == "cls":
            sample = image
        elif self.mode == "seg":
            sample["image"] = image
        
        return sample


# Randomized Quantization
"""
Randomized Quantization: A Generic Augmentation for Data Agnostic Self-supervised Learning
"""
# the numpy version
"""
It is assumed that the format of input is np.array with the shape of (h,w,c).
"""
class RandomizedQuantizationImg(object):
    def __init__(self,
                 region_num, 
                 collapse_to_val="inside_random",
                 spacing="random",
                 p=0.5,
                 mode="seg"):
        
        """
        region_num: the value of the channel will be divide into region_num parts.
        """
        
        self.region_num = region_num
        self.collapse_to_val = collapse_to_val
        self.spacing = spacing
        
        self.p = p 
        self.mode = mode
    
    def get_params(self,x):
        """
        x:(h,w,c) np.array
        """
        h,w,c = x.shape
        
        min_val = np.min(x.reshape((-1,c)),axis=0,keepdims=True) # (1,c) minmal values over spatial dims
        max_val = np.max(x.reshape((-1,c)),axis=0,keepdims=True) # (1,c) maximal values over spatial dims
        
        min_val = min_val[0] # (c)
        max_val = max_val[0] # (c)
        
        total_region_percentile_number_per_channel = (np.ones(c) * (self.region_num-1)).astype(np.int32)
        
        return min_val, max_val, total_region_percentile_number_per_channel
        
    
    def __call__(self,sample):
        
        # When coding, if the EPSILON is defined with 1, there is an error, and it will be correct when EPSILON is 1.0.
        EPSILON = 1.
        
        if self.mode == "cls":
            image =sample
        elif self.mode == "seg":
            image = sample["image"]
            
        # get params and convert the image arr to the required format (h,w,c)->(c,h,w)
        min_val,max_val,total_region_percentile_number_per_channel = self.get_params(image)
        h,w,c = image.shape
        image = image.transpose(2,0,1) # (c,h,w)
        image_orig = copy.deepcopy(image)
        
        # region percentile for each channel
        if self.spacing == "random":
            region_percentiles = np.random.rand(total_region_percentile_number_per_channel.sum()) # shape (sum_val,) 
        elif self.spacing == "uniform":
            region_percentiles = np.tile(np.arange(start=1/(total_region_percentile_number_per_channel[0]+1),
                                                   stop=1,
                                                   step=1/(total_region_percentile_number_per_channel[0]+1)),[c]) # shape (sum_val,)
           
        region_percentiles_per_channel = region_percentiles.reshape([-1,self.region_num-1]) # shape (c,self.region_num-1)
        # ordered region ends
        region_percentiles_pos = (region_percentiles_per_channel * (max_val-min_val).reshape((c,1)) + min_val.reshape((c,1))).reshape((c,-1,1,1)) # (c,self.region_num-1,1,1)
        # np.sort get an ascending order
        ordered_region_right_ends_for_checking = np.sort(np.concatenate((region_percentiles_pos,max_val.reshape(c,1,1,1)+EPSILON),axis=1),axis=1) # (c,self.region_num-1,1,1)->(c,self.region_num,1,1)
        ordered_region_right_ends = np.sort(np.concatenate((region_percentiles_pos,max_val.reshape(c,1,1,1)+1e-6),axis=1),axis=1) # (c,self.region_num,1,1)
        ordered_region_left_ends = np.sort(np.concatenate((min_val.reshape(c,1,1,1),region_percentiles_pos),axis=1),axis=1)
        ordered_region_mid = (ordered_region_right_ends + ordered_region_left_ends)/2 # (c,self.region_num,1,1)
        
        # associate region id
        is_inside_each_region = (image.reshape(c,1,h,w)<ordered_region_right_ends_for_checking)*(image.reshape(c,1,h,w) >=ordered_region_left_ends)
        assert (is_inside_each_region.sum(1) == 1).all()# sanity check: each pixel falls into one sub_range
        associated_region_id = np.argmax(is_inside_each_region.astype(np.int32),axis=1,keepdims=True) # (c,1,h,w)
        
        if self.collapse_to_val == "middle":
            # middle points as the proxy for all values in corresponding regions
            _tmp = np.repeat(ordered_region_mid, repeats=h,axis=2)
            _arr = np.repeat(_tmp, repeats=w,axis=3) # (c,self.region_num,h,w)

        elif self.collapse_to_val == "inside_random":
            # random points inside each region as the proxy for all values in corresponding regions
            proxy_percentiles_per_region = np.random.rand((total_region_percentile_number_per_channel+1.).sum().astype(np.int32)) # (sum_val,) 0~1
            proxy_percentiles_per_channel =  proxy_percentiles_per_region.reshape([-1,self.region_num]) # (c,self.region_num)
            ordered_region_rand = ordered_region_left_ends + proxy_percentiles_per_channel.reshape(c,-1,1,1)*(ordered_region_right_ends-ordered_region_left_ends) # left_ends + rand*delta, (c,self.region_num,1,1)
            _tmp = np.repeat(ordered_region_rand, repeats=h,axis=2)
            _arr = np.repeat(_tmp, repeats=w,axis=3)
        else:
            raise NotImplementedError
        
        proxy_vals = np.take_along_axis(arr=_arr, indices=associated_region_id, axis=1) # (c,1,h,w)
        image = proxy_vals.astype(image.dtype) # (c,1,h,w)
        if self.p<1:
            image = np.where(np.random.rand(c,1,1,1)<self.p,image,image_orig) # (c,1,h,w)
        
        image = image[:,0,:,:].transpose(1,2,0) # (h,w,c)
        
        if self.mode == "cls":
            return image
        elif self.mode == "seg":
            sample["image"] = image
            return sample
        
        return -1

# ------ demo for pixel level ssl ------
# paper: indexnet
class RandomCropOverlapped(object):
    def __init__(self,image_size = 256, patch_size = 16, min_overlapped_ratio = 0.05, mode = "seg", mask = None):
        assert image_size % patch_size == 0
        
        self.mode = mode
        
        self.mask = mask # None,"indexnet"
        
        self.patch_size = patch_size
        self.image_size = image_size
        
        self.mim_overlapped_ratio = min_overlapped_ratio
        
        grid_size_per_axis = int(image_size / patch_size)
        self.grid_size_per_axis = grid_size_per_axis
        overlapped_grid_size = math.ceil(math.sqrt(min_overlapped_ratio)*grid_size_per_axis)
        
        # outer: outer rectangle
        grid_size_per_axis_outer = int(grid_size_per_axis * 2 - overlapped_grid_size)
        self.grid_size_per_axis_outer = grid_size_per_axis_outer
        self.output_size_outer = patch_size * grid_size_per_axis_outer
        
    def __call__(self,sample):
        """
        input:
            sample: a dict contains image and label
        output:
            sample1, sample2 and the _coords(a list contain top_view1, left_view1, top_view2, left_view2)
        """
        
        if self.mode == "cls":
            image = sample
        elif self.mode == "seg":
            image = sample["image"]
            label = sample["label"]
        
        h,w = image.shape[0:2]

        # get outer rectangle
        top = np.random.randint(0, h-self.output_size_outer)
        left = np.random.randint(0, w - self.output_size_outer)

        
        delta_grid_size = self.grid_size_per_axis_outer - self.grid_size_per_axis
        # top_view1, left_view1, top_view2, left_view2
        # relative _coords, delta_grid_size+1 is exclusive.
        _coords = [np.random.randint(0, delta_grid_size+1) * self.patch_size for i in range(4)]

        # view1
        image_view1 = image[top+_coords[0] : top+_coords[0]+self.image_size,
                            left+_coords[1] : left+_coords[1]+self.image_size,:]
        label_view1 = label[top+_coords[0] : top+_coords[0]+self.image_size,
                            left+_coords[1] : left+_coords[1]+self.image_size,:]
        # view2
        image_view2 = image[top+_coords[2] : top+_coords[2]+self.image_size,
                            left+_coords[3] : left+_coords[3]+self.image_size,:]
        label_view2 = label[top+_coords[2] : top+_coords[2]+self.image_size,
                            left+_coords[3] : left+_coords[3]+self.image_size,:]
        
        # update _coords 
        _coords = [top,left] + _coords # 6 elements
        
        sample1 = {"image":image_view1,"label":label_view1}
        sample2 = {"image":image_view2,"label":label_view2}
        
        
        if self.mask is not None:
            cropped_image, _coords, mask1, mask2 = self.generate_mask(_coords=_coords, image=image)
            sample1["mask"] = mask1
            sample2["mask"] = mask2
            
            return sample1, sample2, _coords, cropped_image
            
            
        return sample1, sample2, _coords, None

    def generate_mask(self,_coords,image):
        """
        input:
            _coords: a list that contains top, left, top_view1, left_view1, top_view2, left_view2.
            Both top and left are the absolute values but the others are relative.
            
            image: the image that fed into the RandomCropOverlapped class.
        output:
            cropped_image
            mask1
            mask2
        """
        
        if self.mask == "indexnet":
            top, left, top_view1, left_view1, top_view2, left_view2 = _coords
            # generate the big area
            # the values below are absolute values
            _top = min(top_view1, top_view2) + top
            _bottom = max(top_view1, top_view2) + top + self.image_size
            
            _left = min(left_view1, left_view2) + left 
            _right = max(left_view1, left_view2) + left + self.image_size
            
            _height = _bottom - _top
            _width = _right - _left
            
            height_grid = int(_height / self.patch_size)
            width_grid = int(_width / self.patch_size)
            
            # notice that the cropped_image has a different coordinates from the image
            cropped_image = image[_top:_bottom, _left: _right, :]
            
            mask_grid = np.arange(height_grid*width_grid,dtype=np.int32).reshape(height_grid,width_grid)
            
            # get the mask1 and mask2 
            # update top_view1...
            top_view1 = top_view1 + top - _top
            top_view2 = top_view2 + top - _top
            
            left_view1 = left_view1 + left - _left
            left_view2 = left_view2 + left - _left
            
            top_grid_view1 = int(top_view1 / self.patch_size)
            left_grid_view1 = int(left_view1 / self.patch_size)
            
            top_grid_view2 = int(top_view2 / self.patch_size)
            left_grid_view2 = int(left_view2 / self.patch_size)
            
            # grid
            mask1_grid = mask_grid[top_grid_view1: top_grid_view1 + self.grid_size_per_axis,
                                   left_grid_view1: left_grid_view1 + self.grid_size_per_axis]
            mask2_grid = mask_grid[top_grid_view2: top_grid_view2 + self.grid_size_per_axis,
                                   left_grid_view2: left_grid_view2 + self.grid_size_per_axis]
            # grid->mask
            mask1 = transform.rescale(image=mask1_grid, scale=self.patch_size,order=0)
            mask2 = transform.rescale(image=mask2_grid, scale=self.patch_size,order=0)
            
            # update the coordinates for output
            top = 0
            left = 0
            _coords = [top,left] + [top_view1, left_view1, top_view2, left_view2]
            
        # notice: the method to generate siamese mask waits to be checked.
        if self.mask == "siamese":
            top, left, top_view1, left_view1, top_view2, left_view2 = _coords
            # generate the big area
            # Coordinates below are absolute values.
            _top = top 
            _bottom = max(top_view1, top_view2) + top + self.image_size
            
            _left = left
            _right = max(left_view1, left_view2) + left + self.image_size
            
            _height = _bottom - _top
            _width = _right - _left
            
            # get the cropped_image
            cropped_image = image[_top:_bottom, _left:_right,:]
            
            grid_h = np.arange(_height, dtype=np.float32)
            grid_w = np.arange(_width, dtype=np.float32)
            
            mask_grid = np.meshgrid(grid_w,grid_h,indexing="xy")
            mask_grid = np.stack(mask_grid,axis=0) # [2, _height, _width]
            
            mask1 = mask_grid[:, 
                              top_view1: top_view1+self.image_size,
                              left_view1: left_view1+self.image_size]
            mask2 = mask_grid[:,
                              top_view2: top_view2+self.image_size,
                              left_view2: left_view2+self.image_size]
            
            _coords[0],_coords[1] = 0, 0
 
        return cropped_image, _coords, np.float32(mask1), np.float32(mask2) 


class Weak_Strong_Consistency_Transform(object):
    """
    Generate two patch from a large image, making sure that two patch an area overlapped.
    """
    def __init__(self,image_size=256,
                 patch_size=16,
                 min_overlapped_ratio=0.05,
                 p = 0.5,
                 mode="seg",mask="indexnet"):
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size/patch_size
        
        # a special randomcrop method
        self.randomcrop = RandomCropOverlapped(image_size,patch_size,min_overlapped_ratio,mode,mask)
        
        # weak augmentation: only geometric 
        self.weak_transform = transforms.Compose([
            HMirror(p=p,mode=mode),
            VMirror(p=p,mode=mode),
            Rotation(mode=mode),
            NpTranspose()
            ])
        
        # strong augmentation: both geometric and spectral
        self.strong_transform = transforms.Compose([
            HMirror(p=p,mode=mode),
            VMirror(p=p,mode=mode),
            Rotation(mode=mode),
            GaussianBlurImage(p=p,mode=mode),
            RandomizedQuantizationImg(region_num=12,
                                      collapse_to_val="inside_random",
                                      spacing="random",
                                      p=p,mode=mode),
            GrayScale(p=p,mode=mode),
            NpTranspose()
            ])
    
            
        return 
    def __call__(self,sample):
        
        
        sample1, sample2, _coords, cropped_image = self.randomcrop(sample)
        
        # weak sample
        sample1 = self.weak_transform(sample1)
        # strong sample
        sample2 = self.strong_transform(sample2)
        
        return sample1, sample2
    

if __name__=="__main__":
    # data
    print("augment.py test")
    