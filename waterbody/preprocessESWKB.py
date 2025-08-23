#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 27 15:53:59 2025

@author: no42pc
"""

import os,shutil

from data_utils import MapTile

import warnings
warnings.filterwarnings("ignore")

if __name__=="__main__":
    # ESWKB dataset
    print("building ESWKB dataset")
    
    """
    # LINUX
    raw_root_path = "/home/no42pc/Documents/dataset/dset-s2"
    new_root_path = "/home/no42pc/Documents/mydata/dset-s2_tile"
    """
    
    # WIN
    raw_root_path = r"C:\Users\no42pc\Documents\data\dset-s2"
    new_root_path = r"C:\Users\no42pc\Documents\mydata\dset-s2_tile"
    
    train_image = "tra_scene" 
    train_label = "tra_truth"
    val_image = "val_scene"
    val_label = "val_truth"
    
    maptile_train = MapTile(
        tile_stride = (256,256),
        tile_size = (512,512),
        drop_last=False,
        ) 
    
    maptile_val = MapTile(
        tile_stride = (256,256),
        tile_size = (256,256),
        drop_last=False,
        ) 
    
    # 0 check
    if os.path.exists(new_root_path):
        shutil.rmtree(new_root_path)
        print(f"Deleted existing directory: {new_root_path}")
    os.makedirs(new_root_path,exist_ok=True)
    
    # 1 tra_scene
    source_path = os.path.join(raw_root_path,train_image) 
    target_path = os.path.join(new_root_path,train_image)
    if os.path.exists(target_path):
        shutil.rmtree(target_path)
    os.mkdir(target_path)
    maptile_train.traverse_and_tile(source_path = source_path, 
                                        target_path = target_path)
    
    # 2 tra_truth
    source_path = os.path.join(raw_root_path,train_label) 
    target_path = os.path.join(new_root_path,train_label)
    if os.path.exists(target_path):
        shutil.rmtree(target_path)
    os.mkdir(target_path)
    maptile_train.traverse_and_tile(source_path = source_path, 
                                        target_path = target_path)
    
    # 3 val_scene
    source_path = os.path.join(raw_root_path,val_image) 
    target_path = os.path.join(new_root_path,val_image)
    if os.path.exists(target_path):
        shutil.rmtree(target_path)
    os.mkdir(target_path)
    maptile_val.traverse_and_tile(source_path = source_path, 
                                      target_path = target_path)
    
    # 4 val_truth
    source_path = os.path.join(raw_root_path,val_label) 
    target_path = os.path.join(new_root_path,val_label)
    if os.path.exists(target_path):
        shutil.rmtree(target_path)
    os.mkdir(target_path)
    maptile_val.traverse_and_tile(source_path = source_path, 
                                        target_path = target_path)
    