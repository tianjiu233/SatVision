#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 27 15:54:58 2025

@author: no42pc
"""

"""
Two datasets are used in this study.
The first is ESKWB dataset from the paper 
'An applicable and automatic method for earth surface water mapping based on multispectral images'

And the second dataset is with the paper 
'Deep-Learning-Based Multispectral Satellite Image Segmentation for Water Body Detection.'

"""

class Config(object):
    def __init__(self):
        # dataset root path
        self.raw_ESWKB_root_path = r"/home/no42pc/Documents/dataset/dset-s2"
        self.new_ESWKB_root_path = None
