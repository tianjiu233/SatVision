#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 11:23:50 2024

@author: no42pc
"""

import math
from skimage import io
import os,glob,shutil

from tqdm import tqdm

import imageio,tifffile

def copy_data_folder(source_folder,target_folder):
    try:
        if os.path.exists(target_folder):
            shutil.rmtree(target_folder)
        #shutil.copytree will create the folder
        shutil.copytree(source_folder,target_folder)
    except Exception as e:
        print(f"The error happens during copy process, and error is {e}")
        
    return

class MapTile(object):
    def __init__(self,
                 tile_stride=(256,256),
                 tile_size=(256,256),
                 drop_last=False,
                 ):
        """
        drop_last:When true, if the final tile is overlapped, it will be ignored.
        """

        self.tile_stride = tile_stride
        self.tile_size = tile_size
        
        self.drop_last = drop_last
        return
    
    def _tile(self,img,source_path,target_path):
        
        # image_file like austin1.tif
        _name,_suffix = img.split(".")
        _path = os.path.join(source_path,img)
        bands_first = False
        
        # skimage load the image with the format of h,w,c
        _mat = io.imread(_path)
        
        if len(_mat.shape)==3:
            
            _h,_w,_c = _mat.shape
            # for some geotiff, the array could be [bands,height,width]
            # 12 is a pre-defined value.
            # If _h is less than the value, the image might be with the format of [bands,height,width]
            if _h<12:
                _mat = _mat.transpose(1,2,0)
                _h,_w,_c = _mat.shape
                bands_first = True
            
        elif len(_mat.shape)==2:
            _h,_w = _mat.shape
            _c = None
        
        _h_stride,_w_stride = self.tile_stride[0],self.tile_stride[1]
        _h_tile,_w_tile = self.tile_size[0],self.tile_size[1]
        
        _rows = math.ceil((_h - _h_tile)*1. / _h_stride)+1
        _cols = math.ceil((_w - _w_tile)*1. / _w_stride)+1
        
        _row_start,_row_end,_col_start,_col_end = 0,0,0,0
        # loop for y/rows
        for i in range(_rows):
            if i == _rows-1:
                if self.drop_last and (_row_end>i*_h_stride):
                    continue
                else:
                    _row_end = _h
                    _row_start = _h - _h_tile
            else:
                _row_start = i * _h_stride
                _row_end = _h_tile + _row_start
                
            for j in range(_cols):
                if j == _cols-1:
                    if self.drop_last and (_col_end>j*_w_stride):
                        continue
                    else:
                        _col_end = _w
                        _col_start = _w - _w_tile
                else:
                    _col_start = j * _w_stride
                    _col_end = _col_start + _w_tile
                
                # get the tile mat
                # _mat_tile = _mat[_col_start:_col_end,_row_start:_row_end] if _c is None else _mat[_col_start:_col_end,_row_start:_row_end,:]
                _mat_tile = _mat[_row_start:_row_end,_col_start:_col_end] if _c is None else _mat[_row_start:_row_end,_col_start:_col_end,:]
                
                _tile = _name + "_tile_" + "row"+str(i)+"col"+str(j)+"."+_suffix
                _tile = os.path.join(target_path,_tile)
                
                if bands_first:
                    # back to the format as it is loaded.
                    _mat_tile=_mat_tile.transpose(2,0,1) # [h,w,c]->[c,h,w] 

                io.imsave(_tile,_mat_tile)

        return
    
    def traverse_and_tile(self,source_path,target_path):
        """
        traverse the source_path in a recursion way and process the folder when it contains the images.
        Both source_path and target_path should exist.
        """
        _dirs = os.listdir(source_path)
        # print(_dirs)
        if len(_dirs)>0:
            _path = os.path.join(source_path,_dirs[0])
            if os.path.isdir(_path):
                # dir
                for i in range(len(_dirs)):
                # for i in range(len(_dirs)):
                    _sub_source = os.path.join(source_path,_dirs[i])
                    _sub_target = os.path.join(target_path,_dirs[i])
                    # print(_sub_source)
                    # print(_sub_target)
                    if os.path.exists(_sub_target):
                        shutil.rmtree(_sub_target)
                    os.mkdir(_sub_target)
                    self.traverse_and_tile(_sub_source,_sub_target)
            else:
                # img
                print("processing dir: ",source_path)
                for i in tqdm(range(len(_dirs))):
                    self._tile(_dirs[i],source_path,target_path)
        return

if __name__=="__main__":
    print("Testing utils.py")
