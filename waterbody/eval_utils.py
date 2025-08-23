# -*- coding: utf-8 -*-
"""
Created on Sat Aug 23 22:08:18 2025

@author: no42pc
"""

import numpy as np
import torch
from tqdm import tqdm
from typing import Iterable
"""
https://github.com/milleniums/High-Resolution-Remote-Sensing-Semantic-Segmentation-PyTorch
https://github.com/CSAILVision/semantic-segmentation-pytorch/blob/master/mit_semseg/utils.py
Computes and stores the average and current value.
"""

def get_confusion_matrix(pred,target,cls_num):
    """
    Both pred and target are 1-d np.array.
    The element conf_mat[i][j] in the matrix represents the number of samples where the true label is i but the predicted label is j.
    """
    mask = (target>=0) & (target<cls_num) # bool
    conf_mat = np.bincount(cls_num*target[mask].astype(int)+pred[mask],minlength=cls_num**2).reshape(cls_num,cls_num)

    return conf_mat

def compute_metrics(conf_mat):
    """
    # compute the evaluation for the net
    conf_mat.sum(axis=1) gt col
    conf_mat.sum(aixs=0) pred row
    
    Delta is added to avoid nan.
    Other method can also be adopted to avoid it.
    """
    
    DELTA = 1e-9
    
    # total accuracy
    accu = np.diag(conf_mat).sum()/conf_mat.sum()
    # accuracy for per class
    accu_per_cls = np.diag(conf_mat)/(conf_mat.sum(axis=1) + DELTA)
    accu_cls = np.nanmean(accu_per_cls)
    
    # iou is a list for every class
    iou = np.diag(conf_mat)/(conf_mat.sum(axis=1)+conf_mat.sum(axis=0)-np.diag(conf_mat) + DELTA)
    mean_iou = np.nanmean(iou)
    # add FWIoU
    # https://blog.csdn.net/lingzhou33/article/details/87901365
    # some errors are in the blog, please to read the original paper
    fw_iou = (np.diag(conf_mat)*np.sum(conf_mat,axis=1))/(conf_mat.sum(axis=1)+conf_mat.sum(axis=0)-np.diag(conf_mat) + DELTA)
    fw_iou = fw_iou.sum()/conf_mat.sum()
    
    # kappa
    pe = np.dot(np.sum(conf_mat,axis=0),np.sum(conf_mat,axis=1))/(conf_mat.sum()**2)
    if pe == 1:
        if accu == 1:
            kappa = 1
        else:
            print("Wrong")
            kappa = 0
    else:
        kappa = (accu - pe)/(1 - pe)
    return accu,accu_per_cls,accu_cls,iou,mean_iou,fw_iou,kappa