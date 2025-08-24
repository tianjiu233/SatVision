#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 22 09:41:18 2025

@author: no42pc
"""

from augment import NpTranspose
from augment import RandomCrop,VMirror,HMirror,Rotation
from augment import GrayScale,Solarization # only for RGB
from augment import GaussianBlurImage
from augment import RandomizedQuantizationImg

from dataclasses import dataclass
from typing import Optional
import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch import distributed as dist

from dist_utils import get_rank,get_world_size
from dist_utils import is_distributed,init_distributed
from dist_utils import MetricLogger,SmoothedValue
from dist_utils import save_model,load_model

from eval_utils import get_confusion_matrix,compute_metrics

from data.ESWKB import ESWKBDataset

from archs.deeplabv3plus import DeepLabV3Plus


"""
to be done:
    1.distributed traning
    2.checkpoint √
    3.measurement √
    4.add scaler of apex
    5.add evaluation √
"""

# python >= 3.7
@dataclass
class args(object):
    
    # hardware configuration
    workers : int = 8
    device : str = "cpu" # cpu,cuda
    distributed : bool =False
    
    # data
    train_data_dir : str = r"C:\Users\no42pc\Documents\mydata\dset-s2_tile"
    spec : str =  "rgb"
    data_in_chs : int = 3
    data_cls_num : int = 2 # binary classification
    data_aug_proba : float = 0.5
    
    # train
    image_size : int = 256
    start_epoch : int = 0 
    epochs : int = 100
    per_device_batch_size : int = 16
    resume : bool = False
    seed : int  = None
    
    # eval
    val_batch_size : int = 32
    
    # checkpoint
    save_dir : str = "./checkpoint/"
    resume : Optional[str] = None
    # resume : str = "./checkpoint/checkpoint-test.pth"
    
    # optim
    base_lr : float = 1e-3
    # momentum : float = 0.9
    weight_decay : float = 1e-4 
    
    # task
    task_mode : str = "seg"
    
    # log (within one epoch)
    print_freq = 10
    
    # eval 
    eval_freq = 1
    

train_t = transforms.Compose([
    RandomCrop(output_size=args.image_size,
               mode=args.task_mode),
    HMirror(p=args.data_aug_proba,mode=args.task_mode),
    VMirror(p=args.data_aug_proba,mode=args.task_mode),
    Rotation(mode=args.task_mode),
    GaussianBlurImage(p=args.data_aug_proba,mode=args.task_mode),
    #Solarization(p=args.data_aug_proba,mode=args.task_mode),
    #RandomizedQuantizationImg(region_num=12,collapse_to_val="inside_random",spacing="random",p=args.data_aug_proba,mode=args.task_mode),
    #GrayScale(p=args.data_aug_proba,mode=args.task_mode),
    NpTranspose(mode=args.task_mode)])

val_t = transforms.Compose([
    NpTranspose(mode=args.task_mode)])


# evaluate 
# A simple mode is employed, only when master.
@torch.no_grad()
def evaluate(data_loader,model,device):
    # criterion = torch.nn.CrossEntropy()
    
    model.eval()
    total_conf_mat = 0
    for iter_step, batch_data in tqdm(enumerate(data_loader,start=0)):
    # for batch_data in metric_logger.log_every(data_loader, print_freq=10,header=header):
        image = batch_data["image"]
        label = batch_data["label"]
        if torch.cuda.is_available():
            image = image.to(device,non_blocking=True)
            label = label.to(device,non_blocking=True)
        
        pred_proba = model(image)
        cls_num = pred_proba.size(1)
        _,pred_cls = torch.max(pred_proba,1)
        
        pred_cls = pred_cls.cpu().numpy().squeeze().astype(np.uint8)
        label = label.cpu().numpy().squeeze().astype(np.uint8)
        
        conf_mat = get_confusion_matrix(pred = pred_cls.flatten(), 
                                        target = label.flatten(), 
                                        cls_num = cls_num)
        total_conf_mat += conf_mat
    accu,accu_per_cls,accu_cls,iou,mean_iou,fw_iou,kappa = compute_metrics(total_conf_mat)
    
    eval_metrics = { "accu":accu,"accu_per_cls":accu_per_cls,"accu_cls":accu_cls,
                    "iou":iou,"mean_iou":mean_iou,"fw_iou":fw_iou,
                    "kappa":kappa
                    }
    
    model.train()
    return eval_metrics

if __name__=="__main__":
    print("main supervised learning process")
    
    # mkdir for checkpoint
    save_dir = Path(args.save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True,exist_ok=True)
        print(f"mkdir save_dir: {save_dir.resolve()}")
    else:
        print(f"save_dir: {save_dir.resolve()} exists")
    
    # first step
    # dist.init_process_group(backend="nccl")
    init_distributed(args)
    
    # dataset
    train_dataset = ESWKBDataset(root_path=args.train_data_dir,spec=args.spec,data_type="Train",data_transform=train_t)
    val_dataset = ESWKBDataset(root_path=args.train_data_dir,spec=args.spec,data_type="Val",data_transform=val_t)
     
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,num_replicas=get_world_size(),rank=get_rank(),shuffle=True) if is_distributed() else None
    # val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset,shuffle=False) if is_distributed() else None
    
    # sampler
    if is_distributed():
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,num_replicas=get_world_size(),rank=get_rank(),shuffle=True)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
    # model
    model = DeepLabV3Plus(in_chs=args.data_in_chs, out_chs=args.data_cls_num)
    model = model.to(args.device)
    model_wo_ddp = model
    if is_distributed():
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(model,device_ids=[args.gpu]) # ddp mode
        model_wo_ddp = model.module
    
    loss_fcn = nn.CrossEntropyLoss()
    # optim_params = model.parameters()
    # optimizer = torch.optim.Adam(optim_params,lr=args.base_lr,weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(model_wo_ddp.parameters(),lr=args.base_lr,weight_decay=args.weight_decay)
    
    
    loss_scaler = torch.amp.GradScaler() if is_distributed() else None 
    
    # load model
    load_model(args,model_wo_ddp=model_wo_ddp,optimizer=optimizer,loss_scaler=loss_scaler)
    
    # dataloader
    # The DistributedSampler has built-in shuffle logic.
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.per_device_batch_size,
                                               num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    # val loader
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch_size)
    
    
    optimizer.zero_grad()
    save_index = -1
    for _epoch in range(args.start_epoch,args.epochs):
        
        model.train(True)
        
        # eval & save
        if (_epoch+1)%args.eval_freq == 0:
            print("model evaluation")
            if get_rank() == 0:
                eval_metrics = evaluate(data_loader = val_loader, 
                                        model = model, 
                                        device = args.device)
                max_key_length = max(len(key) for key in eval_metrics.keys())
                for key, value in eval_metrics.items():
                    print(f"{key.ljust(max_key_length)}: {value}")
                if eval_metrics["accu"]>save_index:
                    save_index = eval_metrics["accu"]
                    save_model(args,epoch=_epoch,model_wo_ddp=model_wo_ddp,optimizer=optimizer,loss_scaler=None)
            # save_model(args,epoch=_epoch,model_wo_ddp=model_wo_ddp,optimizer=optimizer,loss_scaler=None)
            
        # log info
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter(name="lr", meter=SmoothedValue(window_size=1,fmt="{value:.6f}"))
        header = "Epoch:[{} / {}]".format(_epoch,args.epochs)
        print_freq = args.print_freq
        
        # train one epoch
        # enumerate(iterable,start)
        # for data_iter_step,batch_data in enumerate(train_loader,0):
        for data_iter_step,batch_data in enumerate(metric_logger.log_every(train_loader, print_freq=print_freq,header=header)):
            image = batch_data["image"].to(args.device)
            label = batch_data["label"].to(args.device)
            
            pred_proba = model(image)
            
            loss = loss_fcn(pred_proba,label)
            loss_val = loss.item()
            # print(loss_val)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            metric_logger.update(loss=loss_val)
            _lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=_lr)
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    