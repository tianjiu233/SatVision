#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 17:58:39 2025

@author: no42pc
"""

"""
codes for the distribution training.
"""
import torch
from torch import nn
from torch import distributed as dist

from pathlib import Path
import os
import time,datetime
from collections import defaultdict,deque

def is_distributed(): return False if not (dist.is_available and dist.is_initialized()) else True

def get_rank(): return dist.get_rank() if is_distributed () else 0

def get_world_size(): return dist.get_world_size() if is_distributed else 0

def init_distributed(args):
    ddp = int(os.environ.get('RANK', -1)) != -1

    if not (ddp and args.distributed):
        args.rank, args.world_size, args.is_master, args.gpu = 0, 1, True, 0
        args.device = torch.device(args.device)
        print("Not using distributed mode!")
    else:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        args.is_master = args.rank == 0
        args.device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(args.device)
        print(f'| distributed init (rank {args.rank}), gpu {args.gpu}', flush=True)
        dist.init_process_group(backend="nccl", 
                                world_size=args.world_size, 
                                rank=args.rank, 
                                device_id=args.device)
        
        # torch.distributed.barrier()
        dist.barrier()
        
        # update config
        # args.per_device_batch_size = int(args.batch_size / torch.cuda.device_count())
        args.batch_size = args.per_device_batch_size * torch.cuda.device_count()
        args.workers = int((args.workers + args.world_size - 1) / args.world_size)

        # fix printing
        def fix_print(is_master):
            import builtins as __builtin__
            builtin_print = __builtin__.print
            
            def print(*args, **kwargs):
                force = kwargs.pop('force', False)
                if is_master or force:
                    builtin_print(*args, **kwargs)
            __builtin__.print = print

        dist.barrier()
        fix_print(args.is_master)

    return is_distributed()

# https://github.com/facebookresearch/mae/tree/main
class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        #if not is_dist_avail_and_initialized():
        #    return
        if not is_distributed():
            return
        
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)   

# https://github.com/facebookresearch/mae/tree/main
class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        # logger.meters["loss"]-->logger.loss
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        # mannually add a metric
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))
"""
# https://github.com/facebookresearch/mae/tree/main
def save_model(args, epoch,model, model_wo_ddp, optimizer, loss_scaler):
    save_dir = Path(args.save_dir)
    epoch_name = str(epoch)
    if loss_scaler is not None:
        checkpoint_paths = [save_dir / ('checkpoint-%s.pth' % epoch_name)]
        for checkpoint_path in checkpoint_paths:
            to_save = {
                'model': model_wo_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }
            # get_rank()==0 means that either the process is the main process or not using the distributed training.
            if get_rank() == 0:
                torch.save(to_save,checkpoint_path)
            # save_on_master(to_save, checkpoint_path)
    else:
        client_state = {'epoch': epoch}
        model.save_checkpoint(save_dir=args.save_dir, tag="checkpoint-%s" % epoch_name, client_state=client_state)
"""
# https://github.com/facebookresearch/mae/tree/main
def save_model(args, epoch, model_wo_ddp, optimizer, loss_scaler):
    save_dir = Path(args.save_dir)
    epoch_name = str(epoch)
    checkpoint_path = save_dir / ('checkpoint-%s.pth' % epoch_name)
    to_save = {
        'model': model_wo_ddp.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'args': args,
    }
    if loss_scaler is not None:
        to_save["scaler"] = loss_scaler.state_dict()
    if get_rank()==0:
        torch.save(to_save,checkpoint_path)

# https://github.com/facebookresearch/mae/tree/main
def load_model(args,model_wo_ddp, optimizer, loss_scaler):
    """
    model_wo_ddp is passed to the function as a parameter. 
    When load_state_dict() is called on it inside the function, 
    the original object itself (not a copy) is modified. 
    Therefore, 
    after the function execution completes, 
    the external model_wo_ddp will already be in the state where its weights have been restored.
    """
    if args.resume is not None:
        checkpoint = torch.load(args.resume,map_location="cpu")
        model_wo_ddp.load_state_dict(checkpoint["model"])
        print("Resume checkpoint %s" % args.resume)
        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
                print("With scaler!")
            print("With optim & epoch!")
    else:
        print("No model.pth need to be loaded.")
    
"""
# https://github.com/facebookresearch/mae/tree/main
def load_model(args,model_wo_ddp, optimizer, loss_scaler):
    checkpoint = torch.load(args.resume,map_location="cpu")
    model_wo_ddp.load_state_dict(checkpoint["model"])
    print("Resume checkpoint %s" % args.resume)
    if 'optimizer' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch'] + 1
        if 'scaler' in checkpoint:
            loss_scaler.load_state_dict(checkpoint['scaler'])
        print("With optim & sched!")
"""
class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name, self.fmt = name, fmt
        self.reset()

    def reset(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    # def __str__(self): return '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'.format(**self.__dict__)
    def __str__(self): return f'{self.name} {self.val:{self.fmt}} (avg {self.avg:{self.fmt}})'
