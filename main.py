import argparse
import datetime
import glob
import json
import math
import os
import random
import time
import pickle
from pathlib import Path
import tempfile

import yaml
import torch
import torch.distributed as dist
import numpy as np
from utils import torch_distributed_zero_first
from tensorboardX import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler
import torch.multiprocessing

import utils.misc as utils
from dataloader import create_data_loader, CLIPTextTokenizer
from model import SSM_video_gen
from train_eval import *
from diffusers.models import AutoencoderKL

# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

torch.multiprocessing.set_sharing_strategy('file_system')


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--config', default='config/cfg.yaml', type=str,
                        help='path to config file')
    parser.add_argument('opts', help='Modify config options using the command-line', default=None,
                        nargs=argparse.REMAINDER)
    
    # output directory
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default none)')
    parser.add_argument('--save_dir', default='trained_models/', type=str,
                        help='directory to save checkpoints')
    
    # training parameters
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument('--seed', default=42, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--device', default='cuda', type=str,
                        help='device to use for training / testing')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='number of data loading workers')
    parser.add_argument('--batch_size', default=2, type=int,
                        help='input batch size for training')
    parser.add_argument('--epochs', default=1000, type=int,
                        help='number of epochs to train')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='start epoch')
    parser.add_argument('--print_freq', default=1000, type=int,
                        help='print frequency')
    parser.add_argument('--save_freq', default=100, type=int,
                        help='save frequency')
    
    # optimizer parameters
    parser.add_argument('--lr', default=0.0001, type=float,
                        help='initial learning rate')
    parser.add_argument('--lrf', default=0.1, type=float,
                        help='learning rate factor')
    
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--amp', action='store_false', default=True,
                        help='use automatic mixed precision training')
    
    args = parser.parse_args()
    assert args.config is not None
    return args


def main(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    utils.init_distributed_mode(args)
    if args.rank in [-1, 0]:
        print(args)
        
    assert args.batch_size % args.world_size == 0, "--batch-size must be divisible by number of GPUs"
    
    if args.device == 'cuda':
        torch.backends.cudnn.benchmark = True
    if args.amp:
        assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."
        assert torch.backends.cudnn.version() >= 7603, "Amp requires cudnn >= 7603"
    
    device = torch.device(args.device)
    
    # load hyper parameters
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    # Setup experiment
    if args.rank in [-1, 0] and args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    
    # create tensorboard writer
    if args.rank in [-1, 0]:
        writer = SummaryWriter()
    
    # create model
    model = SSM_video_gen(config, is_train=True).to(device)
    tokenizer = CLIPTextTokenizer(model_dir=config["textencoder"])
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    
    start_epoch = args.start_epoch
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    
    # load checkpoint
    if args.resume.endswith('.pth'):
        print(f"Loading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        
        try:
            model.load_state_dict(checkpoint['model'], strict=True)
        except KeyError as e:
            s = "%s is not compatible with %s. Specify --resume=.../model.pth" % (args.resume, args.config)
            raise KeyError(s) from e
        
        start_epoch = checkpoint['epoch'] + 1
        scaler.load_state_dict(checkpoint['scaler'])
        del checkpoint
        print(f"Loaded checkpoint: {args.resume}")
    
    # freeze text encoder if needed
    if config["textencoder_freeze"]:
        for n, p in model.textencoder.named_parameters():
            if "textencoder" in n:
                if p.requires_grad:
                    print(f"Freezing {n}, from {p.requires_grad} to False")
                p.requires_grad = False
                
    # move to distributed mode
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    model_without_ddp = model.module
    
    # create optimizer and scheduler and model info
    p_to_optimize, n_p, n_p_o, layers = [], 0, 0, 0
    for p in model.module.parameters():
        n_p += p.numel()
        if p.requires_grad:
            p_to_optimize.append(p)
            n_p_o += p.numel()
            layers += 1
    print(f"Total number of parameters: {n_p}, number of parameters to optimize: {n_p_o}, number of layers: {layers}")
    
    args.lr *= max(1., args.world_size * args.batch_size / 64)
    optimizer = torch.optim.AdamW(p_to_optimize, lr=args.lr)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler.last_epoch = start_epoch
    
    # create data loaders
    dataset, _ = create_data_loader(
        data_dir=config["data_dir"],
        video_size=(config["video_height"], config["video_width"]),
        fps=config["fps"],
        max_num_frames=config["max_num_frames"],
    )
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=args.rank,
        shuffle=True,
        seed=args.seed
        )
    nw = min([os.cpu_count(), int(args.batch_size // dist.get_world_size()) if int(args.batch_size // dist.get_world_size()) > 1 else 0, 8])  # number of workers
    print(f"Using {nw} workers")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(args.batch_size // dist.get_world_size()),
        sampler=sampler,
        num_workers=nw,
        pin_memory=True,
        drop_last=True,
    )
    
    # start training
    start_time = time.time()
    total_time = start_time
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        loss_dict = train_one_epoch(
            model=model, tokenizer=tokenizer, data_loader=dataloader,
            optimizer=optimizer, device=device, epoch=epoch, max_norm=0.1,
            scaler=scaler, print_freq=args.print_freq, vae=vae,
        )
        scheduler.step()
        
        # log to tensorboard
        if args.rank in [-1, 0]:
            for k, v in loss_dict.items():
                writer.add_scalar(k, v, epoch)
        
        if args.rank in [-1, 0]:
            if (epoch % args.save_freq == 0) or (epoch == args.epochs - 1):
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'scaler': scaler.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                }, os.path.join(args.save_dir, f"model_{epoch}.pth"))
        
        if args.rank in [-1, 0]:
            print(f"Epoch: {epoch}, Time: {time.time() - start_time}")
            start_time = time.time()
    
    total_time = time.time() - total_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if args.rank in [-1, 0]:
        writer.close()
        print(f"Training time {total_time_str}")

if __name__ == '__main__':
    args = parse_args()
    main(args)