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
from dataloader import create_data_loader, CLIPTextTokenizer, PTDataset
from model import create_model
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
    parser.add_argument('--resume', default='trained_models/model_70.pth', type=str, metavar='PATH',
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
    parser.add_argument('--batch_size', default=20, type=int,
                        help='input batch size for training')
    parser.add_argument('--epochs', default=1000, type=int,
                        help='number of epochs to train')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='start epoch')
    parser.add_argument('--print_freq', default=20, type=int,
                        help='print frequency')
    parser.add_argument('--save_freq', default=5, type=int,
                        help='save frequency')
    
    # optimizer parameters
    parser.add_argument('--lr', default=0.0001, type=float,
                        help='initial learning rate')
    parser.add_argument('--lrf', default=0.1, type=float,
                        help='learning rate factor')
    parser.add_argument('--clip_max_norm', default=.1, type=float,
                        help='gradient clipping max norm')
    
    # distributed training parameter
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
    # print the gpu model name to check if it is A100
    print(torch.cuda.get_device_name())
    
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
    ssm_model, dit_model, diffusion = create_model(config, is_train=True)
    tokenizer = CLIPTextTokenizer(model_dir=config["textencoder"])
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    
    start_epoch = args.start_epoch
    scaler = scaler = torch.amp.GradScaler('cuda', enabled=args.amp) if args.amp else None
    
    # load checkpoint
    if args.resume.endswith('.pth'):
        print(f"Loading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        
        try:
            ssm_model.load_state_dict(checkpoint['ssm_model'], strict=True)
            dit_model.load_state_dict(checkpoint['dit_model'], strict=True)
        except KeyError as e:
            s = "%s is not compatible with %s. Specify --resume=.../model.pth" % (args.resume, args.config)
            raise KeyError(s) from e
        
        for p_model, p_checkpoint in zip(ssm_model.parameters(), checkpoint['ssm_model'].values()):
            assert torch.equal(p_model, p_checkpoint), "Model and checkpoint parameters are not equal"
        print("SSM model loaded correctly")
        for p_model, p_checkpoint in zip(dit_model.parameters(), checkpoint['dit_model'].values()):
            assert torch.equal(p_model, p_checkpoint), "Model and checkpoint parameters are not equal"
        print("DiT model loaded correctly")
        
        start_epoch = checkpoint['epoch'] + 1
        scaler.load_state_dict(checkpoint['scaler'])
        del checkpoint
        print(f"Loaded checkpoint: {args.resume}")
    
    ssm_model.to(device)
    dit_model.to(device)
    
    # freeze text encoder if needed
    if config["textencoder_freeze"]:
        for n, p in ssm_model.textencoder.named_parameters():
            if "textencoder" in n:
                if p.requires_grad:
                    print(f"Freezing {n}, from {p.requires_grad} to False")
                p.requires_grad = False
                
    # move to distributed mode
    ssm_model = torch.nn.parallel.DistributedDataParallel(ssm_model, device_ids=[args.gpu])
    dit_model = torch.nn.parallel.DistributedDataParallel(dit_model, device_ids=[args.gpu])
    vae = torch.nn.parallel.DistributedDataParallel(vae, device_ids=[args.gpu])
    ssm_model_without_ddp = ssm_model.module
    dit_model_without_ddp = dit_model.module
    
    # create optimizer and scheduler and model info
    p_to_optimize, n_p, n_p_o, layers = [], 0, 0, 0
    for p in ssm_model.module.parameters():
        n_p += p.numel()
        if p.requires_grad:
            p_to_optimize.append(p)
            n_p_o += p.numel()
            layers += 1
    for p in dit_model.module.parameters():
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
    dataset = PTDataset(
        data_dir=config["data_dir"]
    )
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=args.rank,
        shuffle=True,
        seed=args.seed
        )
    nw = min([os.cpu_count(), int(args.batch_size // dist.get_world_size()) if int(args.batch_size // dist.get_world_size()) > 1 else 0, 2])  # number of workers
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(args.batch_size // dist.get_world_size()),
        sampler=sampler,
        # num_workers=nw,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Using {nw} workers")
    
    # start training
    start_time = time.time()
    total_time = start_time
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        loss_dict = train_one_epoch(
            ssm_model=ssm_model, tokenizer=tokenizer, 
            diffusion=diffusion, DiT_model=dit_model,
            data_loader=dataloader, optimizer=optimizer, 
            device=device, epoch=epoch, max_norm=args.clip_max_norm,
            scaler=scaler, print_freq=args.print_freq, vae=vae,
            mini_frames=config["mini_frames"],
        )
        scheduler.step()
        
        # log to tensorboard
        if args.rank in [-1, 0]:
            for k, v in loss_dict.items():
                writer.add_scalar(k, v, epoch)
        
        if args.rank in [-1, 0]:
            if (epoch % args.save_freq == 0) or (epoch == args.epochs - 1):
                utils.save_on_master({
                    'ssm_model': ssm_model_without_ddp.state_dict(),
                    'dit_model': dit_model_without_ddp.state_dict(),
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