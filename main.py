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
from dataloader import create_data_loader
from model import SSM_video_gen
from train_eval import *

# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

torch.multiprocessing.set_sharing_strategy('file_system')


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--config', default='config/config.yaml', type=str,
                        help='path to config file')
    parser.add_argument('opts', help='Modify config options using the command-line', default=None,
                        nargs=argparse.REMAINDER)
    
    # output directory
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default none)')
    parser.add_argument('--save_dir', default='trained_models/', type=str,
                        help='directory to save checkpoints')
    
    # training parameters
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--device', default='cuda', type=str,
                        help='device to use for training / testing')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='number of data loading workers')
    parser.add_argument('--batch_size', default=4, type=int,
                        help='input batch size for training')
    parser.add_argument('--epochs', default=1000, type=int,
                        help='number of epochs to train')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='start epoch')
    parser.add_argument('--print_freq', default=10, type=int,
                        help='print frequency')
    parser.add_argument('--save_freq', default=100, type=int,
                        help='save frequency')
    
    # optimizer parameters
    parser.add_argument('--lr', default=0.0001, type=float,
                        help='initial learning rate')
    parser.add_argument('--lrf', default=0.1, type=float,
                        help='learning rate factor')
    
    # distributed training parameters
    parser.add_argument('--world_size', default=8, type=int,
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