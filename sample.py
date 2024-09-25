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
    parser.add_argument('--resume', default='trained_models/model_48.pth', type=str, metavar='PATH',
                        help='path to latest checkpoint (default none)')
    parser.add_argument('--save_dir', default='sample/', type=str,
                        help='directory to save checkpoints')
    parser.add_argument('--test-dir', default='data/train_data/labels/', type=str,
                        help='directory to read test results')
    parser.add_argument('--test_num', default=3, type=int,
                        help='number of test results to generate')
    
    # training parameters
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")  # Choice doesn't affect training
    parser.add_argument('--seed', default=42, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--device', default='cuda', type=str,
                        help='device to use for training / testing')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='number of data loading workers')
    parser.add_argument('--batch_size', default=20, type=int,
                        help='input batch size for training')

    parser.add_argument('--print_freq', default=20, type=int,
                        help='print frequency')
    
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
    # Setup
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # print the gpu model name to check if it is A100
    print(torch.cuda.get_device_name())
    
    # load hyper parameters
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    # Setup experiment
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    
    # create model
    model = SSM_video_gen(config, is_train=True)
    tokenizer = CLIPTextTokenizer(model_dir=config["textencoder"])
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    
    scaler = scaler = torch.amp.GradScaler('cuda', enabled=args.amp) if args.amp else None
    
    # load checkpoint
    assert args.resume.endswith('.pth'), "Only .pth files are supported"
    print(f"Loading checkpoint: {args.resume}")
    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
    
    try:
        model.load_state_dict(checkpoint['model'], strict=True)
    except KeyError as e:
        s = "%s is not compatible with %s. Specify --resume=.../model.pth" % (args.resume, args.config)
        raise KeyError(s) from e
    
    scaler.load_state_dict(checkpoint['scaler'])
    
    # Check if the model is loaded correctly
    for p_model, p_checkpoint in zip(model.parameters(), checkpoint['model'].values()):
        assert torch.equal(p_model, p_checkpoint), "Model not loaded correctly"
    print("Model loaded correctly")
    del checkpoint
    
    model.to(device)
    
    # read text captions from a list of .txt files
    test_list = [os.path.join(args.test_dir, f) for f in os.listdir(args.test_dir) if f.endswith('.txt')]
    test_list = random.sample(test_list, args.test_num)
    
    captions = []
    for txt_file in test_list:
        with open(txt_file, 'r') as f:
            captions.append(f.readline().strip())
    
    # generate videos
    sampling(
        model=model, tokenizer=tokenizer, 
        data_loader=captions, device=device,
        scaler=scaler, print_freq=1, vae=vae, fps=config["fps"],
        output_dir=args.save_dir,
    )
    
if __name__ == '__main__':
    args = parse_args()
    main(args)