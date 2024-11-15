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
from diffusers import AutoencoderKLCogVideoX
import cv2
import torch.utils.data as data
from torchvision.utils import save_image
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--config', default='config/cfg.yaml', type=str,
                        help='path to config file')
    parser.add_argument('opts', help='Modify config options using the command-line', default=None,
                        nargs=argparse.REMAINDER)
    
    # output directory
    parser.add_argument('--resume', default='trained_models/model_2.pth', type=str, metavar='PATH',
                        help='path to latest checkpoint (default none)')
    parser.add_argument('--save_dir', default='sample/', type=str,
                        help='directory to save checkpoints')
    parser.add_argument('--test-dir', default='data/train_data/', type=str,
                        help='directory to read test results')
    parser.add_argument('--test_num', default=3, type=int,
                        help='number of test results to generate')
    
    # training parameters
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument('--seed', default=42, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--device', default='cuda:1', type=str,
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
    # torch.manual_seed(args.seed)
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    
    torch.set_grad_enabled(False)
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    # print the gpu model name to check if it is A100
    print(torch.cuda.get_device_name())
    
    # load hyper parameters
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # create model
    vae = AutoencoderKLCogVideoX.from_pretrained("THUDM/CogVideoX-2b", subfolder="vae", torch_dtype=torch.float16).to(device)
    
    # create data loaders
    dataset = PTDataset(
        data_dir=config["data_dir"]
    )
    sampler = data.RandomSampler(dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        pin_memory=True,
        drop_last=True,
    )
    
    # get one sample from the dataloader
    for sample in dataloader:
        videos = sample["mp4"][:, :28, ...].to(device).half()
        videos = videos.transpose(1, 2) # (B, F, C, H, W) -> (B, C, F, H, W)
        
        # use the vae to encode the image and then decode it
        with torch.no_grad():
            enc_v = vae.encode(videos).latent_dist.sample().mul_(0.18215)
            print("Encoded image range: ", enc_v.min(), enc_v.max())
            dec_v = vae.decode(enc_v / 0.18215).sample.squeeze(0)
            print("Decoded image range: ", dec_v.min(), dec_v.max())
        
        # generate a histogram of the encoded image values
        plt.hist(dec_v.cpu().numpy().flatten(), bins=100)
        plt.savefig("histogram.png")
        
        # save the decoded video
        dec_v = dec_v.permute(1, 2, 3, 0).cpu().numpy()
        dec_v = ((dec_v - dec_v.min()) / (dec_v.max() - dec_v.min()) * 255).astype('uint8')
        dec_v = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in dec_v]
        video_path = "decoded_video.mp4"
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 8, (dec_v[0].shape[1], dec_v[0].shape[0]))
        for frame in dec_v:
            out.write(frame)
        out.release()
        return

if __name__ == '__main__':
    args = parse_args()
    main(args)