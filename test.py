import torch
from transformers import CLIPTextModelWithProjection, CLIPTextModel, AutoTokenizer
from transformers import CLIPTokenizer
import os
from dataloader import PTDataset, CLIPTextTokenizer
import yaml
import torch.utils.data
from tqdm import tqdm
import numpy as np


if __name__ == "__main__":
    dataset = PTDataset(data_dir="data/train_data")
    print(len(dataset))
    sampler = torch.utils.data.SequentialSampler(dataset)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=8,
        sampler=sampler, num_workers=0,
        drop_last=False)
    
    for batch in tqdm(loader):
        print(batch["mp4"].shape)