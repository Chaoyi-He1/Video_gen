import torch
from transformers import CLIPTextModelWithProjection, CLIPTextModel, AutoTokenizer
from transformers import CLIPTokenizer
import os
from dataloader import create_data_loader, CLIPTextTokenizer
import yaml
import torch.utils.data
from tqdm import tqdm


if __name__ == "__main__":
    config = "config/cfg.yaml"
    with open(config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    dataset, _ = create_data_loader(
        data_dir=config["data_dir"],
        video_size=(config["video_height"], config["video_width"]),
        fps=config["fps"],
        max_num_frames=config["max_num_frames"],
    )
    sampler = torch.utils.data.SequentialSampler(dataset)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=8,
        sampler=sampler, num_workers=4,
        drop_last=False)
    
    for batch in tqdm(loader):
        print(batch)