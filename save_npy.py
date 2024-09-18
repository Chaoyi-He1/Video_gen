import torch
from transformers import CLIPTextModelWithProjection, CLIPTextModel, AutoTokenizer
from transformers import CLIPTokenizer
import os
from dataloader import create_data_loader, CLIPTextTokenizer
import yaml
import torch.utils.data
from tqdm import tqdm
import numpy as np
import os


if __name__ == "__main__":
    config = "config/cfg.yaml"
    save_dir = "data/train_data/videos_pt"
    overwrite = False
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
        dataset, batch_size=1,
        sampler=sampler, num_workers=8,
        drop_last=False)
    
    for i, batch in tqdm(enumerate(loader)):
        if i >= 8000:
            break
        mp4 = batch["mp4"][0].to(torch.uint8).numpy()
        video_name = batch["video_name"][0]
        if len(mp4) > 0:
            if not os.path.exists(os.path.join(save_dir, f"{video_name}.npz")) or overwrite:
                np.savez_compressed(os.path.join(save_dir, f"{video_name}.pt"), array=mp4)
