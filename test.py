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
import cv2


if __name__ == "__main__":
    config = "config/cfg.yaml"
    save_dir = "data/train_data/videos_pt_low"
    overwrite = False
    with open(config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    dataset, _ = create_data_loader(
        data_dir=config["data_dir"],
        video_size=(config["video_height"], config["video_width"]),
        fps=8,
        max_num_frames=56,
    )
    sampler = torch.utils.data.RandomSampler(dataset)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1,
        sampler=sampler, num_workers=8,
        drop_last=False)
    
    for i, batch in tqdm(enumerate(loader)):
        if i >= 3:
            break
        mp4 = batch["mp4"][0]   
        mp4 = mp4.permute(0, 2, 3, 1).contiguous().to(torch.uint8).numpy()# mps is a numpy with shape (1, config["max_num_frames"], 3, config["video_height"], config["video_width"])
        # transfer it to video and save it
        # mp4 = mp4[:112:2, :, :, :]
        print(mp4.shape)
        video = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in mp4]
        video_path = "sample/sample_video_{}.mp4".format(i)
        # check if the folder exists
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), config["fps"], (config["video_width"], config["video_height"]))
        for frame in video:
            out.write(frame)
        out.release()
        print("Video saved!")
    