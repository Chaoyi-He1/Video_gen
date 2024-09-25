import torch
import torch.nn as nn
from utils.misc import *
from typing import Iterable
from dataloader.text_tokenizer import CLIPTextTokenizer
from diffusers.models import AutoencoderKL
import cv2


def sampling(
    model: torch.nn.Module, tokenizer: CLIPTextTokenizer,
    data_loader: Iterable, device: torch.device, 
    scaler=None, print_freq: int = 1, vae: AutoencoderKL = None,
    fps: float = 8, output_dir: str = "sample/",
):
    model.eval()
    assert vae is not None, "VAE model is required for sampling"
    
    for i, sample in enumerate(data_loader):
        captions = tokenizer.tokenize(sample).to(device)
        b = captions["input_ids"].shape[0]
        
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            with torch.no_grad():
                samples = model(None, **captions)
                samples = vae.decode(samples / 0.18215).sample
                bf, c, h, w = samples.shape
                f = bf // b
                samples = samples.view(b, f, c, h, w)
        
        for j in range(b):
            # Save the video in the batch
            video = samples[j].permute(0, 2, 3, 1).cpu().numpy()
            # video = (video * 255).astype('uint8')
            video = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in video]
            video_path = f"{output_dir}/sample_{i}_{j}.mp4"
            # check if the folder exists
            os.makedirs(os.path.dirname(video_path), exist_ok=True)
            out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            for frame in video:
                out.write(frame)
            out.release()
            # save the captions
            with open(f"{output_dir}/sample_{i}_{j}.txt", 'w') as f:
                f.write(sample)
    print("Sampling done!")
    return None
            