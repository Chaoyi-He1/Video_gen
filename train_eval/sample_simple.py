import torch
import torch.nn as nn
from utils.misc import *
from typing import Iterable
from dataloader.text_tokenizer import CLIPTextTokenizer
from diffusers.models import AutoencoderKL
import cv2
from diffusion.diffusion_utils import Diffusion_utils
from torchvision.utils import save_image

def sampling(
    ssm_model: torch.nn.Module, tokenizer: CLIPTextTokenizer,
    diffusion: Diffusion_utils, DiT_model: torch.nn.Module,
    data_loader: Iterable, device: torch.device, 
    scaler=None, print_freq: int = 1, vae: AutoencoderKL = None,
    fps: float = 8, output_dir: str = "sample/", channels: int = 4,
):
    ssm_model.eval()
    DiT_model.eval()
    assert vae is not None, "VAE model is required for sampling"
    
    for i, sample in enumerate(data_loader):
        ssm_model.to(device)
        DiT_model.to(device)
        
        captions = tokenizer.tokenize(sample).to(device)
        
        with torch.amp.autocast('cuda', enabled=scaler is not None):
            with torch.no_grad():
                ssm_out = ssm_model(**captions)
                b, f, _ = ssm_out.shape
                
                x_shape = (channels, f, DiT_model.input_size, DiT_model.input_size)
                
                # Sample images:
                samples = diffusion.sample(
                    shape=x_shape, context=ssm_out, sample=1, bestof=False, step=4, 
                    model=DiT_model, 
                )
                samples = samples[0, ...]  # (b, c, f, h, w)
        
                if torch.isnan(samples).any():
                    print("NaN samples")
                    continue
                
                DiT_model.to("cpu")
                ssm_model.to("cpu")
                # samples = samples.transpose(1, 2).view(b*f, 4, DiT_model.input_size, DiT_model.input_size)
                samples = vae.decode(samples / 0.18215).sample
                img = samples[0, :, 0, ...]
                save_image(img, f"{output_dir}/sample_{i}.png")
                print("samples range: ", samples.min(), samples.max())
                # samples = (samples - samples.min()) / (samples.max() - samples.min())
                
                # check if nan exists
                if torch.isnan(samples).any():
                    print("NaN samples")
                    continue
                b, c, f, h, w = samples.shape
                # assert f == bf // b, "Batch size mismatch"
                # samples = samples.view(b, f, c, h, w) # (b*f, c, h, w) -> (b, f, c, h, w)
        
        for j in range(b):
            # Save the video in the batch
            video = samples[j].permute(1, 2, 3, 0).cpu().numpy()
            video = ((video - video.min()) / (video.max() - video.min()) * 255).astype('uint8')
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
            