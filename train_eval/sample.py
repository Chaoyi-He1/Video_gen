import torch
import torch.nn as nn
from utils.misc import *
from typing import Iterable
from dataloader.text_tokenizer import CLIPTextTokenizer
from diffusers.models import AutoencoderKL
import cv2
from diffusion.respace import SpacedDiffusion


def sampling(
    ssm_model: torch.nn.Module, tokenizer: CLIPTextTokenizer,
    diffusion: SpacedDiffusion, DiT_model: torch.nn.Module,
    data_loader: Iterable, device: torch.device, 
    scaler=None, print_freq: int = 1, vae: AutoencoderKL = None,
    fps: float = 8, output_dir: str = "sample/",
):
    ssm_model.eval()
    DiT_model.eval()
    assert vae is not None, "VAE model is required for sampling"
    
    for i, sample in enumerate(data_loader):
        captions = tokenizer.tokenize(sample).to(device)
        
        with torch.amp.autocast('cuda', enabled=scaler is not None):
            with torch.no_grad():
                ssm_out = ssm_model(**captions)
                b, f, _ = ssm_out.shape
                
                z = torch.randn(b, 4, f, DiT_model.input_size, DiT_model.input_size, device=ssm_out.device)
                # Setup classifier-free guidance:
                z = torch.cat([z, z], 0)
                
                # repeat self.dit.y_embedder.cfg_embedding b*f times and concatenate with ssm_out
                y_null = DiT_model.y_embedder.cfg_embedding.repeat(b*f, 1).view(b, f, -1)
                y = torch.cat([ssm_out, y_null], 0)
                model_kwargs = dict(y=y, cfg_scale=4.0)
                
                # Sample images:
                samples = diffusion.p_sample_loop(
                    DiT_model.forward_with_cfg, z.shape, z, clip_denoised=False, 
                    model_kwargs=model_kwargs, progress=True, device=z.device
                )
                samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        
                if torch.isnan(samples).any():
                    print("NaN samples")
                    continue
                
                samples = samples.transpose(1, 2).view(b*f, 4, DiT_model.input_size, DiT_model.input_size)
                samples = torch.cat([vae.decode(samples[i:i+4, ...] / 0.18215).sample for i in range(0, b*f, 4)], dim=0)
                print("samples range: ", samples.min(), samples.max())
                samples = (samples - samples.min()) / (samples.max() - samples.min())
                
                # check if nan exists
                if torch.isnan(samples).any():
                    print("NaN samples")
                    continue
                bf, c, h, w = samples.shape
                assert f == bf // b, "Batch size mismatch"
                samples = samples.view(b, f, c, h, w)
        
        for j in range(b):
            # Save the video in the batch
            video = samples[j].permute(0, 2, 3, 1).cpu().numpy()
            video = (video * 255).astype('uint8')
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
            