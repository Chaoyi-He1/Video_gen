import torch
import torch.nn as nn
from utils.misc import *
from typing import Iterable
from dataloader.text_tokenizer import CLIPTextTokenizer
from diffusers.models import AutoencoderKL
import torch.amp


def train_one_epoch(
    model: torch.nn.Module, tokenizer: CLIPTextTokenizer,
    data_loader: Iterable, optimizer: torch.optim.Optimizer,
    device: torch.device, epoch: int, max_norm: float = 0.1,
    scaler=None, print_freq: int = 100, vae: AutoencoderKL = None,
    mini_frames: int = 40,
):
    model.train()
    metric_logger = MetricLogger(delimiter="; ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', SmoothedValue(window_size=10, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    for batch in metric_logger.log_every(data_loader, print_freq, header):
        videos = batch["mp4"].to(device)
        captions = tokenizer.tokenize(batch["txt"]).to(device)
        
        b, f, c, h, w = videos.shape
        assert b*f % mini_frames == 0, f"Batch x Frames ({b*f}) should be divisible by mini_frames ({mini_frames})"
        
        with torch.amp.autocast('cuda', enabled=scaler is not None): 
            with torch.no_grad():
                videos = videos.view(b*f, c, h, w)
                videos = torch.cat([vae.module.encode(videos[i: i+mini_frames, ...]).latent_dist.sample().mul_(0.18215) 
                                    for i in range(0, b*f, mini_frames)], dim=0) 
                
                videos = videos.view(b, f, c+1, h//8, w//8)
        
        # Loop over smaller mini-batches (chunks)
        for i in range(0, b*f, mini_frames):
            with torch.amp.autocast('cuda', enabled=scaler is not None):
                # Move the forward pass inside the loop
                ssm_out, video = model(videos, **captions)
                
                # Get a random timestep
                t = torch.randint(0, model.module.diffusion.num_timesteps, (mini_frames,), device=video.device)

                # Compute the loss
                model_kwargs = dict(y=ssm_out[i: i + mini_frames])
                loss_dict = model.module.diffusion.training_losses(model.module.dit, video[i: i + mini_frames], t, model_kwargs)
                loss = loss_dict["loss"].mean()
                
                if loss.isnan():
                    print("NaN loss")
                    optimizer.zero_grad()
                    # clear cache and gradients in the model
                    torch.cuda.empty_cache()
                    model.zero_grad()
                    continue
                
                optimizer.zero_grad()
                
                # Backward pass
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                if max_norm > 0:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                
                # Optimizer step
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                # Update metrics
                metric_logger.update(loss=loss.item())
                
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])    
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: v.global_avg for k, v in metric_logger.meters.items()}
            