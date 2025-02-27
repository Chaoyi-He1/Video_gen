import torch
import torch.nn as nn
from utils.misc import *
from typing import Iterable
from dataloader.text_tokenizer import CLIPTextTokenizer
from diffusers.models import AutoencoderKL, AutoencoderKLCogVideoX
import torch.amp
from diffusion.respace import SpacedDiffusion


def train_one_epoch(
    ssm_model: torch.nn.Module, tokenizer: CLIPTextTokenizer,
    diffusion: SpacedDiffusion, DiT_model: torch.nn.Module,
    data_loader: Iterable, optimizer: torch.optim.Optimizer,
    device: torch.device, epoch: int, max_norm: float = 0.1,
    scaler=None, print_freq: int = 100, vae: AutoencoderKLCogVideoX = None,
    mini_frames: int = 40,
):
    ssm_model.train()
    DiT_model.train()
    metric_logger = MetricLogger(delimiter="; ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    for batch in metric_logger.log_every(data_loader, print_freq, header):
        # # clear cache and gradients in the model
        # torch.cuda.empty_cache()
        # ssm_model.zero_grad()
        # DiT_model.zero_grad()
        
        videos = batch["mp4"].half().to(device)
        captions = tokenizer.tokenize(batch["txt"]).to(device)
        
        b, f, c, h, w = videos.shape
        # assert b*f % mini_frames == 0, f"Batch x Frames ({b*f}) should be divisible by mini_frames ({mini_frames})"
        
        # send vae to device for video encoding, then send it back to cpu for gpu memory saving
        vae.to(device)
        vae.eval()
        with torch.amp.autocast('cuda', enabled=scaler is not None): 
            with torch.no_grad():
                videos = videos.transpose(1, 2) # (b, f, c, h, w) -> (b, c, f, h, w)
                # videos = torch.cat([vae.module.encode(videos[i: i+mini_frames, ...]).latent_dist.sample().mul_(0.18215) 
                #                     for i in range(0, b*f, mini_frames)], dim=0) 
                videos = vae.encode(videos).latent_dist.sample().mul_(0.18215)
        # send vae back to cpu and free gpu memory
        vae.to('cpu')
        
        # Loop over smaller mini-batches (chunks)
        with torch.amp.autocast('cuda', enabled=scaler is not None):
            # Move the forward pass inside the loop
            ssm_out = ssm_model(**captions)
            
            # check if nan exists in ssm_out
            if torch.isnan(ssm_out).any():
                print("NaN ssm_out")
                continue
            
            # Get a random timestep
            t = torch.randint(0, diffusion.num_timesteps, (b,), device=videos.device)

            # Compute the loss
            model_kwargs = dict(y=ssm_out)
            loss_dict = diffusion.training_losses(DiT_model, videos, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            
            if loss.isnan():
                print("NaN loss")
                optimizer.zero_grad()
                # clear cache and gradients in the model
                torch.cuda.empty_cache()
                ssm_model.zero_grad()
                DiT_model.zero_grad()
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
                # Gradient clipping for ssm_model and DiT_model
                torch.nn.utils.clip_grad_norm_(ssm_model.parameters(), max_norm)
                torch.nn.utils.clip_grad_norm_(DiT_model.parameters(), max_norm)
            
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
            