'''
Use this script to train the text model for one epoch
by doing the auto-regressive training
'''
import torch
import torch.nn as nn
from utils.misc import *
from typing import Iterable
from dataloader.text_tokenizer import CLIPTextTokenizer
from diffusers.models import AutoencoderKL, AutoencoderKLCogVideoX
import torch.amp
from diffusion.respace import SpacedDiffusion
import torch.nn.functional as F

def train_one_epoch(
    ssm_model: torch.nn.Module, tokenizer: CLIPTextTokenizer,
    data_loader: Iterable, optimizer: torch.optim.Optimizer,
    device: torch.device, epoch: int, max_norm: float = 0.1,
    scaler=None, print_freq: int = 100, decoder: nn.Module = None,
):
    ssm_model.train()
    decoder.train()
    metric_logger = MetricLogger(delimiter="; ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    for batch in metric_logger.log_every(data_loader, print_freq, header):
        # # clear cache and gradients in the model
        # torch.cuda.empty_cache()
        # ssm_model.zero_grad()
        # DiT_model.zero_grad()
        
        captions = tokenizer.tokenize(batch["txt"]).to(device)
        
        with torch.amp.autocast('cuda', enabled=scaler is not None):
            with torch.no_grad():
                # get the pooled representation of the text
                pooled_text = ssm_model.module.textencoder(**captions).pooler_output.unsqueeze(1).detach()
        
        with torch.amp.autocast('cuda', enabled=scaler is not None):
            # Move the forward pass inside the loop
            ssm_out = ssm_model(**captions)
            
            # check if nan exists in ssm_out
            if torch.isnan(ssm_out).any():
                print("NaN ssm_out")
                continue
            
            decoded = decoder(ssm_out)
            
            loss = F.mse_loss(decoded, pooled_text)
            
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            if max_norm != 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(ssm_model.parameters(), max_norm)
            
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            metric_logger.update(loss=loss.item())
            
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.synchronize_between_processes()
        print("Average loss: ", metric_logger.loss.global_avg)
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
            
            