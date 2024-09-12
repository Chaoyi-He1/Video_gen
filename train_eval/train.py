import torch
import torch.nn as nn
from utils.misc import *
from typing import Iterable
from dataloader.text_tokenizer import CLIPTextTokenizer


def train_one_epoch(
    model: torch.nn.Module, tokenizer: CLIPTextTokenizer,
    data_loader: Iterable, optimizer: torch.optim.Optimizer,
    device: torch.device, epoch: int, max_norm: float = 0.01,
    scaler=None, print_freq: int = 100
):
    model.train()
    metric_logger = MetricLogger(delimiter="; ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', SmoothedValue(window_size=10, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    for batch in metric_logger.log_every(data_loader, print_freq, header):
        videos = batch["mp4"].to(device)
        captions = tokenizer.tokenize(batch["txt"]).to(device)
        
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(videos, **captions)
            loss = loss_dict["loss"].mean()
        
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            if max_norm > 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
        
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: v.global_avg for k, v in metric_logger.meters.items()}
            