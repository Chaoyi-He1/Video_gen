from .CLIPTextEncoder import SSM_input_projection
from .ex_bi_mamba2 import BiMamba2_1D
from .Diffusion_Transformer import DiT_models
import torch.nn as nn
import torch
from diffusion import create_diffusion


class SSM_block(nn.Module):
    def __init__(self, config: dict, is_train: bool=True) -> None:
        super(SSM_block, self).__init__()
        self.clip_config = {
            "input_dim": config["input_dim"],
            "num_frames": config["num_frames"],
            "textencoder": config["textencoder"],
            "textencoder_freeze": config["textencoder_freeze"],
            "textencoder_projection": config["textencoder_projection"],
            "decoder_nhead": config["decoder_nhead"],
            "decoder_dim_feedforward": config["decoder_dim_feedforward"],
            "decoder_dropout": config["decoder_dropout"],
            "decoder_activation": config["decoder_activation"],
            "decoder_num_layers": config["decoder_num_layers"],
        }
        
        self.mamba_config = {
            "cin": config["input_dim"],
            "cout": config["input_dim"],
            "d_model": config["input_dim"],
            "n_layer": config["mamba_n_layer"],
            "d_state": config["mamba_d_state"],
        }
        
        self.textencoder = SSM_input_projection(self.clip_config)
        
        if self.textencoder.textencoder.config.hidden_size != config["input_dim"]:
            self.mamba_config["cin"] = self.textencoder.textencoder.config.hidden_size
            self.mamba_config["cout"] = self.textencoder.textencoder.config.hidden_size
            self.mamba_config["d_model"] = self.textencoder.textencoder.config.hidden_size
            
            print("Warning: input_dim is not equal to the hidden_size of the text encoder. \
                   input_dim={} is updated to hidden_size of the text encoder: {}.".format(
                       config["input_dim"], 
                       self.textencoder.textencoder.config.hidden_size))
        
        self.mamba = BiMamba2_1D(**self.mamba_config)
    
    def forward(self, **Token_text):
        TextEncoderOutput = self.textencoder(**Token_text)
        
        ssm_in = TextEncoderOutput  # (batch, num_frames, dim)
        ssm_out = self.mamba(ssm_in.transpose(1, 2))
        ssm_out = ssm_out.transpose(1, 2)  # (batch, num_frames, dim)
        
        # concat ssm_out with previous ssm_out
        history = torch.cat([torch.zeros_like(ssm_out[:, :1, :]), ssm_out[:, :-1, :]], 1)
        ssm_out = torch.cat([ssm_out, history], 2)
        
        return ssm_out


def create_model(config: dict, is_train: bool=True):
    ssm_block = SSM_block(config, is_train)
    
    dit_config = {
        "input_size": config["dit_input_size"],
        "in_channels": config["dit_in_channels"],
        "learn_sigma": config["dit_learn_sigma"],
        "latent_size": config["input_dim"] * 2,
    }
    
    dit_name = config["dit_name"]
    
    if ssm_block.textencoder.textencoder.config.hidden_size != config["input_dim"]:
        dit_config["latent_size"] = ssm_block.textencoder.textencoder.config.hidden_size
        print("Warning: input_dim is not equal to the hidden_size of the text encoder. \
               input_dim={} is updated to hidden_size of the text encoder: {}.".format(
                   config["input_dim"], 
                   ssm_block.textencoder.textencoder.config.hidden_size))
    
    dit_block = DiT_models[dit_name](**dit_config)
    diffusion = create_diffusion(timestep_respacing="") if is_train else \
        create_diffusion(str(config["num_sampling_steps"]))
    
    return ssm_block, dit_block, diffusion