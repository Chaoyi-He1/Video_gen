from .CLIPTextEncoder import SSM_input_projection
from .ex_bi_mamba2 import BiMamba2_1D
from .Diffusion_Transformer import DiT_models
import torch.nn as nn
import torch
from diffusion import create_diffusion


class SSM_video_gen(nn.Module):
    def __init__(self, config: dict, is_train: bool=True) -> None:
        super(SSM_video_gen, self).__init__()
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
        
        self.dit_config = {
            "input_size": config["dit_input_size"],
            "in_channels": config["dit_in_channels"],
            "learn_sigma": config["dit_learn_sigma"],
            "latent_size": config["input_dim"],
        }
        
        self.dit_name = config["dit_name"]
        
        self.textencoder = SSM_input_projection(self.clip_config)
        
        if self.textencoder.textencoder.config.hidden_size != config["input_dim"]:
            self.mamba_config["cin"] = self.textencoder.textencoder.config.hidden_size
            self.mamba_config["cout"] = self.textencoder.textencoder.config.hidden_size
            self.mamba_config["d_model"] = self.textencoder.textencoder.config.hidden_size
            self.dit_config["latent_size"] = self.textencoder.textencoder.config.hidden_size
            print("Warning: input_dim is not equal to the hidden_size of the text encoder. \
                   input_dim={} is updated to hidden_size of the text encoder: {}.".format(
                       config["input_dim"], 
                       self.textencoder.textencoder.config.hidden_size))
        
        self.mamba = BiMamba2_1D(**self.mamba_config)
        self.dit = DiT_models[self.dit_name](**self.dit_config)
        
        self.diffusion = create_diffusion(timestep_respacing="") if is_train else \
            create_diffusion(str(config["num_sampling_steps"]))
            # default: 1000 steps, linear noise schedule for training, MSE loss
        self.is_train = is_train
    
    def forward_training(self, video: torch.Tensor, **Token_text):
        b, f, c, h, w = video.shape 
        
        TextEncoderOutput = self.textencoder(**Token_text)
        
        ssm_in = TextEncoderOutput.permute(0, 2, 1) # (batch, dim, num_frames)
        ssm_out = self.mamba(ssm_in)
        ssm_out = ssm_out.permute(0, 2, 1)  # (batch, num_frames, dim)
        
        # combine ssm_out first two dimensions, as well as video first two dimensions
        ssm_out = ssm_out.view(b*f, -1)
        video = video.view(b*f, c, h, w)
        
        t = torch.randint(0, self.diffusion.num_timesteps, (video.shape[0],), device=video.device)
        model_kwargs = dict(y=ssm_out)
        loss_dict = self.diffusion.training_losses(self.dit, video, t, model_kwargs)
        return loss_dict
    
    def forward_generation(self, **Token_text):
        TextEncoderOutput = self.textencoder(**Token_text)
        
        ssm_in = TextEncoderOutput.permute(0, 2, 1) # (batch, dim, num_frames)
        ssm_out = self.mamba(ssm_in)
        ssm_out = ssm_out.permute(0, 2, 1)  # (batch, num_frames, dim)
        
        b, f, _ = ssm_out.shape
        # combine ssm_out first two dimensions, as well as video first two dimensions
        ssm_out = ssm_out.view(b*f, -1)
        
        z = torch.randn(b*f, 4, self.dit.input_size, self.dit.input_size, device=ssm_out.device)
        
        # Setup classifier-free guidance:
        z = torch.cat([z, z], 0)
        # repeat self.dit.y_embedder.cfg_embedding b*f times and concatenate with ssm_out
        y_null = self.dit.y_embedder.cfg_embedding.repeat(b*f, 1)
        y = torch.cat([ssm_out, y_null], 0)
        model_kwargs = dict(y=y, cfg_scale=4.0)
        
        # Sample images:
        samples = self.diffusion.p_sample_loop(
            self.dit.forward_with_cfg, z.shape, z, clip_denoised=False, 
            model_kwargs=model_kwargs, progress=True, device=z.device
        )
        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        return samples
    
    def forward(self, video: torch.Tensor=None, **Token_text):
        if self.training and video is not None:
            return self.forward_training(video, **Token_text)
        else:
            return self.forward_generation(**Token_text)
        