from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection
import torch.nn as nn
import torch


def CreateTextEncoder(model_name="openai/clip-vit-large-patch16", freeze=False, projection=False):
    model_candidate = [
        "openai/clip-vit-base-patch32",
        "openai/clip-vit-base-patch16",
        "openai/clip-vit-large-patch32",
        "openai/clip-vit-large-patch16",
    ]
    
    assert model_name in model_candidate, f"model_name should be one of {model_candidate}"
    
    TextEncoder = CLIPTextModel.from_pretrained(model_name) if not projection \
        else CLIPTextModelWithProjection.from_pretrained(model_name)
        
    if freeze:
        for param in TextEncoder.parameters():
            param.requires_grad = False
    return TextEncoder


class SSM_input_projection(nn.Module):
    def __init__(self, config: dict):
        super(SSM_input_projection, self).__init__()
        self.input_dim = config["input_dim"]
        self.num_frames = config["num_frames"]
        
        self.textencoder = CreateTextEncoder(
            model_name=config["textencoder"],
            freeze=config["freeze"],
            projection=config["projection"],
        )
        
        self.querries = nn.Conv1d(1, self.num_frames, 1)
        
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.textencoder.config.hidden_size,
            nhead=config["nhead"],
            dim_feedforward=config["dim_feedforward"],
            dropout=config["dropout"],
            activation=config["activation"],
            batch_first=True,
        )
        self.projection_block = nn.TransformerDecoder(
            self.decoder_layer, 
            num_layers=config["num_layers"]
        )
    
    def forward(self, Token_text: dict):
        TextEncoderOutput = self.textencoder(**Token_text)
        last_hidden_state = TextEncoderOutput.last_hidden_state
        
        querries = self.querries(last_hidden_state.unsqueeze(1)).squeeze(1)
        
        tgt_mask = (1 - torch.tril(torch.ones(self.num_frames, self.num_frames))).bool().to(querries.device)
        
        SSM_input = self.projection_block(
            tgt=querries,
            memory=last_hidden_state,
            tgt_mask=tgt_mask
        )
        
        return SSM_input