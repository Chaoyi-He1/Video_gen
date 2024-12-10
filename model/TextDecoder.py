'''
Used for the CLIPTextEncoder's training,
the TextDecoder is used to decode the encoder's output from the CLIPTextEncoder
CLIPTextEncoder output: (B, F, D) where B is the batch size, F is the number of frames, and D is the dimension
The TextDecoder will decode the output to the CLIP encoder's pooler output
'''
import torch.nn as nn
import torch
import torch.nn.functional as F
from .CLIPTextEncoder import SSM_input_projection

class TextDecoder(nn.Module):
    def __init__(self, config: dict) -> None:
        super(TextDecoder, self).__init__()
        self.query = nn.Parameter(torch.randn(1, 1, config["input_dim"]))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config["input_dim"],
            nhead=config["decoder_nhead"],
            dim_feedforward=config["decoder_dim_feedforward"],
            dropout=config["decoder_dropout"],
            activation=config["decoder_activation"],
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, 
            num_layers=4
        )   
        self.proj = nn.Linear(2 * config["input_dim"], config["input_dim"])
        self.initialize_weights()
    
    def initialize_weights(self):
        for p in self.parameters():
            if p.requires_grad and len(p.shape) > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, ssm_out):
        '''
        ssm_out: (B, F, D) where B is the batch size, F is the number of frames, and D is the dimension
        D = 2 * input_dim
        '''
        B, F, D = ssm_out.shape
        query = self.query.repeat(B, 1, 1)
        ssm_out = self.proj(ssm_out)
        ssm_out = ssm_out.transpose(0, 1)
        query = query.transpose(0, 1)
        out = self.decoder(query, ssm_out)
        out = out.transpose(0, 1)
        return out
        