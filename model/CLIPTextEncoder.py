from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection
import torch.nn as nn
import torch


def CreateTextEncoder(model_name="openai/clip-vit-large-patch16", freeze=False, projection=False):
    model_candidate = [
        "openai/clip-vit-base-patch32",
        "openai/clip-vit-base-patch16",
        "openai/clip-vit-large-patch14",
        "openai/clip-vit-large-patch14-336",
    ]
    
    assert model_name in model_candidate, f"model_name should be one of {model_candidate}"
    
    TextEncoder = CLIPTextModel.from_pretrained(model_name) if not projection \
        else CLIPTextModelWithProjection.from_pretrained(model_name)
        
    if freeze:
        for param in TextEncoder.parameters():
            param.requires_grad = False
    return TextEncoder


class ConcatSquashLinear(nn.Module):
    def __init__(self, dim_in, dim_out, dim_ctx):
        super(ConcatSquashLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(dim_ctx, dim_out, bias=False)
        self._hyper_gate = nn.Linear(dim_ctx, dim_out)

    def forward(self, ctx, x):
        gate = torch.sigmoid(self._hyper_gate(ctx))
        bias = self._hyper_bias(ctx)
        # if x.dim() == 3:
        #     gate = gate.unsqueeze(1)
        #     bias = bias.unsqueeze(1)
        ret = self._layer(x) * gate + bias
        return ret


class SSM_input_projection(nn.Module):
    def __init__(self, config: dict):
        super(SSM_input_projection, self).__init__()
        
        self.num_frames = config["num_frames"]
        
        self.textencoder = CreateTextEncoder(
            model_name=config["textencoder"],
            freeze=config["textencoder_freeze"],
            projection=config["textencoder_projection"],
        )
        
        self.input_dim = self.textencoder.config.hidden_size
        
        self.querries = nn.Parameter(torch.randn(1, self.num_frames, self.input_dim))
        self.querries_proj = nn.ModuleList([
            ConcatSquashLinear(self.input_dim, self.input_dim, self.input_dim)] * config["num_querries_concats_layers"] + [
            nn.LayerNorm(self.input_dim)])
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.textencoder.config.hidden_size,
            nhead=config["decoder_nhead"],
            dim_feedforward=config["decoder_dim_feedforward"],
            dropout=config["decoder_dropout"],
            activation=config["decoder_activation"],
        )
        self.projection_block = nn.TransformerDecoder(
            decoder_layer, 
            num_layers=config["decoder_num_layers"]
        )
    
    def forward(self, **Token_text):
        TextEncoderOutput = self.textencoder(**Token_text)
        last_hidden_state = TextEncoderOutput.last_hidden_state.permute(1, 0, 2).contiguous()
        pooled_output = TextEncoderOutput.pooler_output.unsqueeze(1)
        
        querries = self.querries
        for i, layer in enumerate(self.querries_proj):
            if i == len(self.querries_proj) - 1:
                querries = layer(querries)
            else:
                querries = layer(pooled_output, querries)
        querries = querries.permute(1, 0, 2).contiguous()
        tgt_mask = (1 - torch.tril(torch.ones(self.num_frames, self.num_frames))).bool().to(querries.device)
        
        SSM_input = self.projection_block(
            tgt=querries,
            memory=last_hidden_state,
            tgt_mask=tgt_mask
        )
        
        return SSM_input.permute(1, 0, 2).contiguous()