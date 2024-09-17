import torch
from transformers import CLIPTextModelWithProjection, CLIPTextModel, AutoTokenizer
from diffusers.models import AutoencoderKL
from transformers import CLIPTokenizer
import os


if __name__ == "__main__":
    os.environ['TRANSFORMERS_CACHE'] = '/scratch/user/chaoyi_he/torch_cache'
    model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
    model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14-336")
    model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch16")
    model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
    
    # model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    # model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14-336")
    # model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch16")
    # model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14-336")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")