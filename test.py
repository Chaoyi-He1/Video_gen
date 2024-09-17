import torch
from transformers import CLIPTextModelWithProjection, CLIPTextModel, AutoTokenizer
from diffusers.models import AutoencoderKL
from transformers import CLIPTokenizer
import os


if __name__ == "__main__":
    