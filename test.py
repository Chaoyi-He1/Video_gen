from diffusion import create_diffusion
import torch

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diffusion = create_diffusion(timestep_respacing="")
    
    from transformers import AutoTokenizer, CLIPTextModel

    model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
    inputs_ = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
    pass