from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection


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