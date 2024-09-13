from transformers import CLIPTokenizer, CLIPProcessor

class CLIPTextTokenizer:
    def __init__(self, model_dir=None):
        if model_dir:
            self.tokenizer = CLIPTokenizer.from_pretrained(model_dir)
        else:
            self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    
    def tokenize(self, text):
        return self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
