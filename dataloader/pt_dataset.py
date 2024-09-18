import torch
import os
import numpy as np


class PTDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.pt_dir = os.path.join(self.data_dir, "videos_pt")
        self.labels_dir = os.path.join(self.data_dir, "labels")
        # List all .pt files in the directory
        self.file_paths = [
            os.path.join(self.pt_dir, f)
            for f in os.listdir(self.pt_dir)
            if f.endswith(".pt")
        ]
        self.captions = self._load_captions(self.file_paths)

    def _load_captions(self, file_paths):
        captions = []
        for file_path in file_paths:
            caption = self._load_caption(file_path)
            captions.append(caption)
        return captions

    def _load_caption(self, file_path):
        caption_path = file_path.replace(".mp4.npz", ".txt").replace(
            "videos_pt", "labels"
        )
        if os.path.exists(caption_path):
            with open(caption_path, "r") as f:
                return f.readline().strip()
        return ""

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # mp4 = torch.load(self.file_paths[idx])
        mp4 = np.load(self.file_paths[idx])["array"]
        mp4 = (torch.from_numpy(mp4) - 127.5) / 127.5
        caption = self.captions[idx]
        return {"mp4": mp4, "txt": caption, "num_frames": mp4.size(0)}
