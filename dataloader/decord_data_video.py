import os
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import decord
from decord import VideoReader

from .utils import pad_last_frame, resize_for_rectangle_crop


class SFTDataset(Dataset):
    def __init__(self, data_dir, video_size, fps, max_num_frames, skip_frms_num=3):
        """
        Initializes the dataset by collecting video paths and their corresponding captions.

        Args:
            data_dir (str): Directory containing video files.
            video_size (tuple): Desired video size (height, width).
            fps (int): Frames per second to sample.
            max_num_frames (int): Maximum number of frames per video.
            skip_frms_num (int, optional): Number of frames to skip at the start and end. Defaults to 3.
        """
        super().__init__()
        self.video_size = video_size
        self.fps = fps
        self.max_num_frames = max_num_frames
        self.skip_frms_num = skip_frms_num

        self.video_paths, self.captions = self._collect_video_data(data_dir)
        print(f"Found video pairs: {len(self.video_paths)}")

        self.decord_ctx = self._initialize_decord()

    def _initialize_decord(self):
        """Initializes the Decord context, preferring GPU if available."""
        if torch.cuda.is_available():
            return decord.cpu()
        else:
            print("GPU not available. Falling back to CPU for Decord.")
            return decord.cpu()

    def _collect_video_data(self, data_dir):
        video_paths = []
        captions = []
        for root, _, filenames in os.walk(data_dir):
            for filename in tqdm(filenames, desc="Collecting videos"):
                if filename.endswith(".mp4"):
                    video_path = os.path.join(root, filename)
                    video_paths.append(video_path)
                    caption = self._load_caption(video_path)
                    captions.append(caption)
        return video_paths, captions

    def _load_caption(self, video_path):
        caption_path = video_path.replace(".mp4", ".txt").replace("videos", "labels")
        if os.path.exists(caption_path):
            with open(caption_path, "r") as f:
                return f.readline().strip()
        return ""

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, index):
        try:
            video_path = self.video_paths[index]
            caption = self.captions[index]
            frames = self._load_video_frames(video_path)

            frames = pad_last_frame(frames, self.max_num_frames)
            frames = frames.permute(0, 3, 1, 2)  # [T, H, W, C] -> [T, C, H, W]
            frames = resize_for_rectangle_crop(frames, self.video_size, reshape_mode="center")
            # frames = (frames - 127.5) / 127.5  # Normalize to [-1, 1]
            video_name = video_path.split("/")[-1]
            return {
                "mp4": frames,
                "txt": caption,
                "num_frames": frames.size(0),
                "fps": self.fps,
                "video_name": video_name
            }
        except Exception:
            print("Skip video ", self.video_paths[index])
            return {
                "mp4": None,
                "txt": None,
                "num_frames": None,
                "fps": self.fps,
                "video_name": None
            }

    def _load_video_frames(self, video_path):
        decord.bridge.set_bridge("torch")
        vr = VideoReader(uri=video_path, ctx=self.decord_ctx)
        actual_fps = vr.get_avg_fps()
        ori_vlen = len(vr)

        # Calculate the desired number of frames based on fps
        desired_num_frames = min(
            self.max_num_frames, int(ori_vlen / actual_fps * self.fps)
        )

        start = self.skip_frms_num
        end = ori_vlen - self.skip_frms_num
        max_fps_ratio = int(actual_fps / self.fps) + 1

        if desired_num_frames < self.max_num_frames:
            # Video has fewer frames than max_num_frames after adjusting fps
            indices = self._generate_indices(start, end, desired_num_frames, max_fps_ratio)
        else:
            # Video has more frames; sample max_num_frames
            indices = self._generate_indices(start, end, self.max_num_frames, max_fps_ratio)

        temp_frms = vr.get_batch(indices.numpy())
        tensor_frms = temp_frms if isinstance(temp_frms, torch.Tensor) else torch.from_numpy(temp_frms)

        return tensor_frms

    def _generate_indices(self, start, end, num_frames, max_fps_ratio=1):

        step = min(max((end - start) // num_frames, 1), max_fps_ratio)
        indices = np.arange(start, end, step).astype(int)[:num_frames]
        indices = torch.clamp(torch.tensor(indices), start, end)
        return indices

    @classmethod
    def create_dataset_function(cls, path, args, **kwargs):
        return cls(data_dir=path, **kwargs)
