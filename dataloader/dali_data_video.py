import os
import numpy as np
import torch
from torch.utils.data import Dataset
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator

class VideoPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, file_list, sequence_length, size, normalized):
        super(VideoPipeline, self).__init__(batch_size, num_threads, device_id, seed=12)
        self.input = fn.readers.video(
            device="gpu",
            file_list=file_list,
            sequence_length=sequence_length,
            shard_id=0,
            num_shards=1,
            random_shuffle=False,
            normalized=normalized,
            image_type=types.RGB,
            dtype=types.UINT8,
            initial_fill=16
        )
        print(size)
        self.resize = fn.resize(
            device="gpu",
            resize_x=size[1],
            resize_y=size[0],
            interp_type=types.INTERP_LINEAR
        )
        self.norm = fn.crop_mirror_normalize(
            device="gpu",
            dtype=types.FLOAT,
            output_layout="FCHW",
            mean=[127.5],
            std=[127.5]
        )

    def define_graph(self):
        output = self.input(name="Reader")
        output = self.resize(output)
        output = self.norm(output)
        return output

class SFTDataset(Dataset):
    def __init__(self, data_dir, video_size, fps, max_num_frames, skip_frms_num=3):
        self.video_size = video_size
        self.fps = fps
        self.max_num_frames = max_num_frames
        self.skip_frms_num = skip_frms_num

        self.video_paths, self.captions = self._collect_video_data(data_dir)
        print(f"Found video pairs: {len(self.video_paths)}")

        # Create a file list for DALI
        self.file_list = "video_files.txt"
        with open(self.file_list, "w") as f:
            for idx, video_path in enumerate(self.video_paths):
                f.write(f"{video_path} {idx}\n")  # Using index as a dummy label

        self.batch_size = 1
        self.pipeline = VideoPipeline(
            batch_size=self.batch_size,
            num_threads=2,
            device_id=0 if torch.cuda.is_available() else -1,
            file_list=self.file_list,
            sequence_length=self.max_num_frames,
            size=video_size,
            normalized=False
        )
        self.pipeline.build()
        self.dali_iterator = DALIGenericIterator(
            self.pipeline,
            output_map=["frames"],
            reader_name="Reader",
            auto_reset=True
        )

    def _collect_video_data(self, data_dir):
        video_paths = []
        captions = []
        for root, _, filenames in os.walk(data_dir):
            for filename in filenames:
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

    def __iter__(self):
        self.iterator = iter(self.dali_iterator)
        self.caption_idx = 0
        return self

    def __next__(self):
        try:
            data = next(self.iterator)
            frames = data[0]["frames"].squeeze(0)  
            caption = self.captions[self.caption_idx]
            self.caption_idx += 1
            return {
                "mp4": frames, 
                "txt": caption,
                "num_frames": frames.size(1),
                "fps": self.fps,
            }
        except StopIteration:
            self.iterator.reset()
            raise StopIteration

    @classmethod
    def create_dataset_function(cls, path, args, **kwargs):
        return cls(data_dir=path, **kwargs)
