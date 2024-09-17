import io
import os
import torchvision.transforms as TT
import numpy as np
import torch
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode
import decord
from decord import VideoReader
from torch.utils.data import Dataset
from tqdm import tqdm

def resize_for_rectangle_crop(arr, image_size, reshape_mode="random"):
    if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
        arr = resize(
            arr,
            size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])],
            interpolation=InterpolationMode.BICUBIC,
        )
    else:
        arr = resize(
            arr,
            size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]],
            interpolation=InterpolationMode.BICUBIC,
        )

    h, w = arr.shape[2], arr.shape[3]
    arr = arr.squeeze(0)

    delta_h = h - image_size[0]
    delta_w = w - image_size[1]

    if reshape_mode == "random" or reshape_mode == "none":
        top = np.random.randint(0, delta_h + 1)
        left = np.random.randint(0, delta_w + 1)
    elif reshape_mode == "center":
        top, left = delta_h // 2, delta_w // 2
    else:
        raise NotImplementedError
    arr = TT.functional.crop(
        arr, top=top, left=left, height=image_size[0], width=image_size[1]
    )
    return arr

def pad_last_frame(tensor, num_frames):
    # T, H, W, C
    if len(tensor) < num_frames:
        pad_length = num_frames - len(tensor)
        # Use the last frame to pad instead of zero
        last_frame = tensor[-1]
        pad_tensor = last_frame.unsqueeze(0).expand(pad_length, *tensor.shape[1:])
        padded_tensor = torch.cat([tensor, pad_tensor], dim=0)
        return padded_tensor
    else:
        return tensor[:num_frames]


class SFTDataset(Dataset):
    def __init__(self, data_dir, video_size, fps, max_num_frames, skip_frms_num=3):
        """
        skip_frms_num: ignore the first and the last xx frames, avoiding transitions.
        """
        super(SFTDataset, self).__init__()

        self.video_size = video_size
        self.fps = fps
        self.max_num_frames = max_num_frames
        self.skip_frms_num = skip_frms_num

        self.video_paths = []
        self.captions = []
        try:
            assert torch.cuda.is_available(), "GPU is not available"
            self.decord_ctx = decord.cpu()
        except Exception as e:
            print(
                f"Failed to initialize decord.gpu(), falling back to decord.cpu(): {e}"
            )
            self.decord_ctx = decord.cpu()

        for root, dirnames, filenames in os.walk(data_dir):
            for filename in tqdm(filenames):
                if filename.endswith(".mp4"):
                    video_path = os.path.join(root, filename)
                    self.video_paths.append(video_path)

                    caption_path = video_path.replace(".mp4", ".txt").replace(
                        "videos", "labels"
                    )
                    if os.path.exists(caption_path):
                        caption = open(caption_path, "r").read().splitlines()[0]
                    else:
                        caption = ""
                    self.captions.append(caption)
        print("Found video pairs: ", len(self.video_paths))

    def __getitem__(self, index):
        decord.bridge.set_bridge("torch")

        video_path = self.video_paths[index]
        vr = VideoReader(uri=video_path, height=-1, width=-1, ctx=self.decord_ctx)
        actual_fps = vr.get_avg_fps()
        ori_vlen = len(vr)

        if ori_vlen / actual_fps * self.fps > self.max_num_frames:
            num_frames = self.max_num_frames
            start = int(self.skip_frms_num)
            end = int(start + num_frames / self.fps * actual_fps)
            end_safty = min(
                int(start + num_frames / self.fps * actual_fps), int(ori_vlen)
            )
            indices = np.arange(start, end, max((end - start) // num_frames, 1)).astype(
                int
            )
            temp_frms = vr.get_batch(np.arange(start, end_safty))
            assert temp_frms is not None
            tensor_frms = (
                torch.from_numpy(temp_frms)
                if type(temp_frms) is not torch.Tensor
                else temp_frms
            )    
            indices = torch.tensor((indices - start).tolist())
            indices = torch.clamp(indices, 0, tensor_frms.size(0) - 1)
            tensor_frms = tensor_frms[indices]
        else:
            if ori_vlen > self.max_num_frames:
                num_frames = self.max_num_frames
                start = int(self.skip_frms_num)
                end = int(ori_vlen - self.skip_frms_num)
                indices = np.arange(
                    start, end, max((end - start) // num_frames, 1)
                ).astype(int)
                temp_frms = vr.get_batch(np.arange(start, end))
                assert temp_frms is not None
                tensor_frms = (
                    torch.from_numpy(temp_frms)
                    if type(temp_frms) is not torch.Tensor
                    else temp_frms
                )
                tensor_frms = tensor_frms[torch.tensor((indices - start).tolist())]
            else:

                def nearest_smaller_4k_plus_1(n):
                    remainder = n % 4
                    if remainder == 0:
                        return n - 3
                    else:
                        return n - remainder + 1

                start = int(self.skip_frms_num)
                end = int(ori_vlen - self.skip_frms_num)
                num_frames = nearest_smaller_4k_plus_1(
                    end - start
                )  # 3D VAE requires the number of frames to be 4k+1
                end = int(start + num_frames)
                temp_frms = vr.get_batch(np.arange(start, end))
                assert temp_frms is not None
                tensor_frms = (
                    torch.from_numpy(temp_frms)
                    if type(temp_frms) is not torch.Tensor
                    else temp_frms
                )

        tensor_frms = pad_last_frame(
            tensor_frms, self.max_num_frames
        )  # the len of indices may be less than num_frames, due to round error
        tensor_frms = tensor_frms.permute(0, 3, 1, 2)  # [T, H, W, C] -> [T, C, H, W]
        tensor_frms = resize_for_rectangle_crop(
            tensor_frms, self.video_size, reshape_mode="center"
        )
        tensor_frms = (tensor_frms - 127.5) / 127.5

        item = {
            "mp4": tensor_frms,
            "txt": self.captions[index],
            "num_frames": num_frames,
            "fps": self.fps,
        }
        return item

    def __len__(self):
        return len(self.video_paths)

    @classmethod
    def create_dataset_function(cls, path, args, **kwargs):
        return cls(data_dir=path, **kwargs)
