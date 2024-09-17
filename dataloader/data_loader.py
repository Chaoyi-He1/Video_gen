import torch
from torch.utils.data import DataLoader
from .data_video import SFTDataset
from .text_tokenizer import CLIPTextTokenizer

def create_data_loader(data_dir, video_size, fps, max_num_frames, skip_frms_num=3, batch_size=4, shuffle=True, num_workers=4):
    """
    Create a DataLoader for the SFTDataset.

    Args:
        data_dir (str): Path to the directory containing video files.
        video_size (Tuple[int, int]): Desired size of the video frames (height, width).
        fps (int): Frames per second for the video.
        max_num_frames (int): Maximum number of frames to load per video.
        skip_frms_num (int, optional): Number of frames to skip at the start and end of the video. Defaults to 3.
        batch_size (int, optional): Number of samples per batch. Defaults to 4.
        shuffle (bool, optional): Whether to shuffle the dataset. Defaults to True.
        num_workers (int, optional): Number of subprocesses to use for data loading. Defaults to 4.

    Returns:
        DataLoader: A DataLoader for the SFTDataset.
    """
    dataset = SFTDataset(
        data_dir=data_dir,
        video_size=video_size,
        fps=fps,
        max_num_frames=max_num_frames,
        skip_frms_num=skip_frms_num
    )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

    return dataset, data_loader


if __name__ == '__main__':
    # Example usage
    data_dir = "../data/train_data"
    video_size = (512, 512)  # Example size
    fps = 25
    max_num_frames = 1000
    batch_size = 1
    skip_frms_num = 3

    dataset, data_loader = create_data_loader(data_dir, video_size, fps, max_num_frames, skip_frms_num=3, batch_size=batch_size, shuffle=True, num_workers=2)
    tokenizer = CLIPTextTokenizer()
    
    # Test the __getitem__ function
    for i in range(10000):
        sample_index = i  
        sample = dataset[sample_index]
        print("Sample video shape:", sample["mp4"].shape)
        print("Sample text:", sample["txt"])
    
    # for batch in data_loader:
        # videos = batch["mp4"] # it should be like this [1, 200, 3, 512, 512]
        # print(videos.shape) 
        # captions = tokenizer.tokenize(batch["txt"])
        # print(videos.shape, captions)