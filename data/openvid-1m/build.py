import os
import pandas as pd
import shutil

def create_directories(base_dir):
    labels_dir = os.path.join(base_dir, "labels")
    videos_dir = os.path.join(base_dir, "videos")
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(videos_dir, exist_ok=True)
    return labels_dir, videos_dir

def process_csv(csv_path, labels_dir, videos_dir, video_source_dir):
    df = pd.read_csv(csv_path)
    count = 0
    for index, row in df.iterrows():
        video_filename = row['video']
        caption = row['caption']
        
        # Check if video file and caption are present
        source_video_path = os.path.join(video_source_dir, video_filename)
        if not os.path.exists(source_video_path) or pd.isna(caption):
            print(f"Skipping index {index + 1}: Missing video or caption.")
            continue
        
        # Write caption to a text file
        label_path = os.path.join(labels_dir, f"{count + 1}.txt")
        with open(label_path, 'w') as f:
            f.write(caption)
        
        # Copy video file to the videos directory
        dest_video_path = os.path.join(videos_dir, f"{count + 1}.mp4")
        shutil.copyfile(source_video_path, dest_video_path)
        count += 1

def build_dataset(is_sample=True, is_HD=False):
    base_dir = "../train_data"
    csv_path = "data/train/OpenVid-1M.csv"
    if is_sample:
        csv_path = "data/train/OpenVid-1M-sample.csv"
    if is_HD:
        csv_path = "data/train/OpenVidHD.csv"
        
    video_source_dir = "./video"
    
    labels_dir, videos_dir = create_directories(base_dir)
    print(csv_path)
    process_csv(csv_path, labels_dir, videos_dir, video_source_dir)

if __name__ == '__main__':
    build_dataset(is_sample=False, is_HD=False)
