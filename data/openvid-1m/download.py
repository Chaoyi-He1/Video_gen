import os
import subprocess
import pandas as pd
from tqdm import tqdm


def post_process_video(video_names, video_folder):
    # Check and delete unmatched video files
    video_folder = video_folder
    for video_file in tqdm(os.listdir(video_folder)):
        try:
            if video_file not in video_names:
                os.remove(os.path.join(video_folder, video_file))
        except Exception as e:
            print(f"Error processing file {video_file}: {e}")
            continue


def video_filter(data_path, min_frames=100):
    df = pd.read_csv(data_path)
    df["frame"] = pd.to_numeric(df["frame"], errors="coerce")
    df.dropna(inplace=True)

    def filter_row(row):
        try:
            return row["frame"] >= min_frames
        except Exception as e:
            print(f"Skipping row due to error: {e}")
            return False

    filtered_df = df[df.apply(filter_row, axis=1)]
    filtered_df.to_csv(data_path, index=False)
    return filtered_df


### download openvid_files
def download_files(
    output_directory, max_files=186, min_frames=300, hd=False, start_from=0
):
    zip_folder = os.path.join(output_directory, "download")
    video_folder = os.path.join(output_directory, "video")
    os.makedirs(zip_folder, exist_ok=True)
    os.makedirs(video_folder, exist_ok=True)

    error_log_path = os.path.join(zip_folder, "download_log.txt")
    assert max_files <= 186, "max_files should be less than 186"

    data_folder = os.path.join(output_directory, "data", "train")
    os.makedirs(data_folder, exist_ok=True)
    data_url = None
    if not hd:
        data_url = "https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/data/train/OpenVid-1M.csv"
    else:
        "https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/data/train/OpenVidHD.csv"
    data_path = os.path.join(data_folder, os.path.basename(data_url))
    # if not os.path.exists(data_path):
    # command = ["wget", "-O", data_path, data_url]
    # subprocess.run(command, check=True)
    # else:
    #     print(f"{data_path} already exists. Skipping download.")

    df = video_filter(data_path, min_frames=min_frames)
    video_names = set(df["video"].tolist())

    for i in range(start_from, max_files):
        url = f"https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/OpenVid_part{i}.zip"
        file_path = os.path.join(zip_folder, f"OpenVid_part{i}.zip")
        if os.path.exists(file_path):
            print(f"file {file_path} exits.")
            continue

        command = ["wget", "-O", file_path, url]
        unzip_command = ["unzip", "-j", file_path, "-d", video_folder]
        try:
            subprocess.run(command, check=True)
            print(f"file {url} saved to {file_path}")
            subprocess.run(unzip_command, check=True)
        except subprocess.CalledProcessError as e:
            error_message = f"file {url} download failed: {e}\n"
            print(error_message)
            with open(error_log_path, "a") as error_log_file:
                error_log_file.write(error_message)

            part_urls = [
                f"https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/OpenVid_part{i}_partaa",
                f"https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/OpenVid_part{i}_partab",
            ]

            for part_url in part_urls:
                part_file_path = os.path.join(zip_folder, os.path.basename(part_url))
                if os.path.exists(part_file_path):
                    print(f"file {part_file_path} exits.")
                    continue

                part_command = ["wget", "-O", part_file_path, part_url]
                try:
                    subprocess.run(part_command, check=True)
                    print(f"file {part_url} saved to {part_file_path}")
                except subprocess.CalledProcessError as part_e:
                    part_error_message = f"file {part_url} download failed: {part_e}\n"
                    print(part_error_message)
                    with open(error_log_path, "a") as error_log_file:
                        error_log_file.write(part_error_message)
            file_path = os.path.join(zip_folder, f"OpenVid_part{i}.zip")
            cat_command = (
                "cat "
                + os.path.join(zip_folder, f"OpenVid_part{i}_part*")
                + " > "
                + file_path
            )
            unzip_command = ["unzip", "-j", file_path, "-d", video_folder]
            os.system(cat_command)
            subprocess.run(unzip_command, check=True)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Removed {file_path} to clear space.")

        post_process_video(video_names=video_names, video_folder=video_folder)


if __name__ == "__main__":
    download_files("./", max_files=186, start_from=0)
