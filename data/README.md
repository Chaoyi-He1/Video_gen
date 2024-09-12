# data

## test download notebook
https://colab.research.google.com/drive/1jM48mDhmHBu97cnZvgZRYaBYg2bLOPGW?usp=sharing

## openvid-1m usage
use `python download.py` to download datasets

the default is sample mode, sample mode will take 70G disk space, the sample will only sample the video for person.

set `download_files("./", max_files=186)`, then it will take 70*186 GB disk space and 1 million videos

there are few options for build dataset, 
`build_dataset(is_sample=True, is_HD=False)`, please check. Remove the is_sample will build all dataset in OpenVid-1M.csv

tldr:
```
cd openvid-1m
python download.py
python build.py
```
