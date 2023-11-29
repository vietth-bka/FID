## Generate PKL files for FaceId
This folder is for creating Pickle file, which is a mandatory step when adding data to the FaceId system. 

- To fulfil this task, just run:
```bash
python3 crt_emb_from_pkl_plus.py
```
- Be careful before running ! The directory is needed to be check several times. As usual, data for the trivial staffs are saved at 
``` bash
synData = '/media/v100/DATA4/thviet/Pure_dataset/synthesis'
```
- Otherwise, for the whole special guests or the Big guys, the path is set at 
```bash
untrained = '/media/v100/DATA4/thviet/Visiting_Data_All'.
```

- But anywhere else you save the new images, you should modify these arguments as the same. Make sure they contains the whole needed datasets.

## Running on V100 (recommended)
These files are currently available on V100 at  
```bash
CUDA_VISIBLE_DEVICES=3 python3 crt_emb_from_pkl_plus.py
```

The rest is the same as above.

## Requirements
- python = 3.8.3
- pytorch = 1.7.0
- torchvision = 0.8.1
- pickle