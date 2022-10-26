# (USynthFace) Unsupervised-Face-Recognition-using-Unlabeled-Synthetic-Data

![overview](https://raw.githubusercontent.com/fdbtrs/Unsupervised-Face-Recognition-using-Unlabeled-Synthetic-Data/main/images/USynthFace_framework.png)


## Pretrained Models
| Model  | Epochs | Pretrained Model|
| ------------- | ------------- | ------------- |
| USynthFace100K    | 40 | [pretrained-model](link) |
| USynthFace100K_m  | 200 | [pretrained-model](link) |
| USynthFace400K_m  | 200 | [pretrained-model](link) |


## Requirements
### Requirements for DiscoFaceGAN Image Generation:
- Python 3.6
- Tensorflow 1.12 with GPU support

We recommend creating a virtual environment with *`requirementsTF.txt`*.  
Download pretrained [DiscoFaceGAN model](https://drive.google.com/uc?id=1nT_cf610q5mxD_jACvV43w4SYBxsPUBq) and save in `DiscoFaceGAN/pretrained/`.

### Requirements for USynthFace Training
- pytorch 1.11.0
- torchvision 0.12.0

We recomment creating a virtual environment with *`requirementsTorch.txt`*


## Training Dataset Preparation
To generate images run in `DiscoFaceGAN/`:
```
$ generate_imgs.sh --save_path "save/path/of/unaligned/images"
```

To align images run:
```
$ align_imgs.sh --in_folder "path/to/image/folder" --out_folder "save/path/of/aligned/images"
```
Set `datapath="../.."` in *`config/config.py`* to folder with aligned DiscoFaceGAN images.

## Evaluation Dataset Preparation
Download evaluation datasets from [insightface](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_) in strict compliance with the license distribution. Evaluation datasets are available e.g. in the training dataset package CASIA-Webface as bin files.  
Set `eval_datasets="../.."` in *`config/config.py`* to your unzipped folder which includes the bin files.

## Train USynthFace
Change *`config/config.py`* and *`train.sh`* to your preferences and execute:
```
$ train.sh
```

## Evaluate USynthFace
In *`evaluation/`* run:
```
$ CUDA_VISIBLE_DEVICES=0 python eval.py --model_folder "path/to/model/folder/" --rec_path "path/to/folder/with/bin/files"
```
Test log is saved in model_folder.