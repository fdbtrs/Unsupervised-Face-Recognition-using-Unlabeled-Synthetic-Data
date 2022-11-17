This is the offical repository of the paper:
## Unsupervised Face Recognition using Unlabeled Synthetic Data
[Arxiv](https://arxiv.org/abs/2211.07371)
## Paper accepted at Face and Gesture 2023 
![USynthFace Overview](images/USynthFace_framework.png?raw=true)

## Pretrained Models
| Model  | Images | LFW | AgeDB-30 | CFP-FP | CA-LFW | CP-LFW | Pretrained Model |
| ----------- | ---- | ----- | ----- | ----- | ----- | ----- | ---------------- |
| USynthFace  | 100K | 92.12 | 71.08 | 78.19 | 76.15 | 71.95 | [download](https://drive.google.com/drive/folders/1t1bkvKqQGHkgdqqGcjsiMmgotQMpL64z?usp=sharing) |
| USynthFace  | 200K | 91.93 | 71.23 | 78.03 | 76.73 | 72.27 | [download](https://drive.google.com/drive/folders/1NTk46zJO9xuaY5-8H0TJOswAcVcTXkYA?usp=share_link) |
| USynthFace  | 400K | 92.23 | 71.62 | 78.56 | 77.05 | 72.03 | [download](https://drive.google.com/drive/folders/1J5E7lMHuKbncPSCE5sskeEBA4ZHn4HrG?usp=share_link) |


## Requirements
### Requirements for DiscoFaceGAN Image Generation:
- Python 3.6
- Tensorflow 1.12 with GPU support

We recommend creating a virtual environment with *`requirementsTF.txt`*.  
Download pretrained [DiscoFaceGAN](https://github.com/microsoft/DiscoFaceGAN), strickly follow [DiscoFaceGAN](https://github.com/microsoft/DiscoFaceGAN) license and save in `DiscoFaceGAN/pretrained/`. 

### Requirements for USynthFace Training
- pytorch 1.11.0
- torchvision 0.12.0

We recomment creating a virtual environment with *`requirementsTorch.txt`*


## Training Dataset Preparation
To generate images run in `DiscoFaceGAN/`:
```
generate_imgs.sh --save_path "save/path/of/unaligned/images"
```

To align images run:
```
align_imgs.sh --in_folder "path/to/image/folder" --out_folder "save/path/of/aligned/images"
```
Set `datapath="../.."` in *`config/config.py`* to folder with aligned DiscoFaceGAN images.

## Evaluation Dataset Preparation
Download evaluation datasets from [insightface](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_) in strict compliance with the license distribution. Evaluation datasets are available e.g. in the training dataset package CASIA-Webface as bin files.  
Set `eval_datasets="../.."` in *`config/config.py`* to your unzipped folder which includes the bin files.

## Train USynthFace
Change *`config/config.py`* and *`train.sh`* to your preferences and execute:
```
train.sh
```

To reproduce the results of the pretrained models, change `number_of_images=` and `output_dir=` in *`config/config.py`*.

## Evaluate USynthFace
In *`evaluation/`* run:
```
CUDA_VISIBLE_DEVICES=0 python eval.py --model_folder "path/to/model/folder/" --rec_path "path/to/folder/with/bin/files"
```
Test log is saved in model_folder.


## References:
- [DiscoFaceGAN](https://github.com/microsoft/DiscoFaceGAN) 
- [Moco](https://github.com/facebookresearch/moco)


## License

```
This project is licensed under the terms of the Attribution-NonCommercial-ShareAlike 4.0 
International (CC BY-NC-SA 4.0) license. 
Copyright (c) 2021 Fraunhofer Institute for Computer Graphics Research IGD Darmstadt
```