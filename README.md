<div align="center">  
  
# CLFT: Camera-LiDAR Fusion Transformer for Semantic Segmentation in Autonomous Driving
</div>


https://github.com/user-attachments/assets/cd51585e-2f66-4ff5-bb5b-689d3fb7d4c0

> **CLFT: Camera-LiDAR Fusion Transformer for Semantic Segmentation in Autonomous Driving**, IEEE Transactions on Intelligent Vehicles, 2024 
> - [Paper in arXiv](https://arxiv.org/abs/2404.17793) | [Paper in IEEE Xplore](https://ieeexplore.ieee.org/document/10666263)


# News
- [02-01-2025] The training paths of CLFT and CLFCN are all available for downloading. Three paths for CLFCN corresponding to rgb, lidar, and cross-fusion modalities. Five paths for CLFT corresponding to base-fusion, large-fusion, hybrid-rgb, hybrid-lidar, and hybrid-fusion. All paths trained at TalTech HPC in the environment specified below. 
- [30-09-2024] The train.py and test.py scripts are available. The models trained for paper's experiments are available for downloading. 
- [25-09-2024] The author finally finished his PhD thesis work and start to maintain the repo. The visual_run.py script is available. The waymo dataset used for the experiments in paper is available for downloading. 
- [16/04/2024] Please note that this repository is still under maintenance. Author is focusing on his PhD thesis at the moment and will clean up code and optimize README gradually. You can write to claude.gujunyi@gmail.com for details. 



## Abstract
Critical research about camera-and-LiDAR-based semantic object segmentation for autonomous driving significantly benefited from the recent development of deep learning. Specifically, the vision transformer is the novel ground-breaker that successfully brought the multi-head-attention mechanism to computer vision applications. Therefore, we propose a vision-transformer-based network to carry out camera-LiDAR fusion for semantic segmentation applied to autonomous driving. Our proposal uses the novel progressive-assemble strategy of vision transformers on a double-direction network and then integrates the results in a cross-fusion strategy over the transformer decoder layers. Unlike other works in the literature, our camera-LiDAR fusion transformers have been evaluated in challenging conditions like rain and low illumination, showing robust performance. The paper reports the segmentation results over the vehicle and human classes in different modalities: camera-only, LiDAR-only, and camera-LiDAR fusion. We perform coherent controlled benchmark experiments of the camera-LiDAR fusion transformer (CLFT) against other networks that are also designed for semantic segmentation. The experiments aim to evaluate the performance of CLFT independently from two perspectives: multimodal sensor fusion and backbone architectures. The quantitative assessments show our CLFT networks yield an improvement of up to 10% for challenging dark-wet conditions when comparing with Fully-Convolutional-Neural-Network-based (FCN) camera-LiDAR fusion neural network. Contrasting to the network with transformer backbone but using single modality input, the all-around improvement is 5-10%. Our full code is available online for an interactive demonstration and application. 

## Method

![architecture](https://github.com/user-attachments/assets/93d8a578-66be-4d49-b096-bf8c82669f76)


## Installation 

The experiments in the paper were carried out on TalTech HPC. 
For CLFT and CLFCN, we programmed upon pytorch directly and avoid too much high-level apis, 
thus we believe the scratch training of the models should be compatible with various environments.

However, if you want to try the models we trained for our paper, there is need to keep several critical packages version
same as ours. We use python 3.9 in all experiments. 

```
numpy-1.26.0
pytorch-2.1.0
tqdm-4.66.1
tensorboard-2.15.0
torchvision-0.16.0
timm-0.9.8
einops-0.7.0
```
Moreover, we provide our conda yml file in the repo. 
We recommend to implement this environment if you are using our models.    


## Dataset
- [Dataset](waymo_dataset/README.md)

## Model
For CLFT:
- [CLFT Models](model_path/clft/README.md)

For CLFCN:
- [CLFCN Models](model_path/clfcn/README.md)

## Visualization 
We provide the [visual_run.py](https://github.com/Claud1234/CLFT/blob/079f003bd6d5f9a5fa0674add1ad5048fd9999b8/visual_run.py) to load the model path and input data, then render out the segmentation and overlay results as PNG images. 

Specify three args for this script. \
-m -> modality. Choices: rgb, lidar, cross_fusion \
-bb -> backbone. Choices: clfcn, clft\
-p -> the txt file contains the paths of input data.

The [waymo_dataset/visual_run_demo.txt](https://github.com/Claud1234/CLFT/blob/079f003bd6d5f9a5fa0674add1ad5048fd9999b8/waymo_dataset/visual_run_demo.txt) contains four samples scattered to four weather subsets, light-dry, light-wet, night-dry, and night-wet. But please note you need to have our waymo dataset downloaded and placed in the 'waymo_dataset' folder. The segmentation and overlay results of these four samples will be saved in 'output' folder and followed the same folder tree specified in this repo. We provide the PNG results of four samples in 'output' folder as well.

Please note this script is for making visualization results in big batch. If you only want to try for one image, we recommend to take a look of the [ipython/make_qualitative_images .ipynb](https://github.com/Claud1234/CLFT/blob/079f003bd6d5f9a5fa0674add1ad5048fd9999b8/ipython/make_qualitative_images%20.ipynb), it helps to better understand the code. 

### CLFT
```
python visual_run.py -m cross_fusion -bb clft -p ./waymo_dataset/visual_run_demo.txt
```
Specify the corresponding CLFT model path in the [config.json 'General' 'model_path'](https://github.com/Claud1234/CLFT/blob/079f003bd6d5f9a5fa0674add1ad5048fd9999b8/config.json#L13)


### CLFCN
```
python visual_run.py -m cross_fusion -bb clfcn -p ./waymo_dataset/visual_run_demo.txt
```
Specify the corresponding CLFCN model path in the [config.json 'General' 'model_path'](https://github.com/Claud1234/CLFT/blob/079f003bd6d5f9a5fa0674add1ad5048fd9999b8/config.json#L13)


## Training
The [train.py](https://github.com/Claud1234/CLFT/blob/079f003bd6d5f9a5fa0674add1ad5048fd9999b8/train.py) script is for training the CLFT and CLFCN models. 

Specify two args for this script. \
-m -> modality. Choices: rgb, lidar, cross_fusion \
-bb -> backbone. Choices: clfcn, clft\

The [waymo_dataset/splits_clft/train_all.txt](https://github.com/Claud1234/CLFT/blob/079f003bd6d5f9a5fa0674add1ad5048fd9999b8/waymo_dataset/splits_clft/train_all.txt) and [waymo_dataset/splits_clft/early_stop_valid.txt](https://github.com/Claud1234/CLFT/blob/079f003bd6d5f9a5fa0674add1ad5048fd9999b8/waymo_dataset/splits_clft/early_stop_valid.txt) are specified in the script. We use 60% of the dataset for training, and 20% for validating. 

It is possible to resume the training from the saved model path. There is need to speficy three parameters to resume the training. 

[config.json 'General' 'resume_training'](https://github.com/Claud1234/CLFT/blob/11c18e97c70bcade0030218736340b183ba6a869/config.json#L8) -> Set the true\
[config.json 'General' 'resume_training_model_path'](https://github.com/Claud1234/CLFT/blob/11c18e97c70bcade0030218736340b183ba6a869/config.json#L9) -> Specify the model path that you want to resume from \
[config.json 'General' 'reset_lr'](https://github.com/Claud1234/CLFT/blob/11c18e97c70bcade0030218736340b183ba6a869/config.json#L10)  -> You can decide to reset the learning rate or not. 
 
### CLFT
```
python train.py -m cross_fusion -bb clft
```
As indicated in paper, there are different CLFT variants CLFT-Base, CLFT-Large, and CLFT-hybird. You have to change the corresponding parameters in [config.json](https://github.com/Claud1234/CLFT/blob/079f003bd6d5f9a5fa0674add1ad5048fd9999b8/config.json#L20) for different variants. 

|                 Variant                 | ['CLFT']['emb_dim'] | ['CLFT']['model_timm']  | ['CLFT']['hooks'] |
|:-----------:|:-------------------:|:-----------------------:|:-----------------:|
|  CLFT-Base  |         768         | 'vit_base_patch16_384'  |   [2, 5, 8, 11]   |
| CLFT-Hybird |         768         | 'vit_base_resnet50_384' |   [2, 5, 8, 11]   |
|  CLFT-Large |        1024         | 'vit_large_patch16_384' |  [5, 11, 17, 23]  |

### CLFCN
```
python train.py -m cross_fusion -bb clfcn 
```


## Testing
The [test.py](https://github.com/Claud1234/CLFT/blob/079f003bd6d5f9a5fa0674add1ad5048fd9999b8/test.py) script is for testing the CLFT and CLFCN models. 

Specify three args for this script. \
-m -> modality. Choices: rgb, lidar, cross_fusion \
-bb -> backbone. Choices: clfcn, clft\
-p -> the txt file contains the paths of input data.

The script compute the IoU, recall, and precision for each class.

### CLFT
```
python test.py -bb clft -m cross_fusion -p ./waymo_dataset/splits_clft/test_day_fair.txt
```
Specify the corresponding CLFT model path in the [config.json 'General''model_path'](https://github.com/Claud1234/CLFT/blob/079f003bd6d5f9a5fa0674add1ad5048fd9999b8/config.json#L13)

There is need to change the [config.json](https://github.com/Claud1234/CLFT/blob/079f003bd6d5f9a5fa0674add1ad5048fd9999b8/config.json#L20) as well, based on the CLFT variants, same as training. 

The input path txt files are available in [waymo_dataset/splits_clft](https://github.com/Claud1234/CLFT/tree/079f003bd6d5f9a5fa0674add1ad5048fd9999b8/waymo_dataset/splits_clft), the rest 20% of dataset were used for test. Four weather subsets are classified in different text files. 


### CLFCN
```
python test.py -bb clfcn -m cross_fusion -p ./waymo_dataset/splits_clft/test_day_fair.txt
```
Specify the corresponding CLFCN model path in the [config.json 'General''model_path'](https://github.com/Claud1234/CLFT/blob/079f003bd6d5f9a5fa0674add1ad5048fd9999b8/config.json#L13)

The input path txt files are available in [waymo_dataset/splits_clft](https://github.com/Claud1234/CLFT/tree/079f003bd6d5f9a5fa0674add1ad5048fd9999b8/waymo_dataset/splits_clft), the rest 20% of dataset were used for test. Four weather subsets are classified in different text files. 

## Bibtex
If anything in this repo has a use for your work, please considering to cite our work. This is very helpful for the author who just finished his PhD and started to build his academic career. 

```
@ARTICLE{gu2024clft,
  author={Gu, Junyi and Bellone, Mauro and Pivoňka, Tomáš and Sell, Raivo},
  journal={IEEE Transactions on Intelligent Vehicles}, 
  title={CLFT: Camera-LiDAR Fusion Transformer for Semantic Segmentation in Autonomous Driving}, 
  year={2024},
  volume={},
  number={},
  pages={1-12},
  doi={10.1109/TIV.2024.3454971}}
```


