<div align="center">  
  
# CLFT: Camera-LiDAR Fusion Transformer for Semantic Segmentation in Autonomous Driving
</div>


https://github.com/user-attachments/assets/cd51585e-2f66-4ff5-bb5b-689d3fb7d4c0

> **CLFT: Camera-LiDAR Fusion Transformer for Semantic Segmentation in Autonomous Driving**, IEEE Transactions on Intelligent Vehicles, 2024 
> - [Paper in arXiv](https://arxiv.org/abs/2404.17793) | [Paper in IEEE Xplore](https://ieeexplore.ieee.org/document/10666263)


# News
- [25-09-2024] The author fianlly finished his PhD thesis work and start to maintain the repo. The visual_run.py script is available. The waymo dataset used for the experimetns in paper is available for downloading. 
- [16/04/2024] Please note that this repository is still under maintance. Author is focusing on his PhD thsis at the moment and will chean up code and optimize README gradually. You can write to claude.gujunyi@gmail.com for details. 



## Abstract
Critical research about camera-and-LiDAR-based semantic object segmentation for autonomous driving significantly benefited from the recent development of deep learning. Specifically, the vision transformer is the novel ground-breaker that successfully brought the multi-head-attention mechanism to computer vision applications. Therefore, we propose a vision-transformer-based network to carry out camera-LiDAR fusion for semantic segmentation applied to autonomous driving. Our proposal uses the novel progressive-assemble strategy of vision transformers on a double-direction network and then integrates the results in a cross-fusion strategy over the transformer decoder layers. Unlike other works in the literature, our camera-LiDAR fusion transformers have been evaluated in challenging conditions like rain and low illumination, showing robust performance. The paper reports the segmentation results over the vehicle and human classes in different modalities: camera-only, LiDAR-only, and camera-LiDAR fusion. We perform coherent controlled benchmark experiments of the camera-LiDAR fusion transformer (CLFT) against other networks that are also designed for semantic segmentation. The experiments aim to evaluate the performance of CLFT independently from two perspectives: multimodal sensor fusion and backbone architectures. The quantitative assessments show our CLFT networks yield an improvement of up to 10% for challenging dark-wet conditions when comparing with Fully-Convolutional-Neural-Network-based (FCN) camera-LiDAR fusion neural network. Contrasting to the network with transformer backbone but using single modality input, the all-around improvement is 5-10%. Our full code is available online for an interactive demonstration and application. 

## Method

![architecture](https://github.com/user-attachments/assets/93d8a578-66be-4d49-b096-bf8c82669f76)


## Installation 

The experiments were carried out on TalTech HPC. For CLFT and CLFCN, we progrmmed upon pytorch directly and avoid too much high-level apis, thus we believe the code should be compatible with various environments. Here list out the package versions on HPC:


## Dataset
- [Dataset](waymo_dataset/README.md)

## Model
- [Model](model_path/README.md)

## Visualization 
We provide the 'visual_run.py' to load the model path and input data, then render out the segmentation and ovelay results as PNG images. 

Specify three args for this script. \
-m -> modalitity. Choices: rgb, lidar, cross_fusion \
-bb -> backbonw. Choices: clfcn, clft\
-p -> the txt file contains the paths of input data.

### CLFT
```
python visual_run.py -m cross_fusion -bb clft -p ./waymo_dataset/visual_run_demo.txt
```
The 'visual_run_demo.txt' is existed in 'waymo_dataset' folder, it contains four samples scattered to four weather subsets, light-dry, light-wet, night-dry, and night-wet. But please note you need to have our waymo dataset downloaded and placed in the 'waymo_dataset' folder. The segmentation and overlay results of these four samples will be saved in 'output' folder and followed the same path tree specified in this repo. We provide the PNG results of four samples in 'output' folder as well.

Specify the model path in the config.json ['General']['model_path']

### FCN

## Training
### CLFT

### FCN


## Testing
### CLFT

### CLFCN

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


