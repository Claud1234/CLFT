<div align="center">  
  
# CLFT: Camera-LiDAR Fusion Transformer for Semantic Segmentation in Autonomous Driving
</div>


https://github.com/user-attachments/assets/cd51585e-2f66-4ff5-bb5b-689d3fb7d4c0

> **CLFT: Camera-LiDAR Fusion Transformer for Semantic Segmentation in Autonomous Driving**, IEEE Transactions on Intelligent Vehicles, 2024 
> - [Paper in arXiv](https://arxiv.org/abs/2404.17793) | [Paper in IEEE Xplore](https://ieeexplore.ieee.org/document/10666263)


# News
- [16/04/2024] Please note that this repository is still under maintance. Author is focusing on his PhD thsis at the moment and will chean up code and optimize README gradually. You can write to claude.gujunyi@gmail.com for details. 
TODO list here:
Provide segmentation videos of Waymo Open Dataset for three models campared in paper. CLFT, [CLFCN](https://doi.org/10.3390/electronics11071119) 
and [Panoptic SegFormer](https://arxiv.org/abs/2109.03814)


## Abstract
Critical research about camera-and-LiDAR-based semantic object segmentation for autonomous driving significantly benefited from the recent development of deep learning. Specifically, the vision transformer is the novel ground-breaker that successfully brought the multi-head-attention mechanism to computer vision applications. Therefore, we propose a vision-transformer-based network to carry out camera-LiDAR fusion for semantic segmentation applied to autonomous driving. Our proposal uses the novel progressive-assemble strategy of vision transformers on a double-direction network and then integrates the results in a cross-fusion strategy over the transformer decoder layers. Unlike other works in the literature, our camera-LiDAR fusion transformers have been evaluated in challenging conditions like rain and low illumination, showing robust performance. The paper reports the segmentation results over the vehicle and human classes in different modalities: camera-only, LiDAR-only, and camera-LiDAR fusion. We perform coherent controlled benchmark experiments of the camera-LiDAR fusion transformer (CLFT) against other networks that are also designed for semantic segmentation. The experiments aim to evaluate the performance of CLFT independently from two perspectives: multimodal sensor fusion and backbone architectures. The quantitative assessments show our CLFT networks yield an improvement of up to 10% for challenging dark-wet conditions when comparing with Fully-Convolutional-Neural-Network-based (FCN) camera-LiDAR fusion neural network. Contrasting to the network with transformer backbone but using single modality input, the all-around improvement is 5-10%. Our full code is available online for an interactive demonstration and application. 

## Method

![architecture](https://github.com/user-attachments/assets/93d8a578-66be-4d49-b096-bf8c82669f76)


## Installation 

The experiments were carried out on TalTech HPC. For CLFT and CLFCN, we progrmmed upon pytorch directly and avoid too much high-level apis, thus we believe the code should be compatible with various environments. Here list out the package versions on HPC:


## Dataset
- [Dataset](waymo_dataset/README.md)

## RUN 

The script 'visual_run.py' will load single camera (PNG) and LiDAR (PKL) file from folder 'test_images', then produce the segmentation result. The 'vehicle' class will be rendered as green color and 'human' class was rendered as red. We provide the example CLFT and FCN [models](https://www.roboticlab.eu/claude/models/) for visualized prediction. 

### CLFT
```
python visua_run.py -m <modality, chocies: 'rgb' 'lidar' 'cross_fusion> -bb dpt
```

Here is the example of the CLFT segmentation prediction:

![dpt_seg_visual](https://github.com/Claud1234/fcn_transformer_object_segmentation/assets/43088344/305b4613-906b-444f-91b5-83d40abfc556)

### FCN
```
python visua_run.py -m <modality, chocies: 'rgb' 'lidar' 'cross_fusion> -bb fcn
```

Here is the example of the FCN segmentation prediction:

## Training and Testing
The parameters related to the training and testing all defined in file 'config.json'. Here list some important defination in this file.

* [General][sensors_modality] --> Model modalities, choose 'rgb' 'lidar' or 'cross_fusion'.
* [General][model_timm] --> The backbone of CLFT variants. We proposed base, large, and hybrid in the paper. choose 'vit_base_patch16_384' 'vit_large_patch16_384' 'vit_base_resnet50_384'
* [General][emb_dim] --> The embedded dimension for CLFT models, 768 for base and hybrid, 1024 for large.
* [General][resume_training] --> The flag to resume training from saved path. Set to false for scratch training.
* [Log] --> Place to save the model paths. 
* [Dataset][name] --> We provide two datasets, choose 'waymo' or 'iseauto'. The pre-processing parameters of these two datasets are different.  

### CLFT
Training.
```
python3 train.py -bb dpt
```
 
If want to resume the training, use the same command but modify the 'resume_training' flag in 'config.json' file.

Testing.
```
python3 test.py -bb dpt
```

### FCN
Training.
```
python3 train.py -bb fcn
```

Testing.
```
python3 test.py -bb fcn
```


## TO BE CONTINUE....


[//]: # (### Training the model from the beginning)

[//]: # (```)

[//]: # (python3 train.py -r no)

[//]: # (```)

[//]: # (### Training the model from the checkpoint)

[//]: # (First make sure the epochs you set in configs module is bigger than the finished )

[//]: # (epochs which are saved in checkpoint.)

[//]: # ()
[//]: # (```)

[//]: # (python3 train.py -r yes -p <path to checkpoint model>)

[//]: # (```)

[//]: # ()
[//]: # (### Test the model with single input files)

[//]: # (```)

[//]: # (python3 test.py)

[//]: # (```)

[//]: # ()
[//]: # (### Evaluate the model)

[//]: # (Specify the validation input-list file in configs module. Validation uses the )

[//]: # (same batch size and device you set in configs module, but will only run one epoch.)

[//]: # ()
[//]: # (```)

[//]: # (python3 eval.py -p <path to checkpoint model>)

[//]: # (```)
