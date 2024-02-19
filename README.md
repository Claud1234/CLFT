# CLFT: Camera-LiDAR Fusion Transformer for Semantic Segmentation in Autonomous Driving
  
This repository contains the code for the paper 'CLFT: Camera-LiDAR Fusion Transformer for 
Semantic Segmentation in Autonomous Driving' that currently submitted to the 
IEEE Transactions on Intelligent Vehicles journal for reviewing. 

In this work, we proposed a transformer network (CLFT) to fuse camera and LiDAR for semantic object segmentation. The expectation toward the CLFT is to outperform a [FCN network](https://doi.org/10.3390/electronics11071119) we proposed in the past. Therefore we carried out the experiment to compare these two networks with same input data. 

The traning and testing scripts for both networks are included in this repository. The dataloader is specifically for our own dataset. If you are interested in repeating our experiments, please contact claude.gujunyi@gmail.com Claude for the waymo and iseAuto dataset we used in the experiments.                                                     

## How to Run
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
