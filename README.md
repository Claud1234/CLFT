# CLFT: Camera-LiDAR Fusion Transformer for Semantic Segmentation in Autonomous Driving
  
This repository contains the code for the paper 'CLFT: Camera-LiDAR Fusion Transformer for 
Semantic Segmentation in Autonomous Driving' that currently submitted to the 
IEEE Transactions on Intelligent Vehicles journal for reviewing. 

In this work, we proposed a transformer network (CLFT) to fuse camera and LiDAR for semantic object segmentation. The expectation toward the CLFT is to outperform a [FCN network](https://doi.org/10.3390/electronics11071119) we proposed in the past. Therefore we carried out the experiment to compare these two networks with same input data. 

The traning and testing scripts for both networks are included in this repository. The dataloader is specifcically for our own dataset. If you are interested in repeating our experiments, please contact claude.gujunyi@gmail.com Claude for the waymo and iseAuto dataset we used in the experiments.                                                     

## How to Run
The script 'visual_run.py' will load single camera (PNG) and LiDAR (PKL) file from folder 'test_images', then produce the segmentation result. The 'vehicle' class will be render as green color and 'human' class was redered as red. We provide the example CLFT and FCN [models](https://www.roboticlab.eu/claude/models/) for visualized prediction. 

### CLFT
```
python visua_run.py -m <modality, chocies: 'rgb' 'lidar' 'cross_fusion> -bb dpt
```

### FCN
```
python visua_run.py -m <modality, chocies: 'rgb' 'lidar' 'cross_fusion> -bb fcn
```

## Training and Evaluation

## TO BE CONTINUE.....


### Training the model from the beginning
```
python3 train.py -r no
```
### Training the model from the checkpoint
First make sure the epochs you set in configs module is bigger than the finished 
epochs which are saved in checkpoint.

```
python3 train.py -r yes -p <path to checkpoint model>
```

### Test the model with single input files
```
python3 test.py
```

### Evaluate the model
Specify the validation input-list file in configs module. Validation uses the 
same batch size and device you set in configs module, but will only run one epoch.

```
python3 eval.py -p <path to checkpoint model>
```
