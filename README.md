# TalTech_DriverlessProject

## Introduction

## Prerequisites
* Please make sure Python version in your machine is at least 3.6

* The pytorch version is 1.8.1(stable). The training can be done either in CPU 
or in GPU. The CUDA version this project has been tested with is 10.2. Please
install the corresponding pytorch libraries over [here](https://pytorch.org/get-started/locally/)

* tdpm
    
* python-opencv
    
* numpy
    
* pickle
    
* PIL

## How To Run
All the configurations were set up in configs module. Check and modify the 
'DATAROOT', 'LOG_DIR' and 'SPLITS' in it. 

Currently this project only can be ran locally in single machine, please specify
'DEVICE' in configs module as 'cpu' for CPU training and 'cuda:<id of GPU> for GPU
training. 


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