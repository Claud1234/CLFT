# TalTech_DriverlessProject

## iseAuto dataset process
Please make sure you have ROS, pcl-ros, openCV, catkin-build-tools installed

The node converts pointcloud and compressed image topics to '.bin' and '.png' file refer to 
[this](https://github.com/leofansq/Tools_RosBag2KITTI) project.

Node subscribes lidar topic '/lidar_front/velodyne_points' and compressed image
topic '/front_camera/image_color_rect/compressed'. Make sure remap the topics if you have 
differnt names.

```
cd catkin_ws
catkin build
source devel/setup.bash
rosrun ttu_autolab bags_to_files
```

There will be a 'png' and 'pcd' folder saved in path '~package/output'. Then run 
'/devel/lib/ttu_autolab/pcd2bin', it will convert pcd files to bin files and save them 
in the same path '~package/output'.

The python script project lidar points to image takes three input arguments, images path,
lidar path and transformation text file.

```
cd catkin_ws/src/ttu_autolab/script
python3 lidar_camera_projection.py ../output/png ../output/bin ../config/calib.txt
```

There will be two folders 'lidar_rgb' and 'lidar_blank' saved in path '~package/output'.
Please manually check the point-image alignment in 'lidar_rgb', is the alignment is not good,
you can go to script to choose project previous and next lidar to the image until have 
a good alignment. 


There are some other scripts  

## Training and Evaluation

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

### Evaluate the model
Specify the validation input-list file in configs module. Validation uses the 
same batch size and device you set in configs module, but will only run one epoch.

```
python3 eval.py -p <path to checkpoint model>
```