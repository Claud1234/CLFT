We used 22000 frames from Waymo Open dataset. As described in the paper, there is a need to project the LiDAR point clouds on the corresponding camera plane. 
Our code read the LiDAR's camera-plane-projection directly. 

The waymo dataset used in this work to train the provided pre-trained models is around 70 GB. 
Unfortunately the public link to download the dataset is not available anymore, please write to claude.gujunyi@gmail.com if you 
are interested in the dataset, we can figure out a way to transfer the dataset. 

After you get the dataset archive, decompress it and put the 'labeled' folder here. 

What inside the 'splits_clft' are the text files contain the paths of waymo frames.

The updated local dataset (15 October 2026) contains 22,000 frames with matching
`camera/*.png`, `annotation/*.png` and `lidar/*.pkl` files. Each pickle provides
`3d_points` and `camera_coordinates`, including the front-camera projection at
1920×1280; camera images and labels are 480×320. Use
`Dataset.lidar_format: "pkl"`, now selected in `config.macos.json`, for camera–XYZ
fusion with paired cropping and augmentation. XYZ normalization uses the
configured standard deviations and keeps signed floating-point values.

The earlier dataset copy contained rendered `lidar_png/` files instead. The
explicit `Dataset.lidar_format: "png"` option is retained for that format;
it does not reproduce XYZ point-cloud preprocessing.
