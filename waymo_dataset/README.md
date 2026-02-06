We used 22000 frames from Waymo Open dataset. As described in the paper, there is a need to project the LiDAR point clouds on the corresponding camera plane. 
Our code read the LiDAR's camera-plane-projection directly. 

The waymo dataset used in this work to train the provided pre-trained models is around 70 GB. 
Unfortunately the public link to download the dataset is not available anymore, please write to claude.gujunyi@gmail.com if you 
are interested in the dataset, we can figure out a way to transfer the dataset. 

After you get the dataset archive, decompress it and put the 'labeled' folder here. 

What inside the 'splits_clft' are the text files contain the paths of waymo frames.   