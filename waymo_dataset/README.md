We used 22000 frames from Waymo Open dataset. As described in the paper, there is a need to project the LiDAR point clouds on the corresponding camera plane. Our code read the LiDAR's camera-plane-projection directly. Thus, we provided the processed dataset that used for training and validating [HERE](https://www.roboticlab.eu/claude/waymo/).

Then decompress the file and put the 'labeled' folder here. 

What inside the 'splits_clft' are the text files contain the paths of waymo frames.   