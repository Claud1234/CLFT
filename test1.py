import configs
import numpy as np
from utils.lidar_process import LidarProcess
from utils.image_augment import ImageProcess
a = configs.LIDAR_MEAN
b = configs.LIDAR_STD

mean_lidar = np.array([-0.17263354, 0.85321806, 24.5527253])
std_lidar = np.array([7.34546552, 1.17227659, 15.83745082])
# print (np.array(a).shape)
# print(np.array(a))
# print(np.array(a) == mean_lidar)

# print (configs.TEST_IMAGE)
# if configs.AUGMENT == 'square_crop':
    # print ('sasa')
# else:
    # print ('ssss')    
cam_path = '/home/claude/1.png'
rgb = ImageProcess(cam_path).square_crop()
print (rgb.size)