<div align="center">  
  
# Model Specialization for Enhanced Semantic Segmentation in Autonomous Driving
</div>

This is the branch for the CLFT model specialization paper that has been submitted to the 2025 IEEE International Conference on Intelligent Transportation Systems (ITSC). Please note there is still need some works to clean up the code, intergrate some scripts to the pipeline, and upload the new version of the dataset and model paths. I am catching up the deadline for another paper at the moment, and will do these works gradually. If you have any question about this branch, don't hesitate to contact me directly claude.gujunyi@gmail.com 

Also the general instuctions of CLFT pipeline is well documented, you could find all of them in Master branch. 



## Abstract
Recent advancements in deep learning have significantly improved camera-LiDAR fusion for semantic segmentation in autonomous driving. However, a persistent challenge remains: object detection performance varies significantly based on the dataset's object size and frame representation. In the real world and general autonomous driving datasets, particular traffic objects such as vehicles always dominate the frequencies and absolute quantities. In our previous work, CLFT (Camera-LiDAR Fusion Transformer), we observed that large objects are effectively segmented. In contrast, small objects suffer from lower detection accuracy due to limited feature representation in our dataset. This work introduces a robust semantic segmentation approach to improve segmentation across different object classes by leveraging separate, class-specific neural networks. We propose training specialized models for small and large objects, optimizing feature extraction and fusion strategies accordingly. These specialized models are then integrated into a unified cross-scale fusion architecture that adaptively selects the most relevant model outputs based on object characteristics, resulting in an 5-10\% increase in network performance across all classes in the model.

## Method
![tmp drawio](https://github.com/user-attachments/assets/f9361eec-36f7-4d16-86f3-6f4486f93950)



