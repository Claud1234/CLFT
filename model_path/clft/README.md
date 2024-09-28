Download our CLFT model paths from [HERE] (https://www.roboticlab.eu/claude/models/) and put them in this directory. We provide five paths for different variants and modalities. 

When validating or visualizing the models, specify the model path in the 
'config.json' ['General']['model_path']. Also change the parameters in ['CLFT] corresponding to the CLFT variants. 

|                 Model                 | ['CLFT']['emb_dim'] | ['CLFT']['model_timm']  | ['CLFT']['hooks'] |
|:-------------------------------------:|:-------------------:|:-----------------------:|:-----------------:|
|  clft_base_fusion_checkpoint_313.pth  |         768         | 'vit_base_patch16_384'  |   [2, 5, 8, 11]   |
| clft_hybird_fusion_checkpoint_374.pth |         768         | 'vit_base_resnet50_384' |   [2, 5, 8, 11]   |
| clft_hybird_lidar_checkpoint_279.pth  |         768         | 'vit_base_resnet50_384' |   [2, 5, 8, 11]   |
|  clft_hybird_rgb_checkpoint_292.pth   |         768         | 'vit_base_resnet50_384' |   [2, 5, 8, 11]   |
|  clft_large_fusion_checkpoint_366.pth |        1024         | 'vit_large_patch16_384' |  [5, 11, 17, 23]  |