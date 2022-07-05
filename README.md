DAANet
====
This is the code repository for our paper "Dimension-Aware Attention for Efficient Mobile Networks" (non-commercial use only).

Our dimension-aware attention can be easily inserted into blocks in MobileNet V2, MobileNeXt, ShuffleNet V2, MobileNet V3 and GhostNet. On the ImageNet classification task, the models are trained on 4 GPUs for 300 epochs. The ImageNet pre-trained models can boost the performance of downstream tasks, such as object detection, person re-identification, and semantic segmentation.

Pytorch
----
Python 3.8.5, torch 1.7.1, torchvison 0.8.2

Prepare
----
The public ImageNet-1K dataset is required.

Optional Models and Attention blocks
----
In train_dist_ema.py (line 235--258), the model can be changed.  
We implement MobileNet V2, MobileNeXt, ShuffleNet V2, MobileNet V3-small and GhostNet.  
The optional attention blocks are SE, CBAM, Coordinate Attention (accepted by CVPR 2021), and our DAA block. 

|  Attention Blocks   | Description  |
|  ----  | ----  |
| se_8  | SE, r=8 |
| cbam_24 | CBAM, r=24 |
| ca_32 | CA, r=32 |
| daa\_7\_8\_1\_1\_1 | DAA, kernel\_size=7, r=8, use\_c=True, use\_x=True, use\_y=True |
	
Command
----
```Train the model for ImageNet classification
./run.sh
```
