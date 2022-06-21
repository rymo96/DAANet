# DAANet
This is the code repository for our paper "Dimension-Aware Attention for Efficient Mobile Networks" (non-commercial use only).

Our dimension-aware attention can be easily inserted into blocks in MobileNet V2, MobileNeXt, ShuffleNet V2, MobileNet V3 and GhostNet. On the ImageNet classification task, the models are trained on 4 GPUs for 300 epochs. The ImageNet pre-trained models can boost the performance of downstream tasks, such as object detection, person re-identification, and semantic segmentation.

Train

Pytorch
	Python 3.8.5, torch 1.7.1, torchvison 0.8.2
	
Prepare
	The ImageNet dataset is required.

Command
	./run.sh
