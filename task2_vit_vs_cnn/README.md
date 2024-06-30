# DL_final_task2
神经网络和深度学习期末作业任务2：

在CIFAR-100数据集上比较基于Transformer和CNN的图像分类模型

基本要求：

（1） 分别基于CNN和Transformer架构实现具有相近参数量的图像分类网络；

（2） 在CIFAR-100数据集上采用相同的训练策略对二者进行训练，其中数据增强策略中应包含CutMix；

（3） 尝试不同的超参数组合，尽可能提升各架构在CIFAR-100上的性能以进行合理的比较。


本实验基于PyTorch Lightning在CIFAR-100数据集上训练并测试具有相近参数量的ViT和ResNet50。

## Installation

```
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 pytorch-lightning  --extra-index-url https://download.pytorch.org/whl/cu113

pip install warmup-scheduler

pip install pytorch-lightning==1.5.10

pip install numpy==1.23.0
```

## Training
Train ViT model with CIFAR-100 dataset (default):
```
CUDA_VISIBLE_DEVICES=0 python main.py --dataset c100 --label-smoothing --cutmix --patch 8 --hidden 384 --mlp-hidden 1536 --num-layers 12 --head 8
```
</br>

Train ResNet50 model with CIFAR-100 dataset:
```
CUDA_VISIBLE_DEVICES=0 python main.py --model-name resnet --dataset c100 --label-smoothing --cutmix
```
</br>


最优模型权重保存在：<a href="https://drive.google.com/drive/folders/1INPSAXANyt3CsBdYjNjTND6p73EMWCoZ?usp=sharing">google drive</a>


## Folder Structure
```bash
.
├── model
│   ├── resnet
│   └── vit
├── utils
│   ├── autoaugment.py
│   ├── dataaug.py(包括RandomCropPaste、CutMix、MixUp数据增强方式)
│   └── utils.py(包括获取criterion、model、dataset等设置)
├── weights
├── data(存储数据集，首次运行`main.py`时自动下载数据集)
└── main.py           
```

## Reference

ViT: [ViT-CIFAR repo](https://github.com/omihub777/ViT-CIFAR); [vision_transformer repo](https://github.com/google-research/vision_transformer)

ViT、ResNet、DataAug: [VIT_CNN_CIFAR_10_PyTorch_Lightning](https://github.com/dqj5182/VIT_CNN_CIFAR_10_PyTorch_Lightning)

