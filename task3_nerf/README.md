# DL_final_task3
神经网络和深度学习期末作业任务3：

基于NeRF的物体重建和新视图合成

基本要求：

（1） 选取身边的物体拍摄多角度图片/视频，并使用COLMAP估计相机参数，随后使用现成的框架进行训练；

（2） 基于训练好的NeRF渲染环绕物体的视频，并在预留的测试图片上评价定量结果。


本实验使用现成框架[NeRF-Pytorch](https://github.com/yenchenlin/nerf-pytorch)重建自己拍摄的两个场景。

## Installation
```
pip install -r requirements.txt
```
## Data Preparation

1）拍摄

使用焦距设置为"手动聚焦"的安卓手机的相机，拍摄了两个视频，分别是一盒薯片的场景、和一个行李箱上摆放一个小挎包的场景。每个视频均采用围绕物体一至两圈后再拍摄顶部的方式。再使用[Bytedig在线视频抽帧工具](https://www.bytedig.com/web/video-frame)将视频分解为帧，确保抽出了关键视角的照片。

2）LLFF格式数据构建

使用COLMAP 3.9.1做相机参数估计。将每组图片导入COLMAP后，依次进行特征提取、特征匹配和稀疏重建操作，确保每张图片都能有效重建后，然后导出每组图片的相机参数、位姿估计、关键点和3D点等数据，把图片数据和相机等参数按照LLFF转换需要的格式排版，记为llfftest_images。

配置[LLFF项目](https://github.com/Fyusion/LLFF.git)的环境，运行如下代码，得到转换后相机等参数数据poses_bounds.npz：

```
python imgs2poses.py /your_path_to/nerf-pytorch/data/nerf_llff_data/llfftest_images
```

## Training

依次对收集的两个场景数据进行训练：

场景一是薯片放置于阳台地板上，目标物体与周围环境相对简单。

```
CUDA_VISIBLE_DEVICES=0 python run_nerf_tb.py --config configs/chips.txt --no_ndc --spherify --factor 1
```
在300k iterations后，可以在输出目录中找到渲染视频`logs/chips_test/chips_test_spiral_300000_rgb.mp4`、测试图片渲染结果。目录中也保存了参数文件、权重和tensorboard可视化结果。

</br>

场景二是小挎包放置在行李箱上，整体位于杂乱的室内，背景中有随意放置的雨伞、纸箱和水桶。

```
CUDA_VISIBLE_DEVICES=0 python run_nerf_tb.py --config configs/suitcase.txt --no_ndc --spherify --factor 1
```
在300k iterations后，可以在输出目录中找到渲染视频`logs/suitcase_test/suitcase_test_spiral_300000_rgb.mp4`、测试图片渲染结果。目录中也保存了参数文件、权重和tensorboard可视化结果。

</br>

相关模型权重、参数文件、渲染视频、测试集渲染结果保存在：<a href="https://drive.google.com/drive/folders/1wmIIN_fFoMgZh7XGzuTnCNTYTxXnQYTe?usp=sharing">google drive</a>


## Reference

[NeRF-Pytorch](https://github.com/yenchenlin/nerf-pytorch)

