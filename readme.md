# 计算视觉期末大作业

宋振华, 赵衍琛, 赵睿

我们复现了CVPR2018中的`《Single-Shot Refinement Neural Network for Object Detection》` 工作, 该工作提出了锚点预测与物体分类两个模块, 在多尺度上同时监督锚点的定位与多物体的分类, 并且在每个尺度上使用变换连接模块实现上述二者的高效信息传递. 我们根据原文的描述, 使用`pytorch`框架复现了其算法. 同时我们尝试使用注意力机制与可变形卷积对其进行了改进, 实验结果表明, 我们所复现算法的性能与原版论文中所描述的大致相同, 我们的改进算法也能够起到一定的提升作用, 算法的实时性仍能保证, 在`NVIDIA GeForce 2080Ti`显卡上, 网络能够做到实时预测, 以15ms(66.7FPS)的用时处理一帧图像. 



## 配置

参考 `datasets/readme.md`, 下载所需的Pascal VOC 2007, Pascal VOC 2012数据集.

参考 `weights/readme.md`, 下载预训练好的 `pytorch`  VGG-16 网络权重.



也可以直接下载完整的项目目录, 包括数据集, 代码, 及预训练好的网络权重.

百度网盘链接: 

链接：https://pan.baidu.com/s/1Ffb_5l1G0_bCrWgc45DjuQ 
提取码：44om 



北大网盘链接: 

https://disk.pku.edu.cn:443/link/28656A03F2D8756E338C0B354ABBCC8B
有效期限：2021-02-22 23:59



为了运行修改后的`deform`和`modulation`网络, 需要进入 `models/deformableConv`目录下, 运行 `sh make.sh`, 编译所需的库. 我们使用 `Ubuntu 18.04`, `gcc 7.5`, `cuda 11.1`, `pytorch 1.0` 进行配置. (如果只运行`RefineDet`网络, 无需进行此步配置)

## 训练

训练`RefineDet`网络 `python train_refinedet.py`

## 训练

训练 `RefineDet` 网络 `python train_refinedet.py`

训练修改后的 `deform` 网络 `python train_refinedet.py --model_type deform`

训练修改后的 `modulation` 网络 `python train_refinedet.py --model_type modulation`

## 测试

测试 `RefineDet` 网络 `python eval_refinedet.py`

测试修改后的 `deform` 网络 `python eval_refinedet.py --model_type deform --trained_model weights/RefineDet320_VOC_refinedet_deform1_final.pth`

测试修改后的 `modulation` 网络 `python eval_refinedet.py --model_type modulation --trained_model weights/RefineDet320_VOC_refinedet_modulation1_final.pth`

 ## 展示

运行`python live-demo.py`, 对视频流数据进行处理

运行`python new-demo.py`, 对单张图片进行处理

