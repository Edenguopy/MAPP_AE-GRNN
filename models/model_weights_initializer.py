"""
模型权重初始化
"""
import torch
from torch import nn
import math


def model_weights_init(m):
    classname = m.__class__.__name__
    # 卷积层
    if classname.find('Conv') != -1:
        # 正态分布初始化
        torch.nn.init.xavier_normal_(m.weight)
    # 归一化层
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0.0)
    # 全连接层
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.0)


