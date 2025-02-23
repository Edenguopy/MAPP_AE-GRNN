"""
该代码文件用于搭建神经网络架构
CNN->GNN->MLP
"""

import numpy as np
import torch
import torch.nn as nn
import utils.graph.graphTools as gt  # 图数据处理工具包
import utils.graph.graphML as gml    # 图机器学习与深度神经网络模块
from model_weights_initializer import model_weights_init  # 模型参数初始化
from models.googlenet import GoogleNet
from models.resnet import *


class AEGRNN(nn.modules):
    def __init__(self, config):  # config存储神经网络配置参数
        super().__init__()
        self.config = config
        self.numAGVs = self.config.num_AGVs  # AGV数量
        # 局部视野FOV尺寸
        FOV_W = self.config.FOV + 2
        FOV_H = self.config.FOV + 2


        numAction = 5  # 动作空间为5，上下左右停
        # 每一层卷积后特征图的高度与宽度
        convW = [FOV_W]  # convW为一列表，convW[0]即输入层，为FOV宽度
        convH = [FOV_H]


        # CNN中各层的通道数，输入为3个通道
        numChannel = [3, 32, 32, 64, 64, 128]
        # 步长
        numStride = [1, 1, 1, 1, 1]
        # 紧接CNN特征压缩 MLP为1层
        dimCompressMLP = 1
        # 特征压缩MLP各层的神经元数量
        numCompressFeatures = [self.config.numInputFeatures]
        # 最大池化窗口 2*2
        nMaxPoolFilterTaps = 2
        # 最大池化层步长 2
        numMaxPoolStride = 2


        """--------GNN配置--------"""
        # 图神经网络各层的节点信号维度，初始化GNN输入特征维度
        dimNodeSignals = [self.config.numInputFeatures]
        # 图滤波器的抽头数（多项式阶数）
        nGraphFilterTaps = [self.config.nGraphFilterTaps]
        # 图注意力机制的头数
        nAttentionHeads = [self.config.nAttentionHeads]

        # --- actionMLP --- 是否使用Dropout
        if self.config.use_dropout:
            dimActionMLP = 2
            numActionFeatures = [self.config.numInputFeatures, numAction]
        else:
            dimActionMLP = 1
            numActionFeatures = [numAction]



        """
        CNN-特征提取
        """ 
        VGG = False
        GoogleNet = False
        if VGG:  # 默认不选择VGG，参数量过大
            self.ConvLayers = self.make_layers(cfg, batch_norm=True)
            self.compressMLP = nn.Sequential(
                nn.Linear(512, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 128)
            )
            numCompressFeatures = [128]

        else:  # ResNe和GoogleNet
            if self.config.CNN_mode == 'GoogleNet':
                self.ConvLayers = GoogleNet(dropout_rate=0.2)
                numFeatureMap = 1152  
            elif self.config.CNN_mode == 'ResNetLarge_withMLP':
                convl = []
                convl.append(ResNet(BasicBlock, [1, 1, 1], out_map=False))
                convl.append(nn.Dropout(0.2))
                convl.append(nn.Flatten())
                convl.append(nn.Linear(in_features=1152, out_features=self.config.numInputFeatures, bias=True))
                self.ConvLayers = nn.Sequential(*convl)
                numFeatureMap = self.config.numInputFeatures
            
            #####################################################################
            #                                                                   #
            #                MLP-特征压缩                            #
            #                                                                   #
            #####################################################################

            numCompressFeatures = [numFeatureMap] + numCompressFeatures

            compressmlp = []
            for l in range(dimCompressMLP):
                compressmlp.append(
                    nn.Linear(in_features=numCompressFeatures[l], out_features=numCompressFeatures[l + 1], bias=True))
                compressmlp.append(nn.ReLU(inplace=True))
                # if self.config.use_dropout:
                #     compressmlp.append(nn.Dropout(p=0.2))


            self.compressMLP = nn.Sequential(*compressmlp)

        self.numFeatures2Share = numCompressFeatures[-1]

        #####################################################################
        #                                                                   #
        #                    graph neural network                           #
        #                                                                   #
        #####################################################################

        self.L = len(nGraphFilterTaps)  # Number of graph filtering layers
        self.F = [numCompressFeatures[-1]] + dimNodeSignals  # Features
        # self.F = [numFeatureMap] + dimNodeSignals  # Features
        self.K = nGraphFilterTaps  # nFilterTaps # Filter taps
        self.P = nAttentionHeads
        self.E = 1  # Number of edge features
        self.bias = True

        gfl = []  # Graph Filtering Layers
        for l in range(self.L):
            # \\ Graph filtering stage:

            if self.config.attentionMode == 'GAT_origin':
                gfl.append(gml.GraphFilterBatchAttentional_Origin(self.F[l], self.F[l + 1], self.K[l], self.P[l], self.E,self.bias,
                                                           concatenate=self.config.AttentionConcat,attentionMode=self.config.attentionMode))

            elif self.config.attentionMode == 'GAT_modified' or self.config.attentionMode == 'KeyQuery':
                gfl.append(gml.GraphFilterBatchAttentional(self.F[l], self.F[l + 1], self.K[l], self.P[l], self.E, self.bias,concatenate=self.config.AttentionConcat,
                                                    attentionMode=self.config.attentionMode))
            elif self.config.attentionMode == 'GAT_Similarity':
                gfl.append(gml.GraphFilterBatchSimilarityAttentional(self.F[l], self.F[l + 1], self.K[l], self.P[l], self.E, self.bias,concatenate=self.config.AttentionConcat,
                                                    attentionMode=self.config.attentionMode))

           
            # gfl.append(gml.GraphFilterBatchAttentional_Origin(self.F[l], self.F[l + 1], self.K[l], self.P[l], self.E, self.bias, concatenate=self.config.AttentionConcat))

            # gfl.append(
            #     gml.GraphFilterBatchSimilarityAttentional(self.F[l], self.F[l + 1], self.K[l], self.P[l], self.E, self.bias,
            #                                     concatenate=self.config.AttentionConcat))

            # There is a 2*l below here, because we have three elements per
            # layer: graph filter, nonlinearity and pooling, so after each layer
            # we're actually adding elements to the (sequential) list.

            # \\ Nonlinearity
            # gfl.append(nn.ReLU(inplace=True))

        # And now feed them into the sequential
        self.GFL = nn.Sequential(*gfl)  # Graph Filtering Layers

        #####################################################################
        #                                                                   #
        #                    MLP --- map to actions                         #
        #                                                                   #
        #####################################################################
        if self.config.AttentionConcat:
            numActionFeatures = [self.F[-1]*self.config.nAttentionHeads] + numActionFeatures
        else:
            numActionFeatures = [self.F[-1]] + numActionFeatures
        actionsfc = []
        for l in range(dimActionMLP):
            if l < (dimActionMLP - 1):
                actionsfc.append(
                    nn.Linear(in_features=numActionFeatures[l], out_features=numActionFeatures[l + 1], bias=True))
                actionsfc.append(nn.ReLU(inplace=True))
            else:
                actionsfc.append(
                    nn.Linear(in_features=numActionFeatures[l], out_features=numActionFeatures[l + 1], bias=True))

            if self.config.use_dropout:
                actionsfc.append(nn.Dropout(p=0.2))
                print('Dropout is add on MLP')

        self.actionsMLP = nn.Sequential(*actionsfc)
        self.apply(model_weights_init)

    def make_layers(self, cfg, batch_norm=False):
        layers = []

        input_channel = 3
        for l in cfg:
            if l == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                continue

            layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

            if batch_norm:
                layers += [nn.BatchNorm2d(l)]

            layers += [nn.ReLU(inplace=True)]
            input_channel = l

        return nn.Sequential(*layers)


    def addGSO(self, S):
        # We add the GSO on real time, this GSO also depends on time and has
        # shape either B x N x N or B x E x N x N
        if self.E == 1:  # It is B x T x N x N
            assert len(S.shape) == 3
            self.S = S.unsqueeze(1)  # B x E x N x N
        else:
            assert len(S.shape) == 4
            assert S.shape[1] == self.E
            self.S = S
        # Remove nan data
        self.S[torch.isnan(self.S)] = 0
        if self.config.GSO_mode == 'dist_GSO_one':
            self.S[self.S > 0] = 1
        elif self.config.GSO_mode == 'full_GSO':
            self.S = torch.ones_like(self.S).to(self.config.device)
        # self.S[self.S > 0] = 1

    def forward(self, inputTensor):

        B = inputTensor.shape[0] # batch size
        # N = inputTensor.shape[1]
        # C =
        (B,N,C,W,H) = inputTensor.shape
        # print(inputTensor.shape)
        # print(B,N,C,W,H)
        # B x G x N
        # extractFeatureMap = torch.zeros(B, self.numFeatures2Share, self.numAgents).to(self.config.device)

        input_currentAgent = inputTensor.reshape(B*N,C,W,H).to(self.config.device)
        # print("input_currentAgent:", input_currentAgent.shape)

        featureMap = self.ConvLayers(input_currentAgent).to(self.config.device)
        # print("featureMap:", featureMap.shape)

        featureMapFlatten = featureMap.view(featureMap.size(0), -1).to(self.config.device)

        # print("featureMapFlatten:", featureMapFlatten.shape)


        compressfeature = self.compressMLP(featureMapFlatten).to(self.config.device)
        # print("compressfeature:", compressfeature.shape)


        extractFeatureMap = compressfeature.reshape(B,N,self.numFeatures2Share).to(self.config.device).permute([0,2,1])
        # extractFeatureMap_old = compressfeature.reshape(B,N,self.numFeatures2Share).to(self.config.device)

        # print("extractFeatureMap_old:", extractFeatureMap_old.shape)


        # extractFeatureMap = extractFeatureMap_old.permute([0,2,1]).to(self.config.device)
        # print("extractFeatureMap:", extractFeatureMap.shape)


        # DCP
        for l in range(self.L):
            # \\ Graph filtering stage:
            # There is a 3*l below here, because we have three elements per
            # layer: graph filter, nonlinearity and pooling, so after each layer
            # we're actually adding elements to the (sequential) list.
            # self.GFL[2 * l].addGSO(self.S) # add GSO for GraphFilter
            self.GFL[l].addGSO(self.S)  # add GSO for GraphFilter

        # B x F x N - > B x G x N,
        sharedFeature = self.GFL(extractFeatureMap)

        (_, num_G, _) = sharedFeature.shape


        sharedFeature_stack =sharedFeature.permute([0,2,1]).to(self.config.device).reshape(B*N,num_G)
        # sharedFeature_permute = sharedFeature.permute([0, 2, 1]).to(self.config.device)
        # sharedFeature_stack = sharedFeature_permute.reshape(B*N,num_G)

        # print(sharedFeature_stack.shape)
        action_predict = self.actionsMLP(sharedFeature_stack)
        # print(action_predict)
        # print(action_predict.shape)


        return action_predict


