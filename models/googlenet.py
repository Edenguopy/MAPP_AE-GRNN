import torch.nn as nn

class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super().__init__()
        # 各分支定义保持不变...
        # （这里保留原始Inception模块实现）

class GoogleNet(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        self.features = nn.Sequential(
            # 初始卷积层
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            
            # Inception模块序列
            Inception(64, 64, 96, 128, 16, 32, 32),
            Inception(256, 128, 128, 192, 32, 96, 64),
            nn.MaxPool2d(3, stride=2, padding=1),
            Inception(480, 192, 96, 208, 16, 48, 64),
            Inception(512, 160, 112, 224, 24, 64, 64),
            
            # 输出处理
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Conv2d(512, 32, kernel_size=1),
            nn.Flatten(),
            nn.Dropout(dropout_rate)
        )
    
    def forward(self, x):
        return self.features(x)