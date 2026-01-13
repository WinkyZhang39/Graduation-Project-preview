import torch
import torch.nn as nn
import torch.nn.functional as F
"""
表达能力差异：
AsymBiChaFuseReduce 先对高层特征进行非线性变换，增强其特征表达能力
AsymBiChaFuse 直接使用原始特征，可能对特征质量要求更高

参数和计算量：
AsymBiChaFuseReduce 额外多一个 1x1 卷积层，参数更多
AsymBiChaFuse 更轻量

适用场景：
AsymBiChaFuseReduce：适合高层特征需要进一步提炼的场景
AsymBiChaFuse：适合高层特征已足够丰富的场景（如预训练模型的特征）
"""
class AsymBiChaFuseReduce(nn.Module):
    def __init__(self, channels=64, r=4):
        super(AsymBiChaFuseReduce, self).__init__()
        self.channels = channels#输入特征图的通道数
        self.bottleneck_channels = int(channels // r)#输出通道数，这里是为了降维
        #kernel_size 卷积核大小 stride 步长 padding 是否填充

        # 对高层特征的预处理，在保持特征图尺寸的同时进行特征变换
        self.feature_high = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(min(32,channels),channels),
            nn.ReLU(inplace=True)
        )

        # Top-down 路径: 高层特征 -> 全局池化 -> 生成对低层特征的权重
        self.topdown = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 对应 mxnet 的 GlobalAvgPool2D
            nn.Conv2d(channels, self.bottleneck_channels, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(min(32,self.bottleneck_channels),self.bottleneck_channels),#对输出通道进行批归一化
            #上两行等效于全连接层
            nn.ReLU(inplace=True),
            nn.Conv2d(self.bottleneck_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(min(32,channels),channels),
            nn.Sigmoid()
        )

        # Bottom-up 路径: 低层特征 -> 生成对高层特征的权重
        self.bottomup = nn.Sequential(
            nn.Conv2d(channels, self.bottleneck_channels, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(min(32,self.bottleneck_channels),self.bottleneck_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.bottleneck_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(min(32,channels),channels),
            nn.Sigmoid()
        )

        # 融合后处理
        self.post = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(min(32,channels),channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, xh, xl):
        xh = self.feature_high(xh)
        
        # 生成注意力权重
        topdown_weight = self.topdown(xh)  # 用于加权低层特征
        bottomup_weight = self.bottomup(xl)  # 用于加权高层特征
        
        # 双向加权融合
        xs = 2 * (xl * topdown_weight) + 2 * (xh * bottomup_weight)
        xs = self.post(xs)
        return xs

class AsymBiChaFuse(nn.Module):
    # 此版本与 AsymBiChaFuseReduce 的区别仅在于没有 self.feature_high 预处理
    def __init__(self, channels=64, r=4):
        super(AsymBiChaFuse, self).__init__()
        self.channels = channels
        self.bottleneck_channels = int(channels // r)

        self.topdown = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, self.bottleneck_channels, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(min(32,self.bottleneck_channels),self.bottleneck_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.bottleneck_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(min(32,channels),channels),
            nn.Sigmoid()
        )

        self.bottomup = nn.Sequential(
            nn.Conv2d(channels, self.bottleneck_channels, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(min(32,self.bottleneck_channels),self.bottleneck_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.bottleneck_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(min(32,channels),channels),
            nn.Sigmoid()
        )

        self.post = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(min(32,channels),channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, xh, xl):
        topdown_weight = self.topdown(xh)
        bottomup_weight = self.bottomup(xl)
        xs = 2 * (xl * topdown_weight) + 2 * (xh * bottomup_weight)
        xs = self.post(xs)
        return xs