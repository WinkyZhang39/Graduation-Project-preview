import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np


class Get_gradient_nopadding(nn.Module):
    def __init__(self):
        super(Get_gradient_nopadding, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)

    def forward(self, x):
        x_list = []
        for i in range(x.shape[1]):
            x_i = x[:, i]
            x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v, padding=1)
            x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h, padding=1)
            x_i = torch.sqrt(torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2) + 1e-6)
            x_list.append(x_i)

        print(x_list[1]-x_list[0])
        x = torch.cat(x_list, dim=1)
        return x


class Get_curvature(nn.Module):
    def __init__(self):
        super(Get_curvature, self).__init__()
        kernel_v1 = [[0, -1, 0],
                     [0, 0, 0],
                     [0, 1, 0]]
        kernel_h1 = [[0, 0, 0],
                     [-1, 0, 1],
                     [0, 0, 0]]
        kernel_h2 = [[0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [1, 0, -2, 0, 1],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0]]
        kernel_v2 = [[0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, -2, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0]]
        kernel_w2 = [[1, 0, -1],
                     [0, 0, 0],
                     [-1, 0, 1]]
        kernel_h1 = torch.FloatTensor(kernel_h1).unsqueeze(0).unsqueeze(0)
        kernel_v1 = torch.FloatTensor(kernel_v1).unsqueeze(0).unsqueeze(0)
        kernel_v2 = torch.FloatTensor(kernel_v2).unsqueeze(0).unsqueeze(0)
        kernel_h2 = torch.FloatTensor(kernel_h2).unsqueeze(0).unsqueeze(0)
        kernel_w2 = torch.FloatTensor(kernel_w2).unsqueeze(0).unsqueeze(0)
        self.weight_h1 = nn.Parameter(data=kernel_h1, requires_grad=False)
        self.weight_v1 = nn.Parameter(data=kernel_v1, requires_grad=False)
        self.weight_v2 = nn.Parameter(data=kernel_v2, requires_grad=False)
        self.weight_h2 = nn.Parameter(data=kernel_h2, requires_grad=False)
        self.weight_w2 = nn.Parameter(data=kernel_w2, requires_grad=False)

    def forward(self, x):
        x_list = []
        for i in range(x.shape[1]):
            x_i = x[:, i]
            x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v1, padding=1)
            x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h1, padding=1)
            x_i_v2 = F.conv2d(x_i.unsqueeze(1), self.weight_v2, padding=2)
            x_i_h2 = F.conv2d(x_i.unsqueeze(1), self.weight_h2, padding=2)
            x_i_w2 = F.conv2d(x_i.unsqueeze(1), self.weight_w2, padding=1)
            sum = torch.pow((torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2)), 3 / 2)
            fg = torch.mul(torch.pow(x_i_v, 2), x_i_v2) + 2 * torch.mul(torch.mul(x_i_v, x_i_h), x_i_w2) + torch.mul(
                torch.pow(x_i_h, 2), x_i_h2)
            fh = torch.mul(torch.pow(x_i_v, 2), x_i_h2) - 2 * torch.mul(torch.mul(x_i_v, x_i_h), x_i_w2) + torch.mul(
                torch.pow(x_i_h, 2), x_i_v2)
            x_i = torch.div(torch.abs(fg - fh), sum + 1e-10)
            x_i = torch.div(torch.abs(fh), sum + 1e-10)
            x_list.append(x_i)
        x = torch.cat(x_list, dim=1)
        return x

class FeatureEncoder(nn.Module):
    def __init__(self, out_dims):
        super(FeatureEncoder, self).__init__()

        self.conv1 = nn.Conv2d(3, out_dims[0], kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_dims[0], out_dims[0], kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(out_dims[0], out_dims[1], kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(out_dims[1], out_dims[1], kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(out_dims[1], out_dims[2], kernel_size=3, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(out_dims[2], out_dims[2], kernel_size=3, padding=1)
        self.relu6 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv7 = nn.Conv2d(out_dims[2], out_dims[3], kernel_size=3, padding=1)
        self.relu7 = nn.ReLU(inplace=True)
        self.conv8 = nn.Conv2d(out_dims[3], out_dims[3], kernel_size=3, padding=1)
        self.relu8 = nn.ReLU(inplace=True)

    def forward(self, x):
        # Stage 1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool1(x)
        x1 = x

        # Stage 2
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.maxpool2(x)
        x2 = x

        # Stage 3
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.conv6(x)
        x = self.relu6(x)
        x = self.maxpool3(x)
        x3 = x

        # Stage 4
        x = self.conv7(x)
        x = self.relu7(x)
        x = self.conv8(x)
        x = self.relu8(x)
        x4 = x

        return x1, x2, x3, x4


class WaveletTransform2D(nn.Module):
    """专用于WPMD的小波变换，只提取LH和HL分量"""
    def __init__(self, wavelet='haar'):
        super(WaveletTransform2D, self).__init__()
        self.wavelet = wavelet
        
        # 定义Haar小波滤波器（用于卷积实现）
        # LL低通滤波器 (近似分量)
        self.ll_kernel = torch.tensor([[[[1, 1], [1, 1]]]], dtype=torch.float32) * 0.5
        
        # LH水平高通滤波器 (垂直边缘)
        self.lh_kernel = torch.tensor([[[[1, -1], [1, -1]]]], dtype=torch.float32) * 0.5
        
        # HL垂直高通滤波器 (水平边缘)
        self.hl_kernel = torch.tensor([[[[1, 1], [-1, -1]]]], dtype=torch.float32) * 0.5
        
        # HH对角线高通滤波器
        self.hh_kernel = torch.tensor([[[[1, -1], [-1, 1]]]], dtype=torch.float32) * 0.5
    
    def forward(self, x):
            """
            输入: [batch, channels, height, width]
            输出: LL, LH, HL, HH 四个分量
            height和width维度都减半，这是因为代码中使用了stride=2的卷积操作
            """
            batch_size, channels, height, width = x.shape
            
            # 确保尺寸是偶数
            if height % 2 != 0 or width % 2 != 0:
                x = F.pad(x, (0, width % 2, 0, height % 2))
                height, width = x.shape[-2:]
            
            # 存储各分量
            LL_list, LH_list, HL_list, HH_list = [], [], [], []
            
            # 获取输入数据类型
            dtype = x.dtype
            
            for c in range(channels):
                x_c = x[:, c:c+1, :, :]
                
                # 创建Haar小波滤波器，使用输入的数据类型
                ll_kernel = torch.tensor([[[[1, 1], [1, 1]]]], dtype=dtype, device=x.device) * 0.5
                lh_kernel = torch.tensor([[[[1, -1], [1, -1]]]], dtype=dtype, device=x.device) * 0.5
                hl_kernel = torch.tensor([[[[1, 1], [-1, -1]]]], dtype=dtype, device=x.device) * 0.5
                hh_kernel = torch.tensor([[[[1, -1], [-1, 1]]]], dtype=dtype, device=x.device) * 0.5
                
                # 使用卷积实现小波变换
                ll = F.conv2d(x_c, ll_kernel, stride=2, padding=0)
                lh = F.conv2d(x_c, lh_kernel, stride=2, padding=0)
                hl = F.conv2d(x_c, hl_kernel, stride=2, padding=0)
                hh = F.conv2d(x_c, hh_kernel, stride=2, padding=0)
                
                LL_list.append(ll)
                LH_list.append(lh)
                HL_list.append(hl)
                HH_list.append(hh)
            
            # 拼接各通道
            LL = torch.cat(LL_list, dim=1)
            LH = torch.cat(LH_list, dim=1)
            HL = torch.cat(HL_list, dim=1)
            HH = torch.cat(HH_list, dim=1)
            
            return LL, LH, HL, HH


class PMD_Module(nn.Module):
    """PMD处理模块，专门处理LH和HL分量"""
    def __init__(self, channels):
        super(PMD_Module, self).__init__()
        self.channels = channels
        
        # 扩散系数函数 g(·)
        self.diffusion_coef = nn.Sequential(
            nn.Conv2d(2*channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2*channels, 3, padding=1),
            nn.Sigmoid()  # 输出范围[0,1]
        )
        
        # 分别为LH和HL创建扩散滤波器
        self.diffusion_filter_LH = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.diffusion_filter_HL = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        
        # 初始化扩散滤波器为Laplacian-like
        self._init_diffusion_filter()
    
    def _init_diffusion_filter(self):
        """初始化扩散滤波器为拉普拉斯算子近似"""
        laplacian_kernel = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad():
            self.diffusion_filter_LH.weight.data = laplacian_kernel.repeat(self.channels, 1, 1, 1) * 0.25
            self.diffusion_filter_HL.weight.data = laplacian_kernel.repeat(self.channels, 1, 1, 1) * 0.25
    
    def forward(self, LH, HL, num_iterations=1):
        """
        对LH和HL分量进行PMD处理
        输入: LH [batch, channels, h, w], HL [batch, channels, h, w]
        输出: 处理后的LH, HL
        """
        # 确保输入为浮点类型
        LH = LH.float()
        HL = HL.float()
        
        # 获取实际的输入通道数和空间尺寸
        actual_channels = LH.shape[1]
        h, w = LH.shape[2], LH.shape[3]
        
        # 合并LH和HL用于计算扩散系数
        combined = torch.cat([LH, HL], dim=1)
        
        # 计算梯度幅度（用于扩散系数）
        # 创建水平梯度卷积核（在宽度方向检测边缘）
        horizontal_kernel = torch.tensor([[[[-1, 0, 1]]]], device=LH.device, dtype=LH.dtype)
        # 创建垂直梯度卷积核（在高度方向检测边缘）
        vertical_kernel = torch.tensor([[[[-1], [0], [1]]]], device=HL.device, dtype=HL.dtype)
        
        # 使用适当的padding保持空间尺寸不变
        grad_LH = torch.abs(F.conv2d(LH, 
                                     horizontal_kernel.repeat(actual_channels, 1, 1, 1), 
                                     padding=(0, 1),
                                     groups=actual_channels))
        
        grad_HL = torch.abs(F.conv2d(HL, 
                                     vertical_kernel.repeat(actual_channels, 1, 1, 1), 
                                     padding=(1, 0),
                                     groups=actual_channels))
        
        # 确保两个梯度的空间尺寸相同
        if grad_LH.shape[2:] != grad_HL.shape[2:]:
            target_h, target_w = min(grad_LH.shape[2], grad_HL.shape[2]), min(grad_LH.shape[3], grad_HL.shape[3])
            grad_LH = grad_LH[:, :, :target_h, :target_w]
            grad_HL = grad_HL[:, :, :target_h, :target_w]
        
        grad_mag = torch.cat([grad_LH, grad_HL], dim=1)
        
        # 计算扩散系数
        g = self.diffusion_coef(grad_mag)
        
        # 分离扩散系数
        g_LH, g_HL = g.chunk(2, dim=1)
        
        # PMD迭代处理
        LH_out = LH.clone()
        HL_out = HL.clone()
        
        for _ in range(num_iterations):
            # 对LH应用扩散
            diff_LH = self.diffusion_filter_LH(g_LH * LH_out)
            LH_out = LH_out + 0.1 * diff_LH  # 学习率参数
            
            # 对HL应用扩散
            diff_HL = self.diffusion_filter_HL(g_HL * HL_out)
            HL_out = HL_out + 0.1 * diff_HL
        
        return LH_out, HL_out


class WPMD_Block(nn.Module):
    """完整的WPMD模块，输出尺寸为输入的四分之一
    
    修改说明：
    1. 小波变换后输出尺寸为一半，然后通过步长为2的池化再减半
    2. 重建后的特征图上采样到一半尺寸，再与残差连接
    3. 残差连接使用4x4池化直接下采样到1/4
    """
    def __init__(self, in_channels, out_channels, downsample=True):
        super(WPMD_Block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = downsample
        
        # 小波变换
        self.wavelet = WaveletTransform2D(wavelet='haar')
        
        # PMD处理模块
        self.pmd = PMD_Module(channels=in_channels)
        
        # 重建模块（处理LH、HL、LL、HH）
        self.reconstruct = nn.Sequential(
            nn.Conv2d(4 * in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 残差连接
        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
        # 如果需要下采样但维度不匹配，使用1x1卷积进行通道调整
        self.adjust_channel = None
        if in_channels != out_channels and not downsample:
            self.adjust_channel = nn.Conv2d(in_channels, out_channels, 1)
        
        # 额外的池化层，用于进一步下采样到1/4
        self.downsample_pool = nn.AvgPool2d(kernel_size=2, stride=2) if downsample else nn.Identity()
    
    def forward(self, x):
        """
        输入: x [batch, in_channels, height, width]
        输出: 如果downsample=True: [batch, out_channels, height/4, width/4]
             如果downsample=False: [batch, out_channels, height, width]
        """
        batch_size, channels, height, width = x.shape
        
        if self.downsample:
            # 1. 小波变换（输出尺寸为一半）
            LL, LH, HL, HH = self.wavelet(x)
            
            # 2. 对LH和HL进行PMD处理
            LH_processed, HL_processed = self.pmd(LH, HL, num_iterations=3)
            
            # 3. 重建结构特征（当前尺寸为一半）
            combined = torch.cat([LL, LH_processed, HL_processed, HH], dim=1)
            structural_feature = self.reconstruct(combined)
            
            # 4. 将结构特征下采样到1/4
            structural_feature = self.downsample_pool(structural_feature)
            
            # 5. 残差连接（下采样原始输入到1/4尺寸）
            if isinstance(self.residual, nn.Identity):
                residual = F.avg_pool2d(x, kernel_size=4, stride=4)
            else:
                residual = self.residual(F.avg_pool2d(x, kernel_size=4, stride=4))
            
            # 6. 融合输出
            output = structural_feature + residual
            
        else:
            # 不进行下采样的版本（保持原始尺寸）
            # 首先进行小波变换（尺寸减半）
            LL, LH, HL, HH = self.wavelet(x)
            
            # 对LH和HL进行PMD处理
            LH_processed, HL_processed = self.pmd(LH, HL, num_iterations=3)
            
            # 重建特征（尺寸减半）
            combined = torch.cat([LL, LH_processed, HL_processed, HH], dim=1)
            structural_feature = self.reconstruct(combined)
            
            # 上采样回原始尺寸
            structural_feature = F.interpolate(structural_feature, 
                                             size=(height, width), 
                                             mode='bilinear', 
                                             align_corners=False)
            
            # 残差连接
            if self.adjust_channel is not None:
                residual = self.adjust_channel(x)
            else:
                residual = self.residual(x) if self.in_channels != self.out_channels else x
            
            output = structural_feature + residual
        
        # print(f"WPMD_Block output shape: {output.shape}")
        return output