import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt

def sharpen_with_pil(img, output_path=None, method="sharpen"):
    """
    使用PIL库的内置滤镜进行锐化
    """
    # 方法1: 使用SHARPEN滤镜
    if method=="sharpen":
        img_sharpen = img.filter(ImageFilter.SHARPEN)
    
    # 方法2: 使用UnsharpMask滤镜（可调参数）
    elif method=="unsharpmask":
        img_sharpen = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    
    # 方法3: 使用自定义卷积核
    elif method=="custom_kernel":
        kernel_sharpen = ImageFilter.Kernel((3, 3), 
            [0, -1, 0,
            -1, 5, -1,
            0, -1, 0], scale=1)
        img_sharpen = img.filter(kernel_sharpen)
    
    return img_sharpen