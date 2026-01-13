import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from WPMD import WPMD_Block, WaveletTransform2D, Get_gradient_nopadding, PMD_Module

"""设置matplotlib支持中文显示"""
try:
    import matplotlib.pyplot as plt
    import matplotlib
            
    # 尝试不同的方法设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
            
    # 检查当前字体是否支持中文
    test_fonts = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS', 'WenQuanYi Micro Hei']
            
    # 获取系统可用的字体
    available_fonts = [f.name for f in matplotlib.font_manager.fontManager.ttflist]
            
    # 选择可用的中文字体
    for font in test_fonts:
        if any(font.lower() in f.lower() for f in available_fonts):
            plt.rcParams['font.sans-serif'] = [font]
            # print(f"使用字体: {font}")
            break
    else:
        print("警告: 未找到合适的中文字体，中文显示可能不正常")
        print("建议: 安装中文字体或使用英文标签")
        
                
except ImportError:
    print("matplotlib未安装，无法显示图像")
except Exception as e:
    print(f"设置字体时出错: {e}")

from PIL import Image, ImageFilter
import matplotlib.pyplot as plt


def sharpen_with_pil(image_path, output_path=None):
    """
    使用PIL库的内置滤镜进行锐化
    """
    # 打开图像
    img = Image.open(image_path)
    
    print("PIL锐化方法:")
    print(f"原始图像大小: {img.size}")
    print(f"原始图像模式: {img.mode}")
    
    # 方法1: 使用SHARPEN滤镜
    img_sharpen1 = img.filter(ImageFilter.SHARPEN)
    
    # 方法2: 使用UnsharpMask滤镜（可调参数）
    img_sharpen2 = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    
    # 方法3: 使用自定义卷积核
    kernel_sharpen = ImageFilter.Kernel((3, 3), 
        [0, -1, 0,
         -1, 5, -1,
         0, -1, 0], scale=1)
    img_sharpen3 = img.filter(kernel_sharpen)
    
    # 显示结果
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('原始图像', fontsize=12)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img_sharpen1)
    axes[0, 1].set_title('SHARPEN滤镜', fontsize=12)
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(img_sharpen2)
    axes[1, 0].set_title('UnsharpMask滤镜\n(radius=2, percent=150)', fontsize=12)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(img_sharpen3)
    axes[1, 1].set_title('自定义锐化核\n(拉普拉斯增强)', fontsize=12)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # # 保存结果
    # if output_path:
    #     img_sharpen1.save(f"{output_path}_sharpen1.jpg")
    #     img_sharpen2.save(f"{output_path}_sharpen2.jpg")
    #     img_sharpen3.save(f"{output_path}_sharpen3.jpg")
    #     print(f"锐化后的图像已保存到: {output_path}_*.jpg")
    
    return img, img_sharpen1, img_sharpen2, img_sharpen3

def load_and_preprocess_image(image_path, size=256):
    """加载并预处理图像"""
    # 读取图像
    img = Image.open(image_path).convert('RGB')
    
    # 定义转换
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    
    # 应用转换并添加batch维度
    img_tensor = transform(img).unsqueeze(0)
    return img, img_tensor

def visualize_wpmd_effects(image_path, device='cpu', seed=42):
    """
    可视化WPMD模块对图像的处理效果
    
    参数:
    - image_path: 输入图像路径
    - device: 运行设备 ('cpu' 或 'cuda')
    """
    # 设置随机种子
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # 1. 加载图像
    original_img, img_tensor = load_and_preprocess_image(image_path)
    img_tensor = img_tensor.to(device)
    
    print(f"输入图像尺寸: {img_tensor.shape}")
    
    # 2. 初始化WPMD模块
    wpmd_module = WPMD_Block(in_channels=3, out_channels=3, downsample=False).to(device)
    
    # 3. 设置模型为评估模式
    wpmd_module.eval()
    
    # 4. 前向传播
    with torch.no_grad():
        processed_tensor = wpmd_module(img_tensor)
    
    print(f"输出图像尺寸: {processed_tensor.shape}")
    
    # 5. 转换为可显示的图像
    original_img_np = img_tensor.squeeze(0).cpu().numpy()
    processed_img_np = processed_tensor.squeeze(0).cpu().numpy()
    
    # 将图像数据从[C, H, W]转换为[H, W, C]并调整范围到[0, 1]
    original_img_np = np.transpose(original_img_np, (1, 2, 0))
    processed_img_np = np.transpose(processed_img_np, (1, 2, 0))
    mix_img_np = 0.5 * original_img_np + 0.5 * processed_img_np
    
    # 6. 可视化
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    fig.suptitle('WPMD effects', fontsize=16)
    
    # 显示原始图像和各个通道
    axes[0].imshow(original_img_np)
    axes[0].set_title('ORIGINAL (RGB)')
    axes[0].axis('off')
    
    # 显示处理后的图像和各个通道
    axes[1].imshow(np.clip(processed_img_np, 0, 1))
    axes[1].set_title('after_WPMD')
    axes[1].axis('off')
    
    axes[2].imshow(np.clip(mix_img_np, 0, 1))
    axes[2].set_title('MIX (50% ORI + 50% WPMD)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # 7. 计算差异
    diff = np.abs(processed_img_np - original_img_np)
    diff_avg = np.mean(diff)
    print(f"平均像素变化: {diff_avg:.4f}")
    
    return original_img_np, processed_img_np, mix_img_np

def test_wavelet_transform(image_path):
    """测试单独的小波变换效果"""
    # 加载图像
    img, img_tensor = load_and_preprocess_image(image_path)
    
    # 初始化小波变换模块
    wavelet_module = WaveletTransform2D().to('cpu')
    
    # 前向传播
    with torch.no_grad():
        LL, LH, HL, HH = wavelet_module(img_tensor)
    
    print(f"LL形状: {LL.shape}")
    print(f"LH形状: {LH.shape}")
    print(f"HL形状: {HL.shape}")
    print(f"HH形状: {HH.shape}")
    
    LH_HL=torch.cat([LH,HL],dim=1)
    
    # 可视化小波分量
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('小波变换后结果', fontsize=16)
    
    components = [LL, LH, HL, HH, LH_HL]
    titles = ['LL (近似)', 'LH (水平边缘)', 'HL (垂直边缘)', 'HH (对角线)', 'LH_HL (组合)']
    
    for idx, (comp, title) in enumerate(zip(components, titles)):
        row = idx // 3
        col = idx % 3
        
        # 取三个通道的平均值
        comp = comp.squeeze(0).cpu().numpy()
        comp_mean = np.mean(comp, axis=0)
        
        axes[row, col].imshow(comp_mean, cmap='gray')
        axes[row, col].set_title(title, fontsize=12)
        axes[row, col].axis('off')
    
    # 隐藏多余的子图
    if len(components) < 6:
        axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()


def test_gradient_module(image_path):
    """测试梯度提取模块"""
    # 加载图像
    _, img_tensor = load_and_preprocess_image(image_path)
    
    # 初始化梯度模块
    gradient_module = Get_gradient_nopadding().to('cpu')
    
    # 前向传播
    with torch.no_grad():
        gradient_map = gradient_module(img_tensor)
    
    print(f"梯度图图像: {gradient_map.shape}")
    
    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(img_tensor.squeeze(0).permute(1, 2, 0))
    axes[0].set_title('原始图像')
    axes[0].axis('off')
    
    # 显示梯度幅值
    gradient_magnitude = torch.mean(gradient_map, dim=1, keepdim=True)
    gradient_magnitude_np = gradient_magnitude.squeeze(0).squeeze(0).cpu().numpy()
    
    axes[1].imshow(gradient_magnitude_np, cmap='hot')
    axes[1].set_title('梯度幅值')
    axes[1].axis('off')
    
    # 显示三个通道的梯度
    gradient_channels = gradient_map.squeeze(0).cpu().numpy()
    for c in range(3):
        if c < 3:
            axes[2].imshow(gradient_channels[c], cmap='gray')
            axes[2].set_title(f'通道{c}梯度')
            axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

# def test_sequence_of_wpmd_and_sharpen(image_path):
#     """测试WPMD模块和锐化操作的组合效果"""
#     # 加载图像
#     img, img_tensor = load_and_preprocess_image(image_path)
    
#     #先wpmd再sharpen
#     _,processed_np,mix_np=visualize_wpmd_effects(image_path)

#     # wpmd_output_pil = transforms.ToPILImage()(mix_np.squeeze(0))
#     if mix_np.ndim == 3:  # 彩色图像 (H, W, C)
#         pil_img = Image.fromarray(mix_np, mode='RGB')  # 或 'RGBA' 如果有 4 通道
#     else:  # 灰度图像 (H, W)
#         pil_img = Image.fromarray(mix_np, mode='L')
#     sharpened_pil_img = pil_img.filter(ImageFilter.SHARPEN)
    
#     fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
#     axes[0].imshow(img_tensor.squeeze(0).permute(1, 2, 0))
#     axes[0].set_title('原始图像')
#     axes[0].axis('off')

#     axes[1].imshow(sharpened_pil_img)
#     axes[1].set_title('WPMD后锐化')
#     axes[1].axis('off')
    
#     plt.tight_layout()
#     plt.show()

# 主测试函数
if __name__ == "__main__":
    # setup_chinese_font()
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 测试图像路径
    # image_path = "D:\毕设\IRSAM-mine\datasets\IRSTD-1k\IRSTD1k_Img\XDU0.png"  # 请替换为您的测试图像路径s
    # image_path="assets\example.png"  # 请替换为您的测试图像路径
    image_path=r"D:\毕设\IRSAM-mine\datasets\NUDT-SIRST1\images\000002.png"
    
    try:
        # 测试1: WPMD整体效果
        print("=" * 50)
        print("测试1: WPMD模块整体效果")
        print("=" * 50)
        original, processed, _ = visualize_wpmd_effects(image_path, device)
        
        # 测试2: 小波变换效果
        print("\n" + "=" * 50)
        print("测试2: 小波分解效果")
        print("=" * 50)
        test_wavelet_transform(image_path)
        
        # # 测试3: 梯度提取效果
        # print("\n" + "=" * 50)
        # print("测试3: 梯度提取效果")
        # print("=" * 50)
        # test_gradient_module(image_path)
        
        print("\n" + "=" * 50)
        print("测试4: PIL锐化方法对比")
        print("=" * 50)
        sharpen_with_pil(image_path)
        
        # print("\n" + "=" * 50)
        # print("测试5: WPMD与锐化组合效果")
        # print("=" * 50)
        # test_sequence_of_wpmd_and_sharpen(image_path)
        
    except FileNotFoundError:
        print(f"错误: 找不到图像文件 {image_path}")
        print("请准备一个测试图像，或将路径修改为正确的图像路径")
        
        # 创建示例图像进行测试
        print("\n创建示例图像进行测试...")
        test_image = np.random.rand(256, 256, 3)
        plt.imshow(test_image)
        plt.title('随机测试图像')
        plt.axis('off')
        plt.show()