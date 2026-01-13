# Copyright by HQ-SAM team
# All rights reserved.

# data loader
from __future__ import print_function, division

import cv2
import numpy as np
import random
from copy import deepcopy
from skimage import io
import os
from glob import glob
from PIL import Image, ImageFilter
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms, utils
from torchvision.transforms.functional import normalize
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler


# --------------------- dataloader online ---------------------####

# def get_im_gt_name_dict(datasets, flag='valid'):
#     print("------------------------------", flag, "--------------------------------")
#     name_im_gt_list = []

#     for i in range(len(datasets)):
#         print("--->>>", flag, " dataset ", i, "/", len(datasets), " ", datasets[i]["name"], "<<<---")
#         tmp_im_list, tmp_gt_list = [], []
#         tmp_im_list = glob(datasets[i]["im_dir"] + os.sep + '*' + datasets[i]["im_ext"])
#         print('-im-', datasets[i]["name"], datasets[i]["im_dir"], ': ', len(tmp_im_list))

#         if datasets[i]["gt_dir"] == "":
#             print('-gt-', datasets[i]["name"], datasets[i]["gt_dir"], ': ', 'No Ground Truth Found')
#             tmp_gt_list = []
#         else:
#             tmp_gt_list = [
#                 datasets[i]["gt_dir"] + os.sep + x.split(os.sep)[-1].split(datasets[i]["im_ext"])[0] + datasets[i][
#                     "gt_ext"] for x in tmp_im_list]
#             print('-gt-', datasets[i]["name"], datasets[i]["gt_dir"], ': ', len(tmp_gt_list))

#         name_im_gt_list.append({"dataset_name": datasets[i]["name"],
#                                 "im_path": tmp_im_list,
#                                 "gt_path": tmp_gt_list,
#                                 "im_ext": datasets[i]["im_ext"],
#                                 "gt_ext": datasets[i]["gt_ext"]})

#     return name_im_gt_list
def get_im_gt_name_dict(datasets, flag='valid'):
    print("------------------------------", flag, "--------------------------------")
    #[{'dataset_name': 'NUDT-SIRST1', 'im_path': ['datasets\\NUDT-SIRST1\\images\\000001.png',
    name_im_gt_list = []

    for i in range(len(datasets)):
        # print("--->>>", flag, " dataset ", i, "/", len(datasets), " ", datasets[i]["name"], "<<<---")
        
        # 检查是否已经提供了文件路径列表
        if "im_path" in datasets[i] and isinstance(datasets[i]["im_path"], list):
            # 如果已经提供了文件路径列表，直接使用
            tmp_im_list = datasets[i]["im_path"]
            if "gt_path" in datasets[i] and isinstance(datasets[i]["gt_path"], list):
                tmp_gt_list = datasets[i]["gt_path"]
            else:
                tmp_gt_list = []
        else:
            # 否则按照原来的方式从目录读取
            tmp_im_list = glob(datasets[i]["im_dir"] + os.sep + '*' + datasets[i]["im_ext"])
            print('-im-', datasets[i]["name"], datasets[i]["im_dir"], ': ', len(tmp_im_list))

            if datasets[i]["gt_dir"] == "":
                print('-gt-', datasets[i]["name"], datasets[i]["gt_dir"], ': ', 'No Ground Truth Found')
                tmp_gt_list = []
            else:
                tmp_gt_list = [
                    datasets[i]["gt_dir"] + os.sep + x.split(os.sep)[-1].split(datasets[i]["im_ext"])[0] + datasets[i][
                        "gt_ext"] for x in tmp_im_list]
                print('-gt-', datasets[i]["name"], datasets[i]["gt_dir"], ': ', len(tmp_gt_list))
        
        print('-im-', datasets[i]["name"], ': ', len(tmp_im_list))
        print('-gt-', datasets[i]["name"], ': ', len(tmp_gt_list))

        name_im_gt_list.append({"dataset_name": datasets[i]["name"],
                                "im_path": tmp_im_list,
                                "gt_path": tmp_gt_list,
                                "im_ext": datasets[i]["im_ext"],
                                "gt_ext": datasets[i]["gt_ext"]})

    return name_im_gt_list
# train_config: [{'dataset_name': 'IRSTD-1k_train', 
# 'im_path': 'datasets\\IRSTD-1k/IRSTD1k_Img', 
# 'gt_path': 'datasets\\IRSTD-1k/IRSTD1k_Label', 
# 'im_ext': '.png', 'gt_ext': '.png'}]
def create_dataloaders(name_im_gt_list, my_transforms=[], batch_size=1, training=False, visualize_augmented=False):
    gos_dataloaders = []
    gos_datasets = []
    
    if len(name_im_gt_list) == 0:
        return gos_dataloaders, gos_datasets

    num_workers_ = 1
    if batch_size > 1:
        num_workers_ = 2
    if batch_size > 4:
        num_workers_ = 4
    if batch_size > 8:
        num_workers_ = 8

    if training:
        for i in range(len(name_im_gt_list)):
            gos_dataset = OnlineDataset([name_im_gt_list[i]], 
                                        transform=transforms.Compose(my_transforms), 
                                        eval_ori_resolution=False,
                                        show_samples=False)
            gos_datasets.append(gos_dataset)
            
            # 可视化数据增强后的样本
            if visualize_augmented and i == 0:  # 只显示第一个数据集
                print(f"\n=== after visualize '{name_im_gt_list[i]['dataset_name']}' ===")
                gos_dataset.visualize_augmented_samples(num_samples=5)

        gos_dataset = ConcatDataset(gos_datasets)
        dataloader = DataLoader(gos_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers_)

        gos_dataloaders = [dataloader]
        gos_datasets = gos_dataset
    else:
        # 对于验证/测试，我们使用 eval_ori_resolution=True
        for i in range(len(name_im_gt_list)):
            gos_dataset = OnlineDataset([name_im_gt_list[i]], 
                                        transform=transforms.Compose(my_transforms), 
                                        eval_ori_resolution=True)
            dataloader = DataLoader(gos_dataset, batch_size=batch_size, num_workers=num_workers_)

            gos_dataloaders.append(dataloader)
            gos_datasets.append(gos_dataset)

    return gos_dataloaders, gos_datasets



class RandomHFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        imidx, image, label, edge, shape = sample['imidx'], sample['image'], sample['label'], sample['edge'], sample['shape']

        # random horizontal flip
        if random.random() >= self.prob:
            image = torch.flip(image, dims=[2])
            label = torch.flip(label, dims=[2])
            edge = torch.flip(edge, dims=[2])

        return {'imidx': imidx, 'image': image, 'label': label, 'edge': edge,  'shape': shape}


class Resize(object):
    def __init__(self, size=[320, 320]):
        self.size = size

    def __call__(self, sample):
        imidx, image, label, edge, shape = sample['imidx'], sample['image'], sample['label'], sample['edge'], sample['shape']

        image = torch.squeeze(F.interpolate(torch.unsqueeze(image, 0), self.size, mode='bilinear'), dim=0)
        label = torch.squeeze(F.interpolate(torch.unsqueeze(label, 0), self.size, mode='bilinear'), dim=0)
        edge = torch.squeeze(F.interpolate(torch.unsqueeze(edge, 0), self.size, mode='bilinear'), dim=0)

        return {'imidx': imidx, 'image': image, 'label': label, 'edge':edge, 'shape': torch.tensor(self.size)}


class RandomCrop(object):
    def __init__(self, size=[288, 288]):
        self.size = size

    def __call__(self, sample):
        imidx, image, label, edge, shape = sample['imidx'], sample['image'], sample['label'], sample['edge'], sample['shape']

        h, w = image.shape[1:]
        new_h, new_w = self.size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[:, top:top + new_h, left:left + new_w]
        label = label[:, top:top + new_h, left:left + new_w]
        edge = edge[:, top:top + new_h, left:left + new_w]

        return {'imidx': imidx, 'image': image, 'label': label, 'edge': edge, 'shape': torch.tensor(self.size)}

class Sharpen(object):
    def __init__(self, method="sharpen", **kwargs):
        """
        初始化锐化类
        Args:
            method (str): 锐化方法，可选 "sharpen", "unsharpmask", "custom_kernel"
            **kwargs: 方法参数（如 radius, percent, threshold 或 kernel）
        """
        self.method = method
        self.kwargs = kwargs

    def __call__(self, img):
        """
        对输入图像应用锐化
        Args:
            img (PIL.Image): 输入图像
        Returns:
            PIL.Image: 锐化后的图像
        """
        imidx, image, label, edge, shape = img['imidx'], img['image'], img['label'], img['edge'], img['shape']
        if not isinstance(image, Image.Image):
            raise TypeError("输入必须是 PIL.Image 对象")

        if self.method == "sharpen":
            image_sharpen = image.filter(ImageFilter.SHARPEN)
        
        elif self.method == "unsharpmask":
            radius = self.kwargs.get("radius", 2)
            percent = self.kwargs.get("percent", 150)
            threshold = self.kwargs.get("threshold", 3)
            image_sharpen = image.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))
        
        elif self.method == "custom_kernel":
            kernel = self.kwargs.get("kernel", [0, -1, 0, -1, 5, -1, 0, -1, 0])
            scale = self.kwargs.get("scale", 1)
            kernel_sharpen = ImageFilter.Kernel((3, 3), kernel, scale=scale)
            image_sharpen = image.filter(kernel_sharpen)
        
        else:
            raise ValueError(f"不支持的锐化方法: {self.method}")

        return {'imidx': imidx, 'image': image_sharpen, 'label': label, 'edge': edge, 'shape': shape}

class Normalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        imidx, image, label, edge, shape = sample['imidx'], sample['image'], sample['label'], sample['edge'], sample['shape']
        image = normalize(image, self.mean, self.std)

        return {'imidx': imidx, 'image': image, 'label': label, 'edge': edge, 'shape': shape}


class LargeScaleJitter(object):
    """
        implementation of large scale jitter from copy_paste
        https://github.com/gaopengcuhk/Pretrained-Pix2Seq/blob/7d908d499212bfabd33aeaa838778a6bfb7b84cc/datasets/transforms.py 
    """

    def __init__(self, output_size=512, aug_scale_min=0.8, aug_scale_max=1.5):
        self.desired_size = torch.tensor(output_size)
        self.aug_scale_min = aug_scale_min
        self.aug_scale_max = aug_scale_max

    def pad_target(self, padding, target):
        target = target.copy()
        if "masks" in target:
            target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[1], 0, padding[0]))
        return target

    def __call__(self, sample):
        imidx, image, label, edge, image_size = sample['imidx'], sample['image'], sample['label'], sample['edge'], sample['shape']

        # resize keep ratio
        out_desired_size = (self.desired_size * image_size / max(image_size)).round().int()

        random_scale = torch.rand(1) * (self.aug_scale_max - self.aug_scale_min) + self.aug_scale_min
        scaled_size = (random_scale * self.desired_size).round()

        scale = torch.minimum(scaled_size / image_size[0], scaled_size / image_size[1])
        scaled_size = (image_size * scale).round().long()

        scaled_image = torch.squeeze(F.interpolate(torch.unsqueeze(image, 0), scaled_size.tolist(), mode='bilinear'),
                                     dim=0)
        scaled_label = torch.squeeze(F.interpolate(torch.unsqueeze(label, 0), scaled_size.tolist(), mode='bilinear'),
                                     dim=0)
        scaled_edge = torch.squeeze(F.interpolate(torch.unsqueeze(edge, 0), scaled_size.tolist(), mode='bilinear'),
                                     dim=0)

        # random crop
        crop_size = (min(self.desired_size, scaled_size[0]), min(self.desired_size, scaled_size[1]))

        margin_h = max(scaled_size[0] - crop_size[0], 0).item()
        margin_w = max(scaled_size[1] - crop_size[1], 0).item()
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + crop_size[0].item()
        crop_x1, crop_x2 = offset_w, offset_w + crop_size[1].item()

        scaled_image = scaled_image[:, crop_y1:crop_y2, crop_x1:crop_x2]
        scaled_label = scaled_label[:, crop_y1:crop_y2, crop_x1:crop_x2]
        scaled_edge = scaled_edge[:, crop_y1:crop_y2, crop_x1:crop_x2]

        # pad
        padding_h = max(self.desired_size - scaled_image.size(1), 0).item()
        padding_w = max(self.desired_size - scaled_image.size(2), 0).item()
        image = F.pad(scaled_image, [0, padding_w, 0, padding_h], value=128)
        label = F.pad(scaled_label, [0, padding_w, 0, padding_h], value=0)
        edge = F.pad(scaled_edge, [0, padding_w, 0, padding_h], value=0)

        return {'imidx': imidx, 'image': image, 'label': label, 'edge': edge, 'shape': torch.tensor(image.shape[-2:])}


# class OnlineDataset(Dataset):
#     def __init__(self, name_im_gt_list, transform=None, eval_ori_resolution=False):

#         self.transform = transform
#         self.dataset = {}
#         # combine different datasets into one
#         dataset_names = []
#         dt_name_list = []  # dataset name per image
#         im_name_list = []  # image name
#         im_path_list = []  # im path
#         gt_path_list = []  # gt path
#         im_ext_list = []  # im ext
#         gt_ext_list = []  # gt ext
#         for i in range(0, len(name_im_gt_list)):
#             # print(len(name_im_gt_list))
#             # print("dataloader:227:++++++++++++++name_im_gt_list[i]: ")
#             # print(name_im_gt_list[i])#irstd-1k_train
#             # {'dataset_name': 'IRSTD-1k_train', 
#             # 'im_path': 'datasets\\IRSTD-1k\\IRSTD1k_Img', 
#             # 'gt_path': 'datasets\\IRSTD-1k\\IRSTD1k_Label', 
#             # 'im_ext': '.png',
#             # 'gt_ext': '.png'}
#             dataset_names.append(name_im_gt_list[i]["dataset_name"])
#             # dataset name repeated based on the number of images in this dataset
#             dt_name_list.extend([name_im_gt_list[i]["dataset_name"] for x in name_im_gt_list[i]["im_path"]])
#             im_name_list.extend(
#                 [x.split(os.sep)[-1].split(name_im_gt_list[i]["im_ext"])[0] for x in name_im_gt_list[i]["im_path"]])
#             # print(name_im_gt_list[i]["im_path"])
#             im_path_list.extend(name_im_gt_list[i]["im_path"])
#             gt_path_list.extend(name_im_gt_list[i]["gt_path"])
#             im_ext_list.extend([name_im_gt_list[i]["im_ext"] for x in name_im_gt_list[i]["im_path"]])
#             gt_ext_list.extend([name_im_gt_list[i]["gt_ext"] for x in name_im_gt_list[i]["gt_path"]])

#         self.dataset["data_name"] = dt_name_list
#         self.dataset["im_name"] = im_name_list
#         self.dataset["im_path"] = im_path_list
#         # print("dataloader:247:im_path_list:")
#         # print(im_path_list)
        
#         self.dataset["ori_im_path"] = deepcopy(im_path_list)
#         self.dataset["gt_path"] = gt_path_list
#         self.dataset["ori_gt_path"] = deepcopy(gt_path_list)
#         self.dataset["im_ext"] = im_ext_list
#         self.dataset["gt_ext"] = gt_ext_list

#         self.eval_ori_resolution = eval_ori_resolution
class OnlineDataset(Dataset):
    def __init__(self, name_im_gt_list, transform=None, eval_ori_resolution=False, show_samples=False):

        self.transform = transform
        self.dataset = {}
        # combine different datasets into one
        dataset_names = []
        dt_name_list = []  # dataset name per image
        im_name_list = []  # image name
        im_path_list = []  # im path
        gt_path_list = []  # gt path
        im_ext_list = []  # im ext
        gt_ext_list = []  # gt ext
        
        for i in range(0, len(name_im_gt_list)):
            dataset_names.append(name_im_gt_list[i]["dataset_name"])
            # dataset name repeated based on the number of images in this dataset
            dt_name_list.extend([name_im_gt_list[i]["dataset_name"] for x in name_im_gt_list[i]["im_path"]])
            im_name_list.extend(
                [x.split(os.sep)[-1].split(name_im_gt_list[i]["im_ext"])[0] for x in name_im_gt_list[i]["im_path"]])
            im_path_list.extend(name_im_gt_list[i]["im_path"])
            gt_path_list.extend(name_im_gt_list[i]["gt_path"])
            im_ext_list.extend([name_im_gt_list[i]["im_ext"] for x in name_im_gt_list[i]["im_path"]])
            gt_ext_list.extend([name_im_gt_list[i]["gt_ext"] for x in name_im_gt_list[i]["gt_path"]])

        self.dataset["data_name"] = dt_name_list
        self.dataset["im_name"] = im_name_list
        self.dataset["im_path"] = im_path_list
        self.dataset["ori_im_path"] = deepcopy(im_path_list)
        self.dataset["gt_path"] = gt_path_list
        self.dataset["ori_gt_path"] = deepcopy(gt_path_list)
        self.dataset["im_ext"] = im_ext_list
        self.dataset["gt_ext"] = gt_ext_list
        
        self.setup_chinese_font()
        
        self.eval_ori_resolution = eval_ori_resolution
        
        #这里显示的是完全没有经过图像增强的图像，因为直接读取的路径显示的原始图像
        # 显示前10个样本的路径信息
        if show_samples:
            print("\n=== 显示前10个训练样本的路径信息，这是没有进行数据增强及小波变换之前的 ===")
            for i in range(min(10, len(im_path_list))):
                print(f"\n样本 {i+1}:")
                print(f"  数据集: {dt_name_list[i]}")
                print(f"  图像路径: {im_path_list[i]}")
                print(f"  标签路径: {gt_path_list[i] if i < len(gt_path_list) else '无标签'}")
            
            # 可选：显示图像内容（需要安装matplotlib）
            try:
                import matplotlib.pyplot as plt
                print("\n=== 显示前10个训练样本的图像和标签预览 ===")
                # 创建一个大的子图
                fig, axes = plt.subplots(2, 5, figsize=(20, 8))
                fig.suptitle('top-10 traing sample preview', fontsize=16)
                for i in range(min(10, len(im_path_list))):
                    row = 0  # 图像行
                    col = i % 5
                    # 读取并显示图像
                    try:
                        im = io.imread(im_path_list[i])
                        if len(im.shape) < 3:
                            im = im[:, :, np.newaxis]
                        if im.shape[2] == 1:
                            im = np.repeat(im, 3, axis=2)
                        
                        axes[row, col].imshow(im)
                        axes[row, col].set_title(f'样本 {i+1}')
                        axes[row, col].axis('off')
                        
                        # 读取并显示标签
                        if i < len(gt_path_list) and os.path.exists(gt_path_list[i]):
                            gt = io.imread(gt_path_list[i])
                            if len(gt.shape) > 2:
                                gt = gt[:, :, 0]
                            axes[1, col].imshow(gt, cmap='gray')
                            axes[1, col].set_title(f'标签 {i+1}')
                            axes[1, col].axis('off')
                        else:
                            axes[1, col].text(0.5, 0.5, '无标签', 
                                             horizontalalignment='center',
                                             verticalalignment='center',
                                             transform=axes[1, col].transAxes)
                            axes[1, col].axis('off')
                            
                    except Exception as e:
                        print(f"加载样本 {i+1} 失败: {e}")
                        axes[row, col].text(0.5, 0.5, '加载失败', 
                                           horizontalalignment='center',
                                           verticalalignment='center',
                                           transform=axes[row, col].transAxes)
                        axes[row, col].axis('off')
                
                plt.tight_layout()
                plt.show()
                
            except ImportError:
                print("\n注意: 需要安装matplotlib来显示图像预览")
                print("请运行: pip install matplotlib")
            except Exception as e:
                print(f"\n显示图像时出错: {e}")
    
    def __len__(self):
        return len(self.dataset["im_path"])

    def __getitem__(self, idx):
        im_path = self.dataset["im_path"][idx]
        gt_path = self.dataset["gt_path"][idx]
        
        im = io.imread(im_path)
        gt = io.imread(gt_path)

        if len(gt.shape) > 2:
            gt = gt[:, :, 0]
        if len(im.shape) < 3:
            im = im[:, :, np.newaxis]
        if im.shape[2] == 1:
            im = np.repeat(im, 3, axis=2)

        edge = cv2.Canny(gt, 100, 200)
        im = torch.tensor(im.copy(), dtype=torch.float32)
        im = torch.transpose(torch.transpose(im, 1, 2), 0, 1)
        gt = torch.unsqueeze(torch.tensor(gt, dtype=torch.float32), 0)
        edge = torch.unsqueeze(torch.tensor(edge, dtype=torch.float32), 0)

        sample = {
            "imidx": torch.from_numpy(np.array(idx)),
            "image": im,
            "label": gt,
            "edge": edge,
            "shape": torch.tensor(im.shape[-2:]),
            "path": self.dataset["im_path"][idx]
        }

        if self.transform:
            sample = self.transform(sample)

        if self.eval_ori_resolution:
            sample["ori_label"] = gt.type(torch.uint8)  # NOTE for evaluation only. And no flip here
            sample['ori_im_path'] = self.dataset["im_path"][idx]
            sample['ori_gt_path'] = self.dataset["gt_path"][idx]

        return sample
    
    
    def visualize_augmented_samples(self, num_samples=5, random_seed=42):
        """可视化数据增强后的样本"""
        import matplotlib.pyplot as plt
        import random
        
        # 设置随机种子以确保可重现性
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
        
        # 随机选择样本索引
        indices = random.sample(range(len(self)), min(num_samples, len(self)))
        
        print(f"\n=== 显示数据增强后的 {len(indices)} 个样本 ===")
        
        # 创建子图
        fig, axes = plt.subplots(2, num_samples, figsize=(num_samples*4, 8))
        if num_samples == 1:
            axes = axes.reshape(2, 1)
        
        fig.suptitle('数据增强后样本 (上: 图像, 下: 标签)', fontsize=16)
        
        for i, idx in enumerate(indices):
            # 获取原始样本（不应用变换）
            im_path = self.dataset["im_path"][idx]
            gt_path = self.dataset["gt_path"][idx]
            
            im = io.imread(im_path)
            gt = io.imread(gt_path)
            
            if len(gt.shape) > 2:
                gt = gt[:, :, 0]
            if len(im.shape) < 3:
                im = im[:, :, np.newaxis]
            if im.shape[2] == 1:
                im = np.repeat(im, 3, axis=2)
            
            edge = cv2.Canny(gt, 100, 200)
            im_tensor = torch.tensor(im.copy(), dtype=torch.float32)
            im_tensor = torch.transpose(torch.transpose(im_tensor, 1, 2), 0, 1)
            gt_tensor = torch.unsqueeze(torch.tensor(gt, dtype=torch.float32), 0)
            edge_tensor = torch.unsqueeze(torch.tensor(edge, dtype=torch.float32), 0)
            
            sample = {
                "imidx": torch.from_numpy(np.array(idx)),
                "image": im_tensor,
                "label": gt_tensor,
                "edge": edge_tensor,
                "shape": torch.tensor(im_tensor.shape[-2:]),
                "path": im_path
            }
            
            # 应用数据增强
            if self.transform:
                augmented_sample = self.transform(sample)
            else:
                augmented_sample = sample
            
            # 显示增强后的图像
            img_aug = augmented_sample['image']
            label_aug = augmented_sample['label']
            
            # 转换图像为numpy格式
            if img_aug.dim() == 3:
                img_np = img_aug.permute(1, 2, 0).cpu().numpy()
            else:
                img_np = img_aug.squeeze().cpu().numpy()
            
            # 关键修复：确保图像值在 [0, 1] 范围内
            # 检查是否应用了 Normalize 变换
            normalize_applied = False
            if hasattr(self.transform, 'transforms'):
                for t in self.transform.transforms:
                    if isinstance(t, Normalize):
                        normalize_applied = True
                        # 反标准化
                        mean = torch.tensor(t.mean).view(3, 1, 1)
                        std = torch.tensor(t.std).view(3, 1, 1)
                        img_aug_denorm = img_aug * std + mean
                        img_np = img_aug_denorm.permute(1, 2, 0).cpu().numpy()
                        break
            
            # 如果没有应用 Normalize，且图像值范围在 [0, 255]，则归一化到 [0, 1]
            if not normalize_applied:
                if img_np.max() > 1.0:
                    img_np = img_np / 255.0
            
            # 确保图像值在 [0, 1] 范围内
            img_np = np.clip(img_np, 0, 1)
            
            # 转换标签
            if label_aug.dim() == 3:
                label_np = label_aug.squeeze().cpu().numpy()
            else:
                label_np = label_aug.cpu().numpy()
            
            # 在显示前添加调试输出
            print(f"img_np shape: {img_np.shape}")
            print(f"img_np min/max: {img_np.min()}, {img_np.max()}")
            print(f"img_np dtype: {img_np.dtype}")
            
            # 显示图像
            axes[0, i].imshow(img_np)
            axes[0, i].set_title(f'样本 {i+1}\n{os.path.basename(im_path)}')
            axes[0, i].axis('off')
            
            # 显示标签
            axes[1, i].imshow(label_np, cmap='gray')
            axes[1, i].set_title(f'标签 {i+1}')
            axes[1, i].axis('off')
            
            print(f"\n样本 {i+1}:")
            print(f"  原始图像大小: {im.shape}")
            print(f"  增强后图像大小: {img_aug.shape}")
            print(f"  图像值范围: [{img_np.min():.3f}, {img_np.max():.3f}]")
            print(f"  标签范围: [{label_np.min():.3f}, {label_np.max():.3f}]")
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def setup_chinese_font(self):
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
            