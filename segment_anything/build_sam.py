# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from functools import partial
from pathlib import Path
import urllib.request
import torch

from .modeling import (
    ImageEncoderViT,
    MaskDecoder,
    PromptEncoder,
    Sam,
    TwoWayTransformer,
    TinyViT,
)


def build_sam_vit_h(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
    )


build_sam = build_sam_vit_h


def build_sam_vit_l(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
    )


def build_sam_vit_b(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )

def build_sam_vit_t(checkpoint=None):
    """
    构建MobileSAM模型（Tiny版本）
    
    参数:
        checkpoint (str, optional): 预训练权重的文件路径
    
    返回:
        mobile_sam: 构建好的MobileSAM模型实例
    """
    # 基本配置参数
    prompt_embed_dim = 256      # 提示嵌入的维度
    image_size = 1024           # 输入图像的尺寸
    vit_patch_size = 16         # Vision Transformer的patch大小
    # 计算图像嵌入的尺寸（图像大小除以patch大小）
    image_embedding_size = image_size // vit_patch_size

    # 构建MobileSAM模型
    mobile_sam = Sam(
        # 图像编码器：使用轻量级的TinyViT
        image_encoder=TinyViT(
            img_size=1024,              # 输入图像大小
            in_chans=3,                 # 输入通道数（RGB）
            num_classes=1000,           # 分类数量
            # 不同阶段的嵌入维度
            embed_dims=[64, 128, 160, 320],
            # 每个阶段的深度（层数）
            depths=[2, 2, 6, 2],
            # 每个阶段的注意力头数
            num_heads=[2, 4, 5, 10],
            # 窗口大小配置
            window_sizes=[7, 7, 14, 7],
            mlp_ratio=4.,              # MLP扩展比率
            drop_rate=0.,              # Dropout率
            drop_path_rate=0.0,        # DropPath率
            use_checkpoint=False,      # 是否使用检查点
            mbconv_expand_ratio=4.0,   # MBConv扩展比率
            local_conv_size=3,         # 局部卷积核大小
            layer_lr_decay=0.8         # 层级学习率衰减
        ),
        
        # 提示编码器：处理各种类型的提示（点、框等）
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,                    # 提示嵌入维度
            image_embedding_size=(image_embedding_size, image_embedding_size),  # 图像嵌入大小
            input_image_size=(image_size, image_size),     # 输入图像大小
            mask_in_chans=16,                              # 掩码输入通道数
        ),
        
        # 掩码解码器：生成分割掩码并预测质量
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,      # 多掩码输出数量
            transformer=TwoWayTransformer(  # 双向Transformer用于特征融合
                depth=2,                   # Transformer深度
                embedding_dim=prompt_embed_dim,  # 嵌入维度
                mlp_dim=2048,             # MLP维度
                num_heads=8,              # 注意力头数量
            ),
            transformer_dim=prompt_embed_dim,  # Transformer维度
            iou_head_depth=3,             # IoU预测头深度
            iou_head_hidden_dim=256,      # IoU预测头隐藏层维度
        ),
        
        # 图像标准化参数（来自ImageNet数据集）
        pixel_mean=[123.675, 116.28, 103.53],  # RGB均值
        pixel_std=[58.395, 57.12, 57.375],     # RGB标准差
    )

    # 将模型设置为评估模式
    mobile_sam.eval()
    
    # 如果提供了预训练权重，则加载
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)  # 加载权重文件
        mobile_sam.load_state_dict(state_dict)  # 将权重加载到模型中
    
    return mobile_sam

sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
    "vit_t": build_sam_vit_t,
}


def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    checkpoint = Path(checkpoint)
    if checkpoint.name == "sam_vit_b_01ec64.pth" and not checkpoint.exists():
        cmd = input("Download sam_vit_b_01ec64.pth from facebook AI? [y]/n: ")
        if len(cmd) == 0 or cmd.lower() == "y":
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            print("Downloading SAM ViT-B checkpoint...")
            urllib.request.urlretrieve(
                "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
                checkpoint,
            )
            print(checkpoint.name, " is downloaded!")
    elif checkpoint.name == "sam_vit_h_4b8939.pth" and not checkpoint.exists():
        cmd = input("Download sam_vit_h_4b8939.pth from facebook AI? [y]/n: ")
        if len(cmd) == 0 or cmd.lower() == "y":
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            print("Downloading SAM ViT-H checkpoint...")
            urllib.request.urlretrieve(
                "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
                checkpoint,
            )
            print(checkpoint.name, " is downloaded!")
    elif checkpoint.name == "sam_vit_l_0b3195.pth" and not checkpoint.exists():
        cmd = input("Download sam_vit_l_0b3195.pth from facebook AI? [y]/n: ")
        if len(cmd) == 0 or cmd.lower() == "y":
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            print("Downloading SAM ViT-L checkpoint...")
            urllib.request.urlretrieve(
                "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
                checkpoint,
            )
            print(checkpoint.name, " is downloaded!")

    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f, map_location=torch.device('cpu'))
        sam.load_state_dict(state_dict, strict=False)
    return sam
