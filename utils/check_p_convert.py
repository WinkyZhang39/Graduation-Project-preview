import torch

# 加载你的.pth文件
checkpoint = torch.load("weights/irsam_model_best.pth", map_location='cpu')

print(f"原始检查点类型: {type(checkpoint)}")

# 如果是完整模型，提取state_dict
if isinstance(checkpoint, torch.nn.Module):
    print("检测到完整模型，提取state_dict...")
    state_dict = checkpoint.state_dict()
elif isinstance(checkpoint, dict):
    print(f"检测到字典，键: {checkpoint.keys()}")
    
    # 尝试提取state_dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        # 如果已经是state_dict
        state_dict = checkpoint
else:
    print(f"未知格式: {type(checkpoint)}")
    exit(1)

# 保存为.pt格式
print(f"保存state_dict，包含{len(state_dict)}个参数")
torch.save(state_dict, "weights/irsam_model_converted.pt")
print("转换完成！保存为 weights/irsam_model_converted.pt")

# 也可以保存为完整的模型（如果需要）
# from segment_anything.build_sam import build_sam_vit_t
# model = build_sam_vit_t()
# model.load_state_dict(state_dict, strict=False)
# torch.save(model, "weights/irsam_model_full.pth")