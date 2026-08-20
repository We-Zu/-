import os
import numpy as np
import torch

def save_weights_npy(model, save_dir, prefix='model'):
    """保存模型所有参数为 .pth 文件（单个文件）"""
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, f'{prefix}.pth'))

def load_weights_npy(model, load_dir, prefix='model'):
    """从 .pth 文件加载参数到模型"""
    file_path = os.path.join(load_dir, f'{prefix}.pth')
    if os.path.exists(file_path):
        state_dict = torch.load(file_path, map_location='cpu')
        model.load_state_dict(state_dict)
    else:
        print(f"Warning: {file_path} not found, skipping.")
    return model