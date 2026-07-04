"""
torchvision兼容性补丁
如果torchvision不可用，使用PIL和numpy替代
"""

try:
    from torchvision.utils import save_image, make_grid
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    
    import torch
    import numpy as np
    from PIL import Image
    
    def save_image(tensor, filename, nrow=8, padding=2, normalize=False, 
                   range=None, scale_each=False, pad_value=0, format=None):
        """
        替代torchvision.utils.save_image的功能
        使用PIL保存图像
        """
        # 处理tensor
        if isinstance(tensor, torch.Tensor):
            # 转换为numpy
            if tensor.is_cuda:
                tensor = tensor.cpu()
            tensor = tensor.detach().numpy()
        
        # 处理维度
        if tensor.ndim == 4:
            # (B, C, H, W) -> 合并为网格
            batch_size = tensor.shape[0]
            nrow = min(nrow, batch_size)
            ncol = (batch_size + nrow - 1) // nrow
            
            # 计算输出尺寸
            C, H, W = tensor.shape[1], tensor.shape[2], tensor.shape[3]
            grid_h = ncol * H + (ncol - 1) * padding
            grid_w = nrow * W + (nrow - 1) * padding
            
            # 创建网格
            grid = np.zeros((C, grid_h, grid_w), dtype=tensor.dtype)
            
            for i in range(batch_size):
                row = i // nrow
                col = i % nrow
                h_start = row * (H + padding)
                w_start = col * (W + padding)
                grid[:, h_start:h_start+H, w_start:w_start+W] = tensor[i]
            
            tensor = grid
        
        # 归一化
        if normalize:
            if range is not None:
                min_val, max_val = range
            else:
                min_val = tensor.min()
                max_val = tensor.max()
            if max_val > min_val:
                tensor = (tensor - min_val) / (max_val - min_val)
        
        # 转换为PIL图像格式 (H, W, C) 或 (H, W)
        if tensor.ndim == 3:
            if tensor.shape[0] == 1:
                # 单通道灰度图
                tensor = tensor[0]
            elif tensor.shape[0] == 3:
                # RGB图像，转换为(H, W, C)
                tensor = np.transpose(tensor, (1, 2, 0))
            else:
                # 其他情况，取第一个通道
                tensor = tensor[0]
        
        # 确保值在[0, 1]范围内
        tensor = np.clip(tensor, 0, 1)
        
        # 转换为uint8
        tensor = (tensor * 255).astype(np.uint8)
        
        # 保存
        if tensor.ndim == 2:
            # 灰度图
            img = Image.fromarray(tensor, mode='L')
        else:
            # RGB图
            img = Image.fromarray(tensor, mode='RGB')
        
        img.save(filename, format=format)
    
    def make_grid(tensor, nrow=8, padding=2, normalize=False, 
                  range=None, scale_each=False, pad_value=0):
        """
        替代torchvision.utils.make_grid的功能
        将多个图像排列成网格
        """
        if isinstance(tensor, torch.Tensor):
            if tensor.is_cuda:
                tensor = tensor.cpu()
            tensor = tensor.detach().numpy()
        
        if tensor.ndim == 3:
            # 单个图像，添加batch维度
            tensor = tensor[np.newaxis, ...]
        
        batch_size = tensor.shape[0]
        nrow = min(nrow, batch_size)
        ncol = (batch_size + nrow - 1) // nrow
        
        # 计算输出尺寸
        C, H, W = tensor.shape[1], tensor.shape[2], tensor.shape[3]
        grid_h = ncol * H + (ncol - 1) * padding
        grid_w = nrow * W + (nrow - 1) * padding
        
        # 创建网格
        grid = np.full((C, grid_h, grid_w), pad_value, dtype=tensor.dtype)
        
        for i in range(batch_size):
            row = i // nrow
            col = i % nrow
            h_start = row * (H + padding)
            w_start = col * (W + padding)
            grid[:, h_start:h_start+H, w_start:w_start+W] = tensor[i]
        
        # 转换为torch tensor
        grid_tensor = torch.from_numpy(grid)
        
        return grid_tensor

