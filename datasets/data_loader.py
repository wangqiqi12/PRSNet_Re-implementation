"""
ShapeNet Binvox 数据加载器
用于将.binvox文件加载为32x32x32的体素数据，并创建PyTorch DataLoader
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from viewer import binvox_rw
from scipy.ndimage import zoom
import random
from typing import List, Tuple, Optional
from scipy.spatial.distance import cdist
from scipy.ndimage import distance_transform_edt


class ShapeNetVoxelDataset(Dataset):
    def __init__(self, 
                 dataset_path: str,
                 voxel_type: str = 'surface',
                 target_size: int = 32,
                 center_voxels: bool = True,
                 augment: bool = False,
                 max_samples: Optional[int] = None):
        """
        ShapeNet 体素数据集
        
        Args:
            dataset_path: 数据集根目录路径
            voxel_type: 'solid' 或 'surface'
            target_size: 目标体素网格大小 (默认32)
            center_voxels: 是否居中体素
            augment: 是否进行数据增强
            max_samples: 最大样本数量 (用于调试)
        """
        self.dataset_path = Path(dataset_path)
        self.voxel_type = voxel_type
        self.target_size = target_size
        self.center_voxels = center_voxels
        self.augment = augment
        
        # 获取所有有效的模型路径
        self.model_paths = self._get_model_paths()
        
        if max_samples is not None:
            self.model_paths = self.model_paths[:max_samples]
        
        print(f"加载了 {len(self.model_paths)} 个{voxel_type}体素模型")
        
    def _get_model_paths(self) -> List[Path]:
        """获取所有有效的模型路径"""
        shapenet_plane_path = self.dataset_path / "shapenet_plane"
        if not shapenet_plane_path.exists():
            raise ValueError(f"数据集路径不存在: {shapenet_plane_path}")
        
        model_paths = []
        for item in shapenet_plane_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                if self.voxel_type == 'solid':
                    voxel_file = item / "models" / "model_normalized.solid.binvox"
                else:
                    voxel_file = item / "models" / "model_normalized.surface.binvox"
                
                if voxel_file.exists():
                    model_paths.append(item)
        
        return sorted(model_paths)
    
    def _load_binvox(self, model_path: Path) -> Optional[np.ndarray]:
        """加载binvox文件"""
        if self.voxel_type == 'solid':
            binvox_file = model_path / "models" / "model_normalized.solid.binvox"
        else:
            binvox_file = model_path / "models" / "model_normalized.surface.binvox"
        
        try:
            with open(binvox_file, 'rb') as f:
                voxel_model = binvox_rw.read_as_3d_array(f)
            return voxel_model.data
        except Exception as e:
            print(f"加载失败 {binvox_file}: {e}")
            return None
    
    def _downsample_voxels(self, voxel_data: np.ndarray) -> np.ndarray:
        """下采样体素数据到目标尺寸"""
        current_size = voxel_data.shape[0]  # 假设是立方体
        if current_size == self.target_size:
            return voxel_data
        
        # 计算缩放因子
        scale_factor = self.target_size / current_size
        
        # 使用scipy进行下采样
        voxel_float = voxel_data.astype(np.float32)
        downsampled = zoom(voxel_float, scale_factor, order=1)
        
        # 确保尺寸正确
        if downsampled.shape != (self.target_size, self.target_size, self.target_size):
            # 如果尺寸不匹配，进行裁剪或填充
            result = np.zeros((self.target_size, self.target_size, self.target_size), dtype=np.float32)
            
            min_size = min(downsampled.shape[0], self.target_size)
            result[:min_size, :min_size, :min_size] = downsampled[:min_size, :min_size, :min_size]
            downsampled = result
        
        # 阈值化为布尔值
        return downsampled > 0.5
    
    def _center_voxels(self, voxel_data: np.ndarray) -> np.ndarray:
        """将体素数据居中"""
        if not np.any(voxel_data):
            return voxel_data
        
        # 获取占用体素的坐标
        occupied_coords = np.where(voxel_data)
        if len(occupied_coords[0]) == 0:
            return voxel_data
        
        x, y, z = occupied_coords
        
        # 计算当前边界框
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        z_min, z_max = z.min(), z.max()
        
        # 计算当前物体的中心
        current_center = np.array([
            (x_min + x_max) / 2,
            (y_min + y_max) / 2,
            (z_min + z_max) / 2
        ])
        
        # 计算网格的中心
        grid_center = np.array([
            voxel_data.shape[0] / 2,
            voxel_data.shape[1] / 2,
            voxel_data.shape[2] / 2
        ])
        
        # 计算偏移量
        offset = grid_center - current_center
        offset = np.round(offset).astype(int)
        
        # 创建新的居中体素数据
        centered_voxels = np.zeros_like(voxel_data)
        
        # 应用偏移，确保不超出边界
        for i in range(len(x)):
            new_x = x[i] + offset[0]
            new_y = y[i] + offset[1]
            new_z = z[i] + offset[2]
            
            # 检查边界
            if (0 <= new_x < voxel_data.shape[0] and 
                0 <= new_y < voxel_data.shape[1] and 
                0 <= new_z < voxel_data.shape[2]):
                centered_voxels[new_x, new_y, new_z] = True
        
        return centered_voxels
    
    def _rotate_voxels_small_angle(self, voxel_data: np.ndarray, max_angle: float = 15, fixed_angle: Optional[float] = None) -> np.ndarray:
        """
        小角度旋转体素数据（绕物体中心）
        
        Args:
            voxel_data: 原始体素数据
            max_angle: 最大旋转角度（度）
            fixed_angle: 固定旋转角度（如果提供，则使用此角度而不是随机角度）
            
        Returns:
            旋转后的体素数据
        """
        from scipy.ndimage import rotate
        
        if not np.any(voxel_data):
            return voxel_data
        
        # 选择旋转轴和角度
        if fixed_angle is not None:
            # 使用固定角度，比如就规定是x
            axis = random.choice(['x', 'x', 'x'])
            angle = fixed_angle
        else:
            # 随机选择旋转轴和角度
            axis = random.choice(['x', 'x', 'x'])
            angle = random.choice([0, 15, 30, 45, 60, 75, 90])
        
        # 将布尔值转换为浮点数进行旋转
        voxel_float = voxel_data.astype(np.float32)
        
        # 根据轴选择旋转平面
        if axis == 'x':
            axes = (1, 2)  # YZ平面
        elif axis == 'y':
            axes = (0, 2)  # XZ平面
        else:  # z
            axes = (0, 1)  # XY平面
        
        # 执行旋转，使用物体的几何中心作为旋转中心
        rotated = rotate(voxel_float, angle, axes=axes, reshape=False, order=1, mode='constant', cval=0.0)
        
        # 阈值化回布尔值
        return rotated > 0.5

    def _augment_voxels(self, voxel_data: np.ndarray) -> np.ndarray:
        """
        数据增强 - 包括平移和小角度旋转
        """
        if not self.augment:
            return voxel_data
        
        voxel_data = self._rotate_voxels_small_angle(voxel_data, max_angle=60)

        voxel_data = self._translate_voxels(voxel_data, max_shift=5)
        
        # # 随机小角度旋转 (±45度以内)
        # if random.random() < 0.5:
        #     voxel_data = self._rotate_voxels_small_angle(voxel_data, max_angle=45)
        
        # # 原有的90度旋转和翻转（降低概率）
        # if random.random() < 0.3:
        #     # 绕Z轴旋转
        #     k = random.randint(0, 3)
        #     voxel_data = np.rot90(voxel_data, k, axes=(0, 1))
        
        # if random.random() < 0.3:
        #     # 绕Y轴旋转
        #     k = random.randint(0, 3)
        #     voxel_data = np.rot90(voxel_data, k, axes=(0, 2))
        
        # if random.random() < 0.3:
        #     # 绕X轴旋转
        #     k = random.randint(0, 3)
        #     voxel_data = np.rot90(voxel_data, k, axes=(1, 2))
        
        # # 随机翻转（降低概率）
        # if random.random() < 0.2:
        #     voxel_data = np.flip(voxel_data, axis=0)
        # if random.random() < 0.2:
        #     voxel_data = np.flip(voxel_data, axis=1)
        # if random.random() < 0.2:
        #     voxel_data = np.flip(voxel_data, axis=2)
        
        return voxel_data
    




    def _translate_voxels(self, voxel_data: np.ndarray, max_shift: int = 2) -> np.ndarray:
        """
        平移体素数据
        
        Args:
            voxel_data: 原始体素数据
            max_shift: 最大平移距离（体素数）
            
        Returns:
            平移后的体素数据
        """
        if not np.any(voxel_data):
            return voxel_data
        
        # 生成随机平移量
        shift_x = random.randint(-max_shift, max_shift)
        shift_y = random.randint(-max_shift, max_shift)
        shift_z = random.randint(-max_shift, max_shift)
        
        # 获取占用体素的坐标
        occupied_coords = np.where(voxel_data)
        if len(occupied_coords[0]) == 0:
            return voxel_data
        
        x, y, z = occupied_coords
        
        # 应用平移
        new_x = x + shift_x
        new_y = y + shift_y
        new_z = z + shift_z
        
        # 创建新的体素数据
        translated_voxels = np.zeros_like(voxel_data)
        
        # 只保留在边界内的体素
        valid_mask = ((new_x >= 0) & (new_x < voxel_data.shape[0]) &
                     (new_y >= 0) & (new_y < voxel_data.shape[1]) &
                     (new_z >= 0) & (new_z < voxel_data.shape[2]))
        
        if np.any(valid_mask):
            translated_voxels[new_x[valid_mask], new_y[valid_mask], new_z[valid_mask]] = True
        
        return translated_voxels
    
    # def _rotate_voxels_small_angle(self, voxel_data: np.ndarray, max_angle: float = 15) -> np.ndarray:
    #     """
    #     小角度旋转体素数据（绕物体中心）
        
    #     Args:
    #         voxel_data: 原始体素数据
    #         max_angle: 最大旋转角度（度）
            
    #     Returns:
    #         旋转后的体素数据
    #     """
    #     from scipy.ndimage import rotate
        
    #     if not np.any(voxel_data):
    #         return voxel_data
        
    #     # 随机选择旋转轴和角度
    #     axis = random.choice(['x', 'y', 'z'])
    #     angle = random.uniform(-max_angle, max_angle)
        
    #     # 将布尔值转换为浮点数进行旋转
    #     voxel_float = voxel_data.astype(np.float32)
        
    #     # 根据轴选择旋转平面
    #     if axis == 'x':
    #         axes = (1, 2)  # YZ平面
    #     elif axis == 'y':
    #         axes = (0, 2)  # XZ平面
    #     else:  # z
    #         axes = (0, 1)  # XY平面
        
    #     # 执行旋转，使用物体的几何中心作为旋转中心
    #     rotated = rotate(voxel_float, angle, axes=axes, reshape=False, order=1, mode='constant', cval=0.0)
        
    #     # 阈值化回布尔值
    #     return rotated > 0.5
    
    def _compute_distance_field(self, voxel_data: np.ndarray) -> np.ndarray:
        """
        计算每个体素位置到3D物体表面的最近距离
        
        Args:
            voxel_data: [32, 32, 32] 布尔体素数据
            
        Returns:
            distance_field: [32, 32, 32] 每个位置到表面的最近距离
        """
        # 如果体素数据为空，返回全部为无穷大的距离场
        if not np.any(voxel_data):
            return np.full_like(voxel_data, float('inf'), dtype=np.float32)
        
        # 方法1: 使用EDT (Euclidean Distance Transform)
        # 对于实心物体，计算到边界的距离
        if self.voxel_type == 'solid':
            # 计算物体内部到边界的距离（负距离）
            internal_dist = distance_transform_edt(voxel_data)
            # 计算物体外部到边界的距离（正距离）
            external_dist = distance_transform_edt(~voxel_data)
            # 合并：内部为负距离，外部为正距离
            distance_field = np.where(voxel_data, -internal_dist, external_dist)
        else:
            # 对于表面体素，直接计算到表面点的距离
            surface_points = np.array(np.where(voxel_data)).T  # [N, 3]
            
            if len(surface_points) == 0:
                return np.full_like(voxel_data, float('inf'), dtype=np.float32)
            
            # 生成所有体素位置的坐标
            x, y, z = np.mgrid[0:voxel_data.shape[0], 
                              0:voxel_data.shape[1], 
                              0:voxel_data.shape[2]]
            all_points = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)  # [32*32*32, 3]
            
            # 计算每个位置到最近表面点的距离
            # 为了效率，分批计算
            batch_size = 1024
            distance_field = np.zeros(voxel_data.shape, dtype=np.float32)
            
            for i in range(0, len(all_points), batch_size):
                batch_points = all_points[i:i+batch_size]
                distances = cdist(batch_points, surface_points)
                min_distances = np.min(distances, axis=1)
                
                # 将距离重新排列回原始形状
                start_idx = i
                end_idx = min(i + batch_size, len(all_points))
                flat_indices = np.arange(start_idx, end_idx)
                
                coords = np.unravel_index(flat_indices, voxel_data.shape)
                distance_field[coords] = min_distances
        
        return distance_field.astype(np.float32)
    
    def __len__(self) -> int:
        return len(self.model_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        获取一个样本
        
        Returns:
            voxel_tensor: torch.Tensor, shape [1, 32, 32, 32] - 体素数据
            distance_tensor: torch.Tensor, shape [32, 32, 32] - 距离场
            model_id: str, 模型ID
        """
        model_path = self.model_paths[idx]
        model_id = model_path.name
        
        # 加载binvox数据
        voxel_data = self._load_binvox(model_path)
        if voxel_data is None:
            # 如果加载失败，返回空的体素数据
            voxel_data = np.zeros((self.target_size, self.target_size, self.target_size), dtype=bool)
        else:
            # 下采样
            voxel_data = self._downsample_voxels(voxel_data)
            
            # 居中
            if self.center_voxels:
                voxel_data = self._center_voxels(voxel_data)
            
            # 数据增强
            voxel_data = self._augment_voxels(voxel_data) # augment False, return itself
        
        # 计算距离场
        distance_field = self._compute_distance_field(voxel_data)
        
        # 转换为浮点数并添加channel维度
        voxel_tensor = torch.from_numpy(voxel_data.astype(np.float32))
        voxel_tensor = voxel_tensor.unsqueeze(0)  # 添加channel维度 [1, 32, 32, 32]
        
        # 确保distance_field是连续的，避免负步长问题
        if not distance_field.flags['C_CONTIGUOUS']:
            distance_field = distance_field.copy()
        distance_tensor = torch.from_numpy(distance_field)  # [32, 32, 32]
        
        return voxel_tensor, distance_tensor, model_id


def create_shapenet_dataloader(dataset_path: str,
                              batch_size: int = 32,
                              voxel_type: str = 'surface',
                              target_size: int = 32,
                              center_voxels: bool = True,
                              augment: bool = False,
                              shuffle: bool = True,
                              num_workers: int = 4,
                              max_samples: Optional[int] = None) -> DataLoader:
    """
    创建ShapeNet数据加载器
    
    Args:
        dataset_path: 数据集路径
        batch_size: 批次大小
        voxel_type: 体素类型 ('solid' 或 'surface')
        target_size: 目标体素网格大小
        center_voxels: 是否居中体素
        augment: 是否进行数据增强
        shuffle: 是否打乱数据
        num_workers: 数据加载进程数
        max_samples: 最大样本数量
    
    Returns:
        DataLoader: PyTorch数据加载器
    """
    dataset = ShapeNetVoxelDataset(
        dataset_path=dataset_path,
        voxel_type=voxel_type,
        target_size=target_size,
        center_voxels=center_voxels,
        augment=augment,
        max_samples=max_samples
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True  # 确保每个batch都有相同大小
    )
    
    return dataloader


def test_dataloader():
    """测试数据加载器"""
    print("测试ShapeNet数据加载器...")
    
    # 数据集路径
    dataset_path = "/Users/wangqiqi/Desktop/PRSNet-Reimplementation/data/ShapeNet-toydata"
    
    # 创建数据加载器
    dataloader = create_shapenet_dataloader(
        dataset_path=dataset_path,
        batch_size=4,  # 小批次用于测试
        voxel_type='surface',
        target_size=32,
        center_voxels=True,
        augment=False,
        shuffle=True,
        num_workers=0,  # 单进程用于调试
        max_samples=16  # 限制样本数量用于快速测试
    )
    
    print(f"数据集大小: {len(dataloader.dataset)}")
    print(f"批次数量: {len(dataloader)}")
    
    # 测试加载一个批次
    print("\n测试加载第一个批次...")
    for batch_idx, (voxel_batch, model_ids) in enumerate(dataloader):
        print(f"批次 {batch_idx}:")
        print(f"  体素数据形状: {voxel_batch.shape}")  # 应该是 [4, 1, 32, 32, 32]
        print(f"  数据类型: {voxel_batch.dtype}")
        print(f"  值域: [{voxel_batch.min().item():.3f}, {voxel_batch.max().item():.3f}]")
        print(f"  模型IDs: {model_ids[:2]}...")  # 显示前两个ID
        
        # 计算每个样本的占用率
        occupancy_rates = []
        for i in range(voxel_batch.shape[0]):
            sample = voxel_batch[i, 0]  # 去掉channel维度
            occupancy = (sample > 0.5).sum().item() / sample.numel()
            occupancy_rates.append(occupancy)
        
        print(f"  占用率: {[f'{rate*100:.1f}%' for rate in occupancy_rates]}")
        
        if batch_idx >= 2:  # 只测试前3个批次
            break
    
    print("\n数据加载器测试完成！")


def test_with_prsnet():
    """测试数据加载器与PRSNet的对接"""
    print("\n测试与PRSNet模型的对接...")
    
    # 导入PRSNet
    import sys
    sys.path.append('/Users/wangqiqi/Desktop/PRSNet-Reimplementation')
    from models.prsnet import PRSNet
    
    # 创建数据加载器
    dataset_path = "/Users/wangqiqi/Desktop/PRSNet-Reimplementation/data/ShapeNet-toydata"
    dataloader = create_shapenet_dataloader(
        dataset_path=dataset_path,
        batch_size=8,
        voxel_type='surface',
        target_size=32,
        center_voxels=True,
        augment=False,
        shuffle=False,
        num_workers=0,
        max_samples=32
    )
    
    # 创建PRSNet模型
    model = PRSNet()
    model.eval()
    
    print(f"PRSNet参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试前向传播
    with torch.no_grad():
        for batch_idx, (voxel_batch, model_ids) in enumerate(dataloader):
            print(f"\n批次 {batch_idx}:")
            print(f"  输入形状: {voxel_batch.shape}")
            
            # 前向传播
            outputs = model(voxel_batch)
            
            print(f"  输出:")
            print(f"    planes形状: {outputs['planes'].shape}")  # [batch_size, 3, 4]
            print(f"    quats形状: {outputs['quats'].shape}")    # [batch_size, 3, 4]
            
            # 检查输出值域
            planes = outputs['planes']
            quats = outputs['quats']
            
            print(f"    planes值域: [{planes.min().item():.3f}, {planes.max().item():.3f}]")
            print(f"    quats值域: [{quats.min().item():.3f}, {quats.max().item():.3f}]")
            
            # 验证四元数归一化
            quat_norms = torch.norm(quats, dim=-1)
            print(f"    四元数范数: [{quat_norms.min().item():.3f}, {quat_norms.max().item():.3f}]")
            
            # 验证平面法向量归一化
            plane_normals = planes[:, :, :3]
            normal_norms = torch.norm(plane_normals, dim=-1)
            print(f"    平面法向量范数: [{normal_norms.min().item():.3f}, {normal_norms.max().item():.3f}]")
            
            if batch_idx >= 1:  # 只测试前2个批次
                break
    
    print("\n与PRSNet对接测试完成！")


if __name__ == "__main__":
    # 运行测试
    test_dataloader()
    test_with_prsnet()
