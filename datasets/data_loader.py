"""
ShapeNet Binvox data loader
Loads .binvox files as 32x32x32 voxel data and creates PyTorch DataLoader
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
        ShapeNet voxel dataset
        
        Args:
            dataset_path: Dataset root directory path
            voxel_type: 'solid' or 'surface'
            target_size: Target voxel grid size (default 32)
            center_voxels: Whether to center voxels
            augment: Whether to perform data augmentation
            max_samples: Maximum number of samples (for debugging)
        """
        self.dataset_path = Path(dataset_path)
        self.voxel_type = voxel_type
        self.target_size = target_size
        self.center_voxels = center_voxels
        self.augment = augment
        
        # Get all valid model paths
        self.model_paths = self._get_model_paths()
        
        if max_samples is not None:
            self.model_paths = self.model_paths[:max_samples]
        
        print(f"Loaded {len(self.model_paths)} {voxel_type} voxel models")
        
    def _get_model_paths(self) -> List[Path]:
        """Get all valid model paths"""
        shapenet_plane_path = self.dataset_path / "shapenet_plane"
        if not shapenet_plane_path.exists():
            raise ValueError(f"Dataset path does not exist: {shapenet_plane_path}")
        
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
        """Load binvox file"""
        if self.voxel_type == 'solid':
            binvox_file = model_path / "models" / "model_normalized.solid.binvox"
        else:
            binvox_file = model_path / "models" / "model_normalized.surface.binvox"
        
        try:
            with open(binvox_file, 'rb') as f:
                voxel_model = binvox_rw.read_as_3d_array(f)
            return voxel_model.data
        except Exception as e:
            print(f"Failed to load {binvox_file}: {e}")
            return None
    
    def _downsample_voxels(self, voxel_data: np.ndarray) -> np.ndarray:
        """Downsample voxel data to target size"""
        current_size = voxel_data.shape[0]  # Assume cubic
        if current_size == self.target_size:
            return voxel_data
        
        # Calculate scale factor
        scale_factor = self.target_size / current_size
        
        # Use scipy for downsampling
        voxel_float = voxel_data.astype(np.float32)
        downsampled = zoom(voxel_float, scale_factor, order=1)
        
        # Ensure correct size
        if downsampled.shape != (self.target_size, self.target_size, self.target_size):
            # If size doesn't match, crop or pad
            result = np.zeros((self.target_size, self.target_size, self.target_size), dtype=np.float32)
            
            min_size = min(downsampled.shape[0], self.target_size)
            result[:min_size, :min_size, :min_size] = downsampled[:min_size, :min_size, :min_size]
            downsampled = result
        
        # Threshold to boolean values
        return downsampled > 0.5
    
    def _center_voxels(self, voxel_data: np.ndarray) -> np.ndarray:
        """Center voxel data"""
        if not np.any(voxel_data):
            return voxel_data
        
        # Get occupied voxel coordinates
        occupied_coords = np.where(voxel_data)
        if len(occupied_coords[0]) == 0:
            return voxel_data
        
        x, y, z = occupied_coords
        
        # Calculate current bounding box
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        z_min, z_max = z.min(), z.max()
        
        # Calculate current object center
        current_center = np.array([
            (x_min + x_max) / 2,
            (y_min + y_max) / 2,
            (z_min + z_max) / 2
        ])
        
        # Calculate grid center
        grid_center = np.array([
            voxel_data.shape[0] / 2,
            voxel_data.shape[1] / 2,
            voxel_data.shape[2] / 2
        ])
        
        # Calculate offset
        offset = grid_center - current_center
        offset = np.round(offset).astype(int)
        
        # Create new centered voxel data
        centered_voxels = np.zeros_like(voxel_data)
        
        # Apply offset, ensure within bounds
        for i in range(len(x)):
            new_x = x[i] + offset[0]
            new_y = y[i] + offset[1]
            new_z = z[i] + offset[2]
            
            # Check bounds
            if (0 <= new_x < voxel_data.shape[0] and 
                0 <= new_y < voxel_data.shape[1] and 
                0 <= new_z < voxel_data.shape[2]):
                centered_voxels[new_x, new_y, new_z] = True
        
        return centered_voxels
    
    def _rotate_voxels_small_angle(self, voxel_data: np.ndarray, max_angle: float = 15, fixed_angle: Optional[float] = None) -> np.ndarray:
        """
        Small angle rotation of voxel data (around object center)
        
        Args:
            voxel_data: Original voxel data
            max_angle: Maximum rotation angle (degrees)
            fixed_angle: Fixed rotation angle (if provided, use this angle instead of random)
            
        Returns:
            Rotated voxel data
        """
        from scipy.ndimage import rotate
        
        if not np.any(voxel_data):
            return voxel_data
        
        # Select rotation axis and angle
        if fixed_angle is not None:
            # Use fixed angle, specify x axis
            axis = random.choice(['x', 'x', 'x'])
            angle = fixed_angle
        else:
            # Randomly select rotation axis and angle
            axis = random.choice(['x', 'x', 'x'])
            angle = random.choice([0, 15, 30, 45, 60, 75, 90])
        
        # Convert boolean to float for rotation
        voxel_float = voxel_data.astype(np.float32)
        
        # Select rotation plane based on axis
        if axis == 'x':
            axes = (1, 2)  # YZ plane
        elif axis == 'y':
            axes = (0, 2)  # XZ plane
        else:  # z
            axes = (0, 1)  # XY plane
        
        # Perform rotation using object's geometric center as rotation center
        rotated = rotate(voxel_float, angle, axes=axes, reshape=False, order=1, mode='constant', cval=0.0)
        
        # Threshold back to boolean
        return rotated > 0.5

    def _augment_voxels(self, voxel_data: np.ndarray) -> np.ndarray:
        """
        Data augmentation - including translation and small angle rotation
        """
        if not self.augment:
            return voxel_data
        
        voxel_data = self._rotate_voxels_small_angle(voxel_data, max_angle=60)

        voxel_data = self._translate_voxels(voxel_data, max_shift=5)
        
        # # Random small angle rotation (±45 degrees)
        # if random.random() < 0.5:
        #     voxel_data = self._rotate_voxels_small_angle(voxel_data, max_angle=45)
        
        # # Original 90-degree rotation and flipping (reduced probability)
        # if random.random() < 0.3:
        #     # Rotate around Z axis
        #     k = random.randint(0, 3)
        #     voxel_data = np.rot90(voxel_data, k, axes=(0, 1))
        
        # if random.random() < 0.3:
        #     # Rotate around Y axis
        #     k = random.randint(0, 3)
        #     voxel_data = np.rot90(voxel_data, k, axes=(0, 2))
        
        # if random.random() < 0.3:
        #     # Rotate around X axis
        #     k = random.randint(0, 3)
        #     voxel_data = np.rot90(voxel_data, k, axes=(1, 2))
        
        # # Random flipping (reduced probability)
        # if random.random() < 0.2:
        #     voxel_data = np.flip(voxel_data, axis=0)
        # if random.random() < 0.2:
        #     voxel_data = np.flip(voxel_data, axis=1)
        # if random.random() < 0.2:
        #     voxel_data = np.flip(voxel_data, axis=2)
        
        return voxel_data
    




    def _translate_voxels(self, voxel_data: np.ndarray, max_shift: int = 2) -> np.ndarray:
        """
        Translate voxel data
        
        Args:
            voxel_data: Original voxel data
            max_shift: Maximum translation distance (in voxels)
            
        Returns:
            Translated voxel data
        """
        if not np.any(voxel_data):
            return voxel_data
        
        # Generate random translation amounts
        shift_x = random.randint(-max_shift, max_shift)
        shift_y = random.randint(-max_shift, max_shift)
        shift_z = random.randint(-max_shift, max_shift)
        
        # Get occupied voxel coordinates
        occupied_coords = np.where(voxel_data)
        if len(occupied_coords[0]) == 0:
            return voxel_data
        
        x, y, z = occupied_coords
        
        # Apply translation
        new_x = x + shift_x
        new_y = y + shift_y
        new_z = z + shift_z
        
        # Create new voxel data
        translated_voxels = np.zeros_like(voxel_data)
        
        # Only keep voxels within bounds
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
        Calculate the nearest distance from each voxel position to the 3D object surface
        
        Args:
            voxel_data: [32, 32, 32] boolean voxel data
            
        Returns:
            distance_field: [32, 32, 32] nearest distance to surface for each position
        """
        # If voxel data is empty, return distance field with all infinity values
        if not np.any(voxel_data):
            return np.full_like(voxel_data, float('inf'), dtype=np.float32)
        
        # Method 1: Using EDT (Euclidean Distance Transform)
        # For solid objects, calculate distance to boundary
        if self.voxel_type == 'solid':
            # Calculate distance from inside object to boundary (negative distance)
            internal_dist = distance_transform_edt(voxel_data)
            # Calculate distance from outside object to boundary (positive distance)
            external_dist = distance_transform_edt(~voxel_data)
            # Merge: inside is negative distance, outside is positive distance
            distance_field = np.where(voxel_data, -internal_dist, external_dist)
        else:
            # For surface voxels, directly calculate distance to surface points
            surface_points = np.array(np.where(voxel_data)).T  # [N, 3]
            
            if len(surface_points) == 0:
                return np.full_like(voxel_data, float('inf'), dtype=np.float32)
            
            # Generate coordinates for all voxel positions
            x, y, z = np.mgrid[0:voxel_data.shape[0], 
                              0:voxel_data.shape[1], 
                              0:voxel_data.shape[2]]
            all_points = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)  # [32*32*32, 3]
            
            # Calculate distance from each position to nearest surface point
            # Process in batches for efficiency
            batch_size = 1024
            distance_field = np.zeros(voxel_data.shape, dtype=np.float32)
            
            for i in range(0, len(all_points), batch_size):
                batch_points = all_points[i:i+batch_size]
                distances = cdist(batch_points, surface_points)
                min_distances = np.min(distances, axis=1)
                
                # Reshape distances back to original shape
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
        Get a sample
        
        Returns:
            voxel_tensor: torch.Tensor, shape [1, 32, 32, 32] - voxel data
            distance_tensor: torch.Tensor, shape [32, 32, 32] - distance field
            model_id: str, model ID
        """
        model_path = self.model_paths[idx]
        model_id = model_path.name
        
        # Load binvox data
        voxel_data = self._load_binvox(model_path)
        if voxel_data is None:
            # If loading fails, return empty voxel data
            voxel_data = np.zeros((self.target_size, self.target_size, self.target_size), dtype=bool)
        else:
            # Downsample
            voxel_data = self._downsample_voxels(voxel_data)
            
            # Center
            if self.center_voxels:
                voxel_data = self._center_voxels(voxel_data)
            
            # Data augmentation
            voxel_data = self._augment_voxels(voxel_data) # augment False, return itself
        
        # Calculate distance field
        distance_field = self._compute_distance_field(voxel_data)
        
        # Convert to float and add channel dimension
        voxel_tensor = torch.from_numpy(voxel_data.astype(np.float32))
        voxel_tensor = voxel_tensor.unsqueeze(0)  # Add channel dimension [1, 32, 32, 32]
        
        # Ensure distance_field is contiguous to avoid negative stride issues
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
    Create ShapeNet data loader
    
    Args:
        dataset_path: Dataset path
        batch_size: Batch size
        voxel_type: Voxel type ('solid' or 'surface')
        target_size: Target voxel grid size
        center_voxels: Whether to center voxels
        augment: Whether to perform data augmentation
        shuffle: Whether to shuffle data
        num_workers: Number of data loading processes
        max_samples: Maximum number of samples
    
    Returns:
        DataLoader: PyTorch data loader
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
        drop_last=True  # Ensure each batch has the same size
    )
    
    return dataloader

