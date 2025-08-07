# PRSNet Re-implementation

A PyTorch re-implementation of **PRS-Net: Planar Reflective Symmetry Detection Net for 3D Models**

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/1910.06511)
[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)](https://pytorch.org/)

## Overview

This project provides a clean, simplified re-implementation of PRS-Net, a deep learning model for detecting planar reflective symmetry in 3D voxelized models. The network takes a 32³ voxel grid as input and predicts symmetry plane parameters using a 3D CNN encoder followed by fully connected layers.

## Features

- 🔥 **Clean PyTorch Implementation**: Modern, readable code structure
- 🚀 **Easy Training Pipeline**: Streamlined training and inference scripts
- 📊 **Built-in Visualization**: Integrated plotting and 3D visualization tools
- 🎯 **Flexible Architecture**: Easily configurable model parameters
- 📱 **Interactive Interface**: Gradio-based web interface for demonstrations

## Results

<div align="center">

### Example 1: Airplane Symmetry Detection
![Example 1](assets/eg1.png)

### Example 2: Complex Object Analysis  
![Example 2](assets/eg2.png)

### Example 3: Multi-view Visualization
![Example 3](assets/eg3.png)

</div>

## Architecture

The PRSNet model consists of:
- **3D CNN Encoder**: 5-layer 3D convolutional network (1→4→8→16→32→64 channels)
- **Dual FC Heads**: Two separate fully connected branches for symmetry plane prediction
- **Input**: 32×32×32 voxel grid
- **Output**: Symmetry plane parameters (normal vector + point on plane)

## Installation

### Prerequisites
- Python 3.8+
- CUDA (optional, for GPU acceleration)

### Setup Environment

```bash
# Clone the repository
git clone <your-repo-url>
cd PRSNet_Re-implementation

# Create virtual environment (recommended)
python -m venv prsnet_env
source prsnet_env/bin/activate  # On Windows: prsnet_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- `torch>=2.0.0` - Deep learning framework
- `torchvision>=0.15.0` - Computer vision utilities
- `numpy>=1.21.0` - Numerical computing
- `scipy>=1.7.0` - Scientific computing
- `gradio>=4.0.0` - Web interface
- `plotly>=5.15.0` - Interactive visualization

## Dataset Structure

The project expects data in the following format:

```
toy_data/
└── shapenet_plane/
    └── [model_id]/
        ├── images/
        ├── models/
        │   ├── model_normalized.obj
        │   ├── model_normalized.solid.binvox
        │   └── model_normalized.surface.binvox
        └── screenshots/
```

## Usage

### Training

```bash
python train.py
```

**Training Configuration:**
- Default batch size: 1
- Learning rate: 1e-3 with ReduceLROnPlateau scheduler
- Device: Auto-detection (CUDA/MPS/CPU)
- Checkpoints saved every 10 epochs

### Inference

```bash
python inference.py
```

**Inference Features:**
- Loads pre-trained model from `ckpts/`
- Visualizes symmetry plane predictions
- Generates interactive 3D plots
- Saves results to `inference_outputs/`

### Visualization Tools

The project includes several visualization utilities in the `viewer/` directory:

- `shapenet_voxel_viewer.py` - Basic voxel visualization
- `shapenet_augmented_voxel_viewer.py` - Enhanced voxel display
- `shapnet_mesh_viewer.py` - 3D mesh visualization
- `binvox_rw.py` - Binvox file I/O utilities

## Model Details

### Loss Function

The training uses a combined loss:
```
Total Loss = Reconstruction Loss + λ * Regularization Loss
```

Where:
- **Reconstruction Loss**: Mean Squared Error between predicted and ground truth
- **Regularization Loss**: Encourages valid symmetry plane parameters
- **λ (reg_weight)**: Balancing parameter (default: 1.0)

### Network Architecture

```
Input: [B, 1, 32, 32, 32]
│
├── 3D CNN Encoder (5 layers)
│   ├── Conv3D(1→4) + LeakyReLU + MaxPool3D
│   ├── Conv3D(4→8) + LeakyReLU + MaxPool3D  
│   ├── Conv3D(8→16) + LeakyReLU + MaxPool3D
│   ├── Conv3D(16→32) + LeakyReLU + MaxPool3D
│   └── Conv3D(32→64) + LeakyReLU + MaxPool3D
│
├── FC Branch 1: [64] → [32] → [16] → [4]
└── FC Branch 2: [64] → [32] → [16] → [4]
│
Output: [B, 8] (Symmetry plane parameters)
```

## File Structure

```
PRSNet_Re-implementation/
├── 📜 README.md                    # This file
├── 🔧 requirements.txt             # Dependencies
├── 🚀 train.py                     # Training script
├── 🔍 inference.py                 # Inference script
├── 📁 models/
│   └── prsnet.py                   # PRSNet model definition
├── 📁 datasets/
│   └── data_loader.py              # Dataset and data loading utilities
├── 📁 viewer/                      # Visualization tools
├── 📁 assets/                      # Example images and results
├── 📁 ckpts/                       # Model checkpoints
└── 📁 toy_data/                    # Sample dataset
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this implementation in your research, please cite the original paper:

```bibtex
@article{prsnet2019,
  title={PRS-Net: Planar Reflective Symmetry Detection Net for 3D Models},
  author={[Authors]},
  journal={arXiv preprint arXiv:1910.06511},
  year={2019}
}
```

## License

This project is for research and educational purposes. Please refer to the original paper for licensing terms.

## Acknowledgments

- Original PRS-Net paper authors
- PyTorch team for the excellent framework
- ShapeNet dataset contributors
