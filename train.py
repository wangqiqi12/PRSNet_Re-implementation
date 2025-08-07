import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from datasets.data_loader import ShapeNetVoxelDataset
from models.prsnet import PRSNet
import plotly.graph_objects as go
import plotly.offline as pyo

class PRSNetTrainer:
    def __init__(self, 
                 train_dataset_path: str = None,
                 val_dataset_path: str = None,
                 batch_size: int = 1,
                 learning_rate: float = 1e-3,
                 reg_weight: float = 1.0,
                 scheduler_type: str = 'plateau',  # 'plateau' or 'step'
                 device: str = 'cpu'):
        
        self.train_dataset_path = train_dataset_path
        self.val_dataset_path = val_dataset_path
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.scheduler_type = scheduler_type
        self.device = torch.device(device)
        self.reg_weight = reg_weight

        print(f"Using device: {self.device}")
        
        self.model = PRSNet().to(self.device)

        self._init_model_weights()
        
        # optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # scheduler
        if self.scheduler_type == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',           
                factor=0.5,          
                patience=50,  # every 50 epoch        
                min_lr=1e-3,         
                threshold=1e-3, 
            )
        elif self.scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=100,   
                gamma=0.7,           
            )
        else:
            raise ValueError(f"Unsupported: {self.scheduler_type}")
        
        print(f"Scheduler: {self.scheduler_type}")
        
        self.training_history = {
            'train_losses': [],
            'val_losses': [],
            'steps': [],
            'epochs': []
        }
        
        # dataloader
        self._setup_data_loaders()
    
    def _init_model_weights(self):
        for fc_name in ['fc1', 'fc2', 'fc3']:
            fc_layers = list(getattr(self.model, fc_name).children())
            if len(fc_layers) > 0:
                last_layer = fc_layers[-1]
                if isinstance(last_layer, nn.Linear):
                    with torch.no_grad():
                        nn.init.zeros_(last_layer.weight)
                        # Special initialization: for the 3 predicted symmetry planes
                        # initialize biases with specific values
                        if fc_name == 'fc1':
                            last_layer.bias.data = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device, dtype=torch.float32)
                        elif fc_name == 'fc2':
                            last_layer.bias.data = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device, dtype=torch.float32)
                        elif fc_name == 'fc3':
                            last_layer.bias.data = torch.tensor([0.0, 0.0, 1.0, 0.0], device=self.device, dtype=torch.float32)
                        
    
    def _setup_data_loaders(self):

        train_dataset = ShapeNetVoxelDataset(
            dataset_path=self.train_dataset_path,
            voxel_type='surface',
            target_size=32,
            center_voxels=True,
            augment=True, # True,  
            # max_samples=1   
        )
        
        val_dataset = ShapeNetVoxelDataset(
            dataset_path=self.val_dataset_path,
            voxel_type='surface',
            target_size=32,
            center_voxels=True,
            augment=True,
            # max_samples=1  
        )
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False
        )
        
        print(f"Data loading completed - Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    def compute_loss(self, outputs, voxel_data, distance_field):

        p1 = outputs['p1']  # [B, 1, 4]
        p2 = outputs['p2']  # [B, 1, 4]
        p3 = outputs['p3']  # [B, 1, 4]
        
        symmetry_loss = self.compute_symmetry_loss(voxel_data, p1)
        symmetry_loss += self.compute_symmetry_loss(voxel_data, p2)
        symmetry_loss += self.compute_symmetry_loss(voxel_data, p3)

        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        total_loss = symmetry_loss + self.reg_weight * self.reg_loss(p1, p2, p3)

        # gt_plane = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).view(1, 1, 4)  # 假设的ground truth平面
        # gt_sym_loss = self.compute_symmetry_loss(voxel_data, gt_plane)
        if isinstance(symmetry_loss,float):
            raise ValueError("symmetry_loss should not be a float, it should be a tensor.")

        return total_loss, {'total_loss': total_loss.item(), 'symmetry_loss': symmetry_loss.item()}

    def reg_loss(self, p1, p2, p3):
        # return torch.tensor(0.0, device=self.device, requires_grad=True)
        """
        Compute regularization loss - orthogonality loss for three plane normal vectors
        """
        B = p1.shape[0]
        device = self.device
        
        batch_losses = []
        
        for b in range(B):
            n1 = p1[b, 0, :3]  # [3]
            n2 = p2[b, 0, :3]  # [3] 
            n3 = p3[b, 0, :3]  # [3]
            
            n1_norm = n1 / (torch.norm(n1) + 1e-9)
            n2_norm = n2 / (torch.norm(n2) + 1e-9)
            n3_norm = n3 / (torch.norm(n3) + 1e-9)
            
            N = torch.stack([n1_norm, n2_norm, n3_norm], dim=0)  # [3, 3]
            
            # N @ N^T - I
            NNT = torch.matmul(N, N.T)  # [3, 3]
            I = torch.eye(3, device=device)  # [3, 3]

            # Frobenius norm squared
            orthogonal_loss = torch.norm(NNT - I, p='fro') ** 2
            
            batch_losses.append(orthogonal_loss)
        
        if batch_losses:
            return torch.stack(batch_losses).mean()
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)


    def compute_symmetry_loss(self, voxel_data, planes, num_samples=100):
        """
        Compute symmetry loss
        1. Normalize voxel coordinates to [-0.5, 0.5]
        2. Use model output plane parameters ax+by+cz+d=0
        3. Randomly sample 100 surface points, compute symmetric points, and calculate distance loss
        """
        B = voxel_data.shape[0]
        device = self.device
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        for b in range(B):
            voxel_b = voxel_data[b, 0]  # [32, 32, 32]
            
            occ_indices = (voxel_b > 0.5).nonzero(as_tuple=False).float()
            if occ_indices.shape[0] == 0:
                continue
            
            # voxels scale: [0, 31] -> [-0.5, 0.5]
            all_surface_points = (occ_indices - 15.5) / 31.0  # [N, 3]
            
            # randomly sample points
            if occ_indices.shape[0] > num_samples:
                sample_indices = torch.randperm(occ_indices.shape[0], device=device)[:num_samples]
                sampled_points = all_surface_points[sample_indices]
            else:
                sampled_points = all_surface_points
            
            # ax + by + cz + d = 0
            a, b, c, d = planes[b, 0, 0], planes[b, 0, 1], planes[b, 0, 2], planes[b, 0, 3]
            
            batch_loss_sum = torch.tensor(0.0, device=device, requires_grad=True)
            
            for i in range(sampled_points.shape[0]):
                point = sampled_points[i]  # [3]
                x, y, z = point[0], point[1], point[2]
                
                # distance = (ax + by + cz + d) / sqrt(a^2 + b^2 + c^2)
                numerator = a * x + b * y + c * z + d
                denominator = torch.sqrt(a**2 + b**2 + c**2) + 1e-8
                signed_distance = numerator / denominator
                
                # Compute the symmetric point
                # Symmetric point = original point - 2 * signed_distance * normal vector
                normal = torch.stack([a, b, c]) / denominator
                sym_point = point - 2 * signed_distance * normal
                
                # Compute the distances from the symmetric point to all surface points
                distances = torch.norm(all_surface_points - sym_point.unsqueeze(0), dim=1)
                min_distance = torch.min(distances)
                
                batch_loss_sum = batch_loss_sum + min_distance
            
            # average loss for this batch
            if sampled_points.shape[0] > 0:
                batch_loss = batch_loss_sum / sampled_points.shape[0]
                total_loss = total_loss + batch_loss
        
        return total_loss / B if B > 0 else torch.tensor(0.0, device=device, requires_grad=True)
    
    def train_epoch(self, epoch: int):
        """one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (voxel_batch, distance_batch, model_ids) in enumerate(self.train_loader):
            # -----------------Training-----------------
            voxel_batch = voxel_batch.to(self.device)
            distance_batch = distance_batch.to(self.device)
            
            outputs = self.model(voxel_batch)

            loss, loss_dict = self.compute_loss(outputs, voxel_batch, distance_batch)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # -----------------Recording-----------------
            total_loss += loss.item()
            num_batches += 1

            current_step = (epoch - 1) * len(self.train_loader) + batch_idx
            self.training_history['steps'].append(current_step)
            self.training_history['train_losses'].append(loss.item())
            
            if (batch_idx + 1) % 5 == 0:  
                print(f'  Step {current_step}: Epoch={epoch}, Batch={batch_idx+1}/{len(self.train_loader)}, Loss={loss.item():.6f} (Sym={loss_dict["symmetry_loss"]:.6f})')
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self):
        
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for voxel_batch, distance_batch, model_ids in self.val_loader:
                voxel_batch = voxel_batch.to(self.device)
                distance_batch = distance_batch.to(self.device)
                
                outputs = self.model(voxel_batch)
                loss, _ = self.compute_loss(outputs, voxel_batch, distance_batch)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_val_loss = total_loss / num_batches
        return avg_val_loss, outputs
    
    def train(self, num_epochs: int, save_dir: str = 'checkpoints', resume_from: str = None):
        """
        Training process
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        start_epoch = 1
        
        # If resume path is provided, load checkpoint
        if resume_from and Path(resume_from).exists():
            print(f"Resuming training from checkpoint: {resume_from}")
            start_epoch, _ = self.load_model(resume_from)
            start_epoch += 1  # Start from next epoch
            print(f"Training will continue from epoch {start_epoch}")
        elif resume_from:
            print(f"Warning: Checkpoint file does not exist: {resume_from}")
            print("Training will start from scratch")
        
        print(f"Starting training for {num_epochs} epochs (from epoch {start_epoch})...")
        
        for epoch in range(start_epoch, num_epochs + 1):
            # train
            train_loss = self.train_epoch(epoch)

            # validate
            val_loss, val_outputs = self.validate()

            # record
            self.training_history['epochs'].append(epoch)
            self.training_history['val_losses'].append(val_loss)
            
            # details
            if epoch % 10 == 0 or epoch == num_epochs:
                # Format the output for the 3 planes
                p1 = val_outputs['p1'][0, 0].cpu().numpy()  # [4]
                p2 = val_outputs['p2'][0, 0].cpu().numpy()  # [4]
                p3 = val_outputs['p3'][0, 0].cpu().numpy()  # [4]
                
                planes_str = f"P1:[{p1[0]:.3f},{p1[1]:.3f},{p1[2]:.3f},{p1[3]:.3f}] P2:[{p2[0]:.3f},{p2[1]:.3f},{p2[2]:.3f},{p2[3]:.3f}] P3:[{p3[0]:.3f},{p3[1]:.3f},{p3[2]:.3f},{p3[3]:.3f}]"
                
                print(f'Epoch {epoch}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}')
                print(f'    Planes: {planes_str}')
            elif epoch % 5 == 0:  # Every 5 epochs output simplified information
                print(f'Epoch {epoch}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}')

            # Save model every 50 epochs
            if epoch % 50 == 0 or epoch == num_epochs:
                self.save_model(save_dir, epoch, val_loss)

            # Generate visualizations every 10 epochs
            if epoch % 10 == 0 or epoch == num_epochs:
                self.plot_loss_curves(save_dir, epoch)
                self.visualize_predictions(save_dir, epoch)

            # Update learning rate scheduler
            if self.scheduler_type == 'plateau':
                self.scheduler.step(val_loss)
            elif self.scheduler_type == 'step':
                self.scheduler.step()

            # Record current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            if epoch % 10 == 0:  # Every 10 epochs display learning rate
                print(f'    Current learning rate: {current_lr:.2e}')

        print(f"Training complete! Final Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    def plot_loss_curves(self, save_dir: Path, epoch: int):
        """Plot loss curves"""
        if not self.training_history['steps']:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Step-wise training loss
        ax1.plot(self.training_history['steps'], self.training_history['train_losses'], 'b-', alpha=0.7)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Training Loss')
        ax1.set_title('Training Loss per Step')
        ax1.grid(True, alpha=0.3)

        # Epoch-wise loss comparison
        if self.training_history['epochs'] and self.training_history['val_losses']:
            # Calculate average training loss for each epoch
            steps_per_epoch = len(self.train_loader)
            epoch_train_losses = []
            
            for ep in self.training_history['epochs']:
                start_step = (ep - 1) * steps_per_epoch
                end_step = ep * steps_per_epoch
                epoch_losses = []
                
                for i, step in enumerate(self.training_history['steps']):
                    if start_step <= step < end_step:
                        epoch_losses.append(self.training_history['train_losses'][i])
                
                if epoch_losses:
                    epoch_train_losses.append(np.mean(epoch_losses))
                else:
                    epoch_train_losses.append(epoch_train_losses[-1] if epoch_train_losses else 0.0)
            
            ax2.plot(self.training_history['epochs'], epoch_train_losses, 'b-o', label='Training')
            ax2.plot(self.training_history['epochs'], self.training_history['val_losses'], 'r-s', label='Validation')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.set_title('Training vs Validation Loss')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = save_dir / f'loss_curves_epoch_{epoch}.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def visualize_predictions(self, save_dir: Path, epoch: int):
        """Visualize prediction results - supports 3 planes, randomly selects 3 samples"""
        self.model.eval()
        
        all_batches = []
        with torch.no_grad():
            for voxel_batch, distance_batch, model_ids in self.val_loader:
                for i in range(len(model_ids)):
                    all_batches.append((
                        voxel_batch[i:i+1].to(self.device),
                        distance_batch[i:i+1].to(self.device), 
                        model_ids[i]
                    ))
        
        num_samples = min(3, len(all_batches))
        if num_samples == 0:
            return
            
        import random
        selected_batches = random.sample(all_batches, num_samples)
        
        for idx, (voxel_batch, distance_batch, model_id) in enumerate(selected_batches):
            output = self.model(voxel_batch)
            
            plane1_params = output['p1'][0, 0].detach().cpu().numpy()  # [4]
            plane2_params = output['p2'][0, 0].detach().cpu().numpy()  # [4]  
            plane3_params = output['p3'][0, 0].detach().cpu().numpy()  # [4]
            
            all_plane_params = [plane1_params, plane2_params, plane3_params]
            
            loss, _ = self.compute_loss(output, voxel_batch, distance_batch)
            
            voxel_data = voxel_batch[0, 0].cpu().numpy()  # [32, 32, 32]
            
            self.create_3d_visualization(
                voxel_data, all_plane_params, model_id, 
                save_dir / f"prediction_epoch_{epoch}_sample_{idx+1}.html",
                f"Epoch {epoch} - Sample {idx+1} - 3 Planes Debug Test",
                loss.item()
            )
    
    def create_3d_visualization(self, voxel_data, all_plane_params, model_id, 
                               output_path, title, symmetry_loss):
        """Create 3D visualization - supports 3 planes"""
        # Get occupied voxel coordinates
        occupied_coords = np.where(voxel_data > 0.5)
        
        if len(occupied_coords[0]) == 0:
            return
            
        x, y, z = occupied_coords

        # Create voxel scatter plot
        voxel_trace = go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(size=4, color=z, colorscale='Viridis', opacity=0.7),
            name='体素'
        )

        # Create 3 symmetric planes
        plane_traces = []
        colors = ['Reds', 'Blues', 'Greens'] 
        plane_names = ['平面1', '平面2', '平面3']
        
        for i in range(3):
            plane_trace = self.create_single_plane_mesh(all_plane_params[i], colors[i], plane_names[i])
            if plane_trace:
                plane_traces.extend(plane_trace)
        
        traces = [voxel_trace] + plane_traces

        # Create layout - display parameter information for 3 planes
        plane_info = []
        for i in range(3):
            p = all_plane_params[i]
            plane_info.append(f"P{i+1}: [{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}, {p[3]:.3f}]")
        
        layout = go.Layout(
            title=f'{title}<br>模型: {model_id}<br>对称损失: {symmetry_loss:.6f}<br>' + '<br>'.join(plane_info),
            scene=dict(
                xaxis_title='X轴',
                yaxis_title='Y轴', 
                zaxis_title='Z轴',
                aspectmode='cube',
                xaxis=dict(range=[0, 32]),
                yaxis=dict(range=[0, 32]),
                zaxis=dict(range=[0, 32])
            ),
            width=1000,
            height=800
        )
        
        fig = go.Figure(data=traces, layout=layout)
        pyo.plot(fig, filename=str(output_path), auto_open=False)
    
    def create_single_plane_mesh(self, plane_params, colorscale='Reds', name='平面'):
        
        a, b, c, d = plane_params
        
        # check if valid
        if abs(a) < 1e-6 and abs(b) < 1e-6 and abs(c) < 1e-6:
            return []  # invalid
        
        try:
            # [-0.5, 0.5] -> [0, 31]
            if abs(c) > 1e-6:
                # solve z: z = -(ax + by + d) / c
                x_norm = np.linspace(-0.5, 0.5, 20)
                y_norm = np.linspace(-0.5, 0.5, 20)
                X_norm, Y_norm = np.meshgrid(x_norm, y_norm)
                Z_norm = -(a * X_norm + b * Y_norm + d) / c
                
                # -> [0, 31] 
                X = (X_norm + 0.5) * 31.0
                Y = (Y_norm + 0.5) * 31.0
                Z = (Z_norm + 0.5) * 31.0
                
            elif abs(b) > 1e-6:
                # solve y: y = -(ax + cz + d) / b
                x_norm = np.linspace(-0.5, 0.5, 20)
                z_norm = np.linspace(-0.5, 0.5, 20)
                X_norm, Z_norm = np.meshgrid(x_norm, z_norm)
                Y_norm = -(a * X_norm + c * Z_norm + d) / b
                
                X = (X_norm + 0.5) * 31.0
                Y = (Y_norm + 0.5) * 31.0
                Z = (Z_norm + 0.5) * 31.0
                
            else:
                # solve x: x = -(by + cz + d) / a
                y_norm = np.linspace(-0.5, 0.5, 20)
                z_norm = np.linspace(-0.5, 0.5, 20)
                Y_norm, Z_norm = np.meshgrid(y_norm, z_norm)
                X_norm = -(b * Y_norm + c * Z_norm + d) / a
                
                X = (X_norm + 0.5) * 31.0
                Y = (Y_norm + 0.5) * 31.0
                Z = (Z_norm + 0.5) * 31.0

            # Create surface mesh
            plane_surface = go.Surface(
                x=X, y=Y, z=Z,
                opacity=0.3,
                colorscale=colorscale,
                showscale=False,
                name=name
            )
            
            return [plane_surface]
        
        except:
            return []


    def save_model(self, save_dir: Path, epoch: int, loss: float = None):
   
        save_dir.mkdir(exist_ok=True)
        
        if loss is not None:
            model_path = save_dir / f'model_epoch_{epoch:04d}_loss_{loss:.6f}.pth'
        else:
            model_path = save_dir / f'model_epoch_{epoch:04d}.pth'
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),  
            'loss': loss,
            'reg_weight': self.reg_weight,
            'learning_rate': self.learning_rate,
            'training_history': self.training_history,  
            'hyperparameters': {  
                'batch_size': self.batch_size,
                'scheduler_type': self.scheduler_type,
            }
        }
        
        torch.save(checkpoint, model_path)
        print(f'Model saved to: {model_path}')
        
        latest_path = save_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, latest_path)
        print(f'Latest checkpoint saved to: {latest_path}')

    def load_model(self, checkpoint_path: str):
        """
        Load model weights
        Args:
            checkpoint_path: Path to the checkpoint file
        Returns:
            tuple: (epoch, loss) Restored training state
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']
        
        epoch = checkpoint.get('epoch', 0)
        loss = checkpoint.get('loss', None)

        print(f'Model loaded from: {checkpoint_path}')
        print(f'Restored state: Epoch={epoch}, Loss={loss:.6f}')
        print(f'Current learning rate: {self.optimizer.param_groups[0]["lr"]:.2e}')

        return epoch, loss

def main():

    toy_dataset_path = "./toy_data"
    train_dataset_path = "./train_data/ShapeNet-toydata"
    val_dataset_path = "./val_data/ShapeNet-toydata"

    # NOTE: if you need to use train.py, please change the dataset path to your own dataset.

    trainer = PRSNetTrainer(
        train_dataset_path=train_dataset_path,
        val_dataset_path=val_dataset_path,
        batch_size=16,          
        learning_rate=2e-3,    
        reg_weight=1.0,         
        device='cpu',
        scheduler_type='plateau',  # 'plateau' or 'step'
    )
    
    trainer.train(
        num_epochs=400, 
        save_dir=f'outputs_batch{trainer.batch_size}_lr{trainer.learning_rate}_reg{trainer.reg_weight}',
        resume_from=None
    )

    print("All things completed!")

if __name__ == "__main__":
    main()
