import torch
import random
import numpy as np
from pathlib import Path
from train import PRSNetTrainer

def inference_and_visualize(checkpoint_path: str, 
                          val_dataset_path: str,
                          output_dir: str = "inference_outputs",
                          num_samples: int = 5):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"=== Starting Inference ===")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Validation set: {val_dataset_path}")
    print(f"Output directory: {output_dir}")
    print(f"Visualization samples: {num_samples}")
    
    trainer = PRSNetTrainer(
        train_dataset_path=val_dataset_path,
        val_dataset_path=val_dataset_path,
        batch_size=1,
        device='cpu'
    )
    
    print(f"\nLoading model weights...")
    epoch, loss = trainer.load_model(checkpoint_path)
    print(f"Model loaded: Epoch {epoch}, Loss {loss:.6f}")
    
    trainer.model.eval()
    
    print(f"\nStarting inference...")
    
    all_batches = []
    all_losses = []
    all_symmetry_losses = []
    
    with torch.no_grad():
        for batch_idx, (voxel_batch, distance_batch, model_ids) in enumerate(trainer.val_loader):
            voxel_batch = voxel_batch.to(trainer.device)
            distance_batch = distance_batch.to(trainer.device)
            
            output = trainer.model(voxel_batch)
            
            loss, loss_dict = trainer.compute_loss(output, voxel_batch, distance_batch)
            
            for i in range(len(model_ids)):
                all_batches.append((
                    voxel_batch[i:i+1],
                    distance_batch[i:i+1], 
                    model_ids[i]
                ))
                all_losses.append(loss.item())
                all_symmetry_losses.append(loss_dict.get('symmetry_loss', 0.0))
    
    print(f"Validation set has {len(all_batches)} samples")
    
    if all_losses:
        avg_loss = np.mean(all_losses)
        std_loss = np.std(all_losses)
        min_loss = np.min(all_losses)
        max_loss = np.max(all_losses)
        
        print(f"\n=== Inference Statistics ===")
        print(f"Average total loss: {avg_loss:.6f} ± {std_loss:.6f}")
        print(f"Loss range: [{min_loss:.6f}, {max_loss:.6f}]")
        
        if all_symmetry_losses:
            avg_sym_loss = np.mean(all_symmetry_losses)
            print(f"Average symmetry loss: {avg_sym_loss:.6f}")
    
    num_samples = min(num_samples, len(all_batches))
    if num_samples == 0:
        print("Error: validation set is empty")
        return
        
    print(f"\nRandomly selecting {num_samples} samples for visualization...")
    selected_indices = random.sample(range(len(all_batches)), num_samples)
    selected_batches = [all_batches[i] for i in selected_indices]
    
    for idx, (voxel_batch, distance_batch, model_id) in enumerate(selected_batches):
        print(f"\nProcessing sample {idx+1}/{num_samples}: {model_id}")
        
        with torch.no_grad():
            output = trainer.model(voxel_batch)
        
        plane1_params = output['p1'][0, 0].detach().cpu().numpy()
        plane2_params = output['p2'][0, 0].detach().cpu().numpy()
        plane3_params = output['p3'][0, 0].detach().cpu().numpy()
        
        all_plane_params = [plane1_params, plane2_params, plane3_params]
        
        loss, loss_dict = trainer.compute_loss(output, voxel_batch, distance_batch)
        
        voxel_data = voxel_batch[0, 0].cpu().numpy()  # [32, 32, 32]
        
        output_path = output_dir / f"inference_sample_{idx+1}_{model_id}.html"
        title = f"Inference - Sample {idx+1} - 3 Planes"
        
        trainer.create_3d_visualization(
            voxel_data, 
            all_plane_params, 
            model_id, 
            output_path,
            title,
            loss.item()
        )
        
        print(f"  Total loss: {loss.item():.6f}")
        print(f"  Symmetry loss: {loss_dict.get('symmetry_loss', 0.0):.6f}")
        print(f"  Regularization loss: {loss_dict.get('regularization_loss', 0.0):.6f}")
        print(f"  Plane 1: [{plane1_params[0]:.3f}, {plane1_params[1]:.3f}, {plane1_params[2]:.3f}, {plane1_params[3]:.3f}]")
        print(f"  Plane 2: [{plane2_params[0]:.3f}, {plane2_params[1]:.3f}, {plane2_params[2]:.3f}, {plane2_params[3]:.3f}]")
        print(f"  Plane 3: [{plane3_params[0]:.3f}, {plane3_params[1]:.3f}, {plane3_params[2]:.3f}, {plane3_params[3]:.3f}]")
        print(f"  Visualization saved: {output_path}")
    
    print(f"\n=== Inference Complete ===")
    print(f"Processed {num_samples} samples")
    print(f"Visualization results saved in: {output_dir}")
    print("Open HTML files to view 3D visualizations")


def main():
    
    checkpoint_path = "./ckpts/model_epoch_0200_loss_0.082724.pth"
    val_dataset_path = "./val_data/ShapeNet-toydata"
    output_dir = "inference_outputs"
    num_samples = 5
    
    if not Path(checkpoint_path).exists():
        print(f"Error: checkpoint file does not exist: {checkpoint_path}")
        return
    
    print(f"Using specified checkpoint: {Path(checkpoint_path).name}")
    
    inference_and_visualize(
        checkpoint_path=checkpoint_path,
        val_dataset_path=val_dataset_path,
        output_dir=output_dir,
        num_samples=num_samples,
    )


if __name__ == "__main__":
    main()
