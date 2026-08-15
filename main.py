import torch
import numpy as np
import argparse
from config import get_default_configs
from lbm_solver import LBMSolver
from data_generator import DataGenerator
from neural_operators.uno import UNO3D
from training import PINOTrainer
from inference import PINOInference
from visualization import Visualizer
import os

def parse_args():
    parser = argparse.ArgumentParser(description='PINO-LBM Turbulence Modeling')
    parser.add_argument('--mode', choices=['train', 'inference', 'visualize'], default='train',
                        help='Execution mode')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint file')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--save_dir', type=str, default='results',
                        help='Directory to save results')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load configurations
    configs = get_default_configs()
    lbm_config = configs['lbm']
    model_config = configs['model']
    training_config = configs['training']
    sim_config = configs['simulation']
    
    # Update configs from args
    training_config.epochs = args.epochs
    training_config.batch_size = args.batch_size
    training_config.device = args.device
    
    print("=" * 60)
    print("PINO-LBM: Physics-Informed Neural Operator for LBM Turbulence")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Mode: {args.mode}")
    print(f"  Device: {args.device}")
    print(f"  Grid: {sim_config.grid_size}")
    print(f"  Epochs: {training_config.epochs}")
    print(f"  Batch Size: {training_config.batch_size}")
    
    # Initialize data generator
    data_gen = DataGenerator(sim_config, lbm_config)
    
    if args.mode == 'train':
        # Generate data
        print("\n" + "-" * 40)
        print("Generating training data...")
        X, y = data_gen.generate_training_data(sim_config.n_train_samples)
        
        # Create dataloaders
        train_loader, val_loader = data_gen.get_dataloaders(
            X, y, batch_size=training_config.batch_size
        )
        
        # Initialize model
        print("\nInitializing UNO3D model...")
        model = UNO3D(
            in_channels=model_config.in_channels,
            out_channels=model_config.out_channels,
            hidden_channels=model_config.hidden_channels,
            modes1=model_config.modes1,
            modes2=model_config.modes2,
            modes3=model_config.modes3
        )
        
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {n_params:,}")
        
        # Train
        print("\n" + "-" * 40)
        print("Starting training...")
        trainer = PINOTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=training_config,
            device=args.device
        )
        
        history = trainer.train(training_config.epochs)
        
        # Visualize results
        print("\n" + "-" * 40)
        print("Generating visualizations...")
        visualizer = Visualizer(save_dir=args.save_dir)
        visualizer.plot_training_history(history)
        
        # Test on validation set
        model.eval()
        X_val, y_val = next(iter(val_loader))
        X_val, y_val = X_val.to(args.device), y_val.to(args.device)
        y_pred = model(X_val)
        
        visualizer.plot_prediction(
            X_val[0].cpu(),
            y_val[0].cpu(),
            y_pred[0].cpu()
        )
        
        print(f"\nTraining complete! Results saved to {args.save_dir}")
    
    elif args.mode == 'inference':
        # Load checkpoint
        if args.checkpoint is None:
            print("Error: Checkpoint required for inference mode")
            return
        
        print(f"\nLoading checkpoint from {args.checkpoint}")
        model = UNO3D(
            in_channels=model_config.in_channels,
            out_channels=model_config.out_channels,
            hidden_channels=model_config.hidden_channels
        )
        model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
        
        # Initialize inference engine
        infer = PINOInference(model, model_config, device=args.device)
        
        # Run inference
        print("\nRunning inference...")
        lbm = LBMSolver(lbm_config)
        lbm.reset()
        
        # Run accelerated simulation
        snapshots = infer.run_rollout(lbm, n_steps=100, time_jump=10)
        
        print(f"Generated {len(snapshots)} prediction snapshots")
        
        # Visualize
        visualizer = Visualizer(save_dir=args.save_dir)
        for i, snap in enumerate(snapshots[:5]):
            visualizer.plot_prediction(
                snap, snap, snap,  # Using same for demonstration
                save_name=f'inference_{i}.png'
            )
    
    elif args.mode == 'visualize':
        # Load results and visualize
        print(f"\nVisualizing results from {args.save_dir}")
        
        # This would load saved results and generate plots
        # For now, we just demonstrate the functionality
        visualizer = Visualizer(save_dir=args.save_dir)
        
        # Example: Create dummy data
        k = np.logspace(0, 2, 50)
        E_dns = k**(-5/3)
        E_pred = k**(-5/3) * (1 + 0.1 * np.random.randn(len(k)))
        E_smag = k**(-5/3) * 0.8
        
        visualizer.plot_energy_spectrum(k, E_dns, E_pred, E_smag)

if __name__ == "__main__":
    main()
