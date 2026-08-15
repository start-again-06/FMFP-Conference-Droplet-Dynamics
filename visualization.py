import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import List, Optional, Tuple
from mpl_toolkits.mplot3d import Axes3D

class Visualizer:
    """
    Visualization utilities for PINO-LBM results.
    """
    def __init__(self, save_dir='results'):
        self.save_dir = save_dir
        import os
        os.makedirs(save_dir, exist_ok=True)
    
    def plot_training_history(self, history, save_name='training_history.png'):
        """Plot training and validation loss curves."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Total loss
        axes[0, 0].plot(history['train_loss'], label='Train Loss')
        axes[0, 0].plot(history['val_loss'], label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss components
        axes[0, 1].plot(history['train_data'], label='Data Loss')
        axes[0, 1].plot(history['train_mom'], label='Moment Loss')
        axes[0, 1].plot(history['train_equiv'], label='Equivariance Loss')
        axes[0, 1].plot(history['train_sgs'], label='SGS Transfer Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Loss Components')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate
        axes[0, 2].plot(history['lr'])
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Learning Rate')
        axes[0, 2].set_title('Learning Rate Schedule')
        axes[0, 2].grid(True)
        
        # Validation loss (log scale)
        axes[1, 0].semilogy(history['val_loss'])
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Validation Loss')
        axes[1, 0].set_title('Validation Loss (Log Scale)')
        axes[1, 0].grid(True)
        
        # Train vs Validation
        axes[1, 1].plot(history['train_loss'], label='Train')
        axes[1, 1].plot(history['val_loss'], label='Val')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('Train vs Validation')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        # Loss ratios
        axes[1, 2].plot(np.array(history['train_mom']) / np.array(history['train_data']), label='Mom/Data')
        axes[1, 2].plot(np.array(history['train_equiv']) / np.array(history['train_data']), label='Equiv/Data')
        axes[1, 2].plot(np.array(history['train_sgs']) / np.array(history['train_data']), label='SGS/Data')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Loss Ratio')
        axes[1, 2].set_title('Loss Component Ratios')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/{save_name}", dpi=150)
        plt.close()
    
    def plot_prediction(self, f_input, f_true, f_pred, save_name='prediction.png'):
        """Plot prediction results."""
        # Compute macroscopic quantities
        def compute_rho(f):
            if isinstance(f, torch.Tensor):
                f = f.numpy()
            # Simplified: sum over velocity components
            return np.sum(f, axis=0)
        
        rho_input = compute_rho(f_input)
        rho_true = compute_rho(f_true)
        rho_pred = compute_rho(f_pred)
        rho_error = np.abs(rho_true - rho_pred)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Input density
        im0 = axes[0, 0].imshow(rho_input[0] if len(rho_input.shape) > 2 else rho_input, cmap='viridis')
        axes[0, 0].set_title('Input Density')
        axes[0, 0].axis('off')
        plt.colorbar(im0, ax=axes[0, 0])
        
        # True density
        im1 = axes[0, 1].imshow(rho_true[0] if len(rho_true.shape) > 2 else rho_true, cmap='viridis')
        axes[0, 1].set_title('True Density')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1])
        
        # Predicted density
        im2 = axes[0, 2].imshow(rho_pred[0] if len(rho_pred.shape) > 2 else rho_pred, cmap='viridis')
        axes[0, 2].set_title('Predicted Density')
        axes[0, 2].axis('off')
        plt.colorbar(im2, ax=axes[0, 2])
        
        # Error
        im3 = axes[1, 0].imshow(rho_error[0] if len(rho_error.shape) > 2 else rho_error, cmap='hot')
        axes[1, 0].set_title('Absolute Error')
        axes[1, 0].axis('off')
        plt.colorbar(im3, ax=axes[1, 0])
        
        # Distribution component 0
        im4 = axes[1, 1].imshow(f_true[0, 0] if len(f_true.shape) > 3 else f_true[0], cmap='viridis')
        axes[1, 1].set_title('True f0')
        axes[1, 1].axis('off')
        plt.colorbar(im4, ax=axes[1, 1])
        
        # Distribution component 0 predicted
        im5 = axes[1, 2].imshow(f_pred[0, 0] if len(f_pred.shape) > 3 else f_pred[0], cmap='viridis')
        axes[1, 2].set_title('Predicted f0')
        axes[1, 2].axis('off')
        plt.colorbar(im5, ax=axes[1, 2])
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/{save_name}", dpi=150)
        plt.close()
    
    def plot_energy_spectrum(self, k, E_dns, E_pred, E_smag=None, save_name='energy_spectrum.png'):
        """Plot energy spectrum comparison."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.loglog(k, E_dns, 'b-', label='DNS', linewidth=2)
        ax.loglog(k, E_pred, 'r-', label='PINO-LBM', linewidth=2)
        if E_smag is not None:
            ax.loglog(k, E_smag, 'g--', label='Smagorinsky', linewidth=2)
        
        # Kolmogorov scaling
        k_range = k[k > 1]
        E_kol = k_range**(-5/3)
        E_kol = E_kol / E_kol[0] * E_dns[0]
        ax.loglog(k_range, E_kol, 'k:', label='k^{-5/3}', alpha=0.5)
        
        ax.set_xlabel('Wavenumber k')
        ax.set_ylabel('Energy Spectrum E(k)')
        ax.set_title('Energy Spectrum Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/{save_name}", dpi=150)
        plt.close()
    
    def plot_structure_functions(self, r, S_dns, S_pred, S_smag=None, save_name='structure_functions.png'):
        """Plot structure functions comparison."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot for multiple orders
        colors = plt.cm.viridis(np.linspace(0, 1, 6))
        for p, color in enumerate(colors[:3], 1):
            ax.loglog(r, S_dns[p-1], '--', color=color, alpha=0.5)
            ax.loglog(r, S_pred[p-1], '-', color=color, label=f'S^{p}')
        
        ax.set_xlabel('Scale r')
        ax.set_ylabel('Structure Function S^p(r)')
        ax.set_title('Structure Functions Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/{save_name}", dpi=150)
        plt.close()
    
    def plot_sgs_transfer(self, pi_dns, pi_pred, save_name='sgs_transfer.png'):
        """Plot SGS energy transfer distribution."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Histogram
        bins = np.linspace(min(pi_dns.min(), pi_pred.min()), max(pi_dns.max(), pi_pred.max()), 50)
        ax.hist(pi_dns.flatten(), bins=bins, density=True, alpha=0.5, label='DNS')
        ax.hist(pi_pred.flatten(), bins=bins, density=True, alpha=0.5, label='PINO-LBM')
        
        ax.set_xlabel('SGS Energy Transfer $\Pi$')
        ax.set_ylabel('Probability Density')
        ax.set_title('SGS Energy Transfer Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/{save_name}", dpi=150)
        plt.close()
