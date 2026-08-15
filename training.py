import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from typing import Dict, Optional
import os
import json

class PINOTrainer:
    """
    Trainer for Physics-Informed Neural Operator.
    """
    def __init__(self, model, train_loader, val_loader, config, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=20, factor=0.5
        )
        
        self.loss_fn = PINOLoss(
            lambda_mom=config.lambda_mom,
            lambda_equiv=config.lambda_equiv,
            lambda_pi=config.lambda_pi
        )
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_data': [],
            'train_mom': [],
            'train_equiv': [],
            'train_sgs': [],
            'lr': []
        }
        
        self.best_val_loss = float('inf')
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        epoch_data = 0.0
        epoch_mom = 0.0
        epoch_equiv = 0.0
        epoch_sgs = 0.0
        
        pbar = tqdm(self.train_loader, desc='Training', leave=False)
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(self.device), y.to(self.device)
            
            self.optimizer.zero_grad()
            y_pred = self.model(x)
            
            # Compute macroscopic variables from predicted distribution
            # Simplified: using zeros for demonstration
            ux = torch.zeros_like(x[:, 0])
            uy = torch.zeros_like(x[:, 0])
            uz = torch.zeros_like(x[:, 0])
            
            loss, components = self.loss_fn(self.model, y_pred, y, ux, uy, uz)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            epoch_loss += components['total']
            epoch_data += components['data']
            epoch_mom += components['moment']
            epoch_equiv += components['equivariance']
            epoch_sgs += components['sgs_transfer']
            
            pbar.set_postfix({
                'loss': f"{components['total']:.6f}",
                'data': f"{components['data']:.6f}",
                'mom': f"{components['moment']:.6f}"
            })
        
        n_batches = len(self.train_loader)
        return {
            'loss': epoch_loss / n_batches,
            'data': epoch_data / n_batches,
            'moment': epoch_mom / n_batches,
            'equivariance': epoch_equiv / n_batches,
            'sgs': epoch_sgs / n_batches
        }
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                y_pred = self.model(x)
                
                ux = torch.zeros_like(x[:, 0])
                uy = torch.zeros_like(x[:, 0])
                uz = torch.zeros_like(x[:, 0])
                
                loss, _ = self.loss_fn(self.model, y_pred, y, ux, uy, uz)
                val_loss += loss.item()
        
        return val_loss / len(self.val_loader)
    
    def train(self, epochs):
        """Full training loop."""
        print(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Save history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_loss)
            self.history['train_data'].append(train_metrics['data'])
            self.history['train_mom'].append(train_metrics['moment'])
            self.history['train_equiv'].append(train_metrics['equivariance'])
            self.history['train_sgs'].append(train_metrics['sgs'])
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            # Print progress
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Loss: {train_metrics['loss']:.6f}, "
                  f"Val Loss: {val_loss:.6f}, "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best_model.pth')
            
            # Regular checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')
        
        return self.history
    
    def save_checkpoint(self, filename):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        torch.save(checkpoint, filename)
        print(f"Checkpoint saved to {filename}")
    
    def load_checkpoint(self, filename):
        """Load model checkpoint."""
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        self.best_val_loss = checkpoint['best_val_loss']
        print(f"Checkpoint loaded from {filename}")
        return checkpoint['config']
