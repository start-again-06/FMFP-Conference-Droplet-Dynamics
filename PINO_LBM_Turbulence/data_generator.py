import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional
from lbm_solver import LBMSolver
from config import SimulationConfig, LBMConfig

class TurbulenceDataset(Dataset):
    """
    Dataset for turbulence modeling with multiple filter widths.
    """
    def __init__(self, data, targets, filter_widths=None):
        self.data = data
        self.targets = targets
        self.filter_widths = filter_widths
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]
        if self.filter_widths is not None:
            return x, y, self.filter_widths[idx]
        return x, y

class DataGenerator:
    """
    Generate training data for SGS turbulence modeling.
    """
    def __init__(self, config: SimulationConfig, lbm_config: LBMConfig):
        self.config = config
        self.lbm_config = lbm_config
        self.device = config.device if hasattr(config, 'device') else 'cpu'
    
    def apply_filter(self, f, filter_width, filter_type='gaussian'):
        """
        Apply spatial filter to distribution functions.
        
        Args:
            f: Distribution functions (Q x nx x ny x nz)
            filter_width: Filter width in grid points
            filter_type: 'gaussian' or 'box'
        
        Returns:
            Filtered distribution functions
        """
        # This is a simplified filtering implementation
        # For Gaussian filtering, we use a convolution with a Gaussian kernel
        import scipy.ndimage as ndi
        
        Q = f.shape[0]
        f_filtered = np.zeros_like(f)
        
        if filter_type == 'gaussian':
            sigma = filter_width / 2.0
            for i in range(Q):
                f_filtered[i] = ndi.gaussian_filter(f[i], sigma=sigma)
        elif filter_type == 'box':
            size = filter_width
            kernel = np.ones((size, size, size)) / (size**3)
            for i in range(Q):
                f_filtered[i] = ndi.convolve(f[i], kernel, mode='wrap')
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")
        
        return f_filtered
    
    def generate_training_pairs(self, n_samples, filter_widths=None, coarse_factor=2):
        """
        Generate filtered DNS pairs for training.
        
        Returns:
            X: Input coarse-grid distribution functions
            y: Target fine-grid distribution functions
            target_stresses: SGS stresses (optional)
        """
        if filter_widths is None:
            filter_widths = [2, 4, 6, 8]
        
        X_list = []
        y_list = []
        stress_list = []
        
        for sample_idx in range(n_samples):
            # Run DNS simulation
            lbm = LBMSolver(self.lbm_config)
            
            # Randomize initial conditions
            u0 = 0.05 + 0.15 * np.random.random()
            lbm.reset(u0=u0)
            
            # Run simulation with random forcing
            force_mag = 1e-6 * np.random.randn()
            force_phase = 2 * np.pi * np.random.random()
            n_steps = 500 + int(500 * np.random.random())
            
            snapshots = lbm.run_simulation(n_steps, save_every=1)
            
            # Select a random snapshot
            snapshot_idx = np.random.randint(len(snapshots) - 10)
            f_dns = snapshots[snapshot_idx]
            
            # Filter for different filter widths
            for fw in filter_widths:
                # Coarse-grain the data
                f_filtered = self.apply_filter(f_dns, fw)
                
                # Sub-sample to coarse grid
                f_coarse = f_filtered[:, ::coarse_factor, ::coarse_factor, ::coarse_factor]
                
                # Generate target (could be filtered DNS or SGS stresses)
                f_target = f_filtered  # For now, use filtered DNS as target
                
                X_list.append(f_coarse)
                y_list.append(f_target)
        
        X = torch.stack([torch.from_numpy(x).float() for x in X_list])
        y = torch.stack([torch.from_numpy(y).float() for y in y_list])
        
        return X, y
    
    def generate_training_data(self, n_samples=1000):
        """
        Generate synthetic training data for PINO-LBM.
        """
        print(f"Generating {n_samples} training samples...")
        
        X_list = []
        y_list = []
        
        for i in range(n_samples):
            # Run LBM simulation with random conditions
            lbm = LBMSolver(self.lbm_config)
            u0 = 0.05 + 0.15 * np.random.random()
            tau = 0.55 + 0.15 * np.random.random()
            lbm.reset(u0=u0)
            lbm.state.tau = tau
            
            # Random forcing
            force_x = 1e-6 * np.random.randn()
            force_y = 1e-6 * np.random.randn()
            force_z = 1e-6 * np.random.randn()
            
            # Evolve to developed state
            for _ in range(200):
                lbm.step(force_x, force_y, force_z)
            
            # Store initial state
            f0 = lbm.get_distributions()
            
            # Evolve forward
            n_steps = 10 + int(20 * np.random.random())
            for _ in range(n_steps):
                lbm.step(force_x, force_y, force_z)
            
            f1 = lbm.get_distributions()
            
            X_list.append(f0)
            y_list.append(f1)
        
        X = torch.stack(X_list)
        y = torch.stack(y_list)
        
        print(f"Generated data shape: X={X.shape}, y={y.shape}")
        return X, y
    
    def get_dataloaders(self, X, y, batch_size=32, train_ratio=0.8):
        """Create train/validation dataloaders."""
        n_samples = len(X)
        n_train = int(n_samples * train_ratio)
        
        # Shuffle
        indices = np.random.permutation(n_samples)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        # Create datasets
        train_dataset = TurbulenceDataset(X[train_indices], y[train_indices])
        val_dataset = TurbulenceDataset(X[val_indices], y[val_indices])
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )
        
        return train_loader, val_loader
