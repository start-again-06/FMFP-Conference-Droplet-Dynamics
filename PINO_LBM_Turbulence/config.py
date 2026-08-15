import torch
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional

@dataclass
class LBMConfig:
    """Lattice Boltzmann Method configuration."""
    nx: int = 64
    ny: int = 64
    nz: int = 64  # For 3D simulations
    tau: float = 0.6  # Relaxation time
    rho0: float = 1.0
    u0: float = 0.1
    Re: float = 6000.0
    nu: float = 0.001  # Kinematic viscosity
    dt: float = 1.0
    dx: float = 1.0
    lattice_type: str = 'D3Q19'  # or 'D2Q9'

@dataclass
class ModelConfig:
    """Neural Operator model configuration."""
    in_channels: int = 19  # D3Q19
    out_channels: int = 19
    hidden_channels: int = 64
    modes1: int = 8
    modes2: int = 8
    modes3: int = 8  # For 3D
    n_layers: int = 3
    activation: str = 'gelu'
    use_batch_norm: bool = True

@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 32
    epochs: int = 200
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    lambda_mom: float = 1.0  # Moment-matching weight
    lambda_equiv: float = 0.1  # Equivariance weight
    lambda_pi: float = 0.4  # SGS energy transfer weight
    lambda_sym: float = 0.05  # Symbolic regression weight
    num_workers: int = 4
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_interval: int = 10
    log_interval: int = 10

@dataclass
class SimulationConfig:
    """Simulation configuration."""
    grid_size: Tuple[int, int, int] = (64, 64, 64)
    time_steps: int = 1000
    save_every: int = 10
    n_train_samples: int = 2000
    n_val_samples: int = 500
    n_test_samples: int = 200
    filter_width: int = 4
    coarse_graining_factor: int = 2
    random_seed: int = 42

def get_default_configs():
    """Return default configurations for all components."""
    return {
        'lbm': LBMConfig(),
        'model': ModelConfig(),
        'training': TrainingConfig(),
        'simulation': SimulationConfig()
    }
