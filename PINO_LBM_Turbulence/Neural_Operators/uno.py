import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class SpectralConv3d(nn.Module):
    """
    3D Spectral Convolution layer (Fourier Neural Operator).
    """
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat)
        )
    
    def forward(self, x):
        batchsize, channels, height, width, depth = x.shape
        
        # Fourier transform
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1], norm='ortho')
        
        # Multiply by learned weights in Fourier space
        out_ft = torch.zeros_like(x_ft)
        
        # Low-frequency modes
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = torch.einsum(
            "bixyz,ioxyz->boxyz",
            x_ft[:, :, :self.modes1, :self.modes2, :self.modes3],
            self.weights1
        )
        
        # High-frequency modes (aliased)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = torch.einsum(
            "bixyz,ioxyz->boxyz",
            x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3],
            self.weights2
        )
        
        # Inverse Fourier transform
        x = torch.fft.irfftn(out_ft, s=(height, width, depth), dim=[-3, -2, -1], norm='ortho')
        return x

class UNOBlock3D(nn.Module):
    """U-shaped Neural Operator block for 3D data."""
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.GELU()
        )
        self.spectral = SpectralConv3d(out_channels, out_channels, modes1, modes2, modes3)
        self.decoder = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.GELU()
        )
        self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        identity = self.skip(x)
        x = self.encoder(x)
        x = self.spectral(x)
        x = self.decoder(x)
        return x + identity

class UNO3D(nn.Module):
    """
    3D U-shaped Neural Operator for LBM surrogate modeling.
    
    Architecture:
    - Encoder: UNOBlock with downsampling
    - Bottleneck: UNOBlock at lowest resolution
    - Decoder: UNOBlock with upsampling and skip connections
    """
    def __init__(self, in_channels=19, out_channels=19, hidden_channels=64,
                 modes1=8, modes2=8, modes3=8):
        super().__init__()
        
        # Encoder
        self.enc1 = UNOBlock3D(in_channels, hidden_channels, modes1, modes2, modes3)
        self.pool1 = nn.MaxPool3d(2)
        
        self.enc2 = UNOBlock3D(hidden_channels, hidden_channels * 2, modes1//2, modes2//2, modes3//2)
        self.pool2 = nn.MaxPool3d(2)
        
        self.enc3 = UNOBlock3D(hidden_channels * 2, hidden_channels * 4, modes1//4, modes2//4, modes3//4)
        
        # Decoder
        self.dec3 = UNOBlock3D(hidden_channels * 4, hidden_channels * 2, modes1//4, modes2//4, modes3//4)
        self.dec2 = UNOBlock3D(hidden_channels * 4, hidden_channels, modes1//2, modes2//2, modes3//2)
        self.dec1 = UNOBlock3D(hidden_channels * 2, out_channels, modes1, modes2, modes3)
        
        # Upsampling
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        
        # Decoder with skip connections
        d3 = self.dec3(e3)
        d2 = self.dec2(torch.cat([self.upsample(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.upsample(d2), e1], dim=1))
        
        return d1
