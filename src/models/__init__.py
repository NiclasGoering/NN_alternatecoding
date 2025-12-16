"""
Neural Network Models

This module exports all model architectures:
- MLP: Standard feedforward MLP (from src.data.models.ffnn)
- MLPResNet: MLP with skip connections and batch normalization
- MLPBatchNorm: MLP with batch normalization only
- LazarusMLP: Deep residual MLP with depth-aware scaling initialization
"""
from src.data.models.ffnn import MLP
from src.models.lazarus import LazarusMLP
from src.models.resnet import MLPResNet
from src.models.batchnorm import MLPBatchNorm

__all__ = [
    "MLP",
    "MLPResNet", 
    "MLPBatchNorm",
    "LazarusMLP",
]

