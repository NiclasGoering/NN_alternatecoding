"""
Training Module

This module exports training-related functions:
- train_with_parameterization: Main training loop
- initialize_parameterization: Initialize model weights per parameterization scheme
- parse_parameterization_name: Parse parameterization name to extract optimizer
- evaluate_loss: Evaluate loss on a dataset
"""
from src.training.trainer import (
    train_with_parameterization,
    evaluate_loss,
)
from src.training.initialization import (
    initialize_parameterization,
    parse_parameterization_name,
)

__all__ = [
    "train_with_parameterization",
    "evaluate_loss",
    "initialize_parameterization",
    "parse_parameterization_name",
]

