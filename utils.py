"""
Utility functions for the time series imputation project.
"""

import os
import torch
import numpy as np
import random


def set_seed(seed):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_model(model, path, filename="model.pth"):
    """
    Save model state dict.
    
    Args:
        model: Model to save
        path: Directory path
        filename: Model filename
    """
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, filename))


def load_model(model, path, filename="model.pth", device='cpu'):
    """
    Load model state dict.
    
    Args:
        model: Model to load weights into
        path: Directory path
        filename: Model filename
        device: Device to load to
        
    Returns:
        Model with loaded weights
    """
    model.load_state_dict(torch.load(os.path.join(path, filename), map_location=device))
    return model


def count_parameters(model):
    """
    Count trainable parameters in model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def log_message(message, log_file=None, print_msg=True):
    """
    Log message to file and/or console.
    
    Args:
        message: Message to log
        log_file: Path to log file (optional)
        print_msg: Whether to print to console
    """
    if print_msg:
        print(message)
    if log_file:
        with open(log_file, 'a') as f:
            f.write(message + '\n')


def compute_metrics(gt, pred, mask):
    """
    Compute imputation metrics.
    
    Args:
        gt: Ground truth values
        pred: Predicted values
        mask: Boolean mask (True=observed, False=missing)
        
    Returns:
        Dictionary of metrics
    """
    # Only compute on missing positions
    missing_mask = ~mask
    
    if isinstance(gt, torch.Tensor):
        gt = gt.cpu().numpy()
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    gt_missing = gt[missing_mask]
    pred_missing = pred[missing_mask]
    
    mse = np.mean((gt_missing - pred_missing) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(gt_missing - pred_missing))
    
    # Avoid division by zero
    gt_range = np.max(gt_missing) - np.min(gt_missing) if len(gt_missing) > 0 else 1.0
    nrmse = rmse / (gt_range + 1e-8)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'nrmse': nrmse
    }
