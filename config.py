"""
Configuration and argument parsing for the time series imputation model.
"""

import argparse
import torch
import os


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Multi-scale Transformer for Time Series Imputation")
    
    # Data paths
    parser.add_argument("--data_path", required=True, type=str, help="Path to the input CSV data file")
    parser.add_argument("--output_path", required=True, type=str, help="Path to save outputs")
    
    # Model architecture
    parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--nhead", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=350, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    # Sequence parameters
    parser.add_argument("--seq_len", type=int, default=64, help="Input sequence length")
    parser.add_argument("--pred_len", type=int, default=0, help="Prediction length for forecasting")
    
    # Masking parameters
    parser.add_argument("--lm", type=int, default=10, help="Mean length of masked segments")
    parser.add_argument("--missing_rate", type=float, default=0.25, help="Rate of missing values")
    parser.add_argument("--missing_type", type=int, default=0, choices=[0, 1, 2],
                        help="0: Random missing, 1: Gaussian missing, 2: Prediction missing")
    
    # Tokenization parameters
    parser.add_argument("--token_t_size", type=int, default=8, help="Temporal token size")
    parser.add_argument("--token_t_overlap", type=int, default=0, help="Temporal token overlap")
    parser.add_argument("--token_d_size", type=int, default=8, help="Feature dimension token size")
    parser.add_argument("--token_d_overlap", type=int, default=0, help="Feature dimension token overlap")
    
    # Loss parameters
    parser.add_argument("--loss_r", type=float, default=0.65, help="Loss weight ratio for masked vs unmasked")
    
    args = parser.parse_args()
    return args


def get_device():
    """Get the device to use for training."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def setup_output_dirs(output_path):
    """Create output directories."""
    os.makedirs(output_path, exist_ok=True)
    multi_output_path = os.path.join(output_path, "multi")
    os.makedirs(multi_output_path, exist_ok=True)
    return multi_output_path
