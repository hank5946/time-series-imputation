"""
Data loading and preprocessing utilities for time series imputation.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from mask import (
    generate_mask_matrix,
    generate_mask_matrix_from_paper,
    generate_prediction_mask,
    generate_prediction_mask_fixed
)


def load_and_normalize_data(data_path):
    """
    Load time series data from CSV and normalize.
    
    Args:
        data_path: Path to CSV file with 'date' column and feature columns
        
    Returns:
        data: Normalized numpy array (T, D)
        timestamps: Array of timestamps
        data_mean: Mean values per feature (for denormalization)
        data_std: Standard deviation per feature (for denormalization)
    """
    df = pd.read_csv(data_path, parse_dates=["date"])
    data = df.iloc[:, 1:].values.astype(np.float32)  # multivariate data
    timestamps = df["date"].values
    
    # Normalize for each column
    data_mean = np.mean(data, axis=0)
    data_std = np.std(data, axis=0)
    data_normalized = (data - data_mean) / data_std
    
    return data_normalized, timestamps, data_mean, data_std


def generate_masks(data, missing_type, missing_rate, lm, seq_len, seed=42):
    """
    Generate mask matrix based on missing type.
    
    Args:
        data: Normalized data array (T, D)
        missing_type: 0=Random, 1=Gaussian, 2=Prediction
        missing_rate: Rate of missing values
        lm: Mean length of masked segments
        seq_len: Sequence length for prediction mask
        seed: Random seed
        
    Returns:
        mask_matrix: Boolean mask array (T, D), True=observed, False=missing
    """
    T, D = data.shape
    mask_matrix = np.ones((T, D), dtype=bool)
    
    for d in range(D):
        if missing_type == 0:
            mask_matrix[:, d] = generate_mask_matrix(T, missing_rate)
        elif missing_type == 1:
            mask_matrix[:, d] = generate_mask_matrix_from_paper(T, lm=lm, r=missing_rate, seed=seed + d)
        elif missing_type == 2:
            mask_matrix[:, d] = generate_prediction_mask(T, seq_len=seq_len, pred_len=lm)
    
    return mask_matrix


def apply_mask(data, mask_matrix):
    """
    Apply mask to data, setting missing values to 0.
    
    Args:
        data: Data array (T, D)
        mask_matrix: Boolean mask array (T, D)
        
    Returns:
        masked_data: Data with missing values set to 0
    """
    masked_data = data.copy()
    masked_data[~mask_matrix] = 0.0
    return masked_data


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series imputation."""
    
    def __init__(self, data, masked_data, mask, seq_len, missing_type=0, pred_len=0):
        """
        Args:
            data: Original normalized data (T, D)
            masked_data: Data with masks applied (T, D)
            mask: Mask matrix (T, D)
            seq_len: Sequence length for each sample
            missing_type: Type of missing pattern
            pred_len: Prediction length (for missing_type=2)
        """
        self.data = data
        self.masked_data = masked_data
        self.mask = mask
        self.seq_len = seq_len
        self.missing_type = missing_type
        self.pred_len = pred_len
        self.D = data.shape[1]

    def __len__(self):
        return self.data.shape[0] - self.seq_len

    def __getitem__(self, idx):
        if self.missing_type == 2:
            # Generate prediction mask on-the-fly
            mask = np.zeros((self.seq_len, self.D), dtype=bool)
            for d in range(self.D):
                mask[:, d] = generate_prediction_mask_fixed(seq_len=self.seq_len, pred_len=self.pred_len)
        else:
            mask = self.mask[idx:idx + self.seq_len]
        
        return (
            self.data[idx:idx + self.seq_len],
            self.masked_data[idx:idx + self.seq_len],
            mask
        )


def create_data_loaders(data, masked_data, mask_matrix, args, seed=42):
    """
    Create train, validation, and test data loaders.
    
    Args:
        data: Normalized data array
        masked_data: Masked data array
        mask_matrix: Mask matrix
        args: Arguments containing seq_len, batch_size, missing_type, pred_len
        seed: Random seed for splitting
        
    Returns:
        train_loader, val_loader, test_loader
    """
    full_dataset = TimeSeriesDataset(
        data, masked_data, mask_matrix, 
        seq_len=args.seq_len,
        missing_type=args.missing_type,
        pred_len=args.pred_len
    )
    
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_set, val_set, test_set = random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, drop_last=True)
    
    return train_loader, val_loader, test_loader


def denormalize(x, mean, std):
    """
    Denormalize data back to original scale.
    
    Args:
        x: Normalized data (can be 1D, 2D, or 3D)
        mean: Mean values per feature
        std: Standard deviation per feature
        
    Returns:
        Denormalized data
    """
    mean = np.asarray(mean)
    std = np.asarray(std)
    
    if x.ndim == 1:
        if x.shape[0] != mean.shape[0]:
            raise ValueError(f"Expected x.shape[0] == mean.shape[0], got {x.shape[0]} and {mean.shape[0]}")
        return x * std + mean
    elif x.ndim == 2:
        if x.shape[1] != mean.shape[0]:
            raise ValueError(f"Expected x.shape[1] == mean.shape[0], got {x.shape[1]} and {mean.shape[0]}")
        return x * std[np.newaxis, :] + mean[np.newaxis, :]
    elif x.ndim == 3:
        if x.shape[2] != mean.shape[0]:
            raise ValueError(f"Expected x.shape[2] == mean.shape[0], got {x.shape[2]} and {mean.shape[0]}")
        return x * std[np.newaxis, np.newaxis, :] + mean[np.newaxis, np.newaxis, :]
    else:
        raise ValueError(f"Unsupported x shape: {x.shape}")
