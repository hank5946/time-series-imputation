"""
Computation utilities for correlation matrix and loss functions.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# Rule based correlation matrix computation
def compute_correlation_and_lag_matrix(X_np, mask_np=None, alpha=0.5, gamma=0.5, max_lag=10, min_corr_threshold=0.1):
    """
    Compute correlation and lag matrix from multivariate time series using:
    - Pearson correlation (masked)
    - Autoregressive influence via lagged cross-correlation

    Args:
        X_np: (T, D) numpy array
        mask_np: (T, D) numpy array (1 if observed, 0 if missing) or None
        alpha: weight for Pearson correlation
        gamma: weight for lagged AR correlation
        max_lag: max shift (positive or negative) for lag estimation

    Returns:
        C_combined: torch.Tensor (D, D)
        lag_matrix: torch.IntTensor (D, D)
    """
    T, D = X_np.shape

    # Initialize outputs
    C_pearson = np.zeros((D, D))
    C_ar = np.zeros((D, D))
    lag_matrix = np.zeros((D, D), dtype=int)

    # Preprocessing: Fill missing with interpolation for AR correlation
    if mask_np is not None:
        X_df = pd.DataFrame(np.where(mask_np == 1, X_np, np.nan))
        X_filled = X_df.interpolate(limit_direction='both').bfill().ffill().values
    else:
        X_filled = X_np.copy()

    # Normalize
    X_filled = (X_filled - X_filled.mean(axis=0)) / (X_filled.std(axis=0) + 1e-6)

    for i in range(D):
        for j in range(D):
            xi = X_filled[:, i]
            xj = X_filled[:, j]

            # Pearson (on masked input)
            if mask_np is not None:
                valid = (mask_np[:, i] == 1) & (mask_np[:, j] == 1)
                if valid.sum() > 1:
                    C_pearson[i, j] = np.corrcoef(X_np[valid, i], X_np[valid, j])[0, 1]
                else:
                    C_pearson[i, j] = 0
            else:
                C_pearson[i, j] = np.corrcoef(xi, xj)[0, 1]

            # Lagged AR correlation via cross-correlation
            lags = np.arange(-max_lag, max_lag + 1)
            corrs = []
            for lag in lags:
                if lag > 0:
                    xi_lag = xi[lag:]
                    xj_lag = xj[:-lag]
                elif lag < 0:
                    xi_lag = xi[:lag]
                    xj_lag = xj[-lag:]
                else:
                    xi_lag = xi
                    xj_lag = xj

                if len(xi_lag) > 1:
                    corr = np.corrcoef(xi_lag, xj_lag)[0, 1]
                else:
                    corr = 0.0
                corrs.append(corr)

            corrs = np.nan_to_num(corrs)
            best_lag_idx = np.argmax(np.abs(corrs))
            best_corr = corrs[best_lag_idx]
            best_lag = lags[best_lag_idx]

            C_ar[i, j] = np.abs(best_corr)
            if np.abs(best_corr) >= min_corr_threshold:
                lag_matrix[i, j] = best_lag
            else:
                lag_matrix[i, j] = 0  # ignore weak lag influence

    np.fill_diagonal(C_pearson, 1.0)
    C_pearson = np.nan_to_num(C_pearson)

    # Combine Pearson and AR correlation
    C_combined = alpha * np.abs(C_pearson) + gamma * C_ar
    C_combined /= np.max(C_combined) + 1e-6

    return torch.tensor(C_combined, dtype=torch.float32), torch.tensor(lag_matrix, dtype=torch.int32)

# criterion
class MaskedWeightedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')  # element-wise loss

    def forward(self, out, gt, mask, r):
        mse_loss = self.mse(out, gt)  # shape: [batch, seq_len, ...]
        
        # Convert mask to float for computation
        mask = mask.float()
        
        # Masked positions (0 in mask)
        masked_loss = mse_loss * (1 - mask) * r * 2
        
        # Unmasked positions (1 in mask)
        unmasked_loss = mse_loss * mask * (1 - r) * 2
        
        total_loss = masked_loss + unmasked_loss
        
        # Final scalar loss (mean over all elements)
        return total_loss.mean()