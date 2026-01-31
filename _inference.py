"""
Inference and evaluation logic for time series imputation model.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import os

from _data import denormalize


def evaluate_model(model, test_loader, device, data_mean, data_std, output_path, D):
    """
    Evaluate model on test set and compute MSE.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to use
        data_mean: Mean values for denormalization
        data_std: Std values for denormalization
        output_path: Path to save visualizations
        D: Number of features/variates
        
    Returns:
        mse: Mean squared error on masked positions
    """
    model.eval()
    mse_total = 0.0
    count = 0
    
    for i, (gt, masked, mask) in enumerate(test_loader):
        gt, masked, mask = gt.to(device), masked.to(device), mask.to(device)
        
        with torch.no_grad():
            out = model(masked, mask)
        
        gt_np = gt.cpu().numpy().squeeze()
        out_np = out.cpu().numpy().squeeze()
        mask_np = mask.cpu().numpy().squeeze()
        
        # Handle dimension issues
        if gt_np.ndim == 1:
            gt_np = gt_np[np.newaxis, :]
        if out_np.ndim == 1:
            out_np = out_np[np.newaxis, :]
        if mask_np.ndim == 1:
            mask_np = mask_np[np.newaxis, :]
        
        # Denormalize for visualization
        gt_denorm = denormalize(gt_np, data_mean, data_std)
        out_denorm = denormalize(out_np, data_mean, data_std)
        
        # Compute MSE on masked positions
        mse = ((gt_np[~mask_np] - out_np[~mask_np]) ** 2).sum()
        mse_total += mse
        count += (~mask_np).sum()
        
        # Save visualizations for first few test cases
        if i < 2:
            save_imputation_plots(
                gt_denorm, out_denorm, mask_np, 
                output_path, i, D
            )
    
    mse = mse_total / count
    print(f"[Multi] Masked MSE on test set: {mse}")
    
    return mse


def save_imputation_plots(gt_denorm, out_denorm, mask_np, output_path, case_idx, D):
    """
    Save imputation result plots for a single test case.
    
    Args:
        gt_denorm: Denormalized ground truth
        out_denorm: Denormalized model output
        mask_np: Mask array
        output_path: Output directory
        case_idx: Test case index
        D: Number of features
    """
    case_dir = os.path.join(output_path, "imputation", f"test_case_{case_idx}")
    os.makedirs(case_dir, exist_ok=True)
    
    for j in range(D):
        plt.figure(figsize=(12, 4))
        plt.plot(gt_denorm[0, :, j], label="GT", alpha=0.8)
        plt.plot(out_denorm[0, :, j], label="Output", alpha=0.8)
        plt.plot(mask_np[0, :, j] * np.max(gt_denorm[:, 0]), label="Mask", alpha=0.5)
        plt.legend()
        plt.title(f"Test Case {case_idx} - Column {j}")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.savefig(os.path.join(case_dir, f"column_{j}.png"))
        plt.close()


    with torch.no_grad():
        out = model(sequence * mask, mask)
    
    return out.squeeze(0).cpu().numpy()
