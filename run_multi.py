"""
Main entry point for time series imputation using Multi-scale Transformer.

This script orchestrates the full pipeline:
1. Configuration and argument parsing
2. Data loading and preprocessing
3. Model creation
4. Training
5. Evaluation and visualization

Usage:
    python run_multi.py --data_path <path_to_csv> --output_path <output_dir>
"""

import torch
import os

# Local imports
from config import get_args, get_device, setup_output_dirs
from _data import (
    load_and_normalize_data,
    generate_masks,
    apply_mask,
    create_data_loaders
)
from _compute import compute_correlation_and_lag_matrix, MaskedWeightedMSELoss
from _model import create_model
from _train import train_model
from _inference import evaluate_model
from _visualization import draw_attention_map


def main():
    # =========================================================================
    # 1. Configuration
    # =========================================================================
    args = get_args()
    device = get_device()
    multi_output_path = setup_output_dirs(args.output_path)
    
    # =========================================================================
    # 2. Data Loading and Preprocessing
    # =========================================================================
    print("Loading and preprocessing data...")
    data, timestamps, data_mean, data_std = load_and_normalize_data(args.data_path)
    T, D = data.shape
    print(f"Data shape: T={T}, D={D}")
    
    # Generate masks
    mask_matrix = generate_masks(
        data, 
        missing_type=args.missing_type,
        missing_rate=args.missing_rate,
        lm=args.lm,
        seq_len=args.seq_len,
        seed=args.seed
    )
    
    # Apply mask to data
    masked_data = apply_mask(data, mask_matrix)
    
    # Compute correlation matrix (optional, for analysis)
    C_rule, lag_matrix = compute_correlation_and_lag_matrix(masked_data, mask_matrix)
    
    # Save correlation matrix
    output_file_matrix = os.path.join(args.output_path, "corr_matrix.txt")
    with open(output_file_matrix, "a") as log_file:
        log_file.write("Correlation matrix:\n")
        log_file.write(str(C_rule) + "\n")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        data, masked_data, mask_matrix, args, seed=args.seed
    )
    print(f"Train: {len(train_loader)} batches, Val: {len(val_loader)} batches, Test: {len(test_loader)} batches")
    
    # =========================================================================
    # 3. Model Creation
    # =========================================================================
    print("Creating model...")
    model = create_model(args, D, device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = MaskedWeightedMSELoss()
    
    # =========================================================================
    # 4. Training
    # =========================================================================
    print("Starting training...")
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        args=args,
        output_path=multi_output_path,
        num_epochs=args.epochs,
        patience=4,
        min_improve=0.005
    )
    
    # =========================================================================
    # 5. Evaluation
    # =========================================================================
    print("Loading best model and evaluating...")
    model.load_state_dict(torch.load(os.path.join(multi_output_path, "best_model.pth")))
    
    mse = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        data_mean=data_mean,
        data_std=data_std,
        output_path=multi_output_path,
        D=D
    )
    
    # Save results
    output_file = os.path.join(args.output_path, "test_results.txt")
    with open(output_file, "a") as log_file:
        log_file.write(f"{args.output_path}\n")
        log_file.write(f"[Multi][missing_type={args.missing_type}] Masked MSE on test set: {mse}\n")
    
    # =========================================================================
    # 6. Visualization
    # =========================================================================
    print("Drawing attention maps...")
    draw_attention_map(
        model=model,
        data_loader=test_loader,
        output_dir=multi_output_path,
        device=device,
        D=D,
        num_samples=3
    )
    
    print("Done!")


if __name__ == "__main__":
    main()
