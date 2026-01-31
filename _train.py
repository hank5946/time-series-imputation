"""
Training logic for time series imputation model.
"""

import torch
import matplotlib.pyplot as plt
import os


def train_one_epoch(model, train_loader, optimizer, criterion, device, loss_r, max_grad_norm=1.0):
    """
    Train model for one epoch.
    
    Args:
        model: The model to train
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to use
        loss_r: Loss weight ratio for masked vs unmasked
        max_grad_norm: Maximum gradient norm for clipping
        
    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0
    
    for gt, masked, mask in train_loader:
        gt, masked, mask = gt.to(device), masked.to(device), mask.to(device)
        
        optimizer.zero_grad()
        out = model(masked, mask)
        loss = criterion(out, gt, mask, loss_r)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device, loss_r):
    """
    Validate model on validation set.
    
    Args:
        model: The model to validate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to use
        loss_r: Loss weight ratio
        
    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for gt, masked, mask in val_loader:
            gt, masked, mask = gt.to(device), masked.to(device), mask.to(device)
            out = model(masked, mask)
            loss = criterion(out, gt, mask, loss_r)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, 
                device, args, output_path, num_epochs=350, patience=4, min_improve=0.005):
    """
    Full training loop with early stopping.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        criterion: Loss function
        device: Device to use
        args: Arguments containing loss_r
        output_path: Path to save model and plots
        num_epochs: Maximum number of epochs
        patience: Early stopping patience
        min_improve: Minimum relative improvement to reset early stopping counter
        
    Returns:
        train_losses: List of training losses
        val_losses: List of validation losses
    """
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    early_stop_counter = 0
    
    for epoch in range(num_epochs):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, 
            device, args.loss_r
        )
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device, args.loss_r)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"[Multi] Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Step scheduler
        scheduler.step()
        
        # Early stopping check
        rel_improve = (best_val_loss - val_loss) / best_val_loss if best_val_loss != float('inf') else 1.0
        
        if rel_improve >= min_improve:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), os.path.join(output_path, "best_model.pth"))
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("[Multi] Early stopping")
                break
    
    # Plot and save loss curve
    plot_loss_curve(train_losses, val_losses, output_path)
    
    return train_losses, val_losses


def plot_loss_curve(train_losses, val_losses, output_path):
    """
    Plot and save training loss curve.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        output_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curve")
    plt.savefig(os.path.join(output_path, "loss_curve.png"))
    plt.close()
