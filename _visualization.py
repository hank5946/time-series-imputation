"""
Visualization utilities for attention maps and other plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch


def draw_attention_map(model, data_loader, output_dir, device, D, num_samples=3):
    """
    Draw attention maps from both intra-variate and cross-variate attention layers.
    
    Args:
        model: The trained MultiScaleMultiTokenTransformerEncoder model
        data_loader: DataLoader to get sample inputs
        output_dir: Directory to save attention map figures
        device: Device to use
        D: Number of features/variates
        num_samples: Number of samples to visualize
    """
    attn_output_dir = os.path.join(output_dir, "attention_maps")
    os.makedirs(attn_output_dir, exist_ok=True)
    
    model.eval()
    
    for sample_idx, (gt, masked, mask) in enumerate(data_loader):
        if sample_idx >= num_samples:
            break
        
        gt, masked, mask = gt.to(device), masked.to(device), mask.to(device)
        
        with torch.no_grad():
            _ = model(masked, mask)
        
        # Iterate through each branch (scale)
        for branch_idx, branch in enumerate(model.branches):
            branch_dir = os.path.join(attn_output_dir, f"sample_{sample_idx}", f"branch_{branch_idx}")
            os.makedirs(branch_dir, exist_ok=True)
            
            # Draw intra-variate attention maps
            _draw_intra_attention(branch, branch_dir, D)
            
            # Draw cross-variate attention maps
            _draw_cross_attention(branch, branch_dir, D)
    
    print(f"Attention maps saved to {attn_output_dir}")


def _draw_intra_attention(branch, branch_dir, D):
    """Draw intra-variate attention maps."""
    for layer_idx, layer in enumerate(branch['intra_layers']):
        if hasattr(layer, 'attn_weights') and layer.attn_weights is not None:
            attn_weights = layer.attn_weights.cpu().numpy()  # (B*D, N_t, N_t)
            
            # Visualize up to 4 variates
            for d_idx in range(min(D, 4)):
                fig, ax = plt.subplots(figsize=(8, 6))
                attn_map = attn_weights[d_idx]  # (N_t, N_t)
                sns.heatmap(attn_map, ax=ax, cmap='viridis',
                            xticklabels=False, yticklabels=False)
                ax.set_title(f"Intra-Attn Layer {layer_idx}, Variate {d_idx}")
                ax.set_xlabel("Key Position")
                ax.set_ylabel("Query Position")
                plt.tight_layout()
                plt.savefig(os.path.join(branch_dir, f"intra_layer{layer_idx}_var{d_idx}.png"), dpi=150)
                plt.close()


def _draw_cross_attention(branch, branch_dir, D):
    """Draw cross-variate attention maps."""
    for layer_idx, layer in enumerate(branch['cross_layers']):
        if hasattr(layer, 'attn_weights') and layer.attn_weights is not None:
            attn_weights = layer.attn_weights.cpu().numpy()  # (B, L, L) where L = D * N_t
            
            # Take the first batch item
            attn_map = attn_weights[0]  # (L, L)
            
            # Full attention map
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(attn_map, ax=ax, cmap='viridis',
                        xticklabels=False, yticklabels=False)
            ax.set_title(f"Cross-Attn Layer {layer_idx}")
            ax.set_xlabel("Key Position (All Variates)")
            ax.set_ylabel("Query Position (All Variates)")
            plt.tight_layout()
            plt.savefig(os.path.join(branch_dir, f"cross_layer{layer_idx}.png"), dpi=150)
            plt.close()
            
            # Summarized variate-to-variate attention map
            _draw_variate_summary(attn_map, branch_dir, layer_idx, D)


def _draw_variate_summary(attn_map, branch_dir, layer_idx, D):
    """Draw summarized variate-to-variate attention map."""
    N_t = attn_map.shape[0] // D
    variate_attn = np.zeros((D, D))
    
    for i in range(D):
        for j in range(D):
            # Average attention from variate i to variate j
            variate_attn[i, j] = attn_map[
                i * N_t:(i + 1) * N_t,
                j * N_t:(j + 1) * N_t
            ].mean()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(variate_attn, ax=ax, cmap='viridis',
                annot=True if D <= 10 else False, fmt='.3f')
    ax.set_title(f"Variate-to-Variate Attention (Layer {layer_idx})")
    ax.set_xlabel("Source Variate")
    ax.set_ylabel("Target Variate")
    plt.tight_layout()
    plt.savefig(os.path.join(branch_dir, f"cross_layer{layer_idx}_variate_summary.png"), dpi=150)
    plt.close()
