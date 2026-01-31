"""
Transformer-based models for time series imputation.
"""

import numpy as np
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""
    
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)


class CustomTransformerEncoderLayer(nn.Module):
    """Custom transformer encoder layer with attention weight storage."""
    
    def __init__(self, d_model, nhead=8, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.attn_weights = None

    def forward(self, src):
        attn_output, attn_weights = self.self_attn(src, src, src, need_weights=True)
        self.attn_weights = attn_weights.detach()
        src = self.norm1(src + self.dropout1(attn_output))
        src2 = self.linear2(self.dropout(torch.relu(self.linear1(src))))
        return self.norm2(src + self.dropout2(src2))


class CustomCrossAttentionLayer(nn.Module):
    """Cross-attention layer for inter-variate attention."""
    
    def __init__(self, d_model, nhead=4, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.attn_weights = None

    def forward(self, query, context, attn_mask=None):
        """
        Args:
            query: (B, N_q, d_model) - sequence that receives attended output
            context: (B, N_kv, d_model) - source sequence for keys and values
            attn_mask: Optional attention mask
        """
        attn_output, attn_weights = self.cross_attn(
            query, context, context, attn_mask=attn_mask, need_weights=True
        )
        self.attn_weights = attn_weights.detach()

        query = self.norm1(query + self.dropout1(attn_output))
        query2 = self.linear2(self.dropout(torch.relu(self.linear1(query))))
        return self.norm2(query + self.dropout2(query2))


class TimeOnlyEmbedding(nn.Module):
    """Temporal patch embedding layer."""
    
    def __init__(self, token_t_size, token_t_overlap, d_model):
        super().__init__()
        self.token_t_size = token_t_size
        self.token_t_overlap = token_t_overlap
        self.d_model = d_model
        self.linear = nn.Linear(token_t_size, d_model)

    def forward(self, x):  # x: (B, T, D)
        B, T, D = x.shape
        temporal_patches = []
        for d_idx in range(D):
            for t_start in range(0, T - self.token_t_size + 1, self.token_t_size - self.token_t_overlap):
                patch = x[:, t_start:t_start + self.token_t_size, d_idx]  # (B, token_t_size)
                temporal_patches.append(self.linear(patch))  # (B, d_model)
        temporal_tokens = torch.stack(temporal_patches, dim=1)  # (B, N_t * D, d_model)
        return temporal_tokens


def reconstruct_from_vertical_patches(temporal_tokens, B, T, D, token_t_size, token_t_overlap, device):
    """
    Reconstruct full sequence from temporal patch tokens.
    
    Args:
        temporal_tokens: (B, N_t * D, token_t_size)
        B: Batch size
        T: Sequence length
        D: Number of features
        token_t_size: Temporal token size
        token_t_overlap: Token overlap
        device: Device to use
        
    Returns:
        Reconstructed sequence (B, T, D)
    """
    recon = torch.zeros((B, T, D), device=device)
    count = torch.zeros((B, T, D), device=device)

    t_starts = list(range(0, T - token_t_size + 1, token_t_size - token_t_overlap))
    n_t = len(t_starts)

    for d_idx in range(D):
        for i, t_start in enumerate(t_starts):
            patch = temporal_tokens[:, d_idx * n_t + i, :].reshape(B, token_t_size, 1).squeeze(-1)
            recon[:, t_start:t_start + token_t_size, d_idx] += patch
            count[:, t_start:t_start + token_t_size, d_idx] += 1

    return recon / count.clamp(min=1e-6)


class MultiScaleMultiTokenTransformerEncoder(nn.Module):
    """Multi-scale transformer encoder for time series imputation."""
    
    def __init__(self, seq_len, num_layers=4, nhead=8, dropout=0.1, 
                 token_t_overlap=0, D=8, device='cpu'):
        super().__init__()
        self.device = device
        self.token_sizes = [
            int(seq_len / 16), 
            int(seq_len / 8), 
            int(seq_len / 4), 
            int(seq_len / 2)
        ]
        self.token_overlap = token_t_overlap
        self.num_layers = num_layers
        self.nhead = nhead
        self.seq_len = seq_len
        self.dropout = dropout
        self.D = D

        self.mask_token = nn.Parameter(torch.randn(1))
        self.branches = nn.ModuleList()

        for token_size in self.token_sizes:
            d_model = token_size * self.nhead

            branch = nn.ModuleDict({
                'embed': TimeOnlyEmbedding(
                    token_t_size=token_size,
                    token_t_overlap=self.token_overlap,
                    d_model=d_model
                ),
                'pos_encoder': PositionalEncoding(d_model),
                'intra_layers': nn.ModuleList([
                    CustomTransformerEncoderLayer(d_model, self.nhead, self.dropout)
                    for _ in range(self.num_layers)
                ]),
                'cross_layers': nn.ModuleList([
                    CustomCrossAttentionLayer(d_model, self.nhead, self.dropout)
                    for _ in range(self.num_layers)
                ]),
                'project': nn.Linear(d_model, token_size)
            })
            self.branches.append(branch)
        
        self.alpha = nn.Parameter(torch.ones(len(self.token_sizes)))

    def forward(self, x, mask):
        B, T, D = x.shape
        x = torch.where(mask, x, self.mask_token.expand_as(x))
        outputs = []

        for idx, token_size in enumerate(self.token_sizes):
            branch = self.branches[idx]
            d_model = token_size * self.nhead

            # 1. Token embedding
            x_embed = branch['embed'](x)
            N_t = x_embed.shape[1] // D

            # 2. Positional encoding
            x_embed = branch['pos_encoder'](x_embed)
            x_embed = x_embed.view(B, D, N_t, d_model).view(B * D, N_t, d_model)

            # 3. Intra-variate attention
            for layer in branch['intra_layers']:
                x_embed = layer(x_embed)

            x_embed = x_embed.view(B, D, N_t, d_model)
            x_embed = x_embed.view(B, D * N_t, d_model)

            # 4. Cross-variate attention
            L = D * N_t
            token_to_variate = torch.arange(L, device=x.device) // N_t
            same_variate = token_to_variate.unsqueeze(0) == token_to_variate.unsqueeze(1)
            attn_mask = same_variate

            for layer in branch['cross_layers']:
                x_embed = layer(x_embed, x_embed, attn_mask=attn_mask)

            # 5. Project back to patch space
            out = branch['project'](x_embed)
            recon = reconstruct_from_vertical_patches(
                out, B, T, D, token_size, self.token_overlap, self.device
            )
            outputs.append(recon)

        # Combine outputs using softmax-weighted sum
        weights = torch.softmax(self.alpha, dim=0)
        out_combined = sum(w * o for w, o in zip(weights, outputs))

        return out_combined


def create_model(args, D, device):
    """
    Factory function to create the model.
    
    Args:
        args: Arguments containing model hyperparameters
        D: Number of features
        device: Device to use
        
    Returns:
        Initialized model
    """
    model = MultiScaleMultiTokenTransformerEncoder(
        seq_len=args.seq_len,
        num_layers=args.num_layers,
        nhead=args.nhead,
        dropout=args.dropout,
        token_t_overlap=args.token_t_overlap,
        D=D,
        device=device
    ).to(device)
    return model


