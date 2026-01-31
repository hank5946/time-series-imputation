# Multi-Scale Transformer for Time Series Imputation

A PyTorch implementation of a multi-scale transformer model for multivariate time series imputation. This model uses multiple temporal scales with intra-variate and cross-variate attention mechanisms to effectively impute missing values in time series data.

## Project Structure

```
multi/
├── run_multi.py          # Main entry point - orchestrates the full pipeline
├── config.py             # Configuration and argument parsing
├── _model.py             # Transformer model architecture
├── _data.py              # Data loading, preprocessing, and dataset classes
├── _mask.py              # Mask generation strategies
├── _compute.py           # Correlation computation and loss functions
├── _train.py             # Training loop and validation
├── _inference.py         # Model evaluation and inference
├── _visualization.py     # Attention maps and plotting utilities
├── _utils.py             # General utility functions
└── README.md             # This file
```

## Requirements

```bash
pip install torch numpy pandas matplotlib seaborn scikit-learn scipy statsmodels
```

## Quick Start

### Basic Usage

```bash
python run_multi.py --data_path /path/to/your/data.csv --output_path /path/to/output/
```

### Data Format

Your input CSV file should have the following format:
- First column: `date` (datetime format)
- Remaining columns: Feature values (numeric)

Example:
```csv
date,feature1,feature2,feature3
2024-01-01,1.2,3.4,5.6
2024-01-02,1.3,3.5,5.7
...
```

## Command Line Arguments

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--data_path` | Path to the input CSV data file |
| `--output_path` | Path to save outputs (models, plots, results) |

### Model Architecture

| Argument | Default | Description |
|----------|---------|-------------|
| `--num_layers` | 4 | Number of transformer layers |
| `--nhead` | 8 | Number of attention heads |
| `--dropout` | 0.1 | Dropout rate |

### Training Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--batch_size` | 16 | Batch size for training |
| `--lr` | 1e-3 | Learning rate |
| `--epochs` | 350 | Maximum number of training epochs |
| `--seed` | 42 | Random seed for reproducibility |

### Sequence Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--seq_len` | 512 | Input sequence length |
| `--pred_len` | 128 | Prediction length (for forecasting tasks) |

### Masking Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--missing_rate` | 0.25 | Rate of missing values (0.0-1.0) |
| `--missing_type` | 1 | Mask type: 0=Random, 1=Gaussian, 2=Prediction |
| `--lm` | 10 | Mean length of masked segments |

### Tokenization Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--token_t_overlap` | 0 | Overlap between temporal tokens |

### Loss Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--loss_r` | 0.75 | Weight ratio for masked vs unmasked loss |

## Example Commands

### Train with default settings
```bash
python run_multi.py \
    --data_path ../Datasets/all_data/ETTh1.csv \
    --output_path ../outputs/experiment1/
```

### Train with custom hyperparameters
```bash
python run_multi.py \
    --data_path ../Datasets/all_data/weather_single.csv \
    --output_path ../outputs/experiment2/ \
    --num_layers 6 \
    --nhead 8 \
    --batch_size 32 \
    --seq_len 128 \
    --missing_rate 0.3 \
    --lr 5e-4
```

### Train with Gaussian missing pattern
```bash
python run_multi.py \
    --data_path ../Datasets/all_data/electricity_single.csv \
    --output_path ../outputs/experiment3/ \
    --missing_type 1 \
    --lm 15
```

## Output Files

After training, the following files will be generated in `<output_path>/multi/`:

```
multi/
├── best_model.pth              # Best model checkpoint
├── loss_curve.png              # Training/validation loss plot
├── imputation/
│   ├── test_case_0/            # Imputation visualizations
│   │   ├── column_0.png
│   │   ├── column_1.png
│   │   └── ...
│   └── test_case_1/
└── attention_maps/
    ├── sample_0/
    │   ├── branch_0/           # Attention maps for each scale
    │   │   ├── intra_layer0_var0.png
    │   │   ├── cross_layer0.png
    │   │   └── cross_layer0_variate_summary.png
    │   └── ...
    └── ...
```

## Using Individual Modules

You can import and use individual modules for custom workflows:

### Load and preprocess data
```python
from _data import load_and_normalize_data, generate_masks, create_data_loaders

data, timestamps, mean, std = load_and_normalize_data("data.csv")
mask = generate_masks(data, missing_type=0, missing_rate=0.25, lm=10, seq_len=64)
```

### Create model
```python
from _model import create_model, MultiScaleMultiTokenTransformerEncoder
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiScaleMultiTokenTransformerEncoder(
    seq_len=64, num_layers=4, nhead=8, dropout=0.1, D=8, device=device
).to(device)
```

### Run inference on single sequence
```python
from _inference import impute_single_sequence

imputed = impute_single_sequence(model, sequence, mask, device)
```

### Visualize attention maps
```python
from _visualization import draw_attention_map

draw_attention_map(model, test_loader, output_dir, device, D=8, num_samples=3)
```

### Compute metrics
```python
from _utils import compute_metrics

metrics = compute_metrics(ground_truth, predictions, mask)
print(f"MSE: {metrics['mse']:.4f}, MAE: {metrics['mae']:.4f}")
```

## Model Architecture

The model uses a **Multi-Scale Multi-Token Transformer** architecture:

1. **Multi-Scale Tokenization**: The input sequence is tokenized at 4 different scales (seq_len/16, seq_len/8, seq_len/4, seq_len/2)

2. **Intra-Variate Attention**: Self-attention within each variate's time tokens to capture temporal dependencies

3. **Cross-Variate Attention**: Attention across different variates to capture inter-variable relationships (with same-variate attention masked)

4. **Learnable Scale Fusion**: Outputs from different scales are combined using learnable softmax weights

## Mask Types

- **Type 0 (Random)**: Randomly selects positions to mask based on missing_rate
- **Type 1 (Gaussian/Geometric)**: Creates contiguous masked segments with geometric distribution
- **Type 2 (Prediction)**: Masks the last portion of each sequence (for forecasting evaluation)

## License

MIT License

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{multiscale_transformer_imputation,
  title={Multi-Scale Transformer for Time Series Imputation},
  year={2026},
  url={https://github.com/your-repo}
}
```
