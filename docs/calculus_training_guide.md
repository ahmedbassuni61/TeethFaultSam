# Calculus Segmentation — Training & Usage Guide

## Prerequisites

- Python 3.8+
- PyTorch 2.0+
- PyTorch3D
- SAM2 checkpoint: `model/sam2/checkpoints/sam2.1_hiera_large.pt`
- (Optional) Pre-trained tooth checkpoint: `ckpts/best.pth`

Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Step 1: Generate Synthetic Data (if not already done)
```bash
python scripts/generate_synthetic_dataset.py \
    --input_dir data_part_1 \
    --output_dir synthetic_calculus \
    --num_workers 4
```

### Step 2: Preprocess into Multi-View Renderings
```bash
python preprocess/preprocess_calculus.py \
    --data_dir synthetic_calculus \
    --save_dir preprocess_calculus_data
```

This renders each mesh from 7 viewpoints and generates binary calculus masks.
Output: `.npz` files in `preprocess_calculus_data/`.

### Step 3: Train the Calculus Head
```bash
# Basic training
python train_calculus.py

# With pre-trained tooth features (recommended)
python train_calculus.py --tooth_checkpoint ckpts/best.pth

# Custom settings
python train_calculus.py \
    --tooth_checkpoint ckpts/best.pth \
    --batch_size 4 \
    --learning_rate 5e-4 \
    --epochs 80
```

Training logs and checkpoints are saved to `results_calculus/<timestamp>/`.

### Step 4: Run Inference
```bash
python inference_calculus.py \
    --input_dir teeth3ds_sample \
    --checkpoint results_calculus/<timestamp>/checkpoints/best_calculus_model.pth \
    --output_dir inference_results/calculus
```

### Step 5: Evaluate
```bash
python evaluate_calculus.py \
    --pred_dir inference_results/calculus \
    --gt_dir synthetic_calculus
```

## Google Colab Usage

```python
# Mount drive and clone repo
from google.colab import drive
drive.mount('/content/drive')

# Copy checkpoints
!mkdir -p /content/TeethFaultSam/ckpts
!mkdir -p /content/TeethFaultSam/model/sam2/checkpoints
!cp "/content/drive/MyDrive/Teeth/ckpts/best.pth" "/content/TeethFaultSam/ckpts/"
!cp "/content/drive/MyDrive/Teeth/ckpts/sam2.1_hiera_large.pt" "/content/TeethFaultSam/model/sam2/checkpoints/"

# Generate synthetic data
!python scripts/generate_synthetic_dataset.py \
    --input_dir data_part_1 \
    --output_dir synthetic_calculus \
    --num_workers 4

# Preprocess
!python preprocess/preprocess_calculus.py \
    --data_dir synthetic_calculus \
    --save_dir preprocess_calculus_data

# Train
!python train_calculus.py --tooth_checkpoint ckpts/best.pth

# Inference
!python inference_calculus.py \
    --input_dir teeth3ds_sample \
    --checkpoint results_calculus/*/checkpoints/best_calculus_model.pth
```

## Configuration

All defaults are in `config/config_calculus.py`. Override via CLI args or JSON config:

```bash
python train_calculus.py --config my_config.json
```

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lora_rank` | 4 | LoRA adapter rank (higher = more capacity, more risk of overfitting) |
| `lora_lr` | 1e-4 | Learning rate for LoRA parameters |
| `learning_rate` | 5e-4 | Learning rate for calculus head |
| `num_calculus_queries` | 4 | Number of PEG prompt queries |
| `dropout_rate` | 0.4 | Dropout rate (higher for small datasets) |
| `weight_decay` | 1e-3 | AdamW weight decay |
| `focal_gamma` | 2.0 | Focal loss gamma for class imbalance |
| `early_stopping_patience` | 15 | Epochs without improvement before stopping |
| `epochs` | 80 | Maximum training epochs |

## Monitoring Training

### TensorBoard
```bash
tensorboard --logdir results_calculus/<timestamp>/tensorboard
```

Key metrics to watch:
- `loss/total`: Overall training loss
- `loss/dice_refine`: Refined mask Dice loss (most important)
- `val/dice`: Validation Dice score
- `val/iou`: Validation IoU
- `val/precision` and `val/recall`: Should both be >0.5

### Expected Training Behavior
- **Epochs 1-8**: Warmup period, loss decreasing rapidly
- **Epochs 8-30**: Main learning phase, Dice should reach ~0.5-0.7
- **Epochs 30-60**: Fine-tuning phase, Dice should stabilize ~0.7-0.85
- **Early stopping**: If val Dice doesn't improve for 15 epochs

## Troubleshooting

### Out of Memory
Reduce batch size:
```bash
python train_calculus.py --batch_size 2
```

### Poor Convergence
1. Check that calculus masks are non-empty in preprocessed data
2. Try loading pre-trained tooth features: `--tooth_checkpoint ckpts/best.pth`
3. Increase LoRA rank: add `"lora_rank": 8` to config JSON

### Overfitting (train loss low, val loss high)
1. Reduce epochs: `--epochs 40`
2. Increase dropout in config: `"dropout_rate": 0.5`
3. Decrease learning rate: `--learning_rate 2e-4`

## Directory Structure After Training

```
3DTeethSAM/
├── config/
│   └── config_calculus.py          # Calculus configuration
├── model/
│   ├── calculus/
│   │   ├── __init__.py
│   │   ├── lora.py                 # LoRA adapters
│   │   ├── calculus_head.py        # CalculusRefiner
│   │   ├── calculus_prompt_generator.py  # CalculusPEG
│   │   └── calculus_system.py      # End-to-end system
│   ├── dataset/
│   │   ├── calculus_dataset.py     # Dataset loader
│   │   └── hdf5_calculus.py        # HDF5 conversion
│   ├── loss/
│   │   └── loss_calculus.py        # Calculus loss functions
│   └── trainer_calculus.py         # Training loop
├── preprocess/
│   ├── preprocess_calculus.py      # Mesh → multi-view preprocessing
│   └── split/
│       └── calculus/               # Auto-generated splits
│           ├── training_calculus.txt
│           └── testing_calculus.txt
├── docs/
│   ├── calculus_architecture.md    # Architecture documentation
│   └── calculus_training_guide.md  # This guide
├── train_calculus.py               # Training entrypoint
├── inference_calculus.py           # Inference pipeline
└── evaluate_calculus.py            # Evaluation metrics
```
