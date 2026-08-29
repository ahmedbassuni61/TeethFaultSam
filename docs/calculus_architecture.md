# Calculus Segmentation Architecture — Technical Documentation

## Overview

This document describes the **calculus segmentation head** added to 3DTeethSAM for detecting and segmenting dental calculus (tartar) deposits on 3D intraoral scans. The calculus head operates as an **independent, parallel branch** alongside the existing tooth instance segmentation pipeline — no existing code was modified.

## Architecture Diagram

```
                    ┌─────────────────────────────────────────────────┐
                    │            SAM2 Hiera-L Backbone                │
                    │         (Frozen + LoRA on Q/V, rank=4)          │
                    └───────────┬─────────────┬───────────────────────┘
                                │             │
                  ┌─────────────▼─────────────▼──────────────┐
                  │      Multi-Level Feature Fusion + CBAM    │
                  │     [B, 4096, 256]  +  [B, 256, 64, 64]  │
                  └─────────────┬─────────────┬──────────────┘
                                │             │
              ┌─────────────────▼─┐     ┌─────▼──────────────────┐
              │   CalculusPEG     │     │  Existing Tooth PEG    │
              │  (4 queries,      │     │  (16 queries,          │
              │   3-layer TF dec) │     │   6-layer TF dec)      │
              └────────┬──────────┘     └────────────────────────┘
                       │
              ┌────────▼──────────┐
              │  SAM2 Mask Decoder │
              │  (4 binary masks)  │
              └────────┬──────────┘
                       │
                       ▼
              ┌────────────────────┐
              │  CalculusRefiner   │
              │  (Half-width       │
              │   Binary ResUNet)  │
              │  RGB + Coarse Mask │
              │  + SAM Embedding   │
              └────────┬──────────┘
                       │
                       ▼
              [B, 2, 512, 512]
              (Background | Calculus)
```

## Module Descriptions

### 1. LoRA Adapters (`model/calculus/lora.py`)

**Low-Rank Adaptation (LoRA)** injects small trainable matrices into the frozen SAM2 backbone's attention layers. Instead of fine-tuning the full 144M-parameter backbone on just 300 samples (which would catastrophically overfit), we add rank-4 matrices to the Q and V projections:

```
y = W_original @ x + (B @ A @ x) * alpha/rank
```

Where:
- `A ∈ R^{rank × d_in}` — initialized with Kaiming uniform
- `B ∈ R^{d_out × rank}` — initialized to zero (LoRA starts as identity)
- `rank = 4`, `alpha = 1.0`

This adds only ~0.3M trainable parameters to the backbone, providing just enough capacity to adapt features for calculus detection while preventing overfitting.

**Injection targets**: All `q_proj` and `v_proj` Linear layers in the Hiera-L backbone attention blocks.

### 2. CalculusPEG (`model/calculus/calculus_prompt_generator.py`)

A simplified **Prompt Embedding Generator** with:
- **4 learnable query embeddings** (vs. 16 for teeth) — calculus deposits typically appear in 2-6 localized regions along the gumline, so fewer queries suffice
- **3-layer Transformer decoder** (vs. 6 for teeth) — the calculus detection task is simpler than multi-instance tooth segmentation
- **Confidence head**: Predicts per-query calculus presence probability `[B, 4]`

Input: Sequential features `[B, 4096, 256]` from MultiLevelFeatureFusion  
Output: Prompt embeddings `[B, 4, 256]` + confidence `[B, 4]`

### 3. CalculusRefiner (`model/calculus/calculus_head.py`)

A **lightweight binary ResUNet** adapted from the existing tooth `ResUNet`:

| Property | Tooth ResUNet | Calculus ResUNet |
|----------|--------------|-----------------|
| Input mask channels | 17 (16 teeth + bg) | 2 (calculus + bg) |
| Encoder widths | 64→128→256→512 | 32→64→128→256 |
| Bottleneck | 1024 | 512 |
| Self-attention | Stages 3 & 4 | Stage 4 only |
| Dropout | 0.3 | 0.4 |
| Output classes | 17 | 2 |
| ~Parameters | ~15M | ~4M |

The half-width design prevents overfitting on the small 300-sample dataset while retaining the proven dual-encoder (RGB + mask) + SAM feature fusion architecture.

### 4. CalculusSegmentationSystem (`model/calculus/calculus_system.py`)

End-to-end forward pass:

1. **Image encoding**: RGB images → SAM2 Hiera-L backbone (frozen + LoRA) → multi-scale features
2. **Feature fusion**: MultiLevelFeatureFusion + CBAM → sequential `[B, 4096, 256]` + spatial `[B, 256, 64, 64]`
3. **Prompt generation**: CalculusPEG → 4 prompt embeddings + confidence
4. **Coarse masks**: SAM2 Mask Decoder → 4 per-query binary masks → combined into `[B, 2, H, W]`
5. **Refinement**: CalculusRefiner(RGB, coarse_masks, SAM_embedding) → refined `[B, 2, H, W]`

## Loss Function (`model/loss/loss_calculus.py`)

The loss handles **severe class imbalance** (calculus covers ~5-15% of visible area):

### Stage 1 (Coarse SAM masks)
- **Focal BCE** (γ=2.0, weight=1.5): Down-weights easy background pixels
- **Soft Dice** (weight=1.0): Region-based overlap metric
- **Confidence BCE** (weight=0.5): Supervises query presence predictions

### Stage 2 (Refined masks)
- **Focal BCE** (weight=1.0)
- **Soft Dice** (weight=1.0)
- **Differentiable Boundary Loss** (weight=0.3): Sobel-based gradient matching for sharp calculus margins

## Data Pipeline

### Synthetic Data Format
Input: `.obj` mesh + `.json` with `fault_labels` (per-vertex binary) and `fault_type: "calculus"`

### Preprocessing (`preprocess/preprocess_calculus.py`)
1. Load mesh and normalize to origin
2. Render 7 canonical views at 512×512
3. Rasterize `fault_labels` to 2D via face-level majority vote
4. Save as `.npz`: `images [7,512,512,3]` + `calculus_masks [7,512,512]`

### HDF5 Packing (`model/dataset/hdf5_calculus.py`)
Converts `.npz` files to single HDF5 for efficient training I/O.

### Dataset (`model/dataset/calculus_dataset.py`)
Returns `{image: [3,H,W], calculus_mask: [2,H,W]}` with light augmentation:
- Brightness jitter (p=0.3, ±10%)
- Contrast adjustment (p=0.3, ±10%)
- Gaussian noise (p=0.2, σ=0.01)

## Training Strategy

### Single-Stage Training (No Hungarian Matching)
Unlike teeth (which require 2-stage training with Hungarian matching for permutation invariance), calculus is binary semantic segmentation — a single training stage suffices.

### Optimizer Setup
- **LoRA params**: AdamW, lr=1e-4, weight_decay=1e-3
- **Calculus head params** (PEG + Refiner): AdamW, lr=5e-4, weight_decay=1e-3
- **Feature fusion**: AdamW, lr=5e-4, weight_decay=1e-3

### Schedule
- Linear warmup: 8 epochs
- Cosine annealing: to epoch 80
- Early stopping: patience=15 on validation Dice

### Regularization (for 300-sample dataset)
- LoRA rank=4 (~0.3M backbone params)
- Dropout=0.4
- Weight decay=1e-3
- ~5.5M total trainable parameters
- Light augmentation only

## Inference Pipeline (`inference_calculus.py`)

1. Load 3D mesh
2. Render 7 multi-view RGB images
3. Run CalculusSegmentationSystem → 2D calculus probability maps
4. **Lift to 3D**: Back-project 2D probabilities to mesh vertices via PyTorch3D rasterization + depth-weighted multi-view voting
5. Threshold → per-vertex binary `fault_labels`
6. Save to JSON

## Evaluation Metrics (`evaluate_calculus.py`)

All metrics computed on 3D mesh vertices:
- **Vertex Dice** = 2·TP / (2·TP + FP + FN)
- **Vertex IoU** = TP / (TP + FP + FN)
- **Precision** = TP / (TP + FP)
- **Recall** = TP / (TP + FN)
- **Accuracy** = (TP + TN) / Total

## File Inventory

| File | Purpose |
|------|---------|
| `config/config_calculus.py` | Training configuration |
| `model/calculus/__init__.py` | Package init |
| `model/calculus/lora.py` | LoRA adapter modules |
| `model/calculus/calculus_prompt_generator.py` | CalculusPEG (4-query Transformer decoder) |
| `model/calculus/calculus_head.py` | CalculusRefiner (half-width binary ResUNet) |
| `model/calculus/calculus_system.py` | End-to-end CalculusSegmentationSystem |
| `model/loss/loss_calculus.py` | Focal BCE + Dice + Boundary loss |
| `model/dataset/calculus_dataset.py` | HDF5CalculusDataset + light augmentation |
| `model/dataset/hdf5_calculus.py` | NPZ→HDF5 conversion + split generation |
| `preprocess/preprocess_calculus.py` | Mesh→multi-view rendering + calculus masks |
| `train_calculus.py` | Training entrypoint |
| `model/trainer_calculus.py` | CalculusTrainer class |
| `inference_calculus.py` | 3D calculus inference pipeline |
| `evaluate_calculus.py` | 3D mesh-level evaluation metrics |
