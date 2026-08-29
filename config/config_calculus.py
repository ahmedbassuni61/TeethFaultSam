import os
import torch
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CalculusConfig:
    """Configuration for calculus segmentation training."""

    # Model checkpoints
    tooth_checkpoint: Optional[str] = 'ckpts/best.pth'
    sam_checkpoint: str = 'model/sam2/checkpoints/sam2.1_hiera_large.pt'
    sam_config: str = 'sam2.1_hiera_l.yaml'

    # Architecture
    embed_dim: int = 256
    num_calculus_queries: int = 4
    num_queries: int = 4          # alias used by CalculusSegmentationSystem
    num_classes: int = 2          # background + calculus
    lora_rank: int = 4
    lora_alpha: float = 1.0
    peg_layers: int = 3
    calculus_peg_layers: int = 3
    dropout_rate: float = 0.4
    refiner_dropout: float = 0.4
    nhead: int = 8

    # Training parameters
    batch_size: int = 4
    epochs: int = 80
    learning_rate: float = 5e-4
    lora_lr: float = 1e-4
    weight_decay: float = 1e-3
    warmup_epochs: int = 8
    grad_clip: float = 1.0
    focal_gamma: float = 2.0
    seed: int = 42

    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = 'bfloat16'

    # Data parameters
    view_indices: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6])
    data_dir: str = 'preprocess_calculus_data'
    h5_path: str = 'preprocess_calculus_data/h5/calculus_dataset.h5'
    train_split_path: str = 'preprocess/split/calculus/training_calculus.txt'
    val_split_path: str = 'preprocess/split/calculus/testing_calculus.txt'

    # Training loop
    early_stopping_patience: int = 15
    save_dir: str = 'results_calculus'
    num_workers: int = 4
    val_interval: int = 2
    log_freq: int = 10
    save_freq: int = 10

    def get(self, key, default=None):
        """Dict-like access for compatibility with existing code patterns."""
        return getattr(self, key, default)

    def get_torch_dtype(self):
        if self.amp_dtype == 'bfloat16':
            return torch.bfloat16
        elif self.amp_dtype == 'float16':
            return torch.float16
        return torch.float32
