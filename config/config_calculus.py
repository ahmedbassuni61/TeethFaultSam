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
    num_queries: int = 4          
    num_classes: int = 2          
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
    warmup_epochs: int = 5
    grad_clip: float = 1.0
    focal_gamma: float = 2.0
    seed: int = 42

    # Epoch configuration
    batches_per_epoch: int = 0  # 0 means use the full dataset. Set > 0 to artificially shorten epochs.

    # WandB
    use_wandb: bool = False
    wandb_project: str = '3DTeethSAM_Calculus'
    wandb_entity: Optional[str] = None

    # HuggingFace Hub
    hf_repo_id: Optional[str] = None            # e.g. "ahmedbassuni61/teethsam"
    hf_upload_every_n_epochs: int = 5            # Upload checkpoint every N epochs
    hf_token: Optional[str] = None               # HF token (or use env var / huggingface-cli login)
    hf_path_in_repo: str = 'Calculus_Training_Logs'  # Target folder in HF repo

    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = 'float16'

    # Data parameters
    view_indices: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6])
    data_dir: str = 'preprocess_calculus_data'
    h5_path: str = 'preprocess_calculus_data/h5/calculus_dataset.h5'
    train_split_path: str = 'preprocess/split/calculus/training_calculus.txt'
    val_split_path: str = 'preprocess/split/calculus/testing_calculus.txt'

    # Training loop
    early_stopping_patience: int = 15
    save_dir: str = 'results_calculus'
    num_workers: int = 0  # 0 fixes Colab shared memory deadlocks
    val_interval: int = 1  # Validate and log every epoch
    log_freq: int = 5      # Log to wandb/tensorboard every N batches
    save_freq: int = 5     # Save checkpoint every N epochs (on top of best validation saves)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def get_torch_dtype(self):
        if self.amp_dtype == 'bfloat16':
            return torch.bfloat16
        elif self.amp_dtype == 'float16':
            return torch.float16
        return torch.float32
