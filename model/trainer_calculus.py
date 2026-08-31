"""
Trainer for calculus segmentation head — PyTorch Lightning backend.

Provides the same public API (train / test / load_checkpoint) as the previous
custom-loop trainer, but delegates all heavy lifting to a ``lightning.pytorch.Trainer``.

Key features over the old trainer:
    • TQDM progress bars with per-batch loss, Dice, IoU
    • Detailed model summary table at startup
    • Built-in AMP, gradient clipping, early stopping, model checkpointing
    • Automatic HuggingFace Hub upload every N epochs for safe resumption
"""

import os
import torch
from pathlib import Path
from datetime import datetime

try:
    import lightning.pytorch as L
    from lightning.pytorch.callbacks import (
        ModelCheckpoint,
        EarlyStopping,
        ModelSummary,
    )
    from lightning.pytorch.loggers import TensorBoardLogger
except ImportError:
    import pytorch_lightning as L
    from pytorch_lightning.callbacks import (
        ModelCheckpoint,
        EarlyStopping,
        ModelSummary,
    )
    from pytorch_lightning.loggers import TensorBoardLogger

from model.lightning_calculus import (
    CalculusLightningModule,
    CalculusDataModule,
    HuggingFaceUploadCallback,
)


class CalculusTrainer:
    """
    High-level wrapper that assembles the Lightning Trainer, DataModule,
    LightningModule, and all callbacks from a single ``CalculusConfig``.

    Usage::

        trainer = CalculusTrainer(config)
        trainer.load_checkpoint("resume.ckpt")   # optional
        trainer.train()
        trainer.test("best_calculus_model.ckpt")  # optional
    """

    def __init__(self, config):
        self.config = config
        self.resume_path = None

        # Seed everything for reproducibility
        L.seed_everything(config.seed, workers=True)

        # Timestamped output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.save_dir = Path(config.save_dir) / timestamp
        self.checkpoint_dir = self.save_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Build components
        self.data_module = CalculusDataModule(config)
        self.module = CalculusLightningModule(config)

        callbacks = self._build_callbacks()
        loggers = self._build_loggers()
        precision = self._get_precision()

        # limit_train_batches: int → N batches; float 1.0 → full dataset
        limit = config.batches_per_epoch if config.batches_per_epoch > 0 else 1.0

        self.pl_trainer = L.Trainer(
            max_epochs=config.epochs,
            callbacks=callbacks,
            logger=loggers,
            precision=precision,
            gradient_clip_val=config.grad_clip if config.grad_clip > 0 else None,
            limit_train_batches=limit,
            check_val_every_n_epoch=config.val_interval,
            log_every_n_steps=config.log_freq,
            enable_model_summary=False,  # we add our own ModelSummary callback
            deterministic=False,
            accelerator='auto',
            devices='auto',
            default_root_dir=str(self.save_dir),
        )

    # ── Public API ───────────────────────────────────────────────────────────

    def train(self):
        """Run the full training loop (with optional checkpoint resumption)."""
        self.pl_trainer.fit(
            self.module,
            datamodule=self.data_module,
            ckpt_path=self.resume_path,
        )

    def test(self, checkpoint_path=None):
        """
        Evaluate on the test split.

        If *checkpoint_path* is a Lightning ``.ckpt`` file it is loaded
        automatically.  Legacy ``.pth`` checkpoints are loaded manually.
        """
        if checkpoint_path:
            ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            is_lightning = 'pytorch-lightning_version' in ckpt or 'state_dict' in ckpt
            del ckpt

            if is_lightning:
                self.pl_trainer.test(self.module, datamodule=self.data_module,
                                     ckpt_path=checkpoint_path)
            else:
                self._load_legacy_weights(checkpoint_path)
                self.pl_trainer.test(self.module, datamodule=self.data_module)
        else:
            self.pl_trainer.test(self.module, datamodule=self.data_module)

    def load_checkpoint(self, path):
        """
        Stage a checkpoint for resumption.

        * **Lightning checkpoint** (``.ckpt``): full resume (model + optimizer +
          scheduler + epoch).
        * **Legacy checkpoint** (``.pth``): model weights only; optimizer and
          scheduler start fresh.
        """
        ckpt = torch.load(path, map_location='cpu', weights_only=False)

        if 'pytorch-lightning_version' in ckpt or 'state_dict' in ckpt:
            self.resume_path = path
            print(f"✅ Lightning checkpoint staged for full resume: {path}")
        elif 'model_state_dict' in ckpt:
            self.module.model.load_state_dict(ckpt['model_state_dict'], strict=False)
            epoch = ckpt.get('epoch', '?')
            print(f"✅ Legacy checkpoint loaded (epoch {epoch}). "
                  f"Model weights restored; optimizer will start fresh.")
        else:
            # Try treating the entire file as a raw state dict
            self.module.model.load_state_dict(ckpt, strict=False)
            print(f"✅ Raw state dict loaded from {path}.")

        del ckpt

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _load_legacy_weights(self, path):
        """Load model weights from a legacy .pth checkpoint."""
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        sd = ckpt.get('model_state_dict', ckpt)
        self.module.model.load_state_dict(sd, strict=False)
        del ckpt

    def _get_precision(self):
        if not self.config.use_amp:
            return '32-true'
        if self.config.amp_dtype == 'bfloat16':
            return 'bf16-mixed'
        return '16-mixed'

    def _build_callbacks(self):
        callbacks = []

        # ── Model Checkpoint (best + last) ───────────────────────────────────
        callbacks.append(ModelCheckpoint(
            dirpath=str(self.checkpoint_dir),
            filename='best_calculus_model',
            monitor='val/dice',
            mode='max',
            save_top_k=1,
            save_last=True,
            every_n_epochs=1,
            verbose=True,
        ))

        # ── Early Stopping ───────────────────────────────────────────────────
        callbacks.append(EarlyStopping(
            monitor='val/dice',
            patience=self.config.early_stopping_patience,
            mode='max',
            verbose=True,
        ))

        # ── Model Summary (depth 2 for detailed view) ───────────────────────
        callbacks.append(ModelSummary(max_depth=2))

        # ── HuggingFace Upload ───────────────────────────────────────────────
        if self.config.hf_repo_id:
            callbacks.append(HuggingFaceUploadCallback(
                repo_id=self.config.hf_repo_id,
                save_dir=str(self.save_dir),
                upload_every_n_epochs=self.config.hf_upload_every_n_epochs,
                token=self.config.hf_token,
            ))

        return callbacks

    def _build_loggers(self):
        loggers = []

        loggers.append(TensorBoardLogger(
            save_dir=str(self.save_dir),
            name='tensorboard',
        ))

        if self.config.use_wandb:
            try:
                try:
                    from lightning.pytorch.loggers import WandbLogger
                except ImportError:
                    from pytorch_lightning.loggers import WandbLogger

                loggers.append(WandbLogger(
                    project=self.config.wandb_project,
                    entity=getattr(self.config, 'wandb_entity', None),
                    config=self.config.__dict__,
                    save_dir=str(self.save_dir),
                ))
            except ImportError:
                print("⚠️  wandb is not installed. Skipping WandB logging.")

        return loggers
