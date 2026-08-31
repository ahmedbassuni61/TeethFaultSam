"""
PyTorch Lightning components for calculus segmentation training.

Contains:
    - CalculusLightningModule: LightningModule wrapping the model and loss
    - CalculusDataModule: LightningDataModule wrapping the dataset and dataloaders
    - HuggingFaceUploadCallback: Callback for periodic checkpoint uploads to HF Hub
"""

import os
import torch
import torch.nn.functional as F
from pathlib import Path

try:
    from lightning.pytorch import LightningModule, LightningDataModule
    from lightning.pytorch.callbacks import Callback
except ImportError:
    from pytorch_lightning import LightningModule, LightningDataModule
    from pytorch_lightning.callbacks import Callback

from torch.utils.data import DataLoader

from model.calculus.calculus_system import CalculusSegmentationSystem
from model.calculus.lora import get_lora_params
from model.loss.loss_calculus import CalculusLoss
from model.loss.lwcaLR import LinearWarmupCosineAnnealingLR
from model.dataset.calculus_dataset import HDF5CalculusDataset
from model.dataset.hdf5_calculus import convert_calculus_to_hdf5, generate_calculus_splits


# ──────────────────────────────────────────────────────────────────────────────
# Lightning Module
# ──────────────────────────────────────────────────────────────────────────────

class CalculusLightningModule(LightningModule):
    """
    PyTorch Lightning wrapper for the calculus segmentation system.

    Handles training, validation, test steps, metric logging, and optimizer setup.
    Progress bars show live per-batch loss, Dice, and IoU.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Build the model
        self.model = CalculusSegmentationSystem(config)

        # Load pre-trained tooth features if checkpoint exists
        if config.tooth_checkpoint and Path(config.tooth_checkpoint).exists():
            self.model.load_tooth_features(config.tooth_checkpoint)
            print(f"✅ Loaded pre-trained tooth features from {config.tooth_checkpoint}")

        # Loss function
        self.criterion = CalculusLoss()

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, images):
        return self.model(images)

    # ── Training ─────────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        images = batch['image']
        gt_masks = batch['calculus_mask']

        sam_masks, refined_masks, confidence = self.model(images)
        loss, loss_dict = self.criterion(sam_masks, refined_masks, confidence, gt_masks)

        # Compute binary metrics
        with torch.no_grad():
            refined_probs = torch.softmax(refined_masks, dim=1)
            metrics = self._compute_binary_metrics(refined_probs, gt_masks)

        # Log to progress bar (on_step) and epoch summary (on_epoch)
        self.log('train/loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train/dice', metrics['dice'], prog_bar=True, on_step=True, on_epoch=True)
        self.log('train/iou', metrics['iou'], prog_bar=True, on_step=True, on_epoch=True)

        # Log individual loss components (epoch-level only, keeps the bar clean)
        for k, v in loss_dict.items():
            self.log(f'train/{k}', v, on_step=False, on_epoch=True)

        return loss

    # ── Validation ───────────────────────────────────────────────────────────

    def validation_step(self, batch, batch_idx):
        images = batch['image']
        gt_masks = batch['calculus_mask']

        sam_masks, refined_masks, confidence = self.model(images)
        loss, loss_dict = self.criterion(sam_masks, refined_masks, confidence, gt_masks)

        refined_probs = torch.softmax(refined_masks, dim=1)
        metrics = self._compute_binary_metrics(refined_probs, gt_masks)

        self.log('val/loss', loss, prog_bar=True, on_epoch=True)
        self.log('val/dice', metrics['dice'], prog_bar=True, on_epoch=True)
        self.log('val/iou', metrics['iou'], prog_bar=True, on_epoch=True)
        self.log('val/precision', metrics['precision'], on_epoch=True)
        self.log('val/recall', metrics['recall'], on_epoch=True)

        return loss

    def on_validation_epoch_end(self):
        """Print a clean epoch summary after each validation."""
        metrics = self.trainer.callback_metrics
        epoch = self.current_epoch + 1

        parts = [f"Epoch {epoch:3d}"]
        for key, label in [
            ('train/loss_epoch', 'Train Loss'),
            ('val/loss', 'Val Loss'),
            ('val/dice', 'Val Dice'),
            ('val/iou', 'Val IoU'),
            ('val/precision', 'Val Prec'),
            ('val/recall', 'Val Recall'),
        ]:
            val = metrics.get(key)
            if val is not None:
                val = val.item() if hasattr(val, 'item') else val
                parts.append(f"{label}: {val:.4f}")

        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        parts.append(f"LR: {lr:.2e}")

        print(f"\n{'─' * 70}")
        print(" │ ".join(parts))
        print(f"{'─' * 70}")

    # ── Test ─────────────────────────────────────────────────────────────────

    def test_step(self, batch, batch_idx):
        images = batch['image']
        gt_masks = batch['calculus_mask']

        sam_masks, refined_masks, confidence = self.model(images)
        loss, loss_dict = self.criterion(sam_masks, refined_masks, confidence, gt_masks)

        refined_probs = torch.softmax(refined_masks, dim=1)
        metrics = self._compute_binary_metrics(refined_probs, gt_masks)

        self.log('test/loss', loss, on_epoch=True)
        self.log('test/dice', metrics['dice'], on_epoch=True)
        self.log('test/iou', metrics['iou'], on_epoch=True)
        self.log('test/precision', metrics['precision'], on_epoch=True)
        self.log('test/recall', metrics['recall'], on_epoch=True)

        return loss

    def on_test_epoch_end(self):
        metrics = self.trainer.callback_metrics
        print(f"\n{'=' * 50}")
        print(f"🧪  Test Results")
        print(f"{'=' * 50}")
        for key in ['test/loss', 'test/dice', 'test/iou', 'test/precision', 'test/recall']:
            val = metrics.get(key)
            if val is not None:
                val = val.item() if hasattr(val, 'item') else val
                print(f"  {key.split('/')[-1].capitalize():>12}: {val:.4f}")
        print(f"{'=' * 50}\n")

    # ── Optimizer & Scheduler ────────────────────────────────────────────────

    def configure_optimizers(self):
        """
        Three parameter groups with separate learning rates:
          1. LoRA adapters   → lora_lr
          2. Feature fusion  → learning_rate
          3. Calculus head   → learning_rate
        """
        lora_params = get_lora_params(self.model.sam_model)
        fusion_params = list(self.model.feature_fusion.parameters())
        calculus_head_params = (
            list(self.model.calculus_peg.parameters()) +
            list(self.model.refine_net.parameters())
        )

        lr = self.config.learning_rate
        lora_lr = self.config.lora_lr
        wd = self.config.weight_decay

        param_groups = []
        if lora_params:
            param_groups.append({'params': lora_params, 'lr': lora_lr, 'weight_decay': wd})
        if fusion_params:
            param_groups.append({'params': fusion_params, 'lr': lr, 'weight_decay': wd})
        if calculus_head_params:
            param_groups.append({'params': calculus_head_params, 'lr': lr, 'weight_decay': wd})

        optimizer = torch.optim.AdamW(param_groups)

        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.config.warmup_epochs,
            max_epochs=self.config.epochs,
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
            },
        }

    # ── Startup Info ─────────────────────────────────────────────────────────

    def on_fit_start(self):
        """Print detailed model and training configuration at the start."""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        lora = sum(p.numel() for p in get_lora_params(self.model.sam_model))
        frozen = total - trainable

        c = self.config
        print(f"\n{'=' * 60}")
        print(f"🏗️   Model Architecture")
        print(f"{'=' * 60}")
        print(f"  Total parameters:     {total:>12,}")
        print(f"  Trainable parameters: {trainable:>12,}")
        print(f"  LoRA parameters:      {lora:>12,}")
        print(f"  Frozen parameters:    {frozen:>12,}")
        print(f"{'─' * 60}")
        print(f"🚀  Training Configuration")
        print(f"{'─' * 60}")
        print(f"  Epochs:            {c.epochs}")
        print(f"  Batch size:        {c.batch_size}")
        print(f"  Learning rate:     {c.learning_rate}")
        print(f"  LoRA LR:           {c.lora_lr}")
        print(f"  Weight decay:      {c.weight_decay}")
        print(f"  Warmup epochs:     {c.warmup_epochs}")
        print(f"  Gradient clip:     {c.grad_clip}")
        print(f"  AMP dtype:         {c.amp_dtype}")
        print(f"  Early stop after:  {c.early_stopping_patience} val epochs w/o improvement")
        if c.batches_per_epoch > 0:
            print(f"  Batches/epoch:     {c.batches_per_epoch}  (shortened)")
        if c.hf_repo_id:
            print(f"{'─' * 60}")
            print(f"☁️   HuggingFace Hub")
            print(f"{'─' * 60}")
            print(f"  Repo:              {c.hf_repo_id}")
            print(f"  Upload every:      {c.hf_upload_every_n_epochs} epochs")
        print(f"{'=' * 60}\n")

    # ── Metrics ──────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_binary_metrics(pred_probs, gt_masks):
        """Compute IoU, Dice, Precision, Recall for the calculus channel."""
        pred_bin = (pred_probs[:, 1:2] > 0.5).float()
        gt_bin = gt_masks[:, 1:2].float()

        eps = 1e-6
        intersection = (pred_bin * gt_bin).sum(dim=(2, 3))
        pred_sum = pred_bin.sum(dim=(2, 3))
        gt_sum = gt_bin.sum(dim=(2, 3))
        union = pred_sum + gt_sum - intersection

        iou = (intersection + eps) / (union + eps)
        dice = (2 * intersection + eps) / (pred_sum + gt_sum + eps)
        precision = (intersection + eps) / (pred_sum + eps)
        recall = (intersection + eps) / (gt_sum + eps)

        return {
            'iou': iou.mean().item(),
            'dice': dice.mean().item(),
            'precision': precision.mean().item(),
            'recall': recall.mean().item(),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Data Module
# ──────────────────────────────────────────────────────────────────────────────

class CalculusDataModule(LightningDataModule):
    """
    Handles HDF5 dataset creation, split generation, and DataLoader construction.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

    def prepare_data(self):
        """Convert NPZ → H5 and generate splits if needed. Runs once on rank 0."""
        h5_path = Path(self.config.h5_path)
        data_dir = self.config.data_dir

        if not h5_path.exists():
            print(f"HDF5 file {h5_path} not found — converting from NPZ files …")
            convert_calculus_to_hdf5(data_dir, str(h5_path))
            print(f"HDF5 created: {h5_path}")

        train_split = Path(self.config.train_split_path)
        val_split = Path(self.config.val_split_path)

        if not train_split.exists() or not val_split.exists():
            print("Split files not found — generating …")
            split_dir = train_split.parent
            generate_calculus_splits(data_dir, str(split_dir), h5_path=str(h5_path))

    def setup(self, stage=None):
        """Create dataset instances."""
        h5 = str(self.config.h5_path)
        views = self.config.view_indices

        if stage in ('fit', None):
            self.train_dataset = HDF5CalculusDataset(
                h5, transform=True, mode='train',
                split_file_path=str(self.config.train_split_path),
                view_indices=views,
            )
            self.val_dataset = HDF5CalculusDataset(
                h5, transform=False, mode='val',
                split_file_path=str(self.config.val_split_path),
                view_indices=views,
            )
            print(f"📂 Train samples: {len(self.train_dataset)}, Val samples: {len(self.val_dataset)}")

        if stage in ('test', None):
            self.test_dataset = HDF5CalculusDataset(
                h5, transform=False, mode='test',
                split_file_path=str(self.config.val_split_path),
                view_indices=views,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )


# ──────────────────────────────────────────────────────────────────────────────
# HuggingFace Upload Callback
# ──────────────────────────────────────────────────────────────────────────────

class HuggingFaceUploadCallback(Callback):
    """
    Uploads a minimal resume checkpoint to HuggingFace Hub every N epochs.

    Only the single ``resume_checkpoint.ckpt`` file is uploaded — this contains
    the full training state (model, optimizer, scheduler, epoch) so training
    can be continued later from the exact same point.

    On training end the best model is also uploaded as ``best_calculus_model.ckpt``.
    """

    def __init__(self, repo_id, save_dir, upload_every_n_epochs=5, token=None):
        super().__init__()
        self.repo_id = repo_id
        self.save_dir = save_dir
        self.upload_every_n_epochs = upload_every_n_epochs
        self.token = token
        self._api = None

    @property
    def api(self):
        if self._api is None:
            from huggingface_hub import HfApi
            self._api = HfApi(token=self.token)
        return self._api

    def _ensure_repo(self):
        """Create the HF repo if it doesn't exist yet."""
        try:
            self.api.create_repo(repo_id=self.repo_id, repo_type='model', exist_ok=True)
        except Exception:
            pass  # repo already exists or other non-fatal issue

    def _upload_resume_checkpoint(self, trainer, tag="periodic"):
        """Save current training state and push to HuggingFace."""
        try:
            self._ensure_repo()
            ckpt_path = os.path.join(self.save_dir, 'resume_checkpoint.ckpt')
            trainer.save_checkpoint(ckpt_path)

            epoch = trainer.current_epoch + 1
            print(f"\n📤 [{tag}] Uploading resume checkpoint (epoch {epoch}) to HuggingFace …")
            self.api.upload_file(
                path_or_fileobj=ckpt_path,
                path_in_repo='resume_checkpoint.ckpt',
                repo_id=self.repo_id,
                repo_type='model',
            )
            print(f"✅ Upload complete → https://huggingface.co/{self.repo_id}")
        except Exception as e:
            print(f"⚠️  HuggingFace upload failed ({tag}): {e}")

    def _upload_best_model(self, trainer):
        """Upload the best validation model if available."""
        try:
            best_path = trainer.checkpoint_callback.best_model_path
            if best_path and os.path.exists(best_path):
                print(f"📤 Uploading best model to HuggingFace …")
                self.api.upload_file(
                    path_or_fileobj=best_path,
                    path_in_repo='best_calculus_model.ckpt',
                    repo_id=self.repo_id,
                    repo_type='model',
                )
                print(f"✅ Best model uploaded → https://huggingface.co/{self.repo_id}")
        except Exception as e:
            print(f"⚠️  Best-model upload failed: {e}")

    # ── Hooks ────────────────────────────────────────────────────────────────

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if epoch % self.upload_every_n_epochs == 0:
            self._upload_resume_checkpoint(trainer, tag=f"epoch {epoch}")

    def on_train_end(self, trainer, pl_module):
        self._upload_resume_checkpoint(trainer, tag="final")
        self._upload_best_model(trainer)
