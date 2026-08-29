"""
Trainer for calculus segmentation head.
Single-stage training (no Hungarian matching needed for binary segmentation).
"""

import torch
import numpy as np
from tqdm import tqdm
import logging
import json
from pathlib import Path
import random
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.loss.lwcaLR import LinearWarmupCosineAnnealingLR
from model.loss.loss_calculus import CalculusLoss
from model.dataset.hdf5_calculus import convert_calculus_to_hdf5, generate_calculus_splits
from model.dataset.calculus_dataset import HDF5CalculusDataset
from model.calculus.calculus_system import CalculusSegmentationSystem
from model.calculus.lora import get_lora_params


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy data types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


class CalculusTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # AMP training
        self.use_amp = config.use_amp
        self.amp_dtype = config.get_torch_dtype()

        self.use_scaler = self.use_amp and self.amp_dtype == torch.float16
        if self.use_scaler:
            self.scaler = torch.amp.GradScaler()

        # Set random seed
        self.set_seed(config.seed)

        # Create directories
        self.setup_directories()

        # Setup logging
        self.logger = self.setup_logging()

        # Initialize dataset and model
        self.setup_data()
        self.setup_model()

    def set_seed(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def setup_directories(self):
        """Create necessary directories."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.save_dir = Path(self.config.save_dir) / timestamp
        self.log_dir = self.save_dir / 'logs'
        self.checkpoint_dir = self.save_dir / 'checkpoints'
        self.tensorboard_dir = self.save_dir / 'tensorboard'

        for dir_path in [self.log_dir, self.checkpoint_dir, self.tensorboard_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(self.tensorboard_dir)

    def setup_logging(self):
        """Setup logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / 'train.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger()

    def setup_data(self):
        """Initialize datasets and dataloaders."""
        h5_path = Path(self.config.h5_path)
        data_dir = self.config.data_dir

        # Auto-convert NPZ to HDF5 if needed
        if not h5_path.exists():
            self.logger.info(f"HDF5 file {h5_path} not found, starting conversion from NPZ files...")
            convert_calculus_to_hdf5(data_dir, str(h5_path))
            self.logger.info(f"HDF5 file created: {h5_path}")

        # Auto-generate splits if needed
        train_split = Path(self.config.train_split_path)
        val_split = Path(self.config.val_split_path)

        if not train_split.exists() or not val_split.exists():
            self.logger.info("Split files not found, generating splits...")
            split_dir = train_split.parent
            generate_calculus_splits(data_dir, str(split_dir))
            self.logger.info(f"Splits generated in {split_dir}")

        view_indices = self.config.view_indices

        # Create datasets
        train_dataset = HDF5CalculusDataset(
            str(h5_path),
            transform=True,
            mode='train',
            split_file_path=str(train_split),
            view_indices=view_indices
        )
        val_dataset = HDF5CalculusDataset(
            str(h5_path),
            transform=False,
            mode='val',
            split_file_path=str(val_split),
            view_indices=view_indices
        )

        batch_size = self.config.batch_size
        num_workers = self.config.num_workers

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False
        )

        self.logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    def setup_model(self):
        """Initialize model, loss function, and optimizer."""
        # Create the calculus model
        self.model = CalculusSegmentationSystem(self.config).to(self.device)

        # Optionally load pre-trained tooth features
        if self.config.tooth_checkpoint and Path(self.config.tooth_checkpoint).exists():
            self.logger.info(f"Loading pre-trained tooth features from {self.config.tooth_checkpoint}")
            self.model.load_tooth_features(self.config.tooth_checkpoint)

        # Loss function
        self.criterion = CalculusLoss().to(self.device)

        # Collect parameter groups
        lora_params = get_lora_params(self.model.sam_model)
        fusion_params = list(self.model.feature_fusion.parameters())
        calculus_head_params = (
            list(self.model.calculus_peg.parameters()) +
            list(self.model.refine_net.parameters())
        )

        lr = self.config.learning_rate
        lora_lr = self.config.lora_lr
        weight_decay = self.config.weight_decay

        param_groups = []
        if lora_params:
            param_groups.append({'params': lora_params, 'lr': lora_lr, 'weight_decay': weight_decay})
        if fusion_params:
            param_groups.append({'params': fusion_params, 'lr': lr, 'weight_decay': weight_decay})
        if calculus_head_params:
            param_groups.append({'params': calculus_head_params, 'lr': lr, 'weight_decay': weight_decay})

        self.optimizer = torch.optim.AdamW(param_groups)

        # Scheduler
        self.scheduler = LinearWarmupCosineAnnealingLR(
            self.optimizer,
            warmup_epochs=self.config.warmup_epochs,
            max_epochs=self.config.epochs
        )

        # Log trainable params
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        self.logger.info(f"LoRA parameters: {sum(p.numel() for p in lora_params):,}")

    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }
        if self.use_scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_path = self.checkpoint_dir / 'best_calculus_model.pth'
            torch.save(checkpoint, best_path)
            self.logger.info(f'Best model saved: {best_path}')

    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint for resuming training or testing."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if self.use_scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        self.logger.info(f"Loaded checkpoint from epoch {start_epoch}")
        return start_epoch

    def _compute_binary_metrics(self, pred_probs, gt_masks):
        """
        Compute binary segmentation metrics for calculus channel.

        Args:
            pred_probs: [B, 2, H, W] probabilities (after softmax)
            gt_masks: [B, 2, H, W] one-hot ground truth
        """
        # Use calculus channel (channel 1)
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
            'recall': recall.mean().item()
        }

    def train_one_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_metrics = {'iou': 0, 'dice': 0, 'precision': 0, 'recall': 0}
        batches = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.config.epochs}')
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            gt_masks = batch['calculus_mask'].to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.amp.autocast(device_type='cuda', dtype=self.amp_dtype):
                    sam_masks, refined_masks, confidence = self.model(images)
                    loss, loss_dict = self.criterion(sam_masks, refined_masks, confidence, gt_masks)

                if self.use_scaler:
                    self.scaler.scale(loss).backward()
                    if self.config.grad_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    if self.config.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                    self.optimizer.step()
            else:
                sam_masks, refined_masks, confidence = self.model(images)
                loss, loss_dict = self.criterion(sam_masks, refined_masks, confidence, gt_masks)
                loss.backward()
                if self.config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.optimizer.step()

            # Compute metrics
            with torch.no_grad():
                refined_probs = torch.softmax(refined_masks, dim=1)
                metrics = self._compute_binary_metrics(refined_probs, gt_masks)

            total_loss += loss.item()
            for k in total_metrics:
                total_metrics[k] += metrics[k]
            batches += 1

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'dice': f"{metrics['dice']:.4f}"
            })

            # TensorBoard logging
            if batch_idx % self.config.log_freq == 0:
                step = (epoch - 1) * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/Loss', loss.item(), step)
                for k, v in loss_dict.items():
                    self.writer.add_scalar(f'Train/{k}', v, step)

        avg_loss = total_loss / max(1, batches)
        avg_metrics = {k: v / max(1, batches) for k, v in total_metrics.items()}

        self.writer.add_scalar('Epoch/Train_Loss', avg_loss, epoch)
        self.writer.add_scalar('Epoch/Train_Dice', avg_metrics['dice'], epoch)
        self.writer.add_scalar('Epoch/Train_IoU', avg_metrics['iou'], epoch)

        return avg_loss, avg_metrics

    @torch.no_grad()
    def validate(self, epoch):
        """Run validation."""
        self.model.eval()
        total_loss = 0
        total_metrics = {'iou': 0, 'dice': 0, 'precision': 0, 'recall': 0}
        batches = 0

        pbar = tqdm(self.val_loader, desc='Validation')
        for batch in pbar:
            images = batch['image'].to(self.device)
            gt_masks = batch['calculus_mask'].to(self.device)

            if self.use_amp:
                with torch.amp.autocast(device_type='cuda', dtype=self.amp_dtype):
                    sam_masks, refined_masks, confidence = self.model(images)
                    loss, loss_dict = self.criterion(sam_masks, refined_masks, confidence, gt_masks)
            else:
                sam_masks, refined_masks, confidence = self.model(images)
                loss, loss_dict = self.criterion(sam_masks, refined_masks, confidence, gt_masks)

            refined_probs = torch.softmax(refined_masks, dim=1)
            metrics = self._compute_binary_metrics(refined_probs, gt_masks)

            total_loss += loss.item()
            for k in total_metrics:
                total_metrics[k] += metrics[k]
            batches += 1

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'dice': f"{metrics['dice']:.4f}"
            })

        avg_loss = total_loss / max(1, batches)
        avg_metrics = {k: v / max(1, batches) for k, v in total_metrics.items()}

        self.writer.add_scalar('Val/Loss', avg_loss, epoch)
        for k, v in avg_metrics.items():
            self.writer.add_scalar(f'Val/{k}', v, epoch)

        return avg_loss, avg_metrics

    def train(self):
        """Main training loop with early stopping."""
        best_val_dice = 0.0
        patience = self.config.early_stopping_patience
        epochs_no_improve = 0

        self.logger.info("=" * 60)
        self.logger.info("Starting Calculus Segmentation Training")
        self.logger.info(f"Epochs: {self.config.epochs}, Batch Size: {self.config.batch_size}")
        self.logger.info(f"LR: {self.config.learning_rate}, LoRA LR: {self.config.lora_lr}")
        self.logger.info(f"Early Stopping Patience: {patience}")
        self.logger.info("=" * 60)

        for epoch in range(1, self.config.epochs + 1):
            train_loss, train_metrics = self.train_one_epoch(epoch)

            # Validate at interval
            if epoch % self.config.val_interval == 0 or epoch == self.config.epochs:
                val_loss, val_metrics = self.validate(epoch)

                self.logger.info(
                    f"Epoch {epoch} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val Dice: {val_metrics['dice']:.4f} | "
                    f"Val IoU: {val_metrics['iou']:.4f}"
                )

                # Early stopping on validation Dice
                if val_metrics['dice'] > best_val_dice:
                    best_val_dice = val_metrics['dice']
                    self.save_checkpoint(epoch, is_best=True)
                    epochs_no_improve = 0
                    self.logger.info(f"New best validation Dice: {best_val_dice:.4f}")
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    self.logger.info(f"Early stopping triggered after {epoch} epochs (no improvement for {patience} validations)")
                    break

            # Periodic save
            if epoch % self.config.save_freq == 0:
                self.save_checkpoint(epoch)

            self.scheduler.step()

        self.writer.close()
        self.logger.info(f"Training complete. Best Val Dice: {best_val_dice:.4f}")

    @torch.no_grad()
    def test(self, checkpoint_path):
        """Run testing with a specific checkpoint."""
        self.load_checkpoint(checkpoint_path)
        self.model.eval()

        total_loss = 0
        total_metrics = {'iou': 0, 'dice': 0, 'precision': 0, 'recall': 0}
        batches = 0

        pbar = tqdm(self.val_loader, desc='Testing')
        for batch in pbar:
            images = batch['image'].to(self.device)
            gt_masks = batch['calculus_mask'].to(self.device)

            sam_masks, refined_masks, confidence = self.model(images)
            loss, _ = self.criterion(sam_masks, refined_masks, confidence, gt_masks)

            refined_probs = torch.softmax(refined_masks, dim=1)
            metrics = self._compute_binary_metrics(refined_probs, gt_masks)

            total_loss += loss.item()
            for k in total_metrics:
                total_metrics[k] += metrics[k]
            batches += 1

        avg_loss = total_loss / max(1, batches)
        final_metrics = {k: v / max(1, batches) for k, v in total_metrics.items()}

        self.logger.info("=" * 40)
        self.logger.info("Test Results:")
        self.logger.info(f"  Loss: {avg_loss:.4f}")
        for k, v in final_metrics.items():
            self.logger.info(f"  {k.capitalize()}: {v:.4f}")
        self.logger.info("=" * 40)

        return avg_loss, final_metrics
