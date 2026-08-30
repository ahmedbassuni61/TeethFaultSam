"""
Training entrypoint for calculus segmentation head.

Usage:
    python train_calculus.py
    python train_calculus.py --tooth_checkpoint ckpts/best.pth --use_wandb
    python train_calculus.py --batches_per_epoch 50 --use_wandb
"""

import os
import argparse
import json

from config.config_calculus import CalculusConfig
from model.trainer_calculus import CalculusTrainer


def main():
    parser = argparse.ArgumentParser(description='Calculus Segmentation Training')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test', 'train_test'],
                        help='Running mode: train, test, train_test')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (JSON format)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint for testing')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint for resuming training')
    parser.add_argument('--tooth_checkpoint', type=str, default=None,
                        help='Path to pre-trained tooth model (for feature_fusion init)')

    # Training overrides
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=None, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--data_dir', type=str, default=None, help='Preprocessed data directory')
    
    # QoL and Logging features
    parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default=None, help='Wandb project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='Wandb entity (username or team)')
    parser.add_argument('--batches_per_epoch', type=int, default=None, 
                        help='Artificially shorten epochs to N batches to see loss faster')

    args = parser.parse_args()

    # Load default config
    config = CalculusConfig()

    # Override with JSON config if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            loaded_config = json.load(f)
            for k, v in loaded_config.items():
                if hasattr(config, k):
                    setattr(config, k, v)

    # CLI overrides
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.tooth_checkpoint is not None:
        config.tooth_checkpoint = args.tooth_checkpoint
    if args.data_dir is not None:
        config.data_dir = args.data_dir
        config.h5_path = os.path.join(args.data_dir, 'h5', 'calculus_dataset.h5')
    
    if args.use_wandb:
        config.use_wandb = True
    if args.wandb_project is not None:
        config.wandb_project = args.wandb_project
    if args.wandb_entity is not None:
        config.wandb_entity = args.wandb_entity
    if args.batches_per_epoch is not None:
        config.batches_per_epoch = args.batches_per_epoch

    print("=" * 60)
    print("Calculus Segmentation Training")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"LoRA rank: {config.lora_rank}, lr: {config.lora_lr}")
    print(f"Epochs: {config.epochs}")
    if config.batches_per_epoch > 0:
        print(f"Batches per epoch: {config.batches_per_epoch} (Shortened)")
    print(f"WandB Logging: {config.use_wandb}")
    print(f"Data dir: {config.data_dir}")
    print(f"Tooth checkpoint: {config.tooth_checkpoint}")
    print("=" * 60)

    # Create trainer
    trainer = CalculusTrainer(config)

    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Training
    if args.mode in ['train', 'train_test']:
        print("Starting calculus segmentation training...")
        trainer.train()
        print("Training completed.")

    # Testing
    if args.mode in ['test', 'train_test']:
        checkpoint_to_test = args.checkpoint
        if args.mode == 'train_test' and not checkpoint_to_test:
            best_ckpt = trainer.checkpoint_dir / 'best_calculus_model.pth'
            if best_ckpt.exists():
                checkpoint_to_test = str(best_ckpt)
                print(f"Using best model for testing: {checkpoint_to_test}")

        if checkpoint_to_test:
            print(f"Testing with checkpoint: {checkpoint_to_test}")
            test_loss, test_metrics = trainer.test(checkpoint_to_test)
        else:
            print("No checkpoint found for testing.")


if __name__ == '__main__':
    main()
