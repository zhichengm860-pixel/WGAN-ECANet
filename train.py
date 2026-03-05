#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WGAN-ECANet Complete Training Script

Implements full WGAN-GP adversarial training:
1. Discriminator training: maximize Wasserstein distance
2. Generator training: generate realistic modulated signals
3. Classifier training: utilize real and generated samples
4. Dynamic gradient penalty: stabilize training process

Usage:
    # Full WGAN training
    python train.py --mode wgan

    # Classifier-only training (fast training)
    python train.py --mode classifier

    # Resume from checkpoint
    python train.py --resume results/training_xxx/checkpoint.pth
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
import os
import argparse
import json
from datetime import datetime

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.models import WGANECANet
from src.data import create_dataloaders
from src.training import WGANTrainer
from src.utils.path_manager import get_path_manager


def main():
    parser = argparse.ArgumentParser(description='WGAN-ECANet Complete Training Script')
    parser.add_argument('--mode', type=str, default='wgan',
                        choices=['wgan', 'classifier'],
                        help='Training mode: wgan(full adversarial training), classifier(classifier only)')
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint path to resume training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-4, help='Unified learning rate (paper: 2e-4)')
    parser.add_argument('--lr_g', type=float, default=None, help='Generator learning rate (default: use --lr)')
    parser.add_argument('--lr_d', type=float, default=None, help='Discriminator learning rate (default: use --lr)')
    parser.add_argument('--lr_c', type=float, default=None, help='Classifier learning rate (default: use --lr)')
    parser.add_argument('--n_critic', type=int, default=5, help='Discriminator training steps per generator step')
    parser.add_argument('--lambda_gp', type=float, default=10.0, help='Gradient penalty coefficient')
    parser.add_argument('--dynamic_gp', action='store_true', help='Use dynamic gradient penalty')
    parser.add_argument('--data_path', type=str, default=None, help='Dataset path')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--generate_samples', action='store_true', help='Generate samples after training')
    parser.add_argument('--freeze_classifier', action='store_true', help='Freeze classifier weights (stage 2 training)')
    parser.add_argument('--pretrained', type=str, default=None, help='Pretrained model path (for stage 2)')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    print("=" * 70)
    print(f"WGAN-ECANet Training - Mode: {args.mode.upper()}")
    print("=" * 70)

    # Use unified learning rate 2e-4 (paper setting)
    lr_g = args.lr_g if args.lr_g is not None else args.lr
    lr_d = args.lr_d if args.lr_d is not None else args.lr
    lr_c = args.lr_c if args.lr_c is not None else args.lr

    config = {
        'mode': args.mode,
        'num_classes': 24,
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr_g': lr_g,
        'lr_d': lr_d,
        'lr_c': lr_c,
        'n_critic': args.n_critic,
        'lambda_gp': args.lambda_gp,
        'use_dynamic_gp': args.dynamic_gp,
        'freeze_classifier': args.freeze_classifier,
        'pretrained': args.pretrained,
        'seed': args.seed
    }

    pm = get_path_manager()
    print(f"\nProject root: {pm.project_root}")

    num_gpus = torch.cuda.device_count()
    print(f"\nDetected {num_gpus} GPUs:")
    for i in range(num_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    if args.data_path:
        hdf5_path = args.data_path
    else:
        hdf5_path = pm.get_hdf5_dataset_path('RadioML 2018.01A', 'GOLD_XYZ_OSC.0001_1024.hdf5')
    print(f"\nDataset path: {hdf5_path}")

    print("\nInitializing data loaders...")
    num_workers = min(8, num_gpus * 4) if num_gpus > 0 else 4
    train_loader, val_loader, test_loader = create_dataloaders(
        hdf5_path=str(hdf5_path),
        batch_size=config['batch_size'],
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        random_state=args.seed,
        num_workers=num_workers,
        use_streaming=True
    )

    print("\nInitializing model...")
    model = WGANECANet(
        num_classes=config['num_classes'],
        use_spectral_norm=True,
        use_eca=True,
        use_multi_scale=True
    )
    model.to(device)

    # Load pretrained classifier (stage 2)
    if args.pretrained:
        print(f"\nLoading pretrained model: {args.pretrained}")
        checkpoint = torch.load(args.pretrained, map_location=device)
        if 'classifier_state_dict' in checkpoint:
            model.classifier.load_state_dict(checkpoint['classifier_state_dict'])
            print("  Classifier weights loaded")
        else:
            print("  Warning: Classifier weights not found in checkpoint")

    # Freeze classifier (stage 2)
    if args.freeze_classifier:
        for param in model.classifier.parameters():
            param.requires_grad = False
        print("  Classifier frozen (not participating in training)")

    total_params = sum(p.numel() for p in model.parameters())
    gen_params = sum(p.numel() for p in model.generator.parameters())
    disc_params = sum(p.numel() for p in model.discriminator.parameters())
    cls_params = sum(p.numel() for p in model.classifier.parameters())

    print(f"Model parameter statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Generator: {gen_params:,}")
    print(f"  Discriminator: {disc_params:,}")
    print(f"  Classifier: {cls_params:,}")

    print("\nInitializing WGAN trainer...")
    trainer = WGANTrainer(
        model=model,
        device=device,
        lr_g=config['lr_g'],
        lr_d=config['lr_d'],
        lr_c=config['lr_c'],
        lambda_gp=config['lambda_gp'],
        n_critic=config['n_critic'],
        use_dynamic_gp=config['use_dynamic_gp']
    )

    total_steps = len(train_loader) * config['num_epochs']
    trainer.set_total_steps(total_steps)
    print(f"Total training steps: {total_steps:,}")

    start_epoch = 0
    best_val_acc = 0.0
    history = {
        'train_loss_D': [], 'train_loss_G': [], 'train_loss_C': [],
        'train_accuracy': [], 'val_accuracy': [], 'wasserstein_distance': [],
        'gradient_penalty': [], 'lambda_gp': []
    }

    if args.resume:
        checkpoint_path = Path(args.resume)
        if checkpoint_path.exists():
            print(f"\nResuming from checkpoint: {checkpoint_path}")
            start_epoch, metrics = trainer.load_checkpoint(str(checkpoint_path))
            start_epoch += 1
            best_val_acc = metrics.get('val_accuracy', 0.0)
            print(f"Resumed from Epoch {start_epoch}, Best validation accuracy: {best_val_acc:.2f}%")

    if args.output_dir:
        save_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = pm.get_results_dir() / f'wgan_training_{timestamp}'
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\nResults save directory: {save_dir}")

    train_discriminator = (args.mode == 'wgan')
    train_generator = (args.mode == 'wgan')
    train_classifier = not args.freeze_classifier  # Do not train classifier when frozen

    print("\nStarting training...")
    print("-" * 70)
    print(f"Training configuration:")
    print(f"  Train discriminator: {train_discriminator}")
    print(f"  Train generator: {train_generator}")
    print(f"  Train classifier: {train_classifier}")
    if train_discriminator:
        print(f"  n_critic: {config['n_critic']}")
        print(f"  Gradient penalty coefficient: {config['lambda_gp']}")
        print(f"  Dynamic gradient penalty: {config['use_dynamic_gp']}")
    print("-" * 70)

    for epoch in range(start_epoch, config['num_epochs']):
        train_metrics = trainer.train_epoch(
            train_loader, epoch,
            train_discriminator=train_discriminator,
            train_generator=train_generator,
            train_classifier=train_classifier
        )

        val_metrics = trainer.validate(val_loader, epoch)

        history['train_loss_D'].append(train_metrics.get('loss_D', 0))
        history['train_loss_G'].append(train_metrics.get('loss_G', 0))
        history['train_loss_C'].append(train_metrics.get('loss_C', 0))
        history['train_accuracy'].append(train_metrics.get('accuracy', 0) * 100)
        history['val_accuracy'].append(val_metrics['val_accuracy'])
        history['wasserstein_distance'].append(train_metrics.get('wasserstein_distance', 0))
        history['gradient_penalty'].append(train_metrics.get('gradient_penalty', 0))
        history['lambda_gp'].append(trainer.current_lambda)

        print(f'\nEpoch {epoch+1}/{config["num_epochs"]}:')
        if train_discriminator:
            print(f'  Loss D: {train_metrics["loss_D"]:.4f}')
            print(f'  Loss G: {train_metrics["loss_G"]:.4f}')
            print(f'  Wasserstein Distance: {train_metrics["wasserstein_distance"]:.4f}')
            print(f'  Gradient Penalty: {train_metrics["gradient_penalty"]:.4f}')
            print(f'  Lambda GP: {trainer.current_lambda:.2f}')
        print(f'  Loss C: {train_metrics["loss_C"]:.4f}')
        print(f'  Train Acc: {train_metrics["accuracy"]*100:.2f}%')
        print(f'  Val Acc: {val_metrics["val_accuracy"]:.2f}%')

        if val_metrics['val_accuracy'] > best_val_acc:
            best_val_acc = val_metrics['val_accuracy']

            save_path = save_dir / 'best_model.pth'
            trainer.save_checkpoint(str(save_path), epoch, val_metrics)
            print(f'  Saving best model (Val Acc: {val_metrics["val_accuracy"]:.2f}%)')

        if (epoch + 1) % 10 == 0:
            checkpoint_path = save_dir / f'checkpoint_epoch_{epoch+1}.pth'
            trainer.save_checkpoint(str(checkpoint_path), epoch, val_metrics)

        with open(save_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

        print("-" * 70)

    print("\n" + "=" * 70)
    print("Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved at: {save_dir / 'best_model.pth'}")
    print("=" * 70)

    if args.generate_samples and train_generator:
        print("\nGenerating sample demonstration...")
        num_samples = 100
        fake_samples, fake_labels = trainer.generate_samples(num_samples)

        samples_path = save_dir / 'generated_samples.npz'
        np.savez(samples_path,
                 samples=fake_samples.cpu().numpy(),
                 labels=fake_labels.cpu().numpy())
        print(f"Generated samples saved to: {samples_path}")

        print("\nEvaluating generated sample quality...")
        model.eval()
        with torch.no_grad():
            outputs = model.classifier(fake_samples)
            predictions = torch.argmax(outputs, dim=1)
            consistency = (predictions == fake_labels).sum().item() / num_samples
            print(f"Generated sample-label consistency: {consistency*100:.2f}%")


if __name__ == '__main__':
    main()
