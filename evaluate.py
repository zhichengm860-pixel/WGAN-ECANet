#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WGAN-ECANet Complete Evaluation Script

Evaluation features:
1. Overall accuracy, Top-3/Top-5 accuracy
2. SNR dimension analysis (-20dB to 30dB)
3. Per-class modulation accuracy
4. Confusion matrix
5. Generated sample quality evaluation (if model includes generator)
"""

import argparse
import torch
import numpy as np
import h5py
from pathlib import Path
import sys
import os
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.path_manager import get_path_manager
from src.models import WGANECANet
from src.data import StreamingRadioMLDataset
from src.utils.metrics import (
    compute_metrics, 
    print_classification_report,
    print_snr_report,
    compute_snr_accuracy
)


class WGANECANetEvaluator:
    """WGAN-ECANet Complete Evaluator"""
    
    def __init__(self, model_path, device=None):
        pm = get_path_manager()
        self.model_path = pm.get_model_path(model_path, check_exists=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() and device != 'cpu'
                                else 'cpu')

        print(f"Using device: {self.device}")
        self._load_model()

    def _load_model(self):
        print(f"Loading model: {self.model_path}")
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

        self.model = WGANECANet(num_classes=24, use_spectral_norm=True)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.model.to(self.device)
        self.model.eval()

        print("Model loaded successfully!")
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total parameters: {total_params:,}")

    def evaluate(self, data_loader, split='test'):
        """Basic evaluation"""
        print(f"Evaluating {split} set performance...")

        all_predictions = []
        all_labels = []
        all_probabilities = []
        all_snr = []

        with torch.no_grad():
            for batch_data in tqdm(data_loader, desc=f"Evaluating {split} set"):
                signals = batch_data['signals'].to(self.device)
                labels = batch_data['labels'].to(self.device)
                
                outputs = self.model(signals, mode='classify')
                logits = outputs['logits']
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(probabilities, dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
                if 'snr' in batch_data:
                    all_snr.extend(batch_data['snr'].numpy())

        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)
        all_snr = np.array(all_snr) if all_snr else None

        metrics = compute_metrics(all_labels, all_predictions, all_probabilities)

        print(f"\n{split} set evaluation results:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        if metrics['top3_accuracy'] is not None:
            print(f"  Top-3 Accuracy: {metrics['top3_accuracy']:.4f}")
        if metrics['top5_accuracy'] is not None:
            print(f"  Top-5 Accuracy: {metrics['top5_accuracy']:.4f}")
        print(f"  F1 Score (macro): {metrics['f1_macro']:.4f}")
        print(f"  F1 Score (weighted): {metrics['f1_weighted']:.4f}")
        
        if all_snr is not None and len(all_snr) > 0:
            print_snr_report(all_snr, all_labels, all_predictions)
            snr_metrics = compute_snr_accuracy(all_snr, all_labels, all_predictions)
            metrics.update(snr_metrics)
            metrics['snr_values'] = all_snr.tolist()

        return metrics, all_labels, all_predictions, all_probabilities

    def evaluate_from_hdf5(self, hdf5_path, split='test', batch_size=256, max_samples=None):
        """Evaluate from HDF5 file"""
        print(f"Evaluating from HDF5 file: {hdf5_path}")
        
        with h5py.File(hdf5_path, 'r') as f:
            total_samples = f['X'].shape[0]
        
        if max_samples:
            total_samples = min(total_samples, max_samples)
        
        from sklearn.model_selection import train_test_split
        
        all_indices = np.arange(total_samples)
        with h5py.File(hdf5_path, 'r') as f:
            labels_chunk = f['Y'][:total_samples]
            all_labels_1d = np.argmax(labels_chunk, axis=1) if labels_chunk.ndim == 2 else labels_chunk
        
        train_indices, temp_indices = train_test_split(
            all_indices, test_size=0.2, random_state=42, stratify=all_labels_1d
        )
        
        temp_labels = all_labels_1d[temp_indices]
        val_indices, test_indices = train_test_split(
            temp_indices, test_size=0.5, random_state=42, stratify=temp_labels
        )
        
        if split == 'val':
            indices = val_indices
        elif split == 'test':
            indices = test_indices
        else:
            indices = test_indices
        
        dataset = StreamingRadioMLDataset(hdf5_path, indices, include_snr=True)
        
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        return self.evaluate(loader, split)

    def evaluate_generator(self, num_samples=1000, num_classes=24):
        """Evaluate generator quality"""
        print(f"\nEvaluating generator quality ({num_samples} samples)...")
        
        self.model.eval()
        
        with torch.no_grad():
            noise = torch.randn(num_samples, self.model.generator.noise_dim, device=self.device)
            labels = torch.randint(0, num_classes, (num_samples,), device=self.device)
            
            fake_samples = self.model.generator(noise, labels)
            
            d_fake = self.model.discriminator(fake_samples)
            avg_disc_score = d_fake.mean().item()
            
            cls_outputs = self.model.classifier(fake_samples)
            cls_predictions = torch.argmax(cls_outputs, dim=1)
            label_consistency = (cls_predictions == labels).sum().item() / num_samples
        
        print(f"  Average discriminator score: {avg_disc_score:.4f}")
        print(f"  Label consistency: {label_consistency*100:.2f}%")
        
        return {
            'avg_discriminator_score': avg_disc_score,
            'label_consistency': label_consistency
        }


def main():
    import os

    parser = argparse.ArgumentParser(description='WGAN-ECANet Complete Evaluation Script')
    parser.add_argument('--model', type=str, default=os.getenv('WGAN_MODEL_PATH', ''),
                       help='Model file path')
    parser.add_argument('--dataset', type=str, default=os.getenv('WGAN_DATASET_PATH', ''),
                       help='Dataset path')
    parser.add_argument('--split', type=str, default='test',
                       choices=['val', 'test'], help='Dataset split')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples')
    parser.add_argument('--device', type=str, default=None,
                       choices=['cuda', 'cpu'], help='Compute device')
    parser.add_argument('--output', type=str, default=None,
                       help='Result output path')
    parser.add_argument('--evaluate-generator', action='store_true',
                       help='Also evaluate generator quality')

    args = parser.parse_args()

    if not args.model:
        print("Error: --model argument or WGAN_MODEL_PATH environment variable is required")
        sys.exit(1)

    if not args.dataset:
        print("Error: --dataset argument or WGAN_DATASET_PATH environment variable is required")
        sys.exit(1)

    pm = get_path_manager()
    evaluator = WGANECANetEvaluator(args.model, args.device)

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = pm.project_root / args.dataset

    metrics, labels, predictions, probabilities = evaluator.evaluate_from_hdf5(
        str(dataset_path), args.split, args.batch_size, args.max_samples
    )
    
    if args.evaluate_generator:
        gen_metrics = evaluator.evaluate_generator()
        metrics.update(gen_metrics)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        import json
        
        save_metrics = {k: v for k, v in metrics.items() if k != 'snr_values'}
        
        with open(output_path, 'w') as f:
            json.dump(save_metrics, f, indent=2)

        print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
