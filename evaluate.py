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
from src.models.enhanced_wgan_ecanet import EnhancedWGANECANet
from src.data.radioml_dataloader import RadioMLDataLoader
from src.utils.metrics import compute_metrics


class WGANECANetEvaluator:

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

        self.model = EnhancedWGANECANet(num_classes=24, use_spectral_norm=True)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.model.to(self.device)
        self.model.eval()

        print("Model loaded successfully!")

    def evaluate(self, data_loader, split='test'):
        print(f"Evaluating {split} set performance...")

        all_predictions = []
        all_labels = []
        all_probabilities = []

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

        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)

        metrics = compute_metrics(all_labels, all_predictions, all_probabilities)

        print(f"\n{split} set evaluation results:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        if metrics['top3_accuracy'] is not None:
            print(f"  Top-3 Accuracy: {metrics['top3_accuracy']:.4f}")
        if metrics['top5_accuracy'] is not None:
            print(f"  Top-5 Accuracy: {metrics['top5_accuracy']:.4f}")
        print(f"  F1 Score (macro): {metrics['f1_macro']:.4f}")
        print(f"  F1 Score (weighted): {metrics['f1_weighted']:.4f}")

        return metrics

    def evaluate_from_hdf5(self, hdf5_path, split='val', batch_size=256, max_samples=None):
        print(f"Evaluating from HDF5 file: {hdf5_path}")

        data_loader = RadioMLDataLoader(hdf5_path, max_samples=max_samples)

        if split == 'val':
            dataset = data_loader.get_validation_dataset()
        elif split == 'test':
            dataset = data_loader.get_test_dataset()
        else:
            dataset = data_loader.get_train_dataset()

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )

        return self.evaluate(loader, split)

    def compute_class_accuracy(self, hdf5_path, split='val'):
        data_loader = RadioMLDataLoader(hdf5_path)

        if split == 'val':
            dataset = data_loader.get_validation_dataset()
        else:
            dataset = data_loader.get_test_dataset()

        class_correct = {}
        class_total = {}

        with torch.no_grad():
            for batch_data in tqdm(dataset, desc="Computing per-class accuracy"):
                signals = batch_data['signals'].unsqueeze(0).to(self.device)
                label = batch_data['label']

                outputs = self.model(signals, mode='classify')
                logits = outputs['logits']
                prediction = torch.argmax(logits, dim=1).item()

                if label not in class_total:
                    class_total[label] = 0
                    class_correct[label] = 0

                class_total[label] += 1
                if prediction == label:
                    class_correct[label] += 1

        class_acc = {}
        for label in class_total:
            class_acc[label] = class_correct[label] / class_total[label]

        print("\nPer-class accuracy:")
        for label in sorted(class_acc.keys()):
            print(f"  Class {label:2d}: {class_acc[label]:.4f}")

        return class_acc


def main():
    import os

    parser = argparse.ArgumentParser(description='WGAN-ECANet Model Evaluation')
    parser.add_argument('--model', type=str, default=os.getenv('WGAN_MODEL_PATH', ''),
                       help='Model file path (or set WGAN_MODEL_PATH env var)')
    parser.add_argument('--dataset', type=str, default=os.getenv('WGAN_DATASET_PATH', ''),
                       help='Dataset path (or set WGAN_DATASET_PATH env var)')
    parser.add_argument('--split', type=str, default='val',
                       choices=['train', 'val', 'test'], help='Dataset split')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples (for testing)')
    parser.add_argument('--device', type=str, default=None,
                       choices=['cuda', 'cpu'], help='Compute device')
    parser.add_argument('--class-accuracy', action='store_true',
                       help='Show per-class accuracy')
    parser.add_argument('--output', type=str, default=None,
                       help='Result output path')

    args = parser.parse_args()

    if not args.model:
        print("Error: --model argument or WGAN_MODEL_PATH environment variable is required")
        print("\nSolutions:")
        print("  1. Use --model argument:")
        print("     python evaluate.py --model \"pretrain models/model5_variant_3_seed_889.pth\"")
        print("  2. Set environment variable:")
        print("     export WGAN_MODEL_PATH=\"pretrain models/model5_variant_3_seed_889.pth\"  # Linux/macOS")
        print("     set WGAN_MODEL_PATH=\"pretrain models/model5_variant_3_seed_889.pth\"  # Windows")
        sys.exit(1)

    if not args.dataset:
        print("Error: --dataset argument or WGAN_DATASET_PATH environment variable is required")
        print("\nSolutions:")
        print("  1. Use --dataset argument:")
        print("     python evaluate.py --dataset \"dataset/RadioML 2018.01A/GOLD_XYZ_OSC.0001_1024.hdf5\"")
        print("  2. Set environment variable:")
        print("     export WGAN_DATASET_PATH=\"dataset/RadioML 2018.01A/GOLD_XYZ_OSC.0001_1024.hdf5\"  # Linux/macOS")
        print("     set WGAN_DATASET_PATH=\"dataset/RadioML 2018.01A/GOLD_XYZ_OSC.0001_1024.hdf5\"  # Windows")
        sys.exit(1)

    pm = get_path_manager()
    evaluator = WGANECANetEvaluator(args.model, args.device)

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = pm.project_root / args.dataset

    if args.class_accuracy:
        class_acc = evaluator.compute_class_accuracy(str(dataset_path), args.split)
    else:
        metrics = evaluator.evaluate_from_hdf5(str(dataset_path), args.split, args.batch_size, args.max_samples)

        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            import json
            with open(output_path, 'w') as f:
                json.dump(metrics, f, indent=2)

            print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
