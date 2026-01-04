"""
Batch Inference Example
Demonstrates how to batch process signals and save results
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import h5py

from src.models.enhanced_wgan_ecanet import EnhancedWGANECANet


def batch_inference(model_path, input_path, output_path, batch_size=256, max_samples=None):
    print("=" * 60)
    print("WGAN-ECANet Batch Inference")
    print("=" * 60)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    if not os.path.isabs(model_path):
        model_path = os.path.join(project_root, model_path)
    print(f"\nLoading model: {model_path}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model = EnhancedWGANECANet(num_classes=24, use_spectral_norm=True)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.to(device)
    model.eval()

    print("Model loaded successfully")

    print(f"\nLoading input data: {input_path}")

    if not os.path.isabs(input_path):
        input_path = os.path.join(project_root, input_path)

    if input_path.endswith('.npy'):
        signals = np.load(input_path)
        total_samples = len(signals)
        use_hdf5 = False
    elif input_path.endswith('.hdf5') or input_path.endswith('.h5'):
        f = h5py.File(input_path, 'r')
        total_samples = f['X'].shape[0]
        use_hdf5 = True
        print(f"Input data shape: {f['X'].shape}")
    else:
        raise ValueError(f"Unsupported file format: {input_path}")

    print(f"Total samples: {total_samples}")

    if max_samples is not None and max_samples < total_samples:
        total_samples = max_samples
        print(f"Limited processing samples: {total_samples}")

    print(f"\nStarting batch inference (batch_size={batch_size})...")

    all_predictions = []
    all_probabilities = []

    num_batches = (total_samples + batch_size - 1) // batch_size

    for i in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_samples)

        if use_hdf5:
            batch_signals = f['X'][start_idx:end_idx]
        else:
            batch_signals = signals[start_idx:end_idx]

        batch_signals = np.transpose(batch_signals, (0, 2, 1))

        batch_tensor = torch.FloatTensor(batch_signals).to(device)

        with torch.no_grad():
            outputs = model(batch_tensor, mode='classify')
            logits = outputs['logits']
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

        all_predictions.append(predictions.cpu().numpy())
        all_probabilities.append(probabilities.cpu().numpy())

    if use_hdf5:
        f.close()

    predictions = np.concatenate(all_predictions, axis=0)
    probabilities = np.concatenate(all_probabilities, axis=0)

    print(f"\nInference completed")
    print(f"Prediction shape: {predictions.shape}")
    print(f"Probability shape: {probabilities.shape}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving results to: {output_path}")

    np.savez(output_path,
             predictions=predictions,
             probabilities=probabilities)

    print("Results saved successfully")

    print("\n" + "=" * 60)
    print("Statistics")
    print("=" * 60)

    unique, counts = np.unique(predictions, return_counts=True)

    print("\nPrediction count for each class:")
    for cls, count in zip(unique, counts):
        percentage = count / len(predictions) * 100
        print(f"  Class {cls:2d}: {count:6d} ({percentage:5.2f}%)")

    max_probs = probabilities.max(axis=1)
    avg_confidence = max_probs.mean()
    print(f"\nAverage confidence: {avg_confidence:.4f}")

    print("\n" + "=" * 60)
    print("Batch inference completed!")
    print("=" * 60)


def main():
    import os

    parser = argparse.ArgumentParser(description='Batch inference example')
    parser.add_argument('--model', type=str, default=os.getenv('WGAN_MODEL_PATH', ''),
                       help='Model file path (or set WGAN_MODEL_PATH env var)')
    parser.add_argument('--input', type=str, required=True,
                       help='Input file path (.npy or .hdf5)')
    parser.add_argument('--output', type=str,
                       default='results/predictions.npz',
                       help='Output file path')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples to process (None means all)')

    args = parser.parse_args()

    if not args.model:
        print("Error: --model argument or WGAN_MODEL_PATH environment variable is required")
        print("\nSolutions:")
        print("  1. Use --model argument:")
        print("     python examples/batch_inference.py --model \"pretrain models/model5_variant_3_seed_889.pth\"")
        print("  2. Set environment variable:")
        print("     export WGAN_MODEL_PATH=\"pretrain models/model5_variant_3_seed_889.pth\"  # Linux/macOS")
        print("     set WGAN_MODEL_PATH=\"pretrain models/model5_variant_3_seed_889.pth\"  # Windows")
        sys.exit(1)

    batch_inference(args.model, args.input, args.output, args.batch_size, args.max_samples)


if __name__ == '__main__':
    main()
