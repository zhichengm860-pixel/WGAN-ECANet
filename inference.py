import argparse
import torch
import numpy as np
import h5py
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.path_manager import get_path_manager
from src.models.enhanced_wgan_ecanet import EnhancedWGANECANet
from src.data.radioml_dataloader import RadioMLDataLoader


class WGANECANetInference:

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

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    def predict(self, signals):
        if signals.shape[-1] == 2 and len(signals.shape) == 3:
            signals = signals.transpose(0, 2, 1)
        signals_tensor = torch.FloatTensor(signals).to(self.device)

        with torch.no_grad():
            outputs = self.model(signals_tensor, mode='classify')
            logits = outputs['logits']
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

        return predictions.cpu().numpy(), probabilities.cpu().numpy()

    def predict_from_file(self, input_path, output_path=None):
        print(f"Loading input file: {input_path}")

        if input_path.endswith('.npy'):
            signals = np.load(input_path)
        elif input_path.endswith('.hdf5') or input_path.endswith('.h5'):
            with h5py.File(input_path, 'r') as f:
                signals = f['X'][:]
        else:
            raise ValueError(f"Unsupported file format: {input_path}")

        if len(signals.shape) == 2:
            signals = signals[np.newaxis, ...]

        print(f"Input signal shape: {signals.shape}")

        predictions, probabilities = self.predict(signals)

        print(f"Prediction completed!")
        print(f"Prediction shape: {predictions.shape}")

        if output_path:
            self._save_results(predictions, probabilities, output_path)
            print(f"Results saved to: {output_path}")

        return predictions, probabilities

    def _save_results(self, predictions, probabilities, output_path):
        np.savez(output_path,
                 predictions=predictions,
                 probabilities=probabilities)

    def predict_from_hdf5(self, hdf5_path, start_idx=0, num_samples=1000):
        print(f"Predicting from HDF5 file: {hdf5_path}")

        data_loader = RadioMLDataLoader(hdf5_path)

        signals, labels = data_loader.load_samples(start_idx, num_samples)
        predictions, probabilities = self.predict(signals)

        correct = (predictions == labels).sum()
        accuracy = correct / len(labels) * 100

        print(f"Accuracy: {accuracy:.2f}% ({correct}/{len(labels)})")

        return predictions, probabilities, accuracy


def main():
    import os

    parser = argparse.ArgumentParser(description='WGAN-ECANet Inference Script')
    parser.add_argument('--model', type=str, default=os.getenv('WGAN_MODEL_PATH', ''),
                       help='Model file path (or set WGAN_MODEL_PATH env var)')
    parser.add_argument('--input', type=str, required=True,
                       help='Input file path (.npy or .hdf5)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path (optional)')
    parser.add_argument('--device', type=str, default=None,
                       choices=['cuda', 'cpu'], help='Compute device')
    parser.add_argument('--start-idx', type=int, default=0,
                       help='HDF5 start index')
    parser.add_argument('--num-samples', type=int, default=1000,
                       help='HDF5 number of samples')

    args = parser.parse_args()

    if not args.model:
        print("Error: --model argument or WGAN_MODEL_PATH environment variable is required")
        print("\nSolutions:")
        print("  1. Use --model argument:")
        print("     python inference.py --model \"pretrain models/model5_variant_3_seed_889.pth\"")
        print("  2. Set environment variable:")
        print("     export WGAN_MODEL_PATH=\"pretrain models/model5_variant_3_seed_889.pth\"  # Linux/macOS")
        print("     set WGAN_MODEL_PATH=\"pretrain models/model5_variant_3_seed_889.pth\"  # Windows")
        sys.exit(1)

    inferencer = WGANECANetInference(args.model, args.device)

    input_path = Path(args.input)
    if not input_path.is_absolute():
        pm = get_path_manager()
        input_path = pm.project_root / args.input

    if args.input.endswith('.hdf5') or args.input.endswith('.h5'):
        predictions, probabilities, accuracy = inferencer.predict_from_hdf5(
            str(input_path), args.start_idx, args.num_samples)
    else:
        predictions, probabilities = inferencer.predict_from_file(
            str(input_path), args.output)


if __name__ == '__main__':
    main()
