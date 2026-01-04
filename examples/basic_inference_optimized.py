"""
Basic Inference Example (Optimized)
Demonstrates how to use a pre-trained model for single signal prediction

Main improvements:
1. Use PathManager for unified path management
2. Support environment variable configuration
3. Cross-platform compatibility
4. Friendly error messages
"""

import sys
import os
import argparse
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.path_manager import get_path_manager
from src.models.enhanced_wgan_ecanet import EnhancedWGANECANet


def main():
    import os

    print("=" * 60)
    print("WGAN-ECANet Basic Inference Example (Optimized)")
    print("=" * 60)

    parser = argparse.ArgumentParser(description='WGAN-ECANet Basic Inference')
    parser.add_argument('--model-path', type=str, default=os.getenv('WGAN_MODEL_PATH', ''),
                       help='Model file path (or set WGAN_MODEL_PATH env var)')
    parser.add_argument('--device', type=str, default=None,
                       choices=['cuda', 'cpu'],
                       help='Compute device (default: auto)')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed logs')

    args = parser.parse_args()

    try:
        pm = get_path_manager(verbose=args.verbose)
        pm.add_to_pythonpath()
    except Exception as e:
        print(f"Error: Path manager initialization failed: {e}")
        print("\nSuggestions:")
        print("  1. Ensure you run this script in the correct directory")
        print("  2. Or use --project-root parameter to specify project root")
        sys.exit(1)

    try:
        model_path = args.model_path
        if not model_path:
            print("Error: --model-path argument or WGAN_MODEL_PATH environment variable is required")
            print("\nSolutions:")
            print("  1. Use --model-path argument:")
            print("     python examples/basic_inference_optimized.py --model-path \"pretrain models/model5_variant_3_seed_889.pth\"")
            print("  2. Set environment variable:")
            print("     export WGAN_MODEL_PATH=\"pretrain models/model5_variant_3_seed_889.pth\"  # Linux/macOS")
            print("     set WGAN_MODEL_PATH=\"pretrain models/model5_variant_3_seed_889.pth\"  # Windows")
            sys.exit(1)

        model_path = pm.get_model_path(model_path, check_exists=True)
        print(f"\nModel path: {model_path}")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nSolutions:")
        print("  1. Download pre-trained model and place in 'pretrain models/' directory")
        print("  2. Or use --model-path parameter to specify model path")
        print("  3. Or set environment variable WGAN_MODEL_PATH")
        sys.exit(1)
    
    if args.device == 'cpu':
        device = torch.device('cpu')
        print("Force using CPU")
    elif args.device == 'cuda':
        if not torch.cuda.is_available():
            print("CUDA not available, will use CPU")
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')
            print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device_name = 'CUDA' if torch.cuda.is_available() else 'CPU'
        print(f"Auto-selected device: {device_name}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    try:
        print(f"\nLoading model...")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        model = EnhancedWGANECANet(num_classes=24, use_spectral_norm=True)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.to(device)
        model.eval()
        
        print("Model loaded successfully")
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        if 'epoch' in checkpoint:
            print(f"  Training epochs: {checkpoint['epoch']}")
        if 'best_accuracy' in checkpoint:
            print(f"  Best accuracy: {checkpoint['best_accuracy']:.4f}")
        
    except Exception as e:
        print(f"Error: Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("Preparing Input Signal")
    print("=" * 60)
    
    print("\nNote: Currently using random signal, please replace with real signal for actual use")
    signal = np.random.randn(2, 1024).astype(np.float32)
    signal_tensor = torch.FloatTensor(signal).unsqueeze(0).to(device)
    
    print(f"Input signal shape: {signal.shape}")
    print(f"  Signal length: {signal.shape[1]}")
    print(f"  Number of channels: {signal.shape[0]} (I/Q two channels)")
    
    print("\n" + "=" * 60)
    print("Running Prediction")
    print("=" * 60)
    
    try:
        with torch.no_grad():
            outputs = model(signal_tensor, mode='classify')
            logits = outputs['logits']
            probabilities = torch.softmax(logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1)
        
        print("Prediction completed")
    except Exception as e:
        print(f"Error: Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("Prediction Results")
    print("=" * 60)
    
    predicted_class = prediction.item()
    confidence = probabilities[0, predicted_class].item()
    
    print(f"\nPredicted class index: {predicted_class}")
    print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
    
    class_names = [
        'OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK',
        '16PSK', '32PSK', '16APSK', '32APSK', '64APSK', '128APSK',
        '16QAM', '32QAM', '64QAM', '128QAM', '256QAM',
        'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC',
        'FM', 'GMSK', 'OQPSK'
    ]
    
    if predicted_class < len(class_names):
        print(f"Predicted modulation type: {class_names[predicted_class]}")
    
    top5_probs, top5_indices = torch.topk(probabilities[0], 5)
    
    print("\nTop-5 predictions:")
    for i, (idx, prob) in enumerate(zip(top5_indices, top5_probs)):
        class_idx = idx.item()
        prob_val = prob.item()
        class_name = class_names[class_idx] if class_idx < len(class_names) else "Unknown"
        print(f"  {i+1}. Class {class_idx:2d} ({class_name:12s}) - Probability: {prob_val:.4f} ({prob_val*100:5.2f}%)")
    
    print("\n" + "=" * 60)
    print("Inference completed!")
    print("=" * 60)
    
    print("\nUsage tips:")
    print("  1. Replace random signal with real signal (I/Q two channels, 1024 samples per channel)")
    print("  2. Signal format: np.array([2, 1024]) or np.array([N, 2, 1024])")
    print("  3. Suggest normalizing the signal")
    print("  4. Use --model-path to specify custom model")
    print("  5. Use --device to specify compute device")


if __name__ == '__main__':
    main()
