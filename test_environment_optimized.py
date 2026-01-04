import sys
import os
from pathlib import Path
from typing import Tuple, List

try:
    from src.utils.path_manager import get_path_manager
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from src.utils.path_manager import get_path_manager


class Colors:
    RESET = '\033[0m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    
    @staticmethod
    def supports_color() -> bool:
        return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()


class TestResult:

    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.message = ""
        self.details = []
    
    def set_passed(self, message: str = ""):
        self.passed = True
        self.message = message
    
    def set_failed(self, message: str, details: List[str] = None):
        self.passed = False
        self.message = message
        if details:
            self.details = details
    
    def __str__(self):
        if self.passed:
            status = "[PASS]"
        else:
            status = "[FAIL]"
        
        result = f"  {status} {self.name}"
        if self.message:
            result += f": {self.message}"
        return result


class EnvironmentTester:
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.use_color = Colors.supports_color()
        self.pm = None
        self.results: List[TestResult] = []
    
    def colorize(self, text: str, color: str) -> str:
        if self.use_color:
            return f"{color}{text}{Colors.RESET}"
        return text
    
    def print_header(self, title: str):
        separator = "=" * 60
        print("\n" + self.colorize(separator, Colors.BOLD))
        print(self.colorize(title, Colors.BOLD))
        print(self.colorize(separator, Colors.BOLD))
    
    def print_subheader(self, title: str):
        print(f"\n{self.colorize(title, Colors.CYAN)}")
    
    def test_path_manager(self) -> TestResult:
        result = TestResult("Path manager initialization")
        
        try:
            self.pm = get_path_manager(verbose=self.verbose)
            self.pm.add_to_pythonpath()
            
            project_root = self.pm.get_project_root()
            
            required_dirs = ['src', 'dataset', 'pretrain models', 'examples']
            missing_dirs = [d for d in required_dirs 
                           if not (project_root / d).exists()]
            
            if missing_dirs:
                raise FileNotFoundError(
                    f"Missing key directories: {', '.join(missing_dirs)}"
                )
            
            result.set_passed(f"Project root: {project_root}")
            
            if self.verbose:
                print(f"  - Project root: {project_root}")
                print(f"  - Source directory: {self.pm.get_src_dir()}")
                print(f"  - Model directory: {self.pm.get_models_dir()}")
                print(f"  - Dataset directory: {self.pm.get_dataset_dir()}")
        
        except Exception as e:
            result.set_failed(str(e), [
                "Solution:",
                "  1. Ensure you run this script in the correct directory",
                "  2. Check if project structure is complete",
                "  3. Set environment variable WGAN_PROJECT_ROOT"
            ])
        
        self.results.append(result)
        return result
    
    def test_imports(self) -> TestResult:
        result = TestResult("Dependency package import")
        details = []
        
        try:
            import torch
            torch_version = torch.__version__
            cuda_available = torch.cuda.is_available()
            details.append(f"✓ PyTorch {torch_version}")
            
            if cuda_available:
                details.append(f"  - CUDA {torch.version.cuda}")
                details.append(f"  - GPU count: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    details.append(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
            else:
                details.append("  - CUDA not available (will use CPU)")
            
            import numpy as np
            details.append(f"✓ NumPy {np.__version__}")
            
            import h5py
            details.append(f"✓ h5py {h5py.__version__}")
            
            from tqdm import tqdm
            details.append("✓ tqdm")
            
            import sklearn
            details.append(f"✓ scikit-learn {sklearn.__version__}")
            
            result.set_passed("All dependency packages installed correctly")
            result.details = details
        
        except ImportError as e:
            result.set_failed(f"Missing dependency package: {e}", [
                "Solution:",
                "  1. Run: pip install -r requirements.txt",
                "  2. Check Python version (requires 3.8+)",
                "  3. Ensure pip version is latest"
            ])
        
        self.results.append(result)
        return result
    
    def test_project_modules(self) -> TestResult:
        result = TestResult("Project module import")
        details = []
        
        try:
            from src.models.enhanced_wgan_ecanet import EnhancedWGANECANet
            details.append("✓ EnhancedWGANECANet model")
            
            from src.data.radioml_dataloader import RadioMLDataLoader, RadioMLDataset
            details.append("✓ RadioMLDataLoader data loader")
            
            from src.utils.metrics import compute_metrics
            details.append("✓ Evaluation metrics utilities")
            
            result.set_passed("All project modules imported successfully")
            result.details = details
        
        except ImportError as e:
            result.set_failed(f"Module import failed: {e}", [
                "Solution:",
                "  1. Ensure you run this script in project root directory",
                "  2. Check if src/ directory exists",
                "  3. Check if __init__.py files exist"
            ])
        
        self.results.append(result)
        return result
    
    def test_model_loading(self) -> TestResult:
        result = TestResult("Model loading")
        details = []
        
        try:
            import torch
            from src.models.enhanced_wgan_ecanet import EnhancedWGANECANet
            
            model_path = self.pm.get_model_path(
                'model5_variant_3_seed_889.pth',
                check_exists=True
            )
            details.append(f"✓ Model file exists: {model_path.name}")
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            checkpoint = torch.load(model_path, map_location=device, 
                                   weights_only=False)
            
            model = EnhancedWGANECANet(num_classes=24, use_spectral_norm=True)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            model.to(device)
            model.eval()
            
            details.append("✓ Model weights loaded successfully")
            
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() 
                                   if p.requires_grad)
            details.append(f"✓ Total parameters: {total_params:,}")
            details.append(f"✓ Trainable parameters: {trainable_params:,}")
            
            if 'epoch' in checkpoint:
                details.append(f"✓ Training epochs: {checkpoint['epoch']}")
            if 'best_accuracy' in checkpoint:
                details.append(f"✓ Best accuracy: {checkpoint['best_accuracy']:.4f}")
            
            result.set_passed("Model loaded successfully")
            result.details = details
        
        except FileNotFoundError as e:
            result.set_failed("Model file not found", [
                "Reason: " + str(e),
                "Solution:",
                "  1. Download pre-trained model",
                "  2. Place in 'pretrain models/' directory",
                "  3. Or set environment variable WGAN_MODEL_PATH"
            ])
        except Exception as e:
            result.set_failed(f"Model loading failed: {e}", [
                "Solution:",
                "  1. Check if model file is corrupted",
                "  2. Confirm PyTorch version compatibility",
                "  3. Try using CPU mode"
            ])
        
        self.results.append(result)
        return result
    
    def test_data_loading(self) -> TestResult:
        result = TestResult("Dataset loading")
        details = []

        import os

        try:
            import h5py
            import numpy as np

            dataset_name = os.getenv('WGAN_DATASET_NAME', 'RadioML 2018.01A')
            dataset_file = os.getenv('WGAN_DATASET_FILE', 'GOLD_XYZ_OSC.0001_1024.hdf5')

            dataset_path = self.pm.get_hdf5_dataset_path(
                dataset_name,
                dataset_file,
                check_exists=True
            )
            details.append(f"Dataset file exists: {dataset_path.name}")

            with h5py.File(dataset_path, 'r') as f:
                X = f['X'][:10]
                Y = f['Y'][:10]
                Z = f['Z'][:10]

                details.append("Dataset format correct")
                details.append(f"Signal shape: {X.shape}")
                details.append(f"Label shape: {Y.shape}")
                details.append(f"SNR shape: {Z.shape}")
                details.append(f"Total samples: {f['X'].shape[0]:,}")

                from src.data.radioml_dataloader import RadioMLDataset
                dataset = RadioMLDataset(X, Y)
                sample = dataset[0]

                details.append("RadioMLDataset class works correctly")
                details.append(f"Sample signal shape: {sample['signals'].shape}")
                details.append(f"Sample label: {sample['label']}")

            result.set_passed("Dataset loaded successfully")
            result.details = details

        except FileNotFoundError as e:
            result.set_failed("Dataset file not found", [
                "Reason: " + str(e),
                "Solution:",
                "  1. Download your dataset",
                "  2. Extract to dataset directory",
                "  3. Set environment variables:",
                "     export WGAN_DATASET_NAME=YourDatasetName  # Linux/macOS",
                "     set WGAN_DATASET_NAME=YourDatasetName  # Windows",
                "     export WGAN_DATASET_FILE=your_dataset_file.hdf5  # Linux/macOS",
                "     set WGAN_DATASET_FILE=your_dataset_file.hdf5  # Windows"
            ])
        except Exception as e:
            result.set_failed(f"Data loading failed: {e}", [
                "Solution:",
                "  1. Check if dataset file is complete",
                "  2. Confirm h5py version compatibility",
                "  3. Check if disk space is sufficient"
            ])
        
        self.results.append(result)
        return result
    
    def test_inference(self) -> TestResult:
        result = TestResult("Inference test")
        
        try:
            import torch
            import numpy as np
            from src.models.enhanced_wgan_ecanet import EnhancedWGANECANet
            
            model_path = self.pm.get_model_path(check_exists=False)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            checkpoint = torch.load(model_path, map_location=device, 
                                   weights_only=False)
            model = EnhancedWGANECANet(num_classes=24, use_spectral_norm=True)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            model.to(device)
            model.eval()
            
            signal = np.random.randn(2, 1024).astype(np.float32)
            signal_tensor = torch.FloatTensor(signal).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(signal_tensor, mode='classify')
                logits = outputs['logits']
                probabilities = torch.softmax(logits, dim=1)
                prediction = torch.argmax(probabilities, dim=1)
            
            predicted_class = prediction.item()
            confidence = probabilities[0, predicted_class].item()
            
            result.set_passed(f"Prediction successful (class={predicted_class}, confidence={confidence:.4f})")
        
        except Exception as e:
            result.set_failed(f"Inference failed: {e}")
        
        self.results.append(result)
        return result
    
    def run_all_tests(self) -> Tuple[bool, List[TestResult]]:
        self.print_header("WGAN-ECANet Environment Test (Optimized)")
        
        print(f"Python version: {sys.version}")
        print(f"Operating system: {os.name}")
        print(f"Platform: {sys.platform}")
        
        tests = [
            ("Path manager", self.test_path_manager),
            ("Dependency import", self.test_imports),
            ("Project module import", self.test_project_modules),
            ("Model loading", self.test_model_loading),
            ("Dataset loading", self.test_data_loading),
            ("Inference test", self.test_inference),
        ]
        
        for test_name, test_func in tests:
            self.print_subheader(f"Testing {test_name}")
            result = test_func()
            print(result)
            
            if result.details and self.verbose:
                for detail in result.details:
                    print(f"    {detail}")
            
            if not result.passed:
                for detail in result.details:
                    print(self.colorize(detail, Colors.YELLOW))
        
        self.print_summary()
        
        all_passed = all(r.passed for r in self.results)
        return all_passed, self.results
    
    def print_summary(self):
        self.print_header("Test Summary")
        
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        all_passed = all(r.passed for r in self.results)
        
        for result in self.results:
            print(result)
        
        print()
        self.print_header("=" * 60)
        
        if all_passed:
            print(self.colorize("[PASS] All tests passed!", Colors.GREEN))
            print("Environment is configured correctly, you can now run evaluation or inference scripts.")
        else:
            print(self.colorize(f"[FAIL] {total - passed}/{total} tests failed!", Colors.RED))
            print("Please fix the issues according to the above suggestions.")
        
        print(self.colorize("=" * 60, Colors.BOLD))


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='WGAN-ECANet Environment Test')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show detailed logs')
    parser.add_argument('--project-root', type=str, default=None,
                       help='Manually specify project root directory')
    
    args = parser.parse_args()
    
    tester = EnvironmentTester(verbose=args.verbose)
    
    all_passed, _ = tester.run_all_tests()
    
    sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()
