import os
import sys
from pathlib import Path
from typing import Optional, Union, List
import warnings


class PathManager:

    ROOT_MARKERS = [
        'README.md',
        'LICENSE',
        'requirements.txt',
        'src',
        'dataset',
        'pretrain models',
        'examples'
    ]
    
    def __init__(self, project_root: Optional[Union[str, Path]] = None, verbose: bool = False):

        self.verbose = verbose
        self.project_root = self._find_project_root(project_root)
        self._validate_project_structure()
        
        if self.verbose:
            print(f"[PathManager] Project root: {self.project_root}")
    
    def _find_project_root(self, custom_root: Optional[Union[str, Path]] = None) -> Path:

        if custom_root is not None:
            custom_root = Path(custom_root).resolve()
            if custom_root.exists():
                return custom_root
            else:
                warnings.warn(f"Specified root directory does not exist: {custom_root}, will automatically search")
        
        env_root = os.environ.get('WGAN_PROJECT_ROOT')
        if env_root:
            env_root = Path(env_root).resolve()
            if env_root.exists():
                if self.verbose:
                    print(f"[PathManager] Read root directory from environment variable WGAN_PROJECT_ROOT")
                return env_root
        
        current_file = Path(__file__).resolve()
        
        test_path = current_file
        for _ in range(5):
            if self._is_project_root(test_path):
                return test_path
            test_path = test_path.parent
        
        if len(sys.argv) > 0:
            script_path = Path(sys.argv[0]).resolve()
            if script_path.exists():
                test_path = script_path
                for _ in range(5):
                    if self._is_project_root(test_path):
                        return test_path
                    test_path = test_path.parent
        
        cwd = Path.cwd()
        test_path = cwd
        for _ in range(10):
            if self._is_project_root(test_path):
                return test_path
            test_path = test_path.parent
        
        warnings.warn("Unable to automatically detect project root directory, will use current working directory")
        return cwd
    
    def _is_project_root(self, path: Path) -> bool:
        if not path.exists():
            return False
        
        marker_count = sum(
            1 for marker in self.ROOT_MARKERS
            if (path / marker).exists()
        )
        return marker_count >= 3
    
    def _validate_project_structure(self):
        required_dirs = ['src', 'pretrain models', 'dataset']
        missing = []
        
        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            if not dir_path.exists():
                missing.append(dir_name)
        
        if missing:
            warnings.warn(
                f"Project structure is incomplete, missing the following directories: {', '.join(missing)}\n"
                f"Current project root: {self.project_root}"
            )
    
    def resolve_path(self, path: Union[str, Path], 
                    base_dir: Optional[Union[str, Path]] = None) -> Path:

        path = Path(path)
        
        if path.is_absolute():
            return path.resolve()
        
        path_str = str(path)
        if path_str.startswith('$') or path_str.startswith('%'):
            for var_name in ['MODEL_PATH', 'DATASET_PATH', 'PROJECT_ROOT', 
                           'WGAN_MODEL_PATH', 'WGAN_DATASET_PATH']:
                env_value = os.environ.get(var_name)
                if env_value:
                    path_str = path_str.replace(f'${var_name}', env_value)
                    path_str = path_str.replace(f'%{var_name}%', env_value)
            path = Path(path_str)
            if path.is_absolute():
                return path.resolve()
        
        if base_dir is None:
            base_dir = self.project_root
        else:
            base_dir = Path(base_dir)
        
        result = (base_dir / path).resolve()
        
        if self.verbose:
            print(f"[PathManager] Resolved path: {path} -> {result}")
        
        return result
    
    def get_project_root(self) -> Path:
        return self.project_root
    
    def get_src_dir(self) -> Path:
        return self.project_root / 'src'
    
    def get_models_dir(self) -> Path:
        return self.project_root / 'pretrain models'
    
    def get_model_path(self, model_name: Union[str, Path] = 'model5_variant_3_seed_889.pth',
                       check_exists: bool = True) -> Path:

        model_path = Path(model_name)
        if model_path.is_absolute():
            result = model_path.resolve()
        else:
            env_path = os.environ.get('WGAN_MODEL_PATH')
            if env_path:
                env_model = Path(env_path) / model_name
                if env_model.exists():
                    result = env_model.resolve()
                    if self.verbose:
                        print(f"[PathManager] Found model from environment variable: {result}")
                    return result
            
            result = self.resolve_path(model_name, base_dir=self.get_models_dir())
            
            if not result.exists():
                result_alt = self.resolve_path(model_name)
                if result_alt.exists():
                    result = result_alt
        
        if check_exists and not result.exists():
            raise FileNotFoundError(
                f"Model file does not exist: {result}\n"
                f"Please ensure:\n"
                f"  1. The model has been downloaded and placed in the correct location\n"
                f"  2. Or use the --model-path parameter to specify the correct path\n"
                f"  3. Or set the environment variable WGAN_MODEL_PATH"
            )
        
        return result
    
    def get_dataset_dir(self) -> Path:
        return self.project_root / 'dataset'
    
    def get_dataset_path(self, dataset_name: str,
                         check_exists: bool = True) -> Path:

        env_path = os.environ.get('WGAN_DATASET_PATH')
        if env_path:
            result = Path(env_path).resolve()
            if result.exists():
                if self.verbose:
                    print(f"[PathManager] Found dataset from environment variable: {result}")
                return result
        
        dataset_dir = self.get_dataset_dir() / dataset_name
        result = dataset_dir.resolve()
        
        if check_exists and not result.exists():
            raise FileNotFoundError(
                f"Dataset directory does not exist: {result}\n"
                f"Please ensure:\n"
                f"  1. The dataset has been downloaded\n"
                f"  2. The directory name is correct (note spaces and version numbers)\n"
                f"  3. Or use the --dataset-path parameter to specify the correct path\n"
                f"  4. Or set the environment variable WGAN_DATASET_PATH"
            )
        
        return result
    
    def get_hdf5_dataset_path(self, dataset_name: str = 'RadioML 2018.01A',
                            filename: str = 'GOLD_XYZ_OSC.0001_1024.hdf5',
                            check_exists: bool = True) -> Path:

        dataset_dir = self.get_dataset_path(dataset_name, check_exists=False)
        result = dataset_dir / filename
        
        if check_exists and not result.exists():
            raise FileNotFoundError(
                f"HDF5 file does not exist: {result}\n"
                f"Please ensure:\n"
                f"  1. The dataset file has been downloaded\n"
                f"  2. The filename is correct: {filename}\n"
                f"  3. Or use the --dataset-path parameter to directly specify the file path"
            )
        
        return result
    
    def get_examples_dir(self) -> Path:
        return self.project_root / 'examples'
    
    def get_results_dir(self, subfolder: Optional[str] = None) -> Path:

        results_dir = self.project_root / 'results'
        if subfolder:
            results_dir = results_dir / subfolder
        return results_dir
    
    def add_to_pythonpath(self):
        src_dir = str(self.get_src_dir())
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
            if self.verbose:
                print(f"[PathManager] Added to sys.path: {src_dir}")
    
    def ensure_dir(self, path: Union[str, Path]):

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self.verbose:
            print(f"[PathManager] Ensured directory exists: {path}")
    
    def normalize_path(self, path: Union[str, Path]) -> Path:

        return Path(path).resolve()
    
    @staticmethod
    def get_separator() -> str:
        return os.sep
    
    @staticmethod
    def is_windows() -> bool:
        return os.name == 'nt'
    
    @staticmethod
    def is_unix_like() -> bool:
        return os.name != 'nt'


_global_path_manager = None


def get_path_manager(project_root: Optional[Union[str, Path]] = None,
                    verbose: bool = False) -> PathManager:

    global _global_path_manager
    
    if _global_path_manager is None:
        _global_path_manager = PathManager(project_root, verbose)
    
    return _global_path_manager


def reset_path_manager():
    global _global_path_manager
    _global_path_manager = None
