# WGAN-ECANet: Radio Signal Modulation Recognition

A deep learning framework for automatic modulation classification (AMC) of radio signals, integrating Wasserstein GANs, multi-scale feature extraction, and efficient channel attention mechanisms.

## Project Overview

WGAN-ECANet is a specialized framework designed for radio signal automatic modulation classification (AMC). This project integrates several advanced techniques including Wasserstein Generative Adversarial Networks with Gradient Penalty (WGAN-GP), dynamic gradient penalty strategies, multi-scale convolutional feature extraction, and dimension-reduction-free ECA (Efficient Channel Attention) mechanisms.

### Core Features

- **24-Class Modulation Recognition**: Supports recognition of 24 different modulation types including ASK, PSK, QAM, APSK, and analog modulations (AM, FM)
- **Multi-scale Feature Extraction**: Captures frequency domain features at different scales using parallel convolution kernels (1x1, 3x3, 5x5, 7x7)
- **ECA Attention Mechanism**: Uses dimension-free 1D convolution to adaptively learn channel dependencies while preserving complete feature information
- **Dynamic Gradient Penalty WGAN-GP**: Dynamically adjusts gradient penalty coefficients based on training progress to improve stability
- **Cross-Dataset Generalization**: Supports RadioML 2016 and 2018 series datasets with strong cross-domain generalization capabilities

### Technical Highlights

- **High-Precision Recognition**: Achieves 63.701% validation accuracy on RadioML 2018.01A dataset
- **Strong Robustness**: Maintains good performance across full SNR range (-20 dB to 30 dB)
- **Parameter Efficient**: Model has approximately 7.6M parameters with ~8.2ms inference latency
- **Easy to Extend**: Modular design supporting custom training scripts and testing workflows

## Project Structure

```
WGAN-ECANet/
├── dataset/                          # Dataset directory
│   ├── RadioML 2016.10A/           # 2016 10A dataset (11 classes, .pkl format)
│   ├── RadioML 2016.10B/           # 2016 10B dataset (11 classes, .dat format)
│   ├── RadioML 2018.01A/           # 2018 01A dataset (24 classes, .hdf5 format)
│   └── dataset_explanation.md       # Dataset documentation
│
├── pretrain_models/                  # Pretrained model directory
│   ├── model5_variant_3_seed_889.pth # Best performing model (63.701% accuracy)
│   └── README.md                    # Pretrained model documentation
│
├── src/                             # Source code directory
│   ├── configs/                      # Configuration files
│   │   └── model_config.json        # Model architecture and training configuration
│   ├── data/                         # Data loading modules
│   │   ├── __init__.py
│   │   └── radioml_dataloader.py   # RadioML dataset loader
│   ├── models/                       # Model definitions
│   │   ├── __init__.py
│   │   └── enhanced_wgan_ecanet.py # WGAN-ECANet model implementation
│   └── utils/                        # Utility functions
│       ├── __init__.py
│       ├── metrics.py               # Evaluation metrics calculation
│       └── path_manager.py         # Unified path management
│
├── examples/                         # Usage examples
│   ├── basic_inference.py           # Basic inference example (single signal)
│   └── batch_inference.py          # Batch inference example
│
├── evaluate.py                      # Model evaluation script
├── inference.py                     # Model inference script
├── test_environment.py              # Environment testing script
├── test_environment_optimized.py   # Optimized environment testing script
├── requirements.txt                 # Dependency list
├── .env.example                     # Environment variable template
├── PATH_README.md                   # Path optimization quick guide
├── QUICK_START.md                   # Quick start guide
├── LICENSE                          # MIT open source license
└── README.md                        # This file
```

### Key Files

- **`src/models/enhanced_wgan_ecanet.py`**: Core model implementation including generator, discriminator, and classifier
- **`src/data/radioml_dataloader.py`**: Unified data loader supporting RadioML 2016 and 2018 series datasets
- **`src/utils/path_manager.py`**: Unified path manager for cross-platform compatibility
- **`src/configs/model_config.json`**: Model architecture configuration with network structure and hyperparameters
- **`evaluate.py`**: Complete model evaluation script supporting accuracy, confusion matrix, SNR robustness analysis
- **`inference.py`**: Inference script supporting single or batch signal prediction
- **`examples/basic_inference.py`**: Basic inference example showing how to load model and make predictions
- **`test_environment_optimized.py`**: Optimized environment testing with detailed diagnostics

## Installation Guide

### System Requirements

- **Operating System**: Windows, Linux, or macOS
- **Python Version**: 3.8 or higher
- **GPU**: NVIDIA GPU recommended (CUDA 11.0+), CPU supported but slower
- **RAM**: At least 8GB (16GB recommended)
- **Disk Space**: At least 50GB available space for datasets and model files

### Installation Steps

#### 1. Clone the Repository

```bash
git clone <repository-url>
cd WGAN-ECANet
```

#### 2. Create Virtual Environment (Recommended)

```bash
# Using conda
conda create -n wgan-ecanet python=3.9
conda activate wgan-ecanet

# Or using venv
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Main dependencies include:
- PyTorch 2.0+ (Deep learning framework)
- NumPy, SciPy (Numerical computation)
- scikit-learn (Machine learning tools)
- h5py (HDF5 file support)
- matplotlib, seaborn (Visualization)
- PyYAML (Configuration file processing)
- tqdm (Progress bar)

#### 4. Verify Installation

Run the environment testing script to check configuration:

```bash
python test_environment_optimized.py
```

This script will check:
- PyTorch is correctly installed
- CUDA is available (if using GPU)
- Required dependencies are installed
- Dataset paths exist

#### 5. Download Datasets

The project supports the following datasets (download separately):

1. **RadioML 2018.01A** (Primary training dataset)
   - Download link: https://www.deepsig.io/datasets
   - File name: GOLD_XYZ_OSC.0001_1024.hdf5
   - Location: `dataset/RadioML 2018.01A/`

2. **RadioML 2016.10A / 2016.10B** (Cross-dataset generalization testing)
   - Download link: https://www.deepsig.io/datasets
   - File names: RML2016.10a_dict.pkl, RML2016.10b.dat
   - Location: `dataset/RadioML 2016.10A/`, `dataset/RadioML 2016.10B/`

Refer to `dataset/dataset_explanation.md` for detailed dataset format information.

### Windows User Notes

- If encountering `num_workers` related errors, set `num_workers` to 0 in configuration files
- Some dependencies may require precompiled wheel files for installation
- Use double backslashes `\\` or forward slashes `/` for path separators

## Usage

### Quick Start

#### 1. Inference with Pretrained Model

The simplest way to use the pretrained model for signal prediction:

```bash
python examples/basic_inference.py
```

This example will:
- Load pretrained model `model5_variant_3_seed_889.pth`
- Generate random signal (replace with real signal in actual use)
- Output predicted class and confidence
- Display Top-5 predictions

For batch inference, use:

```bash
python examples/batch_inference.py
```

#### 2. Evaluate Pretrained Model Performance

Use `evaluate.py` to evaluate model performance on test set:

```bash
python evaluate.py --model "pretrain_models/model5_variant_3_seed_889.pth" --dataset "dataset/RadioML 2018.01A/GOLD_XYZ_OSC.0001_1024.hdf5"
```

The evaluation script will output:
- Overall accuracy
- Precision, recall, F1-score for each modulation type
- Confusion matrix
- Accuracy curves under different SNR conditions

#### 3. Model Inference

Use `inference.py` to predict on custom signals:

```bash
python inference.py --model "pretrain_models/model5_variant_3_seed_889.pth" --signal "path/to/your/signal.npy"
```

Supported input formats:
- NumPy array files (`.npy`)
- HDF5 files (`.hdf5`)
- Audio files (`.wav`, requires additional configuration)

### Data Input Format

The model requires input signals in the following format:

- **Shape**: `[batch_size, 2, 1024]` or `[2, 1024]` (single signal)
  - Channel 0: I (in-phase) component
  - Channel 1: Q (quadrature) component
  - Signal length: 1024 samples
- **Data Type**: float32 or float64
- **Normalization**: Recommended to normalize to [-1, 1] or [0, 1] range

### Output Format

Model outputs include:
- **logits**: `[batch_size, 24]`, unnormalized classification scores
- **features**: Intermediate layer feature vectors (for visualization or further analysis)
- **prediction**: Predicted class index (0-23)
- **probabilities**: Normalized class probabilities (softmax output)

Supported 24 modulation types:

| Index | Modulation | Index | Modulation | Index | Modulation |
|-------|------------|-------|------------|-------|------------|
| 0 | OOK | 8 | 16PSK | 16 | 128QAM |
| 1 | 4ASK | 9 | 32PSK | 17 | 256QAM |
| 2 | 8ASK | 10 | 16APSK | 18 | AM-SSB-WC |
| 3 | BPSK | 11 | 32APSK | 19 | AM-SSB-SC |
| 4 | QPSK | 12 | 64APSK | 20 | AM-DSB-WC |
| 5 | 8PSK | 13 | 128APSK | 21 | AM-DSB-SC |
| 6 | 16PSK | 14 | 16QAM | 22 | FM |
| 7 | 32PSK | 15 | 32QAM | 23 | GMSK |

## Model Extension

This project is designed with a modular architecture, supporting extension and custom training based on existing pretrained models.

### Training Based on Pretrained Models

#### 1. Create New Training Scripts

Create custom training scripts in the project root or `examples/` directory, following these steps:

**Step 1: Import necessary modules**
```python
import torch
from src.models.enhanced_wgan_ecanet import EnhancedWGANECANet
from src.data.radioml_dataloader import RadioMLDataLoader
from src.utils.path_manager import get_path_manager
```

**Step 2: Configure training parameters**
- Dataset path and type
- Batch size, learning rate
- Number of epochs
- Data augmentation strategy
- Regularization parameters

**Step 3: Load data**
Use `RadioMLDataLoader` to create training, validation, and test data loaders, supporting:
- Class-balanced sampling
- SNR range filtering
- Data splitting (train/validation/test)
- Data augmentation (noise injection, phase offset, time shift, etc.)

**Step 4: Load pretrained model**
```python
pm = get_path_manager()
model = EnhancedWGANECANet(num_classes=24, use_spectral_norm=True)
model_path = pm.get_model_path('model5_variant_3_seed_889.pth')
checkpoint = torch.load(model_path, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
```

**Step 5: Set up training loop**
- Define loss functions (classification loss, GAN loss, gradient penalty)
- Choose optimizer (Adam, SGD, etc.)
- Implement training loop (forward propagation, backpropagation, parameter updates)
- Add validation steps to monitor overfitting

**Step 6: Save training results**
- Save model checkpoints periodically
- Log training metrics (loss, accuracy, etc.)
- Save best model

#### 2. Write Testing Scripts

Custom testing scripts for evaluating model performance, reference `evaluate.py` implementation:

**Functional modules**:
- Load trained model
- Perform inference on test set
- Calculate various evaluation metrics (accuracy, precision, recall, F1-score)
- Generate confusion matrices and visualizations
- Analyze performance under different SNR conditions
- Generate detailed evaluation reports

**Evaluation metrics**:
- Overall accuracy
- Macro and weighted average
- Per-class metrics
- Confusion matrix
- ROC curves and AUC values

**Visualization outputs**:
- Confusion matrix heatmaps
- SNR-accuracy curves
- Class performance comparison charts
- Feature space visualization (t-SNE, PCA)

#### 3. Model Fine-tuning

Fine-tuning based on pretrained models, applicable to:
- **New datasets**: Use pretrained model for fine-tuning on other datasets
- **Partial data**: Quick adaptation on small datasets
- **Specific tasks**: Optimization for specific SNR ranges or modulation types

**Fine-tuning steps**:
1. Freeze some layers (optional): Freeze parameters of generator or discriminator
2. Lower learning rate: Use lower learning rate than during pretraining (e.g., 1e-4 or 1e-5)
3. Select layers to optimize: Only train classifier or partial network layers
4. Early stopping: Monitor validation accuracy to avoid overfitting

#### 4. Model Architecture Modification

To modify model architecture, edit `src/models/enhanced_wgan_ecanet.py`:

**Modifiable components**:
- Multi-scale convolution kernel sizes (currently 1x1, 3x3, 5x5, 7x7)
- ECA attention parameters (gamma, b values)
- Spectral normalization toggle
- Residual connections
- Network depth and width

**Notes**:
- Retraining required after modifications
- Ensure input/output dimensions match
- Adjust configuration file `src/configs/model_config.json`
- Pretrained weights may not load directly (use `strict=False`)

### Extension Development Suggestions

#### 1. Add New Data Augmentation

Add new augmentation methods in data loader:
- Frequency masking
- Time warping
- Mixup, CutMix, and other mixing strategies

#### 2. Integrate Other Attention Mechanisms

Replace or supplement ECA attention:
- SE (Squeeze-and-Excitation) attention
- CBAM (Convolutional Block Attention Module)
- Self-Attention mechanisms

#### 3. Implement New Loss Functions

Besides standard cross-entropy loss, try:
- Focal Loss (for class imbalance)
- Label Smoothing
- Contrastive Loss
- Triplet Loss

#### 4. Cross-Dataset Generalization Experiments

Refer to implementation in `results/cross_dataset/`:
- Pretrain on RadioML 2018.01A
- Fine-tune on RadioML 2016.10A/B
- Evaluate zero-shot or few-shot generalization capability

## Path Configuration

The project uses `src/utils/path_manager.py` for unified path management, providing cross-platform compatibility.

### Configuration Methods

#### Method 1: Default Configuration (Recommended)

No manual configuration required, run directly:

```bash
python test_environment_optimized.py
python examples/basic_inference.py
```

The script automatically detects project structure.

#### Method 2: Environment Variables

Copy environment variable template:

```bash
cp .env.example .env
```

Edit `.env` file:

```bash
WGAN_MODEL_PATH=pretrain_models/model5_variant_3_seed_889.pth
WGAN_DATASET_PATH=dataset/RadioML 2018.01A
WGAN_DEVICE=cuda
```

#### Method 3: Command Line Arguments

```bash
python examples/basic_inference.py --model-path /path/to/model.pth
```

## Important Notes

### Usage Limitations

1. **Dataset Dependency**
   - Model training requires RadioML series datasets
   - Pretrained model trained on RadioML 2018.01A
   - Cross-dataset usage requires retraining or fine-tuning

2. **SNR Limitations**
   - Accuracy approaches random guessing (<10%) at very low SNR (< -12 dB)
   - Best performance occurs at high SNR (>= 18 dB)
   - Recommended usage within SNR range -20 dB to 30 dB

3. **Class Imbalance**
   - Significant performance differences across modulation types
   - Higher-order modulations (64QAM, 128QAM, 256QAM) have lower recall rates
   - Analog modulations (GMSK, FM) have unstable performance

4. **Computational Resource Requirements**
   - Training requires GPU (recommended VRAM >= 8GB)
   - Inference can run on CPU but slower
   - Pay attention to VRAM limitations during batch inference

5. **Model Compatibility**
   - Pretrained model requires `use_spectral_norm=True`
   - Input signals must be I/Q dual-channel format
   - Pretrained weights may not load after architecture modifications

### Special Requirements

1. **Random Seed Setting**
   - Fix random seed during training for reproducibility
   - Recommended seed: 889 (best performance)
   - Both data loading and model initialization are affected by seed

2. **Data Preprocessing**
   - Signal length must be 1024 (or modify model configuration)
   - Normalization recommended
   - Data augmentation should be consistent with training

3. **Model Saving and Loading**
   - Use `torch.save` to save complete checkpoints
   - Use `weights_only=False` when loading to read metadata
   - Check if `model_state_dict` key exists

4. **CUDA Version Compatibility**
   - Ensure PyTorch and CUDA versions match
   - Windows users recommended to use precompiled PyTorch wheels
   - Multi-GPU training requires additional configuration

### Common Issues

**Q1: "CUDA out of memory" error during runtime**
- Reduce batch size
- Reduce data augmentation operations
- Use gradient accumulation or mixed precision training

**Q2: File not found error when loading dataset**
- Check if dataset path is correct
- Confirm dataset file is completely downloaded
- Refer to dataset documentation for format requirements

**Q3: Model accuracy lower than expected**
- Check if pretrained weights are loaded correctly
- Confirm input signal format and normalization
- Verify SNR range is reasonable
- Check if data augmentation is excessive

**Q4: Slow performance on Windows**
- Set `num_workers` to 0
- Use precompiled PyTorch version
- Close unnecessary background programs to free memory

**Q5: How to use on custom datasets**
- Ensure data format is I/Q dual-channel
- Unify signal length to 1024 (or modify model)
- Perform appropriate normalization
- Consider fine-tuning model on new data

### Performance Benchmarks

Pretrained model performance on RadioML 2018.01A dataset:

| Metric | Value |
|--------|-------|
| Validation Accuracy | 63.701% |
| Test Accuracy | 64.804% |
| Parameters | 7.6M |
| Inference Latency | 8.2ms |
| Throughput | 1220 samples/sec |

SNR robustness:
- -20 dB: 4.87%
- -10 dB: 15.39%
- 0 dB: 62.72%
- 10 dB: 97.72%
- 20 dB: 98.24%

## Citation

If this project is helpful for your research, please consider citing:

```bibtex
@article{wgan-ecanet-2025,
  title={WGAN-ECANet: A Radio Signal Modulation Recognition Method Integrating Multi-scale Attention and Dynamic Gradient Penalty},
  author={Author Name},
  journal={IEEE Transactions on Wireless Communications},
  year={2025}
}
```

## License

This project is open source under the MIT License. See [LICENSE](LICENSE) file for details.

## Contact

- Project homepage: <repository-url>
- Issue reporting: Through GitHub Issues
- Email: <email-address>

## Acknowledgments

Thanks to the RadioML dataset providers and contributors to PyTorch and related open source communities.

---

**Last Updated**: January 2025
