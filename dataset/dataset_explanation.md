# Dataset Documentation (RadioML Series)

This document details the data format, classes, and differences of the datasets used in this project (RadioML 2016.10A / 2016.10B / 2018.01A), and explains their loading processes, train/validation/test split strategies, and specific roles in the project. The content of this document is consistent with the project source code implementation (see `src/data/*.py`).

Directory Structure (Existing Files):
|- RadioML 2016.10A/
  - RML2016.10a_dict.pkl (625,898.1KB)
|- RadioML 2016.10B/
  - RML2016.10b.dat (3,418,538.5KB)
|- RadioML 2018.01A/
  - GOLD_XYZ_OSC.0001_1024.hdf5 (20,946,433.9KB)
  - LICENSE.TXT (20.5KB)
  - classes-fixed.json (0.2KB)
  - classes-fixed.txt (1.6KB)
  - classes.txt (0.3KB)
  - datasets.desktop (0.2KB)

## 1. Dataset Overview and Differences

|- RadioML 2016.10A (.pkl) and 2016.10B (.dat)
  - Storage structure: Python dictionary, keys are `(modulation, snr)`, values are sample arrays under that condition.
  - Sample shape: `[num_samples, 2, signal_length]` (2 represents I/Q two channels).
  - Class set (code default): `['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']` total 11 classes.
  - Label format: Directly use class index (mapped from modulation name).
  - SNR: Provided by `snr` in the key, expanded to vector by sample.
  - File size: 2016.10A is 625,898.1KB, 2016.10B is 3,418,538.5KB

|- RadioML 2018.01A (.hdf5)
  - Storage structure: HDF5 file, typically three datasets:
    - `X`: `[N, 1024, 2]`, I/Q dual channels, length 1024.
    - `Y`: `[N, 24]`, one-hot labels (24 classes).
    - `Z`: `[N]` or `[N, 1]`, SNR values (padded with 0 if missing).
  - Class set (default or read from `classes-fixed.json`) total 24 classes:
    - `['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK', '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM', '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC', 'FM', 'GMSK', 'OQPSK']`
  - Label format: Converted from one-hot to class index after loading.

Main Differences:
|- File format: 2016 is pkl/dat dictionary format, 2018 is hdf5 standard matrix format.
|- Number of classes: 2016 default 11 classes; 2018 is 24 classes, covering a wider range of modulation types and amplitude-phase modulation orders.
|- Label encoding: 2016 generated from class index; 2018 native is one-hot, converted to index during loading.
|- Shape specification: Unified loader will convert data to `[N, C, L]` (C=1 or 2, L=signal_length).
|- File size: 2016.10A is 625,898.1KB, 2016.10B is 3,418,538.5KB, 2018.01A is 20,946,433.9KB.

## 2. Loaders and Usage Paths in Project

The project provides two sets of loaders, prioritizing the "Unified Loader" to be compatible with all three formats:

|- Unified Loader (Recommended): `src/data/unified_radioml_loader.py`
  - Class: `UnifiedRadioMLLoader`
  - Dataset class: `UnifiedRadioMLDataset` (output shape `[N, C, L]`, labels are class indices, can return SNR)
  - Supported formats: `.pkl` (2016.10A), `.dat` (2016.10B), `.hdf5` (2018.01A)
  - Memory friendly: Supports `max_samples_per_class` for class-balanced sampling and memory limits
  - Data split: `train/val/test` randomly split by ratio (can set random seed)
  - Entry function: `create_unified_radioml_data_loaders(config, device)`

|- 2018.01A Dedicated Loader: `src/data/radioml_loader.py`
  - Class: `RadioMLDataLoader`, `RadioMLDataset`
  - Specialized for HDF5 (2018.01A), also supports class balancing and splitting
  - Entry function: `create_radioml_data_loaders(config, device)`
  - Can be selected when `config['data']['use_unified_loader']=False` and data is hdf5

|- Top-level Entry (Auto-select loader): `src/data/data_loader.py`
  - Function: `create_data_loaders(config, device)`
  - Selection strategy:
    - If `config['data']['radioml_path']` points to `.pkl` or `.dat`: use unified loader
    - If points to `.hdf5`: default use unified loader; can switch to dedicated loader when `use_unified_loader=False`
    - If using preprocessed `.npy`: go through `data_dir` fallback path, directly load `train/val/test` arrays

Call Path Examples:
|- Experiment main flow: In `src/experiments/main_experiment.py`, call `create_data_loaders(config, device)`, get `train_loader / val_loader / test_loader`, used for model training and evaluation.
|- Augmentation trainer: `src/training/improved_trainer.py` directly instantiates `UnifiedRadioMLLoader` and builds datasets and DataLoader, for improved training flow (e.g., weighted sampling, memory limits, etc.).

## 3. Data Loading Output and Model Input

|- DataLoader output is usually a triple `(signals, labels, snr)` or a tuple `(signals, labels)` (when SNR is not returned).
|- `signals` shape: Unified to `[B, C, L]` (C=2 means I/Q, L usually 1024).
|- `labels`: Class index (long integer).
|- `snr`: Floating point, if present, participates in robustness evaluation, analysis, or optional training assistance.

Model side usage:
|- WGAN-GP-ECANet (`src/models/wgan_gp_ecanet.py`) classifier part directly receives the above `signals`, and calculates classification loss with `labels`; SNR can be used for robustness evaluation (see `src/evaluation/evaluator.py`).

## 4. Data Split and Sampling Strategy

|- Split ratio: Default `train:val:test = 0.7 : 0.15 : 0.15`, controlled by `train_ratio / val_ratio / test_ratio` in `config['data']`.
|- Randomness: `config['experiment']['seed']` controls the random seed for shuffling and splitting.
|- Class balance and memory control: `max_samples_per_class` limits the maximum number of samples per class, and in 2018.01A adopts the strategy of "sampling by class + global shuffling"; 2016 data will be subject to this limit in each `(mod, snr)` group before merging.

## 5. Configuration Examples

|- Load 2018.01A (default unified loader):
```yaml
data:
  dataset_path: "dataset/RadioML 2018.01A/GOLD_XYZ_OSC.0001_1024.hdf5"
  radioml_path: "dataset/RadioML 2018.01A/GOLD_XYZ_OSC.0001_1024.hdf5"
  use_unified_loader: true
  signal_length: 1024
  num_classes: 24
  input_channels: 2
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
  max_samples_per_class: 1000
  split_ratios:
    train: 0.7
    val: 0.15
    test: 0.15
  min_snr: -10
  classes_path: "dataset/RadioML 2018.01A/classes-fixed.json"   # Optional, provide class names
  augmentation:
    time_warp: true
    freq_mask: true
    time_mask: true
    noise_injection: true
    scaling: true
    shift: true
    mixup: true
    cutmix: true
    spec_augment: true
    random_erase: true
    augment_prob: 0.9
  regularization:
    dropout_rate: 0.6
    weight_decay: 0.0005
    label_smoothing: 0.4
    batch_norm_momentum: 0.05
    layer_norm: true
    gradient_clip_norm: 0.8
training:
  batch_size: 128
  num_workers: 4
experiment:
  seed: 42
```

|- Load 2016.10A (.pkl):
```yaml
data:
  dataset_path: "dataset/RadioML 2016.10A/RML2016.10a_dict.pkl"
  radioml_path: "dataset/RadioML 2016.10A/RML2016.10a_dict.pkl"
  use_unified_loader: true
  signal_length: 128
  num_classes: 11
  input_channels: 2
  max_samples_per_class: 2000          # Recommended setting
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
  split_ratios:
    train: 0.7
    val: 0.15
    test: 0.15
  min_snr: -10
  classes_path: "dataset/RadioML 2016.10A/classes.json"   # Optional, provide class names
  augmentation:
    time_warp: true
    freq_mask: true
    time_mask: true
    noise_injection: true
    scaling: true
    shift: true
    mixup: true
    cutmix: true
    spec_augment: true
    random_erase: true
    augment_prob: 0.9
  regularization:
    dropout_rate: 0.6
    weight_decay: 0.0005
    label_smoothing: 0.4
    batch_norm_momentum: 0.05
    layer_norm: true
    gradient_clip_norm: 0.8
training:
  batch_size: 128
  num_workers: 4
```

|- Load 2016.10B (.dat):
```yaml
data:
  dataset_path: "dataset/RadioML 2016.10B/RML2016.10b.dat"
  radioml_path: "dataset/RadioML 2016.10B/RML2016.10b.dat"
  use_unified_loader: true
  signal_length: 128
  num_classes: 11
  input_channels: 2
  max_samples_per_class: 2000
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
  split_ratios:
    train: 0.7
    val: 0.15
    test: 0.15
  min_snr: -10
  classes_path: "dataset/RadioML 2016.10B/classes.json"   # Optional, provide class names
  augmentation:
    time_warp: true
    freq_mask: true
    time_mask: true
    noise_injection: true
    scaling: true
    shift: true
    mixup: true
    cutmix: true
    spec_augment: true
    random_erase: true
    augment_prob: 0.9
  regularization:
    dropout_rate: 0.6
    weight_decay: 0.0005
    label_smoothing: 0.4
    batch_norm_momentum: 0.05
    layer_norm: true
    gradient_clip_norm: 0.8
training:
  batch_size: 128
  num_workers: 4
```

|- Use preprocessed NPY data (fallback path):
```yaml
data:
  data_dir: "path/to/preprocessed"  # Directory contains train_signals.npy, train_labels.npy, val_*.npy, test_*.npy
training:
  batch_size: 128
```

## 6. Specific Roles of Each Dataset

|- RadioML 2016.10A / 2016.10B
  - Used for basic modulation type recognition tasks, fewer classes (11), suitable for quickly validating model structure, parameter tuning, and convergence comparison.
  - Can be quickly switched in unified loader to verify model consistency and generalization for data from different sources/formats.

|- RadioML 2018.01A
  - Richer classes (24 classes), including multi-order QAM/PSK/APSK, etc., suitable for comprehensively evaluating classifier capabilities and WGAN generator fitting to complex modulation distributions.
  - Paired with `classes-fixed.json` can unify class order, ensuring result comparability.
  - File size is 20,946,433.9KB, the largest dataset.

|- SNR Dimension (all datasets)
  - Used for robustness evaluation and optional training analysis (e.g., evaluating accuracy by SNR segments, plotting performance curves with SNR).

## 7. Data Preprocessing and Notes

|- Shape unification: Automatically standardized to `[N, C, L]` after loading; 2018.01A's `[N, 1024, 2]` will be transposed to `[N, 2, 1024]`; 2016's `[N, 2, L]` is directly compatible.
|- Label unification: Regardless of source, final are class indices (converted from one-hot or mapped from modulation name).
|- SNR shape: If `[N, 1]` will be flattened to `[N]`.
|- Large data volume: Recommend setting `max_samples_per_class` to control memory and improve iteration speed; under Windows/CPU environment, setting `num_workers` to 0 is safer.
|- Class file: If `RadioML 2018.01A/classes-fixed.json` exists, it will be used for class name loading, ensuring consistency with training/evaluation reports.

## 8. Code Reference Locations

|- Unified loader and dataset:
  - `src/data/unified_radioml_loader.py` (`UnifiedRadioMLLoader`, `UnifiedRadioMLDataset`, `create_unified_radioml_data_loaders`)
|- 2018.01A dedicated loader:
  - `src/data/radioml_loader.py` (`RadioMLDataLoader`, `RadioMLDataset`, `create_radioml_data_loaders`)
|- Auto-select entry:
  - `src/data/data_loader.py` (`create_data_loaders`)
|- Training/evaluation usage:
  - `src/experiments/main_experiment.py` (build data loaders, evaluation)
  - `src/training/improved_trainer.py` (directly use unified loader and support weighted sampling)
  - `src/evaluation/evaluator.py` (reading convention of `(signals, labels, snr)` during evaluation)

If you need to add new datasets or custom preprocessing, it is recommended to extend based on `UnifiedRadioMLLoader`, maintaining consistent shape and label protocols, to avoid introducing extra branch logic in models and trainers.
