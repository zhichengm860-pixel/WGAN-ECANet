"""
RadioML Data Loader

Provides three data loading methods:
1. RadioMLDataLoader - One-time loading (suitable for small datasets)
2. StreamingDataLoader - Streaming loading (suitable for large datasets, memory efficient)
3. create_dataloaders - Convenience function to automatically create train/val/test loaders
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, List, Callable
from sklearn.model_selection import train_test_split
import gc
import random


# ============================================================================
# Data Augmentation Functions
# ============================================================================

class SignalAugmentation:
    """
    Signal Data Augmentation (Enhanced - Prevent Overfitting)
    - Gaussian white noise (noise_factor=0.08)
    - Phase shift (80% probability, -30° to 30°)
    - Amplitude scaling (random scale 0.7-1.3)
    - Time shift (random shift up to 50 samples)
    """

    def __init__(self, noise_factor: float = 0.08, phase_shift_prob: float = 0.8,
                 phase_shift_range: Tuple[float, float] = (-30, 30),
                 amplitude_scale_range: Tuple[float, float] = (0.7, 1.3),
                 time_shift_max: int = 50):
        """
        Args:
            noise_factor: Gaussian noise factor (enhanced: 0.08)
            phase_shift_prob: Phase shift probability (enhanced: 0.8)
            phase_shift_range: Phase shift range in degrees (enhanced: -30° to 30°)
            amplitude_scale_range: Amplitude scaling range (new: 0.7-1.3)
            time_shift_max: Maximum time shift samples (new: 50)
        """
        self.noise_factor = noise_factor
        self.phase_shift_prob = phase_shift_prob
        self.phase_shift_range = phase_shift_range
        self.amplitude_scale_range = amplitude_scale_range
        self.time_shift_max = time_shift_max

    def add_gaussian_noise(self, signal: torch.Tensor) -> torch.Tensor:
        """Add Gaussian white noise"""
        noise = torch.randn_like(signal) * self.noise_factor
        return signal + noise

    def phase_shift(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Phase shift
        Signal format: (2, 1024) where [0,:] is I channel, [1,:] is Q channel
        """
        if random.random() > self.phase_shift_prob:
            return signal

        # Random phase shift angle (-10° to 10°)
        shift_degrees = random.uniform(self.phase_shift_range[0], self.phase_shift_range[1])
        shift_radians = np.deg2rad(shift_degrees)

        # Get I/Q channels
        i_channel = signal[0, :]  # (1024,)
        q_channel = signal[1, :]  # (1024,)

        # Convert to complex signal
        complex_signal = i_channel + 1j * q_channel

        # Apply phase shift
        shifted_signal = complex_signal * np.exp(1j * shift_radians)

        # Convert back to I/Q
        signal[0, :] = shifted_signal.real
        signal[1, :] = shifted_signal.imag

        return signal

    def amplitude_scale(self, signal: torch.Tensor) -> torch.Tensor:
        """Amplitude scaling augmentation"""
        scale = random.uniform(self.amplitude_scale_range[0], self.amplitude_scale_range[1])
        return signal * scale

    def time_shift(self, signal: torch.Tensor) -> torch.Tensor:
        """Time shift augmentation"""
        shift = random.randint(-self.time_shift_max, self.time_shift_max)
        if shift == 0:
            return signal
        # Circular shift
        return torch.roll(signal, shifts=shift, dims=1)

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Apply data augmentation

        Args:
            signal: Signal tensor, shape (2, 1024)
        Returns:
            Augmented signal
        """
        # Add Gaussian noise
        signal = self.add_gaussian_noise(signal)

        # Phase shift
        signal = self.phase_shift(signal)

        # Amplitude scaling
        signal = self.amplitude_scale(signal)

        # Time shift
        signal = self.time_shift(signal)

        return signal


# ============================================================================
# Modulation Type Definitions
# ============================================================================

MODULATION_CLASSES = [
    'OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK',
    '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM',
    '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC',
    'AM-DSB-SC', 'FM', 'GMSK', 'OQPSK'
]


# ============================================================================
# Dataset Classes
# ============================================================================

class RadioMLDataset(Dataset):
    """
    RadioML Dataset Class

    Supports:
    - Creating from numpy arrays in memory
    - Automatic shape conversion (1024, 2) -> (2, 1024)
    - One-hot label to class index conversion
    - Automatic normalization to [-1, 1] range (matching Tanh output)
    """

    def __init__(self, signals: np.ndarray, labels: np.ndarray, transform=None, normalize: bool = True):
        """
        Args:
            signals: Signal array, shape (N, 1024, 2) or (N, 2, 1024)
            labels: Label array, shape (N,) or (N, num_classes) one-hot
            transform: Optional data augmentation function
            normalize: Whether to normalize to [-1, 1] range
        """
        # Ensure shape is (N, 2, 1024)
        if signals.shape[-1] == 2 and len(signals.shape) == 3:
            signals = signals.transpose(0, 2, 1)

        signals = signals.astype(np.float32)

        # Normalize to [-1, 1] range (matching Generator Tanh output)
        self.normalize = normalize
        if normalize:
            # Calculate global statistics
            self.mean = signals.mean()
            self.std = signals.std()
            # Normalize to [0, 1], then map to [-1, 1]
            signals_normalized = (signals - self.mean) / (self.std + 1e-8)
            # Use tanh approximation to map to [-1, 1]
            signals = np.tanh(signals_normalized)
            print(f"Data normalization: mean={self.mean:.4f}, std={self.std:.4f}")

        self.signals = torch.FloatTensor(signals)

        # Convert labels
        if labels.ndim == 2:
            self.labels = torch.LongTensor(np.argmax(labels, axis=1))
        else:
            self.labels = torch.LongTensor(labels)

        self.transform = transform

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx]
        label = self.labels[idx]

        # Ensure correct shape
        if signal.dim() == 2 and signal.shape[1] == 2:
            signal = signal.transpose(0, 1)

        if self.transform:
            signal = self.transform(signal)

        return {
            'signals': signal,
            'labels': label,
            'label': label.item()
        }


class StreamingRadioMLDataset(Dataset):
    """
    Streaming Dataset Class - Read from HDF5 file on demand, memory efficient

    Use case: Large datasets (e.g., RadioML 2018.01A, 2.55M samples)
    """

    def __init__(self, hdf5_path: str, indices: np.ndarray, include_snr: bool = True, transform=None):
        """
        Args:
            hdf5_path: HDF5 file path
            indices: Array of sample indices to use
            include_snr: Whether to include SNR information
            transform: Optional data augmentation function
        """
        self.hdf5_path = hdf5_path
        self.indices = indices
        self.include_snr = include_snr
        self.transform = transform
        self._file = None

    def _get_file(self):
        """Get HDF5 file handle"""
        if self._file is None:
            self._file = h5py.File(self.hdf5_path, 'r')
        return self._file

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        f = self._get_file()
        original_idx = self.indices[idx]

        signal = f['X'][original_idx]
        label = f['Y'][original_idx]

        if signal.shape[-1] == 2:
            signal = signal.transpose(1, 0)

        label = np.argmax(label) if label.ndim > 0 else int(label)

        signal_tensor = torch.FloatTensor(signal)

        # Apply data augmentation
        if self.transform:
            signal_tensor = self.transform(signal_tensor)

        result = {
            'signals': signal_tensor,
            'labels': torch.tensor(label, dtype=torch.long)
        }

        if self.include_snr and 'Z' in f:
            snr = f['Z'][original_idx]
            snr = float(snr) if snr.ndim == 0 else float(snr[0])
            result['snr'] = torch.tensor(snr, dtype=torch.float32)

        return result

    def __del__(self):
        if self._file is not None:
            self._file.close()


# ============================================================================
# Data Loader Classes
# ============================================================================

class RadioMLDataLoader:
    """
    RadioML Data Loader - Load all data into memory at once

    Use case: Small datasets or when memory is sufficient

    Usage:
        loader = RadioMLDataLoader('dataset.hdf5')
        train_ds, val_ds, test_ds = loader.get_stratified_split()
    """

    def __init__(self, hdf5_path: str, max_samples: Optional[int] = None):
        """
        Args:
            hdf5_path: HDF5 file path
            max_samples: Maximum number of samples to load (for testing)
        """
        self.hdf5_path = hdf5_path
        self.max_samples = max_samples
        self._load_data()

    def _load_data(self):
        """Load data into memory"""
        print(f"Loading data: {self.hdf5_path}")

        with h5py.File(self.hdf5_path, 'r') as f:
            if self.max_samples is not None:
                total_samples = f['X'].shape[0]
                num_to_load = min(self.max_samples, total_samples)
                print(f"Limited sample count: {total_samples} -> {num_to_load}")
                self.X = f['X'][:num_to_load]
                self.Y = f['Y'][:num_to_load]
            else:
                self.X = f['X'][:]
                self.Y = f['Y'][:]

        print(f"Data shape: {self.X.shape}")
        print(f"Label shape: {self.Y.shape}")

    def get_stratified_split(self, train_ratio: float = 0.8, val_ratio: float = 0.1,
                             test_ratio: float = 0.1, random_state: int = 42
                             ) -> Tuple[RadioMLDataset, RadioMLDataset, RadioMLDataset]:
        """Stratified split of dataset"""
        labels = np.argmax(self.Y, axis=1) if self.Y.ndim == 2 else self.Y

        X_train, X_temp, Y_train, Y_temp = train_test_split(
            self.X, self.Y,
            test_size=(val_ratio + test_ratio),
            random_state=random_state,
            stratify=labels
        )

        val_test_ratio = test_ratio / (val_ratio + test_ratio)
        labels_temp = np.argmax(Y_temp, axis=1) if Y_temp.ndim == 2 else Y_temp
        X_val, X_test, Y_val, Y_test = train_test_split(
            X_temp, Y_temp,
            test_size=val_test_ratio,
            random_state=random_state,
            stratify=labels_temp
        )

        train_dataset = RadioMLDataset(X_train, Y_train)
        val_dataset = RadioMLDataset(X_val, Y_val)
        test_dataset = RadioMLDataset(X_test, Y_test)

        print(f"\nStratified split: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

        return train_dataset, val_dataset, test_dataset


# ============================================================================
# Convenience Functions
# ============================================================================

def create_dataloaders(hdf5_path: str, batch_size: int = 256,
                       train_ratio: float = 0.8, val_ratio: float = 0.1,
                       test_ratio: float = 0.1, random_state: int = 42,
                       num_workers: int = 4, use_streaming: bool = True,
                       include_snr: bool = True,
                       use_augmentation: bool = True
                       ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Convenience function to create data loaders

    Args:
        hdf5_path: HDF5 file path
        batch_size: Batch size
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        random_state: Random seed
        num_workers: Number of data loading workers
        use_streaming: Whether to use streaming loading (recommended for large datasets)
        include_snr: Whether to include SNR information
        use_augmentation: Whether to use data augmentation (training set only)

    Returns:
        train_loader, val_loader, test_loader

    Usage:
        train_loader, val_loader, test_loader = create_dataloaders(
            'dataset.hdf5', batch_size=256, num_workers=8
        )
    """
    print(f"Creating data loaders (streaming={use_streaming}, include_snr={include_snr})...")

    # Get dataset info
    with h5py.File(hdf5_path, 'r') as f:
        total_samples = f['X'].shape[0]
        num_classes = f['Y'].shape[1] if f['Y'].ndim > 1 else len(MODULATION_CLASSES)

    print(f"  Total samples: {total_samples:,}")
    print(f"  Num classes: {num_classes}")

    # Read labels in chunks for stratified sampling
    print("Reading labels for stratified split...")
    chunk_size = 500000
    all_indices = []
    all_labels = []

    with h5py.File(hdf5_path, 'r') as f:
        for start in range(0, total_samples, chunk_size):
            end = min(start + chunk_size, total_samples)
            labels_chunk = f['Y'][start:end]
            labels_1d = np.argmax(labels_chunk, axis=1) if labels_chunk.ndim == 2 else labels_chunk

            all_indices.extend(range(start, end))
            all_labels.extend(labels_1d)

            print(f"  Progress: {end:,}/{total_samples:,} ({end/total_samples*100:.1f}%)")

    all_indices = np.array(all_indices)
    all_labels = np.array(all_labels)

    # Stratified split
    print("Performing stratified split...")
    train_indices, temp_indices, _, temp_labels = train_test_split(
        all_indices, all_labels,
        test_size=(val_ratio + test_ratio),
        random_state=random_state,
        stratify=all_labels
    )

    val_test_ratio = test_ratio / (val_ratio + test_ratio)
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=val_test_ratio,
        random_state=random_state,
        stratify=temp_labels
    )

    # Release label memory
    del all_labels, temp_labels
    gc.collect()

    print(f"\nSplit complete:")
    print(f"  Train: {len(train_indices):,} ({len(train_indices)/total_samples*100:.1f}%)")
    print(f"  Val:   {len(val_indices):,} ({len(val_indices)/total_samples*100:.1f}%)")
    print(f"  Test:  {len(test_indices):,} ({len(test_indices)/total_samples*100:.1f}%)")

    # Create data augmenter (training set only)
    train_transform = SignalAugmentation() if use_augmentation else None

    # Create datasets
    if use_streaming:
        train_dataset = StreamingRadioMLDataset(hdf5_path, train_indices, include_snr, train_transform)
        val_dataset = StreamingRadioMLDataset(hdf5_path, val_indices, include_snr, None)
        test_dataset = StreamingRadioMLDataset(hdf5_path, test_indices, include_snr, None)
    else:
        # One-time loading
        with h5py.File(hdf5_path, 'r') as f:
            train_X = f['X'][train_indices]
            train_Y = f['Y'][train_indices]
            val_X = f['X'][val_indices]
            val_Y = f['Y'][val_indices]
            test_X = f['X'][test_indices]
            test_Y = f['Y'][test_indices]

        train_dataset = RadioMLDataset(train_X, train_Y, train_transform)
        val_dataset = RadioMLDataset(val_X, val_Y, None)
        test_dataset = RadioMLDataset(test_X, test_Y, None)

    # Create data loaders (disable persistent_workers to avoid memory usage)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=False
    )

    return train_loader, val_loader, test_loader


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'MODULATION_CLASSES',
    'RadioMLDataset',
    'StreamingRadioMLDataset',
    'RadioMLDataLoader',
    'create_dataloaders'
]
