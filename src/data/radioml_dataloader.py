import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple


class RadioMLDataset(Dataset):

    MODULATION_CLASSES = [
        'OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK',
        '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM',
        '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC',
        'AM-DSB-SC', 'FM', 'GMSK', 'OQPSK'
    ]

    def __init__(self, signals: np.ndarray, labels: np.ndarray,
                 transform=None):
        if signals.shape[-1] == 2 and len(signals.shape) == 3:
            signals = signals.transpose(0, 2, 1)

        self.signals = torch.FloatTensor(signals)

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

        if signal.dim() == 2 and signal.shape[1] == 2:
            signal = signal.transpose(0, 1)

        if self.transform:
            signal = self.transform(signal)

        return {
            'signals': signal,
            'labels': label,
            'label': label.item()
        }


class RadioMLDataLoader:

    def __init__(self, hdf5_path: str, max_samples: Optional[int] = None):
        self.hdf5_path = hdf5_path
        self.max_samples = max_samples
        self._load_data()

    def _load_data(self):
        print(f"Loading data: {self.hdf5_path}")

        with h5py.File(self.hdf5_path, 'r') as f:
            if self.max_samples is not None:
                total_samples = f['X'].shape[0]
                num_to_load = min(self.max_samples, total_samples)
                print(f"Limited sample count: {total_samples} -> {num_to_load}")
                self.X = f['X'][:num_to_load]
                self.Y = f['Y'][:num_to_load]
                self.Z = f['Z'][:num_to_load]
            else:
                self.X = f['X'][:]
                self.Y = f['Y'][:]
                self.Z = f['Z'][:]

        print(f"Data shape: {self.X.shape}")
        print(f"Label shape: {self.Y.shape}")

    def load_samples(self, start_idx: int = 0, num_samples: int = 1000
                  ) -> Tuple[np.ndarray, np.ndarray]:
        end_idx = min(start_idx + num_samples, len(self.X))
        signals = self.X[start_idx:end_idx]
        labels = np.argmax(self.Y[start_idx:end_idx], axis=1)

        return signals, labels

    def get_train_dataset(self, num_samples: Optional[int] = None
                        ) -> RadioMLDataset:
        num = num_samples if num_samples else len(self.X)
        return RadioMLDataset(self.X[:num], self.Y[:num])

    def get_validation_dataset(self, num_samples: Optional[int] = None
                            ) -> RadioMLDataset:
        num = num_samples if num_samples else len(self.X) // 5
        start = len(self.X) - num
        return RadioMLDataset(self.X[start:], self.Y[start:])

    def get_test_dataset(self, num_samples: Optional[int] = None
                       ) -> RadioMLDataset:
        num = num_samples if num_samples else len(self.X) // 10
        start = len(self.X) - num
        return RadioMLDataset(self.X[start:], self.Y[start:])
