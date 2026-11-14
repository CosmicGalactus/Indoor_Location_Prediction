"""
PyTorch Dataset and Data Augmentation for Wi-Fi Fingerprinting
=============================================================
Provides PyTorch Dataset implementations, coordinate scalers, and domain-informed
data augmentations (AP Dropout and Gaussian RSSI Jitter).
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class CoordinateScaler:
    """Standardizes 2D coordinates for stable multi-task neural network training."""
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, coords: np.ndarray):
        coords = np.asarray(coords, dtype=np.float32)
        self.mean_ = np.mean(coords, axis=0)
        self.std_ = np.std(coords, axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, coords: np.ndarray) -> np.ndarray:
        coords = np.asarray(coords, dtype=np.float32)
        return (coords - self.mean_) / self.std_

    def fit_transform(self, coords: np.ndarray) -> np.ndarray:
        return self.fit(coords).transform(coords)

    def inverse_transform(self, norm_coords: np.ndarray) -> np.ndarray:
        norm_coords = np.asarray(norm_coords, dtype=np.float32)
        return (norm_coords * self.std_) + self.mean_


class WiFiFingerprintDataset(Dataset):
    """
    Multi-Task PyTorch Dataset for Wi-Fi Fingerprints.
    
    Parameters
    ----------
    X : np.ndarray
        Array of shape (N, n_features) of normalized RSSI values [0, 1].
    y_bldg : np.ndarray, optional
        Target Building IDs (0, 1, 2).
    y_floor : np.ndarray, optional
        Target Floor Levels (0, 1, 2, 3, 4).
    y_coords : np.ndarray, optional
        Target coordinates (Latitude, Longitude), preferably scaled.
    is_training : bool, default=False
        Whether to enable data augmentation during sample retrieval.
    ap_dropout_prob : float, default=0.15
        Probability of dropping an active AP (setting to 0.0) during training.
    noise_std : float, default=0.02
        Standard deviation of Gaussian noise added to active signals during training.
    """
    def __init__(
        self,
        X: np.ndarray,
        y_bldg: np.ndarray = None,
        y_floor: np.ndarray = None,
        y_coords: np.ndarray = None,
        is_training: bool = False,
        ap_dropout_prob: float = 0.15,
        noise_std: float = 0.02,
    ):
        self.X = np.asarray(X, dtype=np.float32)
        self.y_bldg = np.asarray(y_bldg, dtype=np.int64) if y_bldg is not None else None
        self.y_floor = np.asarray(y_floor, dtype=np.int64) if y_floor is not None else None
        self.y_coords = np.asarray(y_coords, dtype=np.float32) if y_coords is not None else None

        self.is_training = is_training
        self.ap_dropout_prob = ap_dropout_prob
        self.noise_std = noise_std

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        x = self.X[idx].copy()

        # Apply Data Augmentations strictly during training
        if self.is_training:
            active_mask = x > 0.0
            
            # 1. AP Dropout: randomly mask a fraction of active access points
            if self.ap_dropout_prob > 0.0 and np.any(active_mask):
                drop_mask = np.random.binomial(1, 1.0 - self.ap_dropout_prob, size=x.shape).astype(bool)
                # Ensure we don't zero out every single AP
                if np.any(active_mask & drop_mask):
                    x = np.where(active_mask & ~drop_mask, 0.0, x)

            # 2. Gaussian Jitter on detected signals (simulating device hardware antenna gain differences)
            if self.noise_std > 0.0 and np.any(active_mask):
                noise = np.random.normal(0.0, self.noise_std, size=x.shape).astype(np.float32)
                x = np.where(x > 0.0, np.clip(x + noise, 0.01, 1.0), 0.0)

        item = {'x': torch.tensor(x, dtype=torch.float32)}

        if self.y_bldg is not None:
            item['y_bldg'] = torch.tensor(self.y_bldg[idx], dtype=torch.long)
        if self.y_floor is not None:
            item['y_floor'] = torch.tensor(self.y_floor[idx], dtype=torch.long)
        if self.y_coords is not None:
            item['y_coords'] = torch.tensor(self.y_coords[idx], dtype=torch.float32)

        return item


def create_dataloaders(
    X_train: np.ndarray,
    y_train_bldg: np.ndarray,
    y_train_floor: np.ndarray,
    y_train_coords: np.ndarray,
    X_val: np.ndarray,
    y_val_bldg: np.ndarray,
    y_val_floor: np.ndarray,
    y_val_coords: np.ndarray,
    batch_size: int = 128,
    ap_dropout_prob: float = 0.15,
    noise_std: float = 0.02,
):
    """Factory creating train and validation DataLoaders with coordinate scaling."""
    scaler = CoordinateScaler()
    y_train_coords_scaled = scaler.fit_transform(y_train_coords)
    y_val_coords_scaled = scaler.transform(y_val_coords)

    train_ds = WiFiFingerprintDataset(
        X_train, y_train_bldg, y_train_floor, y_train_coords_scaled,
        is_training=True, ap_dropout_prob=ap_dropout_prob, noise_std=noise_std
    )
    val_ds = WiFiFingerprintDataset(
        X_val, y_val_bldg, y_val_floor, y_val_coords_scaled,
        is_training=False
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader, scaler

