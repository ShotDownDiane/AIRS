"""
Data loading, preprocessing, and dataset classes for REMX-TF.

Supports:
  - Real PeMS04 / PeMS08 .npz files
  - Synthetic data generation for unit-testing / CI

Traffic data format (from STAEformer):
  .npz with key 'data': shape [T_total, N, C]
  Adjacency: .csv distance matrix or pre-built .npy

Weather / incident data:
  .npy arrays of shape [T_total, N, C_w] and [T_total, N]
  If not found, synthetic versions are generated.
"""

import os
import logging
from typing import Optional, Tuple, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Normalization helpers
# ─────────────────────────────────────────────────────────────────────────────

class StandardScaler:
    """Per-sensor z-score normalizer (fit on training split only)."""

    def __init__(self):
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray):
        """x: [T, N, C]"""
        self.mean = x.mean(axis=0, keepdims=True)   # [1, N, C]
        self.std  = x.std(axis=0, keepdims=True) + 1e-8
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean

    def inverse_transform_tensor(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.mean, dtype=x.dtype, device=x.device)
        std  = torch.tensor(self.std,  dtype=x.dtype, device=x.device)
        return x * std + mean

    def save(self, path: str):
        np.savez(path, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path: str) -> "StandardScaler":
        d = np.load(path)
        sc = cls()
        sc.mean = d["mean"]
        sc.std  = d["std"]
        return sc


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data generation (for testing without real PeMS files)
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_traffic(
    T: int,
    N: int,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic traffic, weather, and incident data.

    Returns:
        traffic:  [T, N, 1]  float32
        weather:  [T, N, 7]  float32
        incident: [T, N]     float32 (binary)
    """
    rng = np.random.default_rng(seed)

    # Time indices
    t_idx = np.arange(T)
    tod   = (t_idx % 288) / 288.0   # time-of-day fraction

    # Base flow: sinusoidal daily pattern + sensor-specific offset
    sensor_base = rng.uniform(20, 80, size=(1, N))
    daily_pattern = 30 * np.sin(2 * np.pi * tod[:, None] - np.pi / 2) + 30
    noise = rng.normal(0, 3, size=(T, N))
    traffic = (sensor_base + daily_pattern + noise).clip(0).astype(np.float32)
    traffic = traffic[:, :, np.newaxis]  # [T, N, 1]

    # Weather: 7 features
    temp       = 15 + 10 * np.sin(2 * np.pi * t_idx / (288 * 365)) + rng.normal(0, 2, T)
    precip     = np.maximum(0, rng.normal(0, 0.5, T))
    visibility = np.clip(10 + rng.normal(0, 2, T), 0.1, 20)
    wind       = np.abs(rng.normal(3, 2, T))
    humidity   = np.clip(rng.normal(60, 15, T), 0, 100)
    weather_cat = rng.integers(0, 4, T).astype(float)
    is_night   = ((t_idx % 288) < 72).astype(float)  # 00:00–06:00

    # Broadcast to [T, N, 7]
    weather = np.stack([
        np.tile(temp[:, None],        (1, N)),
        np.tile(precip[:, None],      (1, N)),
        np.tile(visibility[:, None],  (1, N)),
        np.tile(wind[:, None],        (1, N)),
        np.tile(humidity[:, None],    (1, N)),
        np.tile(weather_cat[:, None], (1, N)),
        np.tile(is_night[:, None],    (1, N)),
    ], axis=-1).astype(np.float32)

    # Incidents: sparse binary flags (~5% of (t, n) pairs)
    incident = (rng.random((T, N)) < 0.05).astype(np.float32)

    return traffic, weather, incident


def generate_synthetic_adjacency(N: int, seed: int = 42) -> np.ndarray:
    """Generate a random sparse adjacency matrix for N sensors."""
    rng = np.random.default_rng(seed)
    # Random positions on a 2D grid
    pos = rng.uniform(0, 100, (N, 2))
    dist = np.sqrt(((pos[:, None] - pos[None, :]) ** 2).sum(-1))
    sigma = 10.0
    adj = np.exp(-(dist ** 2) / (2 * sigma ** 2))
    adj[dist > 30] = 0.0
    np.fill_diagonal(adj, 0.0)
    return adj.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Weather preprocessing helpers
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_weather(weather: np.ndarray) -> np.ndarray:
    """
    Apply per-feature normalization to weather array [T, N, 7].

    Feature layout:
      0: temperature     → z-normalize
      1: precip_rate     → log1p then z-normalize
      2: visibility      → z-normalize
      3: wind_speed      → z-normalize
      4: humidity        → z-normalize
      5: weather_cat     → keep as-is (0–3 integer)
      6: is_night        → keep as-is (binary)
    """
    w = weather.copy()
    # log1p for precipitation
    w[..., 1] = np.log1p(w[..., 1])
    # z-normalize features 0–4
    for i in range(5):
        mu  = w[..., i].mean()
        std = w[..., i].std() + 1e-8
        w[..., i] = (w[..., i] - mu) / std
    return w.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Sliding-window dataset
# ─────────────────────────────────────────────────────────────────────────────

class TrafficDataset(Dataset):
    """
    Sliding-window dataset for traffic forecasting.

    Each sample:
        x_traffic:  [N, T, C_in]
        x_weather:  [N, T, C_w]
        x_incident: [N, T]
        y:          [N, T', C_in]
        tod_idx:    [T]   (time-of-day index 0..287)
        dow_idx:    [T]   (day-of-week index 0..6)
    """

    def __init__(
        self,
        traffic:  np.ndarray,   # [T_total, N, C_in]
        weather:  np.ndarray,   # [T_total, N, C_w]
        incident: np.ndarray,   # [T_total, N]
        in_steps:  int = 12,
        out_steps: int = 12,
        start_ts:  int = 0,     # absolute start timestamp (for tod/dow)
        augment:   bool = False,
        aug_jitter_prob:  float = 0.2,
        aug_sensor_prob:  float = 0.1,
        aug_sensor_frac:  float = 0.1,
        aug_weather_noise: float = 0.05,
        seed: int = 42,
    ):
        self.traffic  = traffic.astype(np.float32)
        self.weather  = weather.astype(np.float32)
        self.incident = incident.astype(np.float32)
        self.in_steps  = in_steps
        self.out_steps = out_steps
        self.start_ts  = start_ts
        self.augment   = augment
        self.aug_jitter_prob   = aug_jitter_prob
        self.aug_sensor_prob   = aug_sensor_prob
        self.aug_sensor_frac   = aug_sensor_frac
        self.aug_weather_noise = aug_weather_noise
        self.rng = np.random.default_rng(seed)

        T_total = traffic.shape[0]
        self.indices = list(range(in_steps, T_total - out_steps + 1))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        t = self.indices[idx]

        # Optional temporal jitter
        if self.augment and self.rng.random() < self.aug_jitter_prob:
            jitter = int(self.rng.choice([-1, 1]))
            t = max(self.in_steps, min(len(self.indices) - 1 + self.in_steps, t + jitter))

        x_traffic  = self.traffic[t - self.in_steps : t]          # [T, N, C]
        x_weather  = self.weather[t - self.in_steps : t]          # [T, N, C_w]
        x_incident = self.incident[t - self.in_steps : t]         # [T, N]
        y          = self.traffic[t : t + self.out_steps]         # [T', N, C]

        # Transpose to [N, T, C] convention
        x_traffic  = x_traffic.transpose(1, 0, 2)   # [N, T, C]
        x_weather  = x_weather.transpose(1, 0, 2)   # [N, T, C_w]
        x_incident = x_incident.transpose(1, 0)     # [N, T]
        y          = y.transpose(1, 0, 2)            # [N, T', C]

        # Sensor dropout augmentation
        if self.augment and self.rng.random() < self.aug_sensor_prob:
            N = x_traffic.shape[0]
            n_drop = max(1, int(N * self.aug_sensor_frac))
            drop_idx = self.rng.choice(N, n_drop, replace=False)
            x_traffic[drop_idx] = 0.0

        # Weather noise augmentation
        if self.augment and self.aug_weather_noise > 0:
            x_weather = x_weather + self.rng.normal(
                0, self.aug_weather_noise, x_weather.shape
            ).astype(np.float32)

        # Time indices for embeddings
        abs_t = self.start_ts + t
        tod_idx = np.array([(abs_t - self.in_steps + i) % 288 for i in range(self.in_steps)],
                           dtype=np.int64)
        dow_idx = np.array([((abs_t - self.in_steps + i) // 288) % 7 for i in range(self.in_steps)],
                           dtype=np.int64)

        return {
            "x_traffic":  torch.from_numpy(x_traffic),
            "x_weather":  torch.from_numpy(x_weather),
            "x_incident": torch.from_numpy(x_incident),
            "y":          torch.from_numpy(y),
            "tod_idx":    torch.from_numpy(tod_idx),
            "dow_idx":    torch.from_numpy(dow_idx),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Data loading entry point
# ─────────────────────────────────────────────────────────────────────────────

def load_pems_data(
    data_dir: str,
    dataset: str,
    in_steps: int = 12,
    out_steps: int = 12,
    batch_size: int = 32,
    num_workers: int = 4,
    use_synthetic: bool = False,
    synthetic_timesteps: int = 17280,
    num_sensors: Optional[int] = None,
    seed: int = 42,
    aug_jitter_prob: float = 0.2,
    aug_sensor_prob: float = 0.1,
    aug_sensor_frac: float = 0.1,
    aug_weather_noise: float = 0.05,
) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler, np.ndarray]:
    """
    Load and prepare PeMS data.

    Returns:
        train_loader, val_loader, test_loader, scaler, adjacency_matrix
    """
    dataset_cfg = {
        "PEMS04": {"N": 307, "npz": "PEMS04.npz", "adj": "PEMS04.csv"},
        "PEMS08": {"N": 170, "npz": "PEMS08.npz", "adj": "PEMS08.csv"},
    }
    if dataset not in dataset_cfg:
        raise ValueError(f"Unknown dataset '{dataset}'")

    cfg = dataset_cfg[dataset]
    N   = num_sensors or cfg["N"]

    # ── Load or generate traffic data ────────────────────────────────────────
    npz_path = os.path.join(data_dir, dataset, cfg["npz"])

    if use_synthetic or not os.path.exists(npz_path):
        if not use_synthetic:
            logger.warning(
                f"Data file not found at {npz_path}. "
                "Falling back to synthetic data. "
                "Set use_synthetic=True to suppress this warning."
            )
        T_total = synthetic_timesteps
        traffic, weather, incident = generate_synthetic_traffic(T_total, N, seed=seed)
        adj = generate_synthetic_adjacency(N, seed=seed)
        logger.info(f"Using synthetic data: T={T_total}, N={N}")
    else:
        logger.info(f"Loading traffic data from {npz_path}")
        raw = np.load(npz_path)
        traffic = raw["data"][:, :N, :1].astype(np.float32)  # [T, N, 1]
        T_total = traffic.shape[0]

        # Load or generate weather
        weather_path = os.path.join(data_dir, dataset, "weather.npy")
        if os.path.exists(weather_path):
            weather = np.load(weather_path).astype(np.float32)[:T_total, :N]
        else:
            logger.warning("Weather data not found; using synthetic weather.")
            _, weather, _ = generate_synthetic_traffic(T_total, N, seed=seed + 1)

        # Load or generate incidents
        incident_path = os.path.join(data_dir, dataset, "incident.npy")
        if os.path.exists(incident_path):
            incident = np.load(incident_path).astype(np.float32)[:T_total, :N]
        else:
            logger.warning("Incident data not found; using synthetic incidents.")
            _, _, incident = generate_synthetic_traffic(T_total, N, seed=seed + 2)

        # Load adjacency
        adj_path = os.path.join(data_dir, dataset, cfg["adj"])
        adj_npy  = os.path.join(data_dir, dataset, "adj.npy")
        if os.path.exists(adj_npy):
            adj = np.load(adj_npy).astype(np.float32)[:N, :N]
        elif os.path.exists(adj_path):
            adj = _load_adj_csv(adj_path, N)
        else:
            logger.warning("Adjacency file not found; using synthetic adjacency.")
            adj = generate_synthetic_adjacency(N, seed=seed)

    # ── Train / val / test split (6:2:2) ─────────────────────────────────────
    n_train = int(T_total * 0.6)
    n_val   = int(T_total * 0.2)

    train_traffic  = traffic[:n_train]
    val_traffic    = traffic[n_train : n_train + n_val]
    test_traffic   = traffic[n_train + n_val:]

    train_weather  = weather[:n_train]
    val_weather    = weather[n_train : n_train + n_val]
    test_weather   = weather[n_train + n_val:]

    train_incident  = incident[:n_train]
    val_incident    = incident[n_train : n_train + n_val]
    test_incident   = incident[n_train + n_val:]

    # ── Normalize traffic (fit on train only) ─────────────────────────────────
    scaler = StandardScaler().fit(train_traffic)
    train_traffic = scaler.transform(train_traffic)
    val_traffic   = scaler.transform(val_traffic)
    test_traffic  = scaler.transform(test_traffic)

    # ── Normalize weather ─────────────────────────────────────────────────────
    train_weather = preprocess_weather(train_weather)
    val_weather   = preprocess_weather(val_weather)
    test_weather  = preprocess_weather(test_weather)

    # ── Build datasets ────────────────────────────────────────────────────────
    common_kw = dict(in_steps=in_steps, out_steps=out_steps, seed=seed)
    aug_kw = dict(
        aug_jitter_prob=aug_jitter_prob,
        aug_sensor_prob=aug_sensor_prob,
        aug_sensor_frac=aug_sensor_frac,
        aug_weather_noise=aug_weather_noise,
    )

    train_ds = TrafficDataset(
        train_traffic, train_weather, train_incident,
        start_ts=0, augment=True, **common_kw, **aug_kw
    )
    val_ds = TrafficDataset(
        val_traffic, val_weather, val_incident,
        start_ts=n_train, augment=False, **common_kw
    )
    test_ds = TrafficDataset(
        test_traffic, test_weather, test_incident,
        start_ts=n_train + n_val, augment=False, **common_kw
    )

    loader_kw = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kw)

    logger.info(
        f"Dataset splits — train: {len(train_ds)}, "
        f"val: {len(val_ds)}, test: {len(test_ds)}"
    )
    return train_loader, val_loader, test_loader, scaler, adj


def _load_adj_csv(path: str, N: int) -> np.ndarray:
    """Load adjacency from PeMS-style CSV (sensor_id_from, sensor_id_to, distance)."""
    import csv
    adj = np.zeros((N, N), dtype=np.float32)
    with open(path) as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if len(row) < 3:
                continue
            i, j, d = int(row[0]), int(row[1]), float(row[2])
            if i < N and j < N:
                sigma = 10.0
                adj[i, j] = np.exp(-(d ** 2) / (2 * sigma ** 2))
    return adj
