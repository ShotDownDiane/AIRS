"""REMX-TF: Configuration."""
from dataclasses import dataclass, asdict
import json, os

@dataclass
class REMXConfig:
    dataset: str = "PEMS04"
    data_dir: str = "data/"
    num_sensors: int = 307
    in_steps: int = 12
    out_steps: int = 12
    in_channels: int = 1
    weather_channels: int = 7
    tod_steps: int = 288
    dow_steps: int = 7
    # Model
    d_model: int = 128
    n_heads: int = 8
    d_ff: int = 512
    enc_layers: int = 4
    dropout: float = 0.1
    memory_size: int = 128  # Fixed learnable memory slots
    # Training
    batch_size: int = 32
    epochs: int = 20
    lr: float = 1e-3
    lr_min: float = 1e-5
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    early_stop_patience: int = 5
    # Data augmentation
    aug_temporal_jitter_prob: float = 0.2
    aug_sensor_dropout_prob: float = 0.1
    aug_sensor_dropout_frac: float = 0.1
    aug_weather_noise_std: float = 0.05
    # Runtime
    seed: int = 42
    num_workers: int = 4
    device: str = "auto"
    output_dir: str = "outputs/"
    log_interval: int = 50
    save_best: bool = True

    def to_dict(self): return asdict(self)
    def save(self, path):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f: json.dump(self.to_dict(), f, indent=2)
    @classmethod
    def load(cls, path):
        with open(path) as f: return cls(**json.load(f))
    def get_device(self):
        import torch
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)
