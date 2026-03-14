"""REMX-TF: Retrieval-Enhanced Memory Transformer for Traffic Forecasting.

Complexity analysis:
- Encoder self-attention: O(B * N * T^2 * d) — T=12, manageable
- Memory cross-attention: O(B * N * T * M * d) — M=64~256, fixed and small
- Decoder: O(B * N * T * d)
- Total per batch: O(B * N * T * (T + M) * d), scales linearly with B and N
"""
import math, torch, torch.nn as nn, torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Unified multi-head attention for both self and cross attention."""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k=None, v=None):
        if k is None: k = q
        if v is None: v = k
        B, Lq, D = q.shape; Lk = k.shape[1]
        H = self.n_heads; dk = self.d_k
        Q = self.W_q(q).view(B, Lq, H, dk).transpose(1, 2)
        K = self.W_k(k).view(B, Lk, H, dk).transpose(1, 2)
        V = self.W_v(v).view(B, Lk, H, dk).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(dk)
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, Lq, D)
        return self.W_o(out)


class FFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )
    def forward(self, x): return self.net(x)


class InputEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        N = config.num_sensors; d = config.d_model
        self.E_s = nn.Embedding(N, d)
        self.E_tod = nn.Embedding(config.tod_steps, d)
        self.E_dow = nn.Embedding(config.dow_steps, d)
        self.traffic_proj = nn.Linear(config.in_channels, d)
        self.weather_proj = nn.Sequential(nn.Linear(config.weather_channels, d), nn.GELU(), nn.Linear(d, d))
        self.incident_proj = nn.Linear(1, d)
        self.alpha_w = nn.Parameter(torch.tensor(0.1))
        self.alpha_i = nn.Parameter(torch.tensor(0.1))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x_traffic, x_weather, x_incident, tod_idx, dow_idx):
        B, N, T, _ = x_traffic.shape; device = x_traffic.device
        sensor_ids = torch.arange(N, device=device)
        E_s = self.E_s(sensor_ids).unsqueeze(0).unsqueeze(2)       # (1, N, 1, d)
        E_tod = self.E_tod(tod_idx).unsqueeze(1)                    # (B, 1, T, d)
        E_dow = self.E_dow(dow_idx).unsqueeze(1)                    # (B, 1, T, d)
        H = self.traffic_proj(x_traffic) + E_s + E_tod + E_dow
        H = H + self.alpha_w * self.weather_proj(x_weather)
        H = H + self.alpha_i * self.incident_proj(x_incident.unsqueeze(-1))
        return self.dropout(H)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FFN(d_model, d_ff, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.enc_layers)
        ])
    def forward(self, x):
        """x: (B, N, T, d) -> (B, N, T, d). Attention over T per sensor."""
        B, N, T, d = x.shape
        x = x.reshape(B * N, T, d)  # Flatten batch and sensors
        for layer in self.layers:
            x = layer(x)
        return x.reshape(B, N, T, d)


class LearnableMemory(nn.Module):
    """Fixed-size learnable memory bank with cross-attention retrieval.

    Complexity: O(B * N * T * memory_size) per forward pass.
    memory_size is a fixed hyperparameter (default 128), independent of dataset size.
    """
    def __init__(self, config):
        super().__init__()
        self.memory_size = config.memory_size
        d = config.d_model
        # Learnable memory slots
        self.memory = nn.Parameter(torch.randn(config.memory_size, d) * 0.02)
        # Cross-attention: encoder output queries memory
        self.cross_attn = MultiHeadAttention(d, config.n_heads, config.dropout)
        self.norm = nn.LayerNorm(d)
        # Gating: blend memory-augmented representation with encoder output
        self.gate = nn.Sequential(nn.Linear(d * 2, d), nn.Sigmoid())

    def forward(self, H_enc):
        """
        H_enc: (B, N, T, d) encoder output
        Returns: (B, N, T, d) memory-augmented representation
        """
        B, N, T, d = H_enc.shape
        # Reshape to (B*N, T, d) for attention
        H_flat = H_enc.reshape(B * N, T, d)
        # Expand memory for batch: (B*N, memory_size, d)
        mem = self.memory.unsqueeze(0).expand(B * N, -1, -1)
        # Cross-attention: query=encoder, key/value=memory
        H_mem = self.cross_attn(self.norm(H_flat), mem, mem)
        # Gated fusion
        gate = self.gate(torch.cat([H_flat, H_mem], dim=-1))
        H_out = gate * H_mem + (1 - gate) * H_flat
        return H_out.reshape(B, N, T, d)


class Decoder(nn.Module):
    """Simple MLP decoder: projects from d_model to out_steps predictions."""
    def __init__(self, config):
        super().__init__()
        d = config.d_model
        self.norm = nn.LayerNorm(d)
        # Per-sensor temporal projection: (T_in, d) -> (T_out, 1)
        self.proj = nn.Sequential(
            nn.Linear(config.in_steps * d, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.out_steps * config.in_channels),
        )

    def forward(self, H):
        """H: (B, N, T, d) -> pred: (B, N, T_out, C)"""
        B, N, T, d = H.shape
        H = self.norm(H)
        H_flat = H.reshape(B, N, T * d)
        pred = self.proj(H_flat)  # (B, N, T_out * C)
        return pred.reshape(B, N, -1, 1)


class REMXModel(nn.Module):
    """REMX-TF: Retrieval-Enhanced Memory Transformer.

    Architecture: InputEmbedding -> Encoder -> LearnableMemory -> Decoder
    All components have bounded complexity independent of dataset size.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = InputEmbedding(config)
        self.encoder = TransformerEncoder(config)
        self.memory = LearnableMemory(config)
        self.decoder = Decoder(config)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, x_traffic, x_weather, x_incident, tod_idx, dow_idx):
        H = self.embedding(x_traffic, x_weather, x_incident, tod_idx, dow_idx)
        H_enc = self.encoder(H)
        H_mem = self.memory(H_enc)
        pred = self.decoder(H_mem)
        return pred


def masked_mae_loss(pred, target, null_val=0.0):
    mask = (target != null_val).float()
    loss = torch.abs(pred - target) * mask
    return loss.sum() / (mask.sum() + 1e-8)
