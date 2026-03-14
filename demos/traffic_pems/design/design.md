# REMX-TF: Technical Design Document

## Retrieval-Enhanced Multi-Modal Transformer for Traffic Forecasting

---

## 1. Problem Formulation and Notation

### 1.1 Core Problem

Given a road sensor network modeled as a graph G = (V, E) with N = |V| sensors, historical traffic observations, and aligned external context (weather, incidents), predict future traffic flow values for all sensors over a forecast horizon.

### 1.2 Notation

| Symbol | Description | Shape / Domain |
|--------|-------------|----------------|
| N | Number of sensors | 307 (PEMS04), 170 (PEMS08) |
| T | Historical window length (time steps) | 12 (= 60 min at 5-min intervals) |
| T' | Forecast horizon (time steps) | 12 (= 60 min) |
| C_in | Number of input traffic features | 1 (flow) |
| X ∈ ℝ^{N×T×C_in} | Historical traffic tensor | |
| Y ∈ ℝ^{N×T'×C_in} | Ground-truth future traffic tensor | |
| Ŷ ∈ ℝ^{N×T'×C_in} | Predicted future traffic tensor | |
| W ∈ ℝ^{N×T×C_w} | Weather feature matrix | C_w = 7 |
| I ∈ {0,1}^{N×T} | Incident indicator matrix | Binary |
| A ∈ ℝ^{N×N} | Adjacency matrix (road connectivity) | Sparse, pre-defined |
| d | Model hidden dimension | 128 |
| k | Number of retrieved neighbors | 8 (default) |
| M | External memory bank | ~2M (key, value) pairs |

### 1.3 Weather Features (C_w = 7)

| Index | Feature | Encoding |
|-------|---------|----------|
| 0 | Temperature (°C) | Continuous, z-normalized |
| 1 | Precipitation rate (mm/hr) | Continuous, log1p-transformed |
| 2 | Visibility (km) | Continuous, z-normalized |
| 3 | Wind speed (m/s) | Continuous, z-normalized |
| 4 | Humidity (%) | Continuous, z-normalized |
| 5 | Weather category (clear/rain/fog/snow) | One-hot → 4 bits, but collapsed to 1 dim via embedding |
| 6 | Is_night flag | Binary |

### 1.4 Formal Objective

Learn a function f_θ such that:

```
Ŷ = f_θ(X, W, I, A, M)
```

Minimize:

```
L_total = L_MAE(Y, Ŷ) + λ · L_div(R)
```

where:
- L_MAE = (1 / (N · T')) Σ_{n,t} |Y_{n,t} - Ŷ_{n,t}|
- L_div = retrieval diversity regularizer (Section 4.4)
- λ = 0.1

---

## 2. Proposed Method: REMX-TF

### 2.1 High-Level Architecture

```
Input (X, W, I)
       │
       ▼
┌─────────────────┐
│  Input Embedding │  ← Spatio-Temporal Adaptive Embedding (from STAEformer)
│  + Weather/Inc   │  ← Multi-modal feature projection
│    Projection    │
└────────┬────────┘
         │  H_0 ∈ ℝ^{N×T×d}
         ▼
┌─────────────────┐
│  Transformer     │  ← L_enc = 4 layers, 8-head self-attention
│  Encoder         │
└────────┬────────┘
         │  H_enc ∈ ℝ^{N×T×d}
         ▼
┌─────────────────┐     ┌──────────────┐
│  Retrieval-      │◄────│  FAISS Memory │
│  Augmented       │     │  Bank M       │
│  Fusion Layer    │     └──────────────┘
└────────┬────────┘
         │  H_fused ∈ ℝ^{N×T×d}
         ▼
┌─────────────────┐
│  Transformer     │  ← L_dec = 4 layers, 8-head cross-attention
│  Decoder         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Output          │  → Ŷ ∈ ℝ^{N×T'×1}
│  Projection      │
└─────────────────┘
```

### 2.2 Module 1: Input Embedding Layer

Following STAEformer, we construct spatio-temporal adaptive embeddings and extend them with multi-modal context.

**Spatial Embedding:**
```
E_s ∈ ℝ^{N×d}   (learnable, one vector per sensor)
```

**Temporal Embedding:**
```
E_tod ∈ ℝ^{288×d}   (time-of-day: 288 = 24h × 60min / 5min)
E_dow ∈ ℝ^{7×d}     (day-of-week)
```

**Spatio-Temporal Adaptive Embedding (STAE):**
```
STAE(n, t) = E_s[n] + E_tod[tod(t)] + E_dow[dow(t)]
```

This is a core contribution of STAEformer: the adaptive embedding captures per-sensor, per-time-slot patterns.

**Traffic Input Projection:**
```
H_traffic = Linear_{C_in → d}(X) + STAE     ∈ ℝ^{N×T×d}
```

**Weather Projection:**
```
H_weather = MLP_{C_w → d}(W)                 ∈ ℝ^{N×T×d}
```

where MLP is: Linear(C_w, d) → GELU → Linear(d, d)

**Incident Projection:**
```
H_incident = Linear_{1 → d}(I.unsqueeze(-1)) ∈ ℝ^{N×T×d}
```

**Combined Input:**
```
H_0 = H_traffic + α_w · H_weather + α_i · H_incident
```

where α_w, α_i ∈ ℝ are learnable scalar gates initialized to 0.1 (allows the model to learn how much weight to give external features).

### 2.3 Module 2: Transformer Encoder

Standard pre-norm Transformer encoder with L_enc = 4 layers.

Each layer l:
```
H'_l = H_{l-1} + MSA(LayerNorm(H_{l-1}))
H_l  = H'_l   + FFN(LayerNorm(H'_l))
```

**Multi-Head Self-Attention (MSA):**
- Heads: 8
- Head dim: d_k = d / 8 = 16
- Attention applied along the temporal dimension independently per sensor (temporal attention), then along the spatial dimension independently per time step (spatial attention), alternating per layer.

Specifically:
- Layers 0, 2: **Temporal self-attention** — for each sensor n, attend across T time steps. Input reshaped to (N, T, d), attention over dim 1.
- Layers 1, 3: **Spatial self-attention** — for each time step t, attend across N sensors. Input reshaped to (T, N, d), attention over dim 1.

**Feed-Forward Network (FFN):**
```
FFN(x) = Linear(d, 4d) → GELU → Dropout(0.1) → Linear(4d, d) → Dropout(0.1)
```

Output: H_enc ∈ ℝ^{N×T×d}

### 2.4 Module 3: External Memory Bank & Retriever

#### 2.4.1 Memory Construction (Offline)

For each historical sample (n, t) in the training set, we construct:

**Key vector** k_{n,t} ∈ ℝ^{d_key} where d_key = 64:
```
k_{n,t} = MLP_key( concat[
    E_s[n],                          # 16-d (projected from d)
    sin_cos_tod(t),                   # 16-d (8 sin/cos pairs)
    sin_cos_doy(t),                   # 8-d  (4 sin/cos pairs)
    weather_embed(W[n,t]),            # 16-d
    incident_flag(I[n,t])             # 8-d  (projected from 1-d)
] )
```

Total raw key input: 64-d → MLP_key: Linear(64, 64) → ReLU → Linear(64, 64)

**Value vector** v_{n,t} ∈ ℝ^{d}:
```
v_{n,t} = MLP_value(X[n, t-11:t+1, :])   # Past 12 steps of traffic for sensor n
```

MLP_value: Linear(12, 64) → GELU → Linear(64, 128) → GELU → Linear(128, d)

#### 2.4.2 FAISS Index

- Index type: `IndexIVFFlat` with nlist=1024, nprobe=32
- Distance metric: L2
- Total entries: ~2M for PEMS04 (307 sensors × ~6700 training time steps)
- Storage: ~2M × (64 + 128) × 4 bytes ≈ 1.5 GB
- Built once before training; rebuilt if weather/incident data changes

#### 2.4.3 Online Retrieval

At inference (and training) time, for each sample (n, t) in the current batch:

1. Compute query key q_{n,t} using the same MLP_key with current context.
2. Retrieve top-k=8 nearest neighbors from FAISS: {(k_i, v_i)}_{i=1}^{k}
3. Stack retrieved values: R_{n,t} ∈ ℝ^{k×d}

**Batch retrieval:** For a batch of size B with N sensors and T time steps, we issue B×N×T queries. To amortize cost, we:
- Pre-compute all queries for the batch
- Use FAISS batch search: `index.search(queries, k)` — this is highly optimized
- Cache retrieval results in the DataLoader (pre-fetch next batch's retrievals in a background thread)

### 2.5 Module 4: Retrieval-Augmented Fusion Layer

This is the core novel component. It fuses the encoder output with retrieved historical contexts.

**Cross-Attention Fusion:**

For each sensor n and time step t:
```
Q = Linear_Q(H_enc[n, t, :])          ∈ ℝ^{d}
K = Linear_K(R_{n,t})                  ∈ ℝ^{k×d}
V = Linear_V(R_{n,t})                  ∈ ℝ^{k×d}

Attn = softmax(Q · K^T / √d_k)        ∈ ℝ^{k}
H_retrieved = Attn · V                 ∈ ℝ^{d}
```

**Gated Fusion:**
```
g = σ(W_g · H_enc[n,t,:] + b_g)       ∈ ℝ^{d}   (element-wise sigmoid gate)
H_fused[n,t,:] = g ⊙ H_enc[n,t,:] + (1 - g) ⊙ H_retrieved
```

W_g ∈ ℝ^{d×d}, b_g ∈ ℝ^{d}, initialized so that g ≈ 0.9 at start (bias init = 2.0), meaning the model initially relies mostly on the encoder and gradually learns to incorporate retrieval.

**Efficient Implementation:**
- Reshape H_enc to (B·N·T, d), R to (B·N·T, k, d)
- Single batched cross-attention operation
- Total FLOPs: O(B·N·T·k·d) — negligible compared to self-attention O(B·N·T²·d)

### 2.6 Module 5: Transformer Decoder

The decoder generates the T'=12 future time steps.

**Decoder Input:**
```
H_dec_input = Linear(d, d)(positional_encoding(1..T'))   ∈ ℝ^{T'×d}
```
Broadcast across N sensors. Add spatial embedding E_s.

**Decoder Layers (L_dec = 4):**

Each layer:
```
H'_l = H_{l-1} + MaskedMSA(LayerNorm(H_{l-1}))          # Causal self-attention over T'
H''_l = H'_l  + CrossAttn(LayerNorm(H'_l), H_fused)      # Cross-attend to fused encoder
H_l   = H''_l + FFN(LayerNorm(H''_l))
```

Cross-attention: queries from decoder, keys/values from H_fused.

Alternating spatial/temporal structure same as encoder:
- Layers 0, 2: Temporal attention (causal mask for self-attn, full for cross-attn)
- Layers 1, 3: Spatial attention

### 2.7 Module 6: Output Projection

```
Ŷ = Linear(d, C_in)(H_dec_final)     ∈ ℝ^{N×T'×1}
```

---

## 3. Training Procedure

### 3.1 Curriculum Training Schedule

| Phase | Epochs | Description | Learning Rate |
|-------|--------|-------------|---------------|
| Phase 1: Warm-up | 1–5 | Train encoder + decoder WITHOUT retrieval fusion (gate g forced to 1.0). This establishes a strong STAEformer-like baseline. | 1e-3 with linear warm-up for 1 epoch |
| Phase 2: Retrieval | 6–20 | Enable retrieval fusion layer. Unfreeze gate bias. Train all parameters jointly. | 5e-4, cosine decay to 1e-5 |

### 3.2 Loss Function

**Primary Loss — Masked MAE:**
```
L_MAE = (1 / |Ω|) Σ_{(n,t) ∈ Ω} |Y_{n,t} - Ŷ_{n,t}|
```
where Ω = set of valid (non-missing) entries.

**Retrieval Diversity Loss:**

Encourages the model to attend to diverse retrieved neighbors rather than collapsing to a single one.

```
L_div = - (1 / (N·T)) Σ_{n,t} H(Attn_{n,t})
```

where H(p) = -Σ_i p_i log(p_i + ε) is the entropy of the attention distribution over k retrieved items. Maximizing entropy (minimizing negative entropy) encourages uniform attention, preventing collapse.

**Total Loss:**
```
L_total = L_MAE + λ · L_div
```

λ = 0.1 (Phase 2 only; λ = 0 in Phase 1)

### 3.3 Optimizer and Hyperparameters

| Hyperparameter | Value | Notes |
|----------------|-------|-------|
| Optimizer | AdamW | β1=0.9, β2=0.999 |
| Weight decay | 1e-4 | |
| Batch size | 32 | Per GPU |
| Gradient clipping | max_norm = 5.0 | |
| d_model | 128 | |
| n_heads | 8 | |
| d_ff | 512 (= 4 × d) | |
| Encoder layers | 4 | |
| Decoder layers | 4 | |
| Dropout | 0.1 | Attention + FFN |
| k (retrieval) | 8 | Ablate: 1, 4, 16 |
| λ (diversity) | 0.1 | |
| α_w init | 0.1 | Learnable weather gate |
| α_i init | 0.1 | Learnable incident gate |
| Gate bias init | 2.0 | σ(2.0) ≈ 0.88 |
| LR Phase 1 | 1e-3 | Linear warm-up 1 epoch |
| LR Phase 2 | 5e-4 | Cosine decay → 1e-5 |
| Total epochs | 20 | 5 + 15 |
| Early stopping | patience=5 | On validation MAE |

### 3.4 Data Augmentation

- **Temporal jitter:** With probability 0.2, shift the input window by ±1 time step.
- **Sensor dropout:** With probability 0.1, zero out a random 10% of sensors in the input (improves robustness).
- **Weather noise:** Add Gaussian noise N(0, 0.05) to normalized weather features during training.

---

## 4. Evaluation Metrics and Baselines

### 4.1 Metrics

| Metric | Formula | Notes |
|--------|---------|-------|
| MAE | (1/n) Σ \|y - ŷ\| | Primary metric |
| RMSE | √((1/n) Σ (y - ŷ)²) | Penalizes large errors |
| MAPE | (100/n) Σ \|y - ŷ\|/\|y\| | Percentage; exclude y < 1 to avoid division issues |

Report at horizons: 3-step (15 min), 6-step (30 min), 12-step (60 min), and average.

### 4.2 Baselines

| Baseline | Category | Implementation Source |
|----------|----------|----------------------|
| **STAEformer** | SOTA Transformer (no externals) | Official GitHub |
| **PDFormer** | Domain-aware Transformer (no externals) | Official GitHub |
| **Graph WaveNet** | Adaptive graph + dilated conv | LibCity / BasicTS |
| **AGCRN** | Adaptive graph + GRU | Official GitHub |
| **DCRNN** | Diffusion conv + GRU | LibCity |
| **SimpleFusion** | STAEformer + concat weather/incident features | Our implementation |
| **REMX-TF (no retrieval)** | Our model without memory bank | Ablation |
| **REMX-TF (no weather)** | Our model without weather features | Ablation |
| **REMX-TF (no incidents)** | Our model without incident features | Ablation |

### 4.3 Evaluation Slices

Beyond overall metrics, we report performance on specific data slices:

| Slice | Definition | Expected % of test set |
|-------|------------|----------------------|
| **Overall** | All test samples | 100% |
| **Adverse weather** | Precipitation > 2 mm/hr OR visibility < 1 km | ~5–10% |
| **Active incidents** | At least one sensor has incident flag = 1 in the window | ~8–15% |
| **Peak hours** | 7–9 AM and 4–7 PM weekdays | ~25% |
| **Off-peak** | All other times | ~75% |
| **Weekday** | Monday–Friday | ~71% |
| **Weekend** | Saturday–Sunday | ~29% |

### 4.4 Statistical Significance

- Run each experiment with 3 random seeds (42, 123, 456).
- Report mean ± std for all metrics.
- Paired t-test (p < 0.05) between REMX-TF and STAEformer on per-sample MAE.

---

## 5. Dataset Requirements

### 5.1 Traffic Data

| Dataset | Sensors | Time Range | Interval | Features | Split |
|---------|---------|------------|----------|----------|-------|
| PEMS04 | 307 | Jan–Feb 2018 (59 days) | 5 min | Flow | 6:2:2 (train:val:test) |
| PEMS08 | 170 | Jul–Aug 2016 (62 days) | 5 min | Flow | 6:2:2 |

**Preprocessing (following STAEformer exactly):**
1. Load raw .npz files (shape: [num_timesteps, num_sensors, num_features])
2. Z-score normalization per sensor: x' = (x - μ) / σ, computed on training set only
3. Sliding window: input 12 steps → predict next 12 steps, stride 1
4. Adjacency matrix: pre-computed from road network distances, thresholded Gaussian kernel

### 5.2 Weather Data

**Source:** NOAA Integrated Surface Database (ISD) — freely available via `https://www.ncei.noaa.gov/data/global-hourly/`

**Stations:**
- PEMS04: San Francisco Bay Area → stations: SFO (WBAN 23234), OAK (WBAN 23230), SJC (WBAN 23293)
- PEMS08: San Bernardino area → stations: ONT (WBAN 23174), RIV (WBAN 03171)

**Mapping:** Each sensor is assigned to its nearest weather station via haversine distance. Weather is recorded hourly; we:
1. Forward-fill to 5-min resolution
2. Linear interpolate temperature, wind speed, visibility
3. Step-function for precipitation, weather category

**Features extracted per timestep:**
- Temperature (°C), precipitation rate (mm/hr), visibility (km), wind speed (m/s), relative humidity (%), weather category (clear/rain/fog/snow), is_night

### 5.3 Incident Data

**Source:** California Highway Patrol (CHP) Computer-Aided Dispatch (CAD) logs — available via Caltrans PeMS website historical incident data download.

**Processing:**
1. Download incident logs for matching date ranges
2. Spatial join: assign each incident to nearest sensor within 1 km radius (using postmile or lat/lon)
3. Temporal encoding: for each (sensor, timestep), set I=1 if an active incident exists within the time window
4. Active duration: from incident start time to clearance time

**Fallback:** If CHP data is unavailable for the exact period, we synthesize incident indicators by detecting anomalous drops in flow (> 2σ below expected for that time-of-day) and label those as pseudo-incidents. This is clearly documented as a limitation.

### 5.4 Data Alignment

All three data sources are aligned to a common timeline:
```
Unified DataFrame:
  index: (sensor_id, timestamp)  — 5-min resolution
  columns: flow, temperature, precip_rate, visibility, wind_speed,
           humidity, weather_cat, is_night, incident_flag
```

Missing weather values: forward-fill up to 30 min, then fill with daily mean.
Missing incident values: assume no incident (I=0).

---

## 6. Compute Requirements

### 6.1 Hardware

| Resource | Specification |
|----------|---------------|
| GPU | 1× NVIDIA A100 80GB (or 1× A6000 48GB) |
| CPU | 8+ cores for data loading and FAISS |
| RAM | 64 GB (FAISS index + data) |
| Storage | 50 GB (data + checkpoints + logs) |

### 6.2 Time Estimates

| Task | Estimated Time |
|------|---------------|
| Data download & preprocessing | 4–6 hours |
| Weather/incident alignment | 2–3 hours |
| FAISS index construction | 30 min |
| Phase 1 training (5 epochs, PEMS04) | 3 hours |
| Phase 2 training (15 epochs, PEMS04) | 8 hours |
| Phase 1+2 training (PEMS08) | 6 hours (smaller dataset) |
| Baseline reproduction (5 models × ~8h) | 40 hours |
| Ablation studies (4 variants × 2 datasets) | 48 hours |
| Evaluation & analysis | 4 hours |
| **Total GPU hours** | **~110 hours** |

### 6.3 Memory Budget (PEMS04, batch=32)

| Component | Memory |
|-----------|--------|
| Model parameters (~3.2M params × 4 bytes) | ~13 MB |
| Activations (forward pass) | ~2.5 GB |
| Gradients + optimizer states | ~1.5 GB |
| FAISS index (in CPU RAM) | ~1.5 GB |
| Retrieved embeddings per batch (32×307×12×8×128×4) | ~1.2 GB |
| Data batch | ~0.5 GB |
| **Total GPU memory** | **~6 GB** |
| **Total CPU RAM** | **~8 GB** (FAISS + data) |

This fits comfortably on any modern GPU. For faster iteration, batch size can be increased to 64.

### 6.4 Parameter Count Estimate

| Module | Parameters |
|--------|-----------|
| Spatial embedding (307 × 128) | 39K |
| Temporal embeddings (288 × 128 + 7 × 128) | 38K |
| Input projections (traffic + weather + incident) | 100K |
| Encoder (4 layers × [MSA + FFN]) | 1.1M |
| MLP_key + MLP_value | 50K |
| Fusion cross-attention + gate | 100K |
| Decoder (4 layers × [MSA + CrossAttn + FFN]) | 1.6M |
| Output projection | 16K |
| **Total** | **~3.0M** |

For comparison, STAEformer has ~2.5M parameters. Our overhead is ~20% from the retrieval fusion components.

---

## 7. Ablation Study Design

### 7.1 Component Ablations

| ID | Variant | What Changes |
|----|---------|-------------|
| A1 | No retrieval | Remove fusion layer entirely; equivalent to STAEformer + weather/incident |
| A2 | No weather | Set α_w = 0; remove weather from memory keys |
| A3 | No incidents | Set α_i = 0; remove incidents from memory keys |
| A4 | No externals | A2 + A3; pure traffic retrieval |
| A5 | SimpleFusion | Concatenate raw weather + incident to traffic input; no retrieval |

### 7.2 Retrieval Hyperparameter Ablations

| ID | k | Expected Effect |
|----|---|----------------|
| R1 | 1 | Minimal retrieval; may underfit |
| R2 | 4 | Moderate retrieval |
| R3 | 8 | Default |
| R4 | 16 | More context; risk of noise |
| R5 | 32 | Heavy retrieval; latency concern |

### 7.3 Diversity Loss Ablation

| ID | λ | Expected Effect |
|----|---|----------------|
| D1 | 0.0 | No diversity encouragement |
| D2 | 0.05 | Light regularization |
| D3 | 0.1 | Default |
| D4 | 0.5 | Strong regularization |

### 7.4 Curriculum Ablation

| ID | Strategy | Description |
|----|----------|-------------|
| C1 | No curriculum | Train with retrieval from epoch 1 |
| C2 | Default | Phase 1 (5 ep) → Phase 2 (15 ep) |
| C3 | Extended warm-up | Phase 1 (10 ep) → Phase 2 (10 ep) |

---

## 8. Expected Results

### 8.1 PEMS04 (Overall, 12-step Average)

| Method | MAE | RMSE | MAPE |
|--------|-----|------|------|
| DCRNN | ~21.2 | ~33.4 | ~14.2% |
| Graph WaveNet | ~19.8 | ~31.0 | ~13.0% |
| AGCRN | ~19.8 | ~32.3 | ~12.9% |
| PDFormer | ~19.3 | ~31.1 | ~12.6% |
| STAEformer | ~19.1 | ~30.8 | ~12.5% |
| SimpleFusion | ~18.9 | ~30.5 | ~12.3% |
| **REMX-TF** | **~18.5** | **~29.8** | **~12.0%** |

### 8.2 Adverse Weather Slice (PEMS04)

| Method | MAE | Δ vs Overall |
|--------|-----|-------------|
| STAEformer | ~22.5 | +3.4 |
| SimpleFusion | ~21.8 | +2.9 |
| **REMX-TF** | **~20.5** | **+2.0** |

Key claim: REMX-TF reduces the performance gap between normal and adverse conditions.

---

## 9. Risk Analysis and Mitigations

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| Weather data unavailable for exact period | Medium | Low | Use nearest available period; document temporal gap |
| Incident data too sparse | High | Medium | Use pseudo-incident detection from flow anomalies |
| FAISS retrieval too slow during training | Medium | Low | Pre-compute retrievals per epoch; cache in DataLoader |
| Retrieval doesn't help overall | High | Low | SimpleFusion ablation still contributes; focus on adverse slices |
| STAEformer reproduction doesn't match paper | Medium | Medium | Use official code; report our reproduction numbers |
| Overfitting on small adverse-weather test set | Medium | Medium | Report confidence intervals; use bootstrap sampling |

---

## 10. Timeline

| Week | Tasks |
|------|-------|
| 1 | Data collection: download PeMS04/08, weather, incidents. Preprocessing scripts. |
| 2 | Implement STAEformer backbone. Reproduce baseline numbers. |
| 3 | Implement memory bank construction, FAISS index, retriever module. |
| 4 | Implement fusion layer, curriculum training. End-to-end training on PEMS04. |
| 5 | Run all baselines. Ablation studies. |
| 6 | PEMS08 experiments. Adverse weather / incident analysis. |
| 7 | Statistical analysis. Visualization. Paper writing. |
| 8 | Paper revision. Code cleanup. Release. |
