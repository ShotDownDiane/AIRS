# Research Ideas for Traffic Flow Prediction on PeMS

Below are six research directions that address explicit gaps identified in the literature survey.  Each idea is evaluated on novelty, feasibility (1‒2 month scope), impact, and technical clarity.

---

## 1. HATS: Hierarchical Adaptive Time-Scale State-space Model for 39K-Sensor PeMS

### Motivation & Gap
Gap addressed: *Scalability to large networks* (Gap 3).  Existing SOTA models are evaluated on ≤900 sensors, whereas the real PeMS network has >39 000 sensors.

### Proposed Approach
1. **Graph Partitioning**:  Apply METIS/Louvain to partition the full PeMS graph into ~100 contiguous sub-graphs (≈400 sensors each).
2. **Local Module (S-Mamba)**:  For each sub-graph train a lightweight selective-state-space model (Mamba) that captures fine-grained ST dynamics.
3. **Inter-Region Router (MoE)**:  A Mixture-of-Experts router learns to fuse the hidden states of neighboring partitions via a sparsely-gated MoE layer; gating weights derived from coarse adjacency and recent flow residuals.
4. **Multi-Time-Scale Heads**:  Parallel heads predict 5-min, 15-min and 60-min horizons; share parameters through FiLM layers that condition on the target horizon.
5. **Training Regime**:  Curriculum schedule – start with partition-only training, then joint fine-tuning with inter-region MoE.  Distributed training with gradient checkpointing to fit in a single 4×A100 server.

### Expected Results & Baselines
Datasets: full PeMS raw feed (39K sensors) and standard PEMS04 subset.  Baselines: Graph WaveNet, AGCRN, STAEformer on PEMS04; Graph-Partitioning-DCRNN (Mallick 19) on large-scale setting.  Expect ≈15-20 % speed-up in inference and ≤2 × memory vs GraphWaveNet while matching or beating MAE.

### Risks & Mitigations
•  Partition borders may cut important flows → use overlap padding & router.
•  MoE routing instability → use load-balancing loss.

### Scores
Novelty 8 / Feasibility 6 / Impact 9 / Technical 8

---

## 2. BENCH-PEMS: Reproducible Benchmarking Suite & Leaderboard

### Motivation & Gap
Gap addressed: *Standardization and fair comparison* (Gap 2).  Lack of shared preprocessing/splits obscures true progress.

### Proposed Approach
1. Curate fixed raw → processed pipelines using open-source *LibCity* + new scripts that replicate original sensor lists and missing-value filters.
2. Provide deterministic data splits for all six PeMS benchmarks and the full 39 K graph.
3. Containerised baselines: DCRNN, GraphWaveNet, AGCRN, GMAN, STAEformer, MLPST implemented under a unified API.
4. Auto-evaluation harness outputs MAE/RMSE/MAPE and logs carbon footprint.
5. Public leaderboard + CI that re-runs submissions on GCP Pre-emptible GPUs.

### Expected Results & Baselines
Deliver a paper + GitHub repo; demonstrate that many reported SOTA gaps shrink under identical splits (e.g.
STAEformer advantage drops from 4 % to 1 %).

### Risks & Mitigations
•  Re-implementations might underperform originals → consult authors / borrow checkpoints.
•  Engineering heavy – mitigate by forking LibCity & BasicTS.

### Scores
Novelty 5 / Feasibility 9 / Impact 8 / Technical 7

---

## 3. RUST: Robust Uncertainty-aware Spatio-Temporal Transformer

### Motivation & Gap
Gaps addressed: *Robustness & Reliability* (Gap 5) and *Interpretability* (Gap 9).

### Proposed Approach
1. Base architecture: STAEformer.
2. **Deep Ensemble + MC Dropout** to output predictive mean & variance.
3. **Adversarial Training**:  Use projected gradient noise on node values and edges (diffusion conv weights) following Zhu 21.
4. **Conformal Calibration**:  Calibrate predictive intervals offline to guarantee 90 % empirical coverage.
5. **Saliency Maps**:  Integrated-gradients over spatio-temporal embeddings highlight influential sensors/time-steps.

### Expected Results & Baselines
Benchmarks: PEMS04 & 08 with 5 % synthetic sensor outages.  Expect similar MAE but ~30 % fewer large errors; 90 % coverage within ±2 σ.

### Risks & Mitigations
•  Ensembles multiply compute cost → share encoder weights, only diversify last two layers.

### Scores
Novelty 7 / Feasibility 8 / Impact 7 / Technical 8

---

## 4. REMX-TF: Retrieval-Enhanced Multi-Modal Transformer for Traffic Forecasting

### Motivation & Gap
Gap addressed: *Multi-Modal and External Factors* (Gap 6).  Weather, incidents, holidays strongly influence traffic yet are rarely used.

### Proposed Approach
1. **External Memory**:  Build a key-value store where keys = (sensorID, day-of-year, time-of-day, weather-code) hashed via LSH; values = recent traffic embeddings.
2. **Retriever**:  During training, for each query time-step retrieve k=8 similar historical contexts (same spatial vicinity & weather similarity).
3. **Fusion**:  Concatenate retrieved embeddings to the input sequence; a gated cross-attention layer lets the model decide how much to use.
4. **Modalities**:  Weather (temperature, precipitation), calendar (holiday flag), incident reports (binary lane-closure).  Encoded via learned embeddings and added to positional encodings.
5. **Architecture**:  STAEformer backbone with retrieval cross-attention inserted before each encoder block (inspired by REALM).

### Expected Results & Baselines
Datasets: PEMS04 merged with NOAA weather API + CHP incident logs (both publicly available).  Baselines: STAEformer (no externals), PDFormer (prop delay).  Expect 5-7 % MAE reduction during adverse weather & 2-3 % overall.

### Risks & Mitigations
•  Data alignment errors → use spatio-temporal nearest-neighbor join with quality flags.
•  Retrieval latency at inference → pre-compute keys and batch fetch.

### Scores
Novelty 8 / Feasibility 7 / Impact 8 / Technical 8

---

## 5. C-GATE: Counterfactual Graph Attention for Traffic Explanation

### Motivation & Gap
Gap addressed: *Interpretability* (Gap 9).  Practitioners need causal explanations ("what if this ramp closes?").

### Proposed Approach
1. Train any STGNN (e.g., GraphWaveNet) jointly with a **counterfactual generator** that learns minimal perturbations to input graph signals producing significant output change.
2. Use edge-mask optimisation with sparsity & plausibility constraints to create human-readable scenarios (closing a subset of roads).
3. Score each sensor by average causal effect (ACE) across sampled counterfactuals; visualise on map.
4. Validate using real incident events: model explanations should highlight affected road segments.

### Expected Results & Baselines
Quantitative: fidelity (how well counterfactual reproduces Δprediction), sparsity, human study with traffic engineers.

### Risks & Mitigations
•  Optimisation can be unstable → use Gumbel-softmax edge masks + curriculum.

### Scores
Novelty 7 / Feasibility 6 / Impact 7 / Technical 7

---

## 6. STAugment: Graph-based Data Augmentation for Generalisation & Transfer

### Motivation & Gap
Gap addressed: *Transferability & Generalisation* (Gap 8).  Current models over-fit to specific networks.

### Proposed Approach
1. **Spatial Mixup**:  Randomly swap features of topologically similar sensors across cities during training.
2. **Temporal CutMix**:  Replace random 30-min segments with same-period data from other days/cities.
3. **Graph Dropout**:  Stochastically remove edges to simulate topology change.
4. Apply augmentations while training a lightweight ST-Mamba; evaluate zero-shot on unseen city (transfer from PEMS04 → METR-LA).

### Expected Results & Baselines
Expect 5-10 % MAE improvement in transfer setting versus vanilla training.

### Risks & Mitigations
•  Augmentations may hurt in-domain accuracy → use adaptive probability schedule.

### Scores
Novelty 6 / Feasibility 8 / Impact 7 / Technical 7

---
