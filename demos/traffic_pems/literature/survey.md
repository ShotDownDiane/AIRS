# Literature Survey: Traffic Flow Prediction on PeMS Dataset

## Executive Summary

Traffic flow prediction is a fundamental task in Intelligent Transportation Systems (ITS) that aims to forecast future traffic conditions (flow, speed, or occupancy) based on historical observations from sensor networks. The Performance Measurement System (PeMS) dataset, collected from over 39,000 loop detectors across California's highway system, has become the de facto standard benchmark for evaluating traffic prediction methods.

Over the past decade, the field has evolved dramatically from traditional statistical methods (ARIMA, VAR) and basic machine learning approaches to sophisticated deep learning architectures. The most significant paradigm shift occurred in 2017-2018 with the introduction of **graph neural networks (GNNs)** for traffic forecasting, which model the road network as a graph where sensors are nodes and road connections are edges. This formulation naturally captures the spatial dependencies inherent in traffic data.

The field has progressed through several waves of innovation:
1. **GCN + RNN era (2017-2019)**: STGCN and DCRNN established the graph-based paradigm
2. **Adaptive graph learning (2019-2021)**: Graph WaveNet, AGCRN removed the need for pre-defined graphs
3. **Attention/Transformer era (2020-2023)**: GMAN, STAEformer showed attention mechanisms can match or exceed GNN-based methods
4. **Simplification and foundation models (2023-present)**: MLP-based methods, LLMs, and state space models challenge architectural complexity

Key PeMS benchmark datasets include:
- **METR-LA**: 207 sensors in Los Angeles, 4 months of data (speed)
- **PeMS-BAY**: 325 sensors in Bay Area, 6 months of data (speed)
- **PEMS03**: 358 sensors (flow)
- **PEMS04**: 307 sensors (flow)
- **PEMS07**: 883 sensors (flow)
- **PEMS08**: 170 sensors (flow)

Standard evaluation metrics are **MAE**, **RMSE**, and **MAPE** at prediction horizons of 15, 30, and 60 minutes (or 3, 6, 12 steps at 5-minute intervals).

---

## Key Papers Organized by Theme/Approach

### 1. Foundational Graph-Based Methods (GCN + RNN/CNN)

#### STGCN — Spatio-Temporal Graph Convolutional Networks (Yu et al., 2017)
- **ArXiv**: [1709.04875](http://arxiv.org/abs/1709.04875v4)
- **Key idea**: First to formulate traffic prediction as a graph problem; uses Chebyshev graph convolution + gated temporal convolution (no RNNs)
- **Architecture**: Sandwich of ST-Conv blocks with ChebNet spatial convolution and 1D gated causal convolution
- **Significance**: Established the graph-based paradigm for traffic forecasting; all subsequent works build upon or compare against STGCN
- **Datasets**: BJER4, PeMSD7

#### DCRNN — Diffusion Convolutional Recurrent Neural Network (Li et al., 2017)
- **ArXiv**: [1707.01926](http://arxiv.org/abs/1707.01926v3)
- **Key idea**: Models traffic as a diffusion process on directed graphs; uses diffusion convolution integrated into GRU encoder-decoder with scheduled sampling
- **Significance**: Introduced METR-LA and PeMS-BAY benchmarks; diffusion convolution concept widely adopted
- **Datasets**: METR-LA, PeMS-BAY

#### SST-GCN — Simplified Spatio-temporal Traffic Forecasting (Roy et al., 2021)
- **ArXiv**: [2104.00055](http://arxiv.org/abs/2104.00055v1)
- **Key idea**: Discriminates between the impact of road junctions at different hop-distances; uses separate GNN layers for different hops
- **Datasets**: Standard traffic benchmarks

### 2. Adaptive Graph Learning Methods

#### Graph WaveNet (Wu et al., 2019; Shleifer et al., 2019)
- **ArXiv (improvements)**: [1912.07390](http://arxiv.org/abs/1912.07390v1)
- **Key idea**: Learns an adaptive adjacency matrix end-to-end via node embeddings; combines diffusion convolution with dilated causal convolutions (WaveNet-style)
- **Significance**: Removed the need for pre-defined graph structure; achieved SOTA on METR-LA and PeMS-BAY; highly influential
- **Datasets**: METR-LA, PeMS-BAY

#### AGCRN — Adaptive Graph Convolutional Recurrent Network (Bai et al., 2020)
- **ArXiv**: [2007.02842](http://arxiv.org/abs/2007.02842v2)
- **Key idea**: Node Adaptive Parameter Learning (NAPL) generates node-specific parameters; Data Adaptive Graph Generation (DAGG) learns graph from data
- **Significance**: Demonstrated that pre-defined graphs are avoidable; lightweight design with competitive performance
- **Datasets**: PEMS03, PEMS04, PEMS07, PEMS08, METR-LA, PeMS-BAY

#### MSTFGRN — Multi-Spatio-temporal Fusion Graph Recurrent Network (Zhao et al., 2022)
- **ArXiv**: [2205.01480](http://arxiv.org/abs/2205.01480v2)
- **Key idea**: Data-driven dynamic adjacency matrices that change with time; multi-scale spatio-temporal fusion
- **Datasets**: PeMS datasets

### 3. Attention and Transformer-Based Methods

#### GMAN — Graph Multi-Attention Network (Zheng et al., 2019)
- **ArXiv**: [1911.08415](http://arxiv.org/abs/1911.08415v2)
- **Key idea**: Encoder-decoder with spatio-temporal attention blocks; transform attention layer between encoder and decoder
- **Significance**: Showed attention-based approaches can outperform GCN+RNN methods, especially for longer horizons
- **Datasets**: PeMS-BAY, METR-LA

#### STTN — Spatial-Temporal Transformer Networks (Xu et al., 2020)
- **ArXiv**: [2001.02908](http://arxiv.org/abs/2001.02908v2)
- **Key idea**: Early application of Transformer to traffic forecasting; parallel spatial and temporal Transformers
- **Significance**: Pioneered Transformer use in traffic domain
- **Datasets**: PeMS datasets

#### A3T-GCN — Attention Temporal Graph Convolutional Network (Zhu et al., 2020)
- **ArXiv**: [2006.11583](http://arxiv.org/abs/2006.11583v1)
- **Key idea**: GCN + GRU + attention for adaptive temporal weighting; recognizes that distant past points can be as important as recent ones
- **Datasets**: Traffic datasets including PeMS

#### STAEformer — Spatio-Temporal Adaptive Embedding Transformer (Liu et al., 2023)
- **ArXiv**: [2308.10425](http://arxiv.org/abs/2308.10425v5)
- **Key idea**: Vanilla Transformer with spatio-temporal adaptive embeddings achieves SOTA; argues that embedding quality matters more than architectural complexity
- **Significance**: Major finding that challenges the trend of increasingly complex architectures; shows diminishing returns of architectural innovation
- **Datasets**: PEMS03, PEMS04, PEMS07, PEMS08, METR-LA, PeMS-BAY

#### PDFormer — Propagation Delay-Aware Dynamic Long-Range Transformer (Jiang et al., 2023)
- **ArXiv**: [2301.07945](http://arxiv.org/abs/2301.07945v3)
- **Key idea**: Models traffic propagation delay explicitly; dynamic spatial attention with long-range dependencies
- **Significance**: Incorporates traffic domain knowledge (propagation delay) into Transformer architecture
- **Datasets**: PEMS03, PEMS04, PEMS07, PEMS08

#### Navigating Spatio-Temporal Heterogeneity (Zhou et al., 2024)
- **ArXiv**: [2408.10822](http://arxiv.org/abs/2408.10822v2)
- **Key idea**: Graph Transformer addressing spatio-temporal heterogeneity; traffic patterns vary across regions and time
- **Datasets**: Standard PeMS benchmarks

### 4. Joint/Unified Spatio-Temporal Modeling

#### Unified Spatio-Temporal Modeling (Roy et al., 2021)
- **ArXiv**: [2104.12518](http://arxiv.org/abs/2104.12518v2)
- **Key idea**: Joint extraction of spatial and temporal features instead of separate factorized modules
- **Datasets**: Standard traffic benchmarks

#### Spatio-Temporal Joint Graph Convolutional Networks (Zheng et al., 2021)
- **ArXiv**: [2111.13684](http://arxiv.org/abs/2111.13684v3)
- **Key idea**: Constructs joint spatio-temporal graph connecting nodes across space and time; dynamic adjacency matrices
- **Datasets**: PEMS03, PEMS04, PEMS07, PEMS08, METR-LA, PeMS-BAY

#### COOL — Conjoint Perspective on STGNNs (Ju et al., 2024)
- **ArXiv**: [2403.01091](http://arxiv.org/abs/2403.01091v1)
- **Key idea**: Models high-order interactions between spatial and temporal dimensions; addresses diverse transitional patterns
- **Datasets**: Standard PeMS benchmarks

### 5. Dynamic Graph Methods

#### DGCRN — Dynamic Graph Convolutional Recurrent Network (Li et al., 2021)
- **ArXiv**: [2104.14917](http://arxiv.org/abs/2104.14917v2)
- **Key idea**: Dynamic graph generation conditioned on traffic states; hyper-network for dynamic parameters; comprehensive benchmarking
- **Significance**: Important benchmarking contribution; addresses lack of fair comparison in the field
- **Datasets**: METR-LA, PeMS-BAY

#### Dynamic Causal Graph Convolutional Network (Lin et al., 2023)
- **ArXiv**: [2306.07019](http://arxiv.org/abs/2306.07019v2)
- **Key idea**: Embeds time-varying dynamic Bayesian networks to capture fine spatio-temporal topology
- **Datasets**: Traffic benchmarks

### 6. Lightweight and MLP-Based Methods

#### MLPST — MLP is All You Need for Spatio-Temporal Prediction (Zhang et al., 2023)
- **ArXiv**: [2309.13363](http://arxiv.org/abs/2309.13363v1)
- **Key idea**: Pure MLP architecture without GNNs or Transformers; focuses on efficiency and practical deployment
- **Significance**: Questions the necessity of complex architectures; competitive results with much simpler models
- **Datasets**: PeMS datasets

#### LSTAN-GERPE — Lightweight Spatio-Temporal Attention Network (Wang & Yang, 2025)
- **ArXiv**: [2505.12136](http://arxiv.org/abs/2505.12136v1)
- **Key idea**: Lightweight attention with graph embedding and rotational position encoding; overcomes GNN local receptive field limitation
- **Datasets**: Traffic benchmarks

### 7. Emerging Paradigms (Foundation Models & State Space Models)

#### Spatial-Temporal Large Language Model for Traffic Prediction (Liu et al., 2024)
- **ArXiv**: [2401.10134](http://arxiv.org/abs/2401.10134v4)
- **Key idea**: Adapts pre-trained LLMs for spatial-temporal traffic forecasting; spatial-temporal tokenization
- **Significance**: Represents the emerging trend of applying foundation models to traffic prediction
- **Datasets**: Standard PeMS benchmarks

#### SpoT-Mamba — Selective State Spaces for STG Forecasting (Choi et al., 2024)
- **ArXiv**: [2406.11244](http://arxiv.org/abs/2406.11244v1)
- **Key idea**: Applies Mamba (selective state space model) to spatio-temporal graph forecasting; linear complexity
- **Significance**: Newest architectural paradigm for traffic forecasting
- **Datasets**: Traffic benchmarks including PeMS

### 8. Scalability and Robustness

#### Graph-Partitioning-Based DCRNN for Large-Scale Forecasting (Mallick et al., 2019)
- **ArXiv**: [1909.11197](http://arxiv.org/abs/1909.11197v4)
- **Key idea**: Graph partitioning to scale DCRNN to large networks with thousands of sensors
- **Significance**: Addresses practical scalability challenges for real-world PeMS deployment
- **Datasets**: Large-scale PeMS data

#### Graph-based Multi-ODE Neural Networks (Liu et al., 2023)
- **ArXiv**: [2305.18687](http://arxiv.org/abs/2305.18687v2)
- **Key idea**: Combines GNNs with Neural ODEs to address over-smoothing in deep architectures
- **Datasets**: Standard traffic benchmarks

#### Adversarial Attacks on Graph-based Traffic Prediction (Zhu et al., 2021)
- **ArXiv**: [2104.09369](http://arxiv.org/abs/2104.09369v1)
- **Key idea**: Studies vulnerability of GCN-based traffic prediction to adversarial attacks
- **Significance**: Highlights robustness concerns for deployed traffic prediction systems
- **Datasets**: Traffic benchmarks

---

## Current State-of-the-Art Methods

Based on the surveyed literature, the current state-of-the-art on PeMS benchmarks can be characterized as follows:

### Top-Performing Approaches (2023-2025)
1. **STAEformer** (2023): Vanilla Transformer + adaptive embeddings — achieves SOTA with surprising simplicity
2. **PDFormer** (2023): Propagation delay-aware Transformer — strong on PEMS03/04/07/08
3. **COOL** (2024): Conjoint spatio-temporal modeling — captures high-order interactions
4. **SpoT-Mamba** (2024): State space model — efficient long-range dependency modeling

### Established Strong Baselines (2019-2022)
5. **Graph WaveNet** (2019): Adaptive graph + dilated convolution — consistently strong baseline
6. **AGCRN** (2020): Adaptive graph + node-specific parameters — efficient and effective
7. **GMAN** (2019): Multi-attention encoder-decoder — strong for long-term prediction
8. **DGCRN** (2021): Dynamic graph + hyper-network — good benchmark paper

### Performance Trends
- On **PEMS04** (representative benchmark), typical MAE values range from ~19-22 for 12-step prediction
- On **METR-LA**, typical MAE values range from ~2.6-3.0 for 12-step prediction
- Performance differences between top methods are often small (< 2% relative improvement)
- The gap between simple baselines (e.g., MLP-based) and complex architectures is narrowing

---

## Open Challenges and Research Gaps

### 1. Diminishing Returns of Architectural Complexity
- STAEformer showed that vanilla Transformers with good embeddings can match complex GNN architectures
- MLPST demonstrated competitive results with pure MLPs
- **Gap**: Need for principled understanding of when and why complex architectures help

### 2. Standardization and Fair Comparison
- DGCRN highlighted the severe lack of fair comparison across methods
- Different papers use different data splits, preprocessing, and evaluation protocols
- **Gap**: Need for standardized benchmarking frameworks (like LibCity/BasicTS)

### 3. Scalability to Large Networks
- Most methods are evaluated on networks with 100-900 sensors
- Real PeMS system has 39,000+ sensors
- **Gap**: Scalable methods that maintain accuracy on very large networks

### 4. Dynamic and Evolving Networks
- Most methods assume fixed network topology
- Real traffic networks evolve (new roads, closures, construction)
- **Gap**: Methods that handle topology changes gracefully

### 5. Robustness and Reliability
- Adversarial attack studies show vulnerability of GNN-based methods
- Missing data, sensor failures, and anomalous events are common in practice
- **Gap**: Robust prediction methods with uncertainty quantification

### 6. Multi-Modal and External Factors
- Most methods use only historical traffic data
- Weather, events, incidents, construction zones significantly affect traffic
- **Gap**: Effective integration of external factors and multi-modal data

### 7. Long-Term and Multi-Scale Prediction
- Most methods focus on short-term (up to 1 hour) prediction
- Practical applications need predictions at multiple time scales
- **Gap**: Accurate prediction beyond 1 hour; multi-scale prediction frameworks

### 8. Transferability and Generalization
- Models trained on one network typically don't transfer to others
- **Gap**: Transfer learning and domain adaptation for traffic prediction

### 9. Interpretability
- Deep learning models are largely black boxes
- Traffic engineers need interpretable predictions for decision-making
- **Gap**: Interpretable traffic prediction models

### 10. Foundation Models for Traffic
- LLM-based approaches are emerging but still in early stages
- Pre-training strategies for traffic data are underexplored
- **Gap**: Effective foundation models that leverage large-scale traffic data across cities

---

## Methodological Evolution Summary

| Era | Period | Key Methods | Spatial Modeling | Temporal Modeling |
|-----|--------|-------------|-----------------|-------------------|
| Classical | Pre-2017 | ARIMA, VAR, SVR, LSTM | None/manual | Statistical/RNN |
| GCN+RNN | 2017-2019 | STGCN, DCRNN | ChebNet, Diffusion Conv | Gated Conv, GRU |
| Adaptive Graph | 2019-2021 | Graph WaveNet, AGCRN | Learned adjacency | Dilated Conv, GRU |
| Attention/Transformer | 2020-2023 | GMAN, STAEformer, PDFormer | Spatial attention | Temporal attention |
| Simplification | 2023-2024 | MLPST, STAEformer | MLP/Embedding | MLP/Attention |
| Foundation Models | 2024-present | ST-LLM, SpoT-Mamba | LLM/SSM | LLM/SSM |

---

## Bibliography

1. Yu, B., Yin, H., & Zhu, Z. (2017). Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting. ArXiv: [1709.04875](http://arxiv.org/abs/1709.04875v4)

2. Li, Y., Yu, R., Shahabi, C., & Liu, Y. (2017). Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting. ArXiv: [1707.01926](http://arxiv.org/abs/1707.01926v3)

3. Wu, Z., Pan, S., Long, G., Jiang, J., & Zhang, C. (2019). Graph WaveNet for Deep Spatial-Temporal Graph Modeling. ArXiv: 1906.00121 (referenced via [1912.07390](http://arxiv.org/abs/1912.07390v1))

4. Zheng, C., Fan, X., Wang, C., & Qi, J. (2019). GMAN: A Graph Multi-Attention Network for Traffic Prediction. ArXiv: [1911.08415](http://arxiv.org/abs/1911.08415v2)

5. Bai, L., Yao, L., Li, C., Wang, X., & Wang, C. (2020). Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting. ArXiv: [2007.02842](http://arxiv.org/abs/2007.02842v2)

6. Xu, M., Dai, W., Liu, C., Gao, X., Lin, W., Qi, G.-J., & Xiong, H. (2020). Spatial-Temporal Transformer Networks for Traffic Flow Forecasting. ArXiv: [2001.02908](http://arxiv.org/abs/2001.02908v2)

7. Zhu, J., Song, Y., Zhao, L., & Li, H. (2020). A3T-GCN: Attention Temporal Graph Convolutional Network for Traffic Forecasting. ArXiv: [2006.11583](http://arxiv.org/abs/2006.11583v1)

8. Roy, A., Roy, K.K., Ali, A.A., Amin, M.A., & Rahman, A.K.M.M. (2021). Unified Spatio-Temporal Modeling for Traffic Forecasting using Graph Neural Network. ArXiv: [2104.12518](http://arxiv.org/abs/2104.12518v2)

9. Li, F., Feng, J., Yan, H., Jin, G., Jin, D., & Li, Y. (2021). Dynamic Graph Convolutional Recurrent Network for Traffic Prediction: Benchmark and Solution. ArXiv: [2104.14917](http://arxiv.org/abs/2104.14917v2)

10. Zheng, C., Fan, X., Pan, S., Jin, H., Peng, Z., Wang, C., & Yu, P.S. (2021). Spatio-Temporal Joint Graph Convolutional Networks for Traffic Forecasting. ArXiv: [2111.13684](http://arxiv.org/abs/2111.13684v3)

11. Zhao, W., Zhang, S., Zhou, B., & Wang, B. (2022). Multi-Spatio-temporal Fusion Graph Recurrent Network for Traffic Forecasting. ArXiv: [2205.01480](http://arxiv.org/abs/2205.01480v2)

12. Jiang, J., Han, C., Zhao, W.X., & Wang, J. (2023). PDFormer: Propagation Delay-Aware Dynamic Long-Range Transformer for Traffic Flow Prediction. ArXiv: [2301.07945](http://arxiv.org/abs/2301.07945v3)

13. Liu, H., Dong, Z., Jiang, R., Deng, J., Deng, J., Chen, Q., & Li, X. (2023). STAEformer: Spatio-Temporal Adaptive Embedding Makes Vanilla Transformer SOTA for Traffic Forecasting. ArXiv: [2308.10425](http://arxiv.org/abs/2308.10425v5)

14. Zhang, Z., Huang, Z., Hu, Z., Zhao, X., Wang, W., Liu, Z., Zhang, J., Qin, S.J., & Zhao, H. (2023). MLPST: MLP is All You Need for Spatio-Temporal Prediction. ArXiv: [2309.13363](http://arxiv.org/abs/2309.13363v1)

15. Liu, Z., Shojaee, P., & Reddy, C.K. (2023). Graph-based Multi-ODE Neural Networks for Spatio-Temporal Traffic Forecasting. ArXiv: [2305.18687](http://arxiv.org/abs/2305.18687v2)

16. Lin, J., Li, Z., Li, Z., Bai, L., Zhao, R., & Chen, J. (2023). Dynamic Causal Graph Convolutional Network for Traffic Prediction. ArXiv: [2306.07019](http://arxiv.org/abs/2306.07019v2)

17. Liu, C., Yang, S., Xu, Q., Li, Z., Long, C., Li, Z., & Zhao, R. (2024). Spatial-Temporal Large Language Model for Traffic Prediction. ArXiv: [2401.10134](http://arxiv.org/abs/2401.10134v4)

18. Ju, W., Zhao, Y., Qin, Y., Yi, S., Yuan, J., Xiao, Z., Luo, X., Yan, X., & Zhang, M. (2024). COOL: A Conjoint Perspective on Spatio-Temporal Graph Neural Network for Traffic Forecasting. ArXiv: [2403.01091](http://arxiv.org/abs/2403.01091v1)

19. Choi, J., Kim, H., An, M., & Whang, J.J. (2024). SpoT-Mamba: Learning Long-Range Dependency on Spatio-Temporal Graphs with Selective State Spaces. ArXiv: [2406.11244](http://arxiv.org/abs/2406.11244v1)

20. Zhou, J., Liu, E., Chen, W., Zhong, S., & Liang, Y. (2024). Navigating Spatio-Temporal Heterogeneity: A Graph Transformer Approach for Traffic Forecasting. ArXiv: [2408.10822](http://arxiv.org/abs/2408.10822v2)

21. Mallick, T., Balaprakash, P., Rask, E., & Macfarlane, J. (2019). Graph-Partitioning-Based Diffusion Convolutional Recurrent Neural Network for Large-Scale Traffic Forecasting. ArXiv: [1909.11197](http://arxiv.org/abs/1909.11197v4)

22. Zhu, L., Feng, K., Pu, Z., & Ma, W. (2021). Adversarial Diffusion Attacks on Graph-based Traffic Prediction Models. ArXiv: [2104.09369](http://arxiv.org/abs/2104.09369v1)

23. Wang, X., & Yang, S.-R. (2025). Lightweight Spatio-Temporal Attention Network with Graph Embedding and Rotational Position Encoding for Traffic Forecasting. ArXiv: [2505.12136](http://arxiv.org/abs/2505.12136v1)

24. Kieu, D., Kieu, T., Han, P., Yang, B., & Jensen, C.S. (2024). TEAM: Topological Evolution-aware Framework for Traffic Forecasting. ArXiv: [2410.19192](http://arxiv.org/abs/2410.19192v3)
