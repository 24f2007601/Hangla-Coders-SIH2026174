# Step classifier baseline is XGBoost over engineered features

Step Recognition is a small windowed classification problem over engineered features (hand-object distances, velocities, joint angles). We decided the baseline is XGBoost on hand-crafted window features: it trains in seconds on CPU, works well with a small custom dataset (~30 clips), is interpretable for the demo, and needs no GPU.

**Considered options:** LightGBM (near-identical tradeoffs); a small GRU/1D-CNN over landmark time series (better temporal modeling but needs more data/compute than a 5-day PoC justifies); ST-GCN or video backbones (research-grade, out of scope).

**Consequences:** feature engineering and dataset generation (recording + labeling short clips, extracting MediaPipe landmarks) is the real bottleneck and must start early. Upgrade to GRU/TCN only if the XGBoost baseline underperforms.