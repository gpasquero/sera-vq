# SERA-VQ: Discrete Codes for Extreme Embedding Compression

SERA-VQ compresses sentence embeddings into a few discrete bytes per vector
using **PCA dimensionality reduction followed by Residual Vector
Quantization (RVQ)**. Each embedding is encoded as a short sequence of
codebook indices (1 byte each, 256 centroids per codebook), so a 1536-byte
`float32` vector collapses to 8–32 bytes while preserving most of its
semantic structure.

The headline finding: **in the very-low-memory regime, discrete codes beat
dense PCA + int8 by a wide margin on retrieval quality.**

## Key result (BEIR / SciFact, nDCG@10)

| Method        | Bytes per embedding | nDCG@10   |
| ------------- | ------------------- | --------- |
| PCA + int8    | 32                  | 0.451     |
| **SERA-VQ**   | **32**              | **0.560** |

At a 48× compression ratio, SERA-VQ recovers retrieval quality that PCA
+ int8 cannot reach until it spends several times more bytes per vector.

## Repo layout

```
sera-vq/
├── experiments/   # standalone experiment scripts (STS-B + BEIR/SciFact)
├── plots/         # figure-generation scripts
├── figures/       # paper figures (PNG)
└── results/       # output CSVs from experiments
```

## How to run

```bash
pip install -r requirements.txt
python experiments/sera_beir_scifact_experiment.py
```

The SciFact experiment downloads the BEIR/SciFact dataset on first run and
writes its results CSV to the working directory. Other experiments
(`sera_vq_experiment.py`, `sera_curve_experiment.py`,
`sera_global_topm_experiment.py`) run on STS-B via HuggingFace `datasets`.
