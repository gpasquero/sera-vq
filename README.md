# SERA-VQ

**Structured Embedding Representation via Residual Approximation and Vector Quantization.**
A small benchmark of memory-efficient sentence-embedding representations on BEIR/SciFact.

This repo started out as "PCA + RVQ beats PCA + int8 at 32 bytes/vector". After adding a missing baseline (Product Quantization) and seed-averaging, the honest version of the story is more nuanced.

---

## What this repo actually shows

On BEIR/SciFact with `all-MiniLM-L6-v2` (384-dim), at 32 bytes per embedding (48× compression over float32):

| Method               | Bytes/vec | nDCG@10           | Codebook RAM |
| -------------------- | --------- | ----------------- | ------------ |
| Raw float32          | 1536      | 0.6451            | —            |
| PQ-96 (PCA-96 + PQ)  | 32        | **0.5798 ± 0.004** | 96 kB        |
| SERA-VQ-96 (PCA + RVQ) | 32      | 0.5544 ± 0.005     | 3 MB         |
| PCA-32 + int8        | 32        | 0.4510             | —            |

(Stochastic methods averaged over 3 seeds. PCA + int8 is deterministic.)

Two things follow:

1. **Discrete codes do beat PCA + int8 at low byte budgets.** At 32 bytes, both PQ and SERA-VQ improve nDCG@10 by ~0.10 over PCA + int8.
2. **PQ beats SERA-VQ at every budget ≥ 16 bytes**, with ~32× less codebook RAM. SERA-VQ only wins at ≤ 8 bytes per vector, where PQ's chunk-splitting collapses.

The original "SERA-VQ is the right answer for low memory" framing does **not** hold. PQ is the better default. SERA-VQ is still a reasonable choice if you really need ≤ 8 B/vec or if your dimensionality doesn't divide cleanly into PQ chunks.

Full numbers, including 8/16/24 B budgets and per-seed values: `results/sera_beir_scifact_results.csv`.

---

## When to use which

| Byte budget        | Best choice in this study | Why                                                                                |
| ------------------ | ------------------------- | ---------------------------------------------------------------------------------- |
| ≥ 64 B             | PCA + int8                | Linear, deterministic, no codebooks, recovers ≥ 95% of raw nDCG.                   |
| 16–32 B            | PQ (PCA + PQ)             | Best discrete-code quality at this range, tiny codebooks, asymmetric distance LUTs. |
| ≤ 8 B              | SERA-VQ (PCA + RVQ)       | RVQ doesn't suffer PQ's chunk-dimension collapse at extreme low byte budgets.       |

---

## Method

All three methods start by L2-normalizing the 384-dim embedding and applying PCA to a target dim *k*.

- **PCA + int8**: store the *k*-dim float vector as int8 with a single global scale. *k* bytes/vec.
- **PQ (Product Quantization)**: split the *k*-dim PCA vector into *M* chunks of size *k*/*M*, fit one 256-centroid KMeans codebook per chunk. *M* bytes/vec.
- **SERA-VQ (PCA + RVQ)**: fit *M* sequential 256-centroid codebooks on residuals of the full *k*-dim PCA vector. *M* bytes/vec, *M* × larger codebooks than PQ.

Each codebook index fits in 1 byte (256 centroids), so per-vector memory equals the number of codebooks.

---

## Repo layout

```
sera-vq/
├── experiments/         # standalone runnable scripts
│   └── sera_beir_scifact_experiment.py   # main benchmark (PQ + SERA-VQ + baselines, 3 seeds)
├── results/             # CSVs (committed, treated as outputs of record)
├── plots/               # plotting scripts (numbers may be stale vs latest CSV)
├── figures/             # historical figures (some predate the PQ baseline)
└── paper/
    └── paper.pdf        # write-up of the comparison
```

---

## Reproduce

```bash
python3 -m venv sera-env
source sera-env/bin/activate
pip install -r requirements.txt
python experiments/sera_beir_scifact_experiment.py
```

This downloads SciFact (≈ 3 MB), encodes 5K documents and 300 queries with `all-MiniLM-L6-v2`, then runs all configurations across 3 seeds. End-to-end takes ~10 min on CPU. Results land in `results/sera_beir_scifact_results.csv`.

---

## Status and known gaps

- [x] BEIR/SciFact, single encoder (`all-MiniLM-L6-v2`), 3 seeds.
- [x] PQ baseline.
- [ ] OPQ (PQ with a learned rotation) — likely closes part of the SERA-VQ ↔ PQ gap.
- [ ] 1-bit binary quantization, Matryoshka representations.
- [ ] Larger BEIR splits (FiQA, NFCorpus, MS MARCO) and stronger encoders (BGE, E5).
- [ ] Asymmetric-distance scoring with PQ lookup tables — current setup reconstructs to dense and uses cosine, which hides PQ's main practical advantage.

The paper (`paper/paper.pdf`) reflects this state, including a full Limitations section.
