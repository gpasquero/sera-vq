import csv
import numpy as np
import matplotlib.pyplot as plt

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from scipy.stats import spearmanr


def cosine_sim(a, b, eps=1e-8):
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + eps)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + eps)
    return np.sum(a * b, axis=1)


def quantize_int8(x, scale=None):
    if scale is None:
        scale = (np.max(np.abs(x)) + 1e-8) / 127.0
    q = np.round(x / scale)
    q = np.clip(q, -127, 127).astype(np.int8)
    return q, scale


def dequantize_int8(q, scale):
    return q.astype(np.float32) * scale


def top_m_keep(x, m):
    if m >= x.shape[1]:
        return x.copy()

    out = np.zeros_like(x)
    idx = np.argpartition(np.abs(x), -m, axis=1)[:, -m:]
    rows = np.arange(x.shape[0])[:, None]
    out[rows, idx] = x[rows, idx]
    return out


def random_orthogonal_rotation(k, seed=42):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(k, k))
    q, _ = np.linalg.qr(a)
    return q.astype(np.float32)


def evaluate_pair(x1, x2, gold):
    scores = cosine_sim(x1, x2)
    return float(spearmanr(scores, gold).correlation)


def add_result(results, method, family, dim, bytes_per_embedding, raw_bytes, spearman):
    results.append({
        "method": method,
        "family": family,
        "dim": dim,
        "bytes_per_embedding": bytes_per_embedding,
        "compression": raw_bytes / bytes_per_embedding,
        "spearman": spearman,
    })


def run():
    print("Loading STS-B...")
    dataset = load_dataset("glue", "stsb")
    train = dataset["train"]
    val = dataset["validation"]

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    train_sentences = list(train["sentence1"]) + list(train["sentence2"])
    val_s1 = list(val["sentence1"])
    val_s2 = list(val["sentence2"])
    gold = np.array(val["label"], dtype=np.float32)

    print("Encoding train sentences...")
    x_train = model.encode(
        train_sentences,
        batch_size=64,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=False,
    ).astype(np.float32)

    print("Encoding validation pairs...")
    x1 = model.encode(
        val_s1,
        batch_size=64,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=False,
    ).astype(np.float32)

    x2 = model.encode(
        val_s2,
        batch_size=64,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=False,
    ).astype(np.float32)

    raw_dim = x1.shape[1]
    raw_bytes = raw_dim * 4

    results = []

    raw_spearman = evaluate_pair(x1, x2, gold)
    add_result(
        results,
        "Raw float32",
        "Raw",
        raw_dim,
        raw_bytes,
        raw_bytes,
        raw_spearman,
    )

    ks = [32, 64, 96, 128, 192, 256, 384]
    ms = [4, 8, 12, 16, 24, 32, 48, 64, 96, 128]
    seeds = [1, 2, 3, 4, 5]

    for k in ks:
        if k > raw_dim:
            continue

        print(f"\n=== PCA k={k} ===")
        pca = PCA(n_components=k, random_state=42)
        pca.fit(x_train)

        z1 = pca.transform(x1).astype(np.float32)
        z2 = pca.transform(x2).astype(np.float32)

        # PCA float32
        sp = evaluate_pair(z1, z2, gold)
        add_result(
            results,
            f"PCA-{k} float32",
            "PCA float32",
            k,
            k * 4,
            raw_bytes,
            sp,
        )

        # PCA int8 dense
        q1, scale = quantize_int8(z1)
        q2, _ = quantize_int8(z2, scale)
        dq1 = dequantize_int8(q1, scale)
        dq2 = dequantize_int8(q2, scale)

        sp = evaluate_pair(dq1, dq2, gold)
        add_result(
            results,
            f"PCA-{k} int8 dense",
            "PCA int8 dense",
            k,
            k,
            raw_bytes,
            sp,
        )

        # PCA top-m int8
        for m in ms:
            if m > k:
                continue

            t1 = top_m_keep(z1, m)
            t2 = top_m_keep(z2, m)

            q1, scale = quantize_int8(t1)
            q2, _ = quantize_int8(t2, scale)
            dq1 = dequantize_int8(q1, scale)
            dq2 = dequantize_int8(q2, scale)

            sp = evaluate_pair(dq1, dq2, gold)

            # Store each active value as uint16 index + int8 value = 3 bytes
            bytes_per_embedding = m * 3

            add_result(
                results,
                f"PCA-{k} topm-{m}",
                "PCA top-m int8",
                k,
                bytes_per_embedding,
                raw_bytes,
                sp,
            )

        # Random rotations + top-m
        for seed in seeds:
            R = random_orthogonal_rotation(k, seed=seed)
            rz1 = z1 @ R
            rz2 = z2 @ R

            for m in ms:
                if m > k:
                    continue

                t1 = top_m_keep(rz1, m)
                t2 = top_m_keep(rz2, m)

                q1, scale = quantize_int8(t1)
                q2, _ = quantize_int8(t2, scale)
                dq1 = dequantize_int8(q1, scale)
                dq2 = dequantize_int8(q2, scale)

                sp = evaluate_pair(dq1, dq2, gold)
                bytes_per_embedding = m * 3

                add_result(
                    results,
                    f"RandRot-{k}-s{seed} topm-{m}",
                    "Random rotation top-m int8",
                    k,
                    bytes_per_embedding,
                    raw_bytes,
                    sp,
                )

    # Save CSV
    csv_path = "sera_curve_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "family",
                "dim",
                "bytes_per_embedding",
                "compression",
                "spearman",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved results to {csv_path}")

    # Print best under budgets
    for budget in [64, 96, 128, 192]:
        filtered = [r for r in results if r["bytes_per_embedding"] <= budget]
        filtered = sorted(filtered, key=lambda r: r["spearman"], reverse=True)

        print(f"\nBest under {budget} bytes/embedding")
        print("=" * 90)
        for r in filtered[:10]:
            print(
                f"{r['method']:40s} "
                f"{r['bytes_per_embedding']:8.2f} bytes "
                f"{r['compression']:8.2f}x "
                f"{r['spearman']:.4f}"
            )

    plot_results(results)


def plot_results(results):
    families = sorted(set(r["family"] for r in results))

    # Plot 1: Spearman vs bytes
    plt.figure(figsize=(10, 6))

    for family in families:
        rs = [r for r in results if r["family"] == family]
        rs = sorted(rs, key=lambda r: r["bytes_per_embedding"])

        x = [r["bytes_per_embedding"] for r in rs]
        y = [r["spearman"] for r in rs]

        plt.scatter(x, y, label=family, alpha=0.8)

    plt.xlabel("Bytes per embedding")
    plt.ylabel("Spearman correlation")
    plt.title("Compression-quality tradeoff on STS-B")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("spearman_vs_bytes.png", dpi=200)
    print("Saved spearman_vs_bytes.png")

    # Plot 2: Spearman vs compression
    plt.figure(figsize=(10, 6))

    for family in families:
        rs = [r for r in results if r["family"] == family]
        rs = sorted(rs, key=lambda r: r["compression"])

        x = [r["compression"] for r in rs]
        y = [r["spearman"] for r in rs]

        plt.scatter(x, y, label=family, alpha=0.8)

    plt.xlabel("Compression ratio")
    plt.ylabel("Spearman correlation")
    plt.title("Compression ratio vs semantic similarity")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("spearman_vs_compression.png", dpi=200)
    print("Saved spearman_vs_compression.png")

    # Plot 3: under 128 bytes only
    plt.figure(figsize=(10, 6))

    for family in families:
        rs = [
            r for r in results
            if r["family"] == family and r["bytes_per_embedding"] <= 128
        ]

        if not rs:
            continue

        rs = sorted(rs, key=lambda r: r["bytes_per_embedding"])

        x = [r["bytes_per_embedding"] for r in rs]
        y = [r["spearman"] for r in rs]

        plt.scatter(x, y, label=family, alpha=0.8)

    plt.xlabel("Bytes per embedding")
    plt.ylabel("Spearman correlation")
    plt.title("Best region: methods under 128 bytes/embedding")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("spearman_under_128_bytes.png", dpi=200)
    print("Saved spearman_under_128_bytes.png")


if __name__ == "__main__":
    run()
