import csv
import numpy as np
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


def evaluate(x1, x2, gold):
    scores = cosine_sim(x1, x2)
    return float(spearmanr(scores, gold).correlation)


def select_global_topm(z_train, m, criterion="mean_abs"):
    """
    Select same m coordinates for all vectors.

    criterion:
    - mean_abs: coordinates with highest average absolute value
    - variance: coordinates with highest variance
    - energy: coordinates with highest average squared value
    """
    if criterion == "mean_abs":
        importance = np.mean(np.abs(z_train), axis=0)
    elif criterion == "variance":
        importance = np.var(z_train, axis=0)
    elif criterion == "energy":
        importance = np.mean(z_train ** 2, axis=0)
    else:
        raise ValueError(f"Unknown criterion: {criterion}")

    idx = np.argsort(importance)[-m:]
    idx = np.sort(idx)
    return idx


def project_global_topm(z, idx):
    return z[:, idx]


def add_result(results, method, family, dim, stored_dim, bytes_per_embedding, raw_bytes, spearman):
    results.append({
        "method": method,
        "family": family,
        "dim": dim,
        "stored_dim": stored_dim,
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

    print("Encoding validation sentence pairs...")
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

    results = []

    raw_dim = x1.shape[1]
    raw_bytes = raw_dim * 4

    raw_sp = evaluate(x1, x2, gold)
    add_result(
        results,
        "Raw float32",
        "Raw",
        raw_dim,
        raw_dim,
        raw_bytes,
        raw_bytes,
        raw_sp,
    )

    ks = [64, 96, 128, 192, 256, 384]
    ms = [16, 24, 32, 48, 64, 96, 128, 192]
    criteria = ["mean_abs", "variance", "energy"]

    for k in ks:
        if k > raw_dim:
            continue

        print(f"\n=== PCA k={k} ===")
        pca = PCA(n_components=k, random_state=42)
        pca.fit(x_train)

        z_train = pca.transform(x_train).astype(np.float32)
        z1 = pca.transform(x1).astype(np.float32)
        z2 = pca.transform(x2).astype(np.float32)

        # PCA float32
        sp = evaluate(z1, z2, gold)
        add_result(
            results,
            f"PCA-{k} float32",
            "PCA float32",
            k,
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

        sp = evaluate(dq1, dq2, gold)
        add_result(
            results,
            f"PCA-{k} int8 dense",
            "PCA int8 dense",
            k,
            k,
            k,
            raw_bytes,
            sp,
        )

        for m in ms:
            if m > k:
                continue

            for criterion in criteria:
                idx = select_global_topm(z_train, m, criterion=criterion)

                g1 = project_global_topm(z1, idx)
                g2 = project_global_topm(z2, idx)

                q1, scale = quantize_int8(g1)
                q2, _ = quantize_int8(g2, scale)

                dq1 = dequantize_int8(q1, scale)
                dq2 = dequantize_int8(q2, scale)

                sp = evaluate(dq1, dq2, gold)

                # Since selected coordinates are global, no per-vector indices.
                bytes_per_embedding = m

                add_result(
                    results,
                    f"GlobalTopM PCA-{k} m={m} {criterion}",
                    f"GlobalTopM-{criterion}",
                    k,
                    m,
                    bytes_per_embedding,
                    raw_bytes,
                    sp,
                )

    results = sorted(results, key=lambda r: (-r["spearman"], r["bytes_per_embedding"]))

    csv_path = "sera_global_topm_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "family",
                "dim",
                "stored_dim",
                "bytes_per_embedding",
                "compression",
                "spearman",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved results to {csv_path}")

    print("\nAll top results")
    print("=" * 110)
    print(
        f"{'Method':58s} {'Dim':>6s} {'Stored':>8s} "
        f"{'Bytes':>8s} {'Comp.':>8s} {'Spearman':>10s}"
    )
    print("-" * 110)

    for r in results[:30]:
        print(
            f"{r['method']:58s} "
            f"{r['dim']:6d} "
            f"{r['stored_dim']:8d} "
            f"{r['bytes_per_embedding']:8.2f} "
            f"{r['compression']:8.2f} "
            f"{r['spearman']:10.4f}"
        )

    for budget in [32, 48, 64, 96, 128]:
        filtered = [r for r in results if r["bytes_per_embedding"] <= budget]
        filtered = sorted(filtered, key=lambda r: r["spearman"], reverse=True)

        print(f"\nBest under {budget} bytes/embedding")
        print("=" * 110)
        for r in filtered[:15]:
            print(
                f"{r['method']:58s} "
                f"{r['bytes_per_embedding']:8.2f} bytes "
                f"{r['compression']:8.2f}x "
                f"{r['spearman']:10.4f}"
            )


if __name__ == "__main__":
    run()
