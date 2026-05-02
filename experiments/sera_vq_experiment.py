import csv
import numpy as np

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from scipy.stats import spearmanr


def cosine_sim(a, b, eps=1e-8):
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + eps)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + eps)
    return np.sum(a * b, axis=1)


def evaluate(x1, x2, gold):
    return float(spearmanr(cosine_sim(x1, x2), gold).correlation)


def quantize_int8(x, scale=None):
    if scale is None:
        scale = (np.max(np.abs(x)) + 1e-8) / 127.0
    q = np.round(x / scale)
    q = np.clip(q, -127, 127).astype(np.int8)
    return q, scale


def dequantize_int8(q, scale):
    return q.astype(np.float32) * scale


class ResidualVectorQuantizer:
    def __init__(self, n_codebooks=8, n_centroids=256, batch_size=2048, random_state=42):
        self.n_codebooks = n_codebooks
        self.n_centroids = n_centroids
        self.batch_size = batch_size
        self.random_state = random_state
        self.codebooks = []

    def fit(self, x):
        residual = x.astype(np.float32).copy()
        self.codebooks = []

        for i in range(self.n_codebooks):
            print(f"  training codebook {i + 1}/{self.n_codebooks}")
            km = MiniBatchKMeans(
                n_clusters=self.n_centroids,
                batch_size=self.batch_size,
                random_state=self.random_state + i,
                n_init="auto",
                max_iter=200,
                verbose=0,
            )
            km.fit(residual)

            centers = km.cluster_centers_.astype(np.float32)
            labels = km.predict(residual)
            approx = centers[labels]

            residual = residual - approx
            self.codebooks.append(centers)

        return self

    def encode(self, x):
        residual = x.astype(np.float32).copy()
        codes = []

        for centers in self.codebooks:
            # brute-force nearest centroid
            # distances: ||x-c||^2 = ||x||^2 + ||c||^2 - 2xc
            x_norm = np.sum(residual ** 2, axis=1, keepdims=True)
            c_norm = np.sum(centers ** 2, axis=1)[None, :]
            dist = x_norm + c_norm - 2 * residual @ centers.T

            labels = np.argmin(dist, axis=1).astype(np.uint16)
            approx = centers[labels]

            residual = residual - approx
            codes.append(labels)

        return np.stack(codes, axis=1)

    def decode(self, codes):
        n = codes.shape[0]
        dim = self.codebooks[0].shape[1]
        out = np.zeros((n, dim), dtype=np.float32)

        for i, centers in enumerate(self.codebooks):
            out += centers[codes[:, i]]

        return out


def add_result(results, method, bytes_per_embedding, raw_bytes, spearman):
    results.append({
        "method": method,
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

    print("Encoding train...")
    x_train = model.encode(
        train_sentences,
        batch_size=64,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    print("Encoding validation...")
    x1 = model.encode(
        val_s1,
        batch_size=64,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    x2 = model.encode(
        val_s2,
        batch_size=64,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    raw_dim = x1.shape[1]
    raw_bytes = raw_dim * 4
    results = []

    raw_sp = evaluate(x1, x2, gold)
    add_result(results, "Raw float32 normalized", raw_bytes, raw_bytes, raw_sp)

    # Strong PCA+int8 baselines
    for k in [32, 64, 96, 128, 192]:
        print(f"\nPCA baseline k={k}")
        pca = PCA(n_components=k, random_state=42)
        pca.fit(x_train)

        z1 = pca.transform(x1).astype(np.float32)
        z2 = pca.transform(x2).astype(np.float32)

        sp = evaluate(z1, z2, gold)
        add_result(results, f"PCA-{k} float32", k * 4, raw_bytes, sp)

        q1, scale = quantize_int8(z1)
        q2, _ = quantize_int8(z2, scale)
        d1 = dequantize_int8(q1, scale)
        d2 = dequantize_int8(q2, scale)

        sp = evaluate(d1, d2, gold)
        add_result(results, f"PCA-{k} int8 dense", k, raw_bytes, sp)

    # SERA-VQ: residual vector quantization on PCA space
    for k in [64, 96, 128]:
        print(f"\nSERA-VQ PCA space k={k}")
        pca = PCA(n_components=k, random_state=42)
        pca.fit(x_train)

        z_train = pca.transform(x_train).astype(np.float32)
        z1 = pca.transform(x1).astype(np.float32)
        z2 = pca.transform(x2).astype(np.float32)

        for n_codebooks in [4, 8, 12, 16, 24, 32]:
            print(f"Training RVQ: k={k}, codebooks={n_codebooks}")

            rvq = ResidualVectorQuantizer(
                n_codebooks=n_codebooks,
                n_centroids=256,  # 1 byte per code
                batch_size=2048,
                random_state=42,
            )

            rvq.fit(z_train)

            c1 = rvq.encode(z1)
            c2 = rvq.encode(z2)

            d1 = rvq.decode(c1)
            d2 = rvq.decode(c2)

            sp = evaluate(d1, d2, gold)

            # 256 centroids => each code is 1 byte.
            # Codebooks are global model storage, not per embedding.
            bytes_per_embedding = n_codebooks

            add_result(
                results,
                f"SERA-VQ PCA-{k} {n_codebooks}x8bit",
                bytes_per_embedding,
                raw_bytes,
                sp,
            )

    results = sorted(results, key=lambda r: (-r["spearman"], r["bytes_per_embedding"]))

    csv_path = "sera_vq_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["method", "bytes_per_embedding", "compression", "spearman"],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved {csv_path}")

    print("\nTop results")
    print("=" * 90)
    print(f"{'Method':45s} {'Bytes':>10s} {'Comp.':>10s} {'Spearman':>10s}")
    print("-" * 90)
    for r in results[:30]:
        print(
            f"{r['method']:45s} "
            f"{r['bytes_per_embedding']:10.2f} "
            f"{r['compression']:10.2f} "
            f"{r['spearman']:10.4f}"
        )

    for budget in [8, 16, 24, 32, 64, 96]:
        filtered = [r for r in results if r["bytes_per_embedding"] <= budget]
        filtered = sorted(filtered, key=lambda r: r["spearman"], reverse=True)

        print(f"\nBest under {budget} bytes/embedding")
        print("=" * 90)
        for r in filtered[:10]:
            print(
                f"{r['method']:45s} "
                f"{r['bytes_per_embedding']:10.2f} "
                f"{r['compression']:10.2f} "
                f"{r['spearman']:10.4f}"
            )


if __name__ == "__main__":
    run()
