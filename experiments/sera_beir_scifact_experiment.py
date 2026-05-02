import os
import csv
import time
import zipfile
import urllib.request
from pathlib import Path

import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans


# ----------------------------
# Utilities
# ----------------------------

def download_and_extract_scifact(data_dir="datasets"):
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True)

    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
    zip_path = data_dir / "scifact.zip"
    out_dir = data_dir / "scifact"

    if out_dir.exists():
        print("SciFact already exists.")
        return out_dir

    print("Downloading SciFact...")
    urllib.request.urlretrieve(url, zip_path)

    print("Extracting SciFact...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(data_dir)

    return out_dir


def load_beir_jsonl(path):
    import json
    rows = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            rows[obj["_id"]] = obj
    return rows


def load_qrels(path):
    qrels = {}
    with open(path, "r", encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            parts = line.strip().split("\t")

            if len(parts) == 4:
                qid, _, docid, score = parts
            elif len(parts) == 3:
                qid, docid, score = parts
            else:
                continue

            score = int(score)

            if score > 0:
                qrels.setdefault(qid, set()).add(docid)

    return qrels


def normalize(x, eps=1e-8):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def search_topk(query_emb, doc_emb, topk=10, batch_size=256):
    results = {}
    doc_matrix = doc_emb.T

    for start in range(0, len(query_emb), batch_size):
        end = start + batch_size
        q = query_emb[start:end]
        scores = q @ doc_matrix
        idx = np.argpartition(-scores, topk, axis=1)[:, :topk]
        sorted_idx = np.take_along_axis(
            idx,
            np.argsort(-np.take_along_axis(scores, idx, axis=1), axis=1),
            axis=1,
        )
        for i, row in enumerate(sorted_idx):
            results[start + i] = row.tolist()

    return results


def recall_at_k(results, qids, docids, qrels, k=10):
    recalls = []
    for qi, qid in enumerate(qids):
        relevant = qrels.get(qid, set())
        if not relevant:
            continue
        retrieved = [docids[j] for j in results[qi][:k]]
        hit_count = len(set(retrieved) & relevant)
        recalls.append(hit_count / len(relevant))
    return float(np.mean(recalls))


def ndcg_at_k(results, qids, docids, qrels, k=10):
    scores = []
    for qi, qid in enumerate(qids):
        relevant = qrels.get(qid, set())
        if not relevant:
            continue
        dcg = 0.0
        for rank, doc_idx in enumerate(results[qi][:k], start=1):
            docid = docids[doc_idx]
            if docid in relevant:
                dcg += 1.0 / np.log2(rank + 1)
        ideal_hits = min(len(relevant), k)
        idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))
        scores.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(scores))


def quantize_int8(x, scale=None):
    if scale is None:
        scale = (np.max(np.abs(x)) + 1e-8) / 127.0
    q = np.round(x / scale)
    q = np.clip(q, -127, 127).astype(np.int8)
    return q, scale


def dequantize_int8(q, scale):
    return q.astype(np.float32) * scale


# ----------------------------
# Residual Vector Quantizer
# ----------------------------

class ResidualVectorQuantizer:
    def __init__(self, n_codebooks=16, n_centroids=256, batch_size=2048, random_state=42):
        self.n_codebooks = n_codebooks
        self.n_centroids = n_centroids
        self.batch_size = batch_size
        self.random_state = random_state
        self.codebooks = []

    def fit(self, x):
        residual = x.astype(np.float32).copy()
        self.codebooks = []
        for i in range(self.n_codebooks):
            km = MiniBatchKMeans(
                n_clusters=self.n_centroids,
                batch_size=self.batch_size,
                random_state=self.random_state + i,
                n_init="auto",
                max_iter=200,
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

    def codebook_bytes(self):
        return sum(c.nbytes for c in self.codebooks)


# ----------------------------
# Product Quantizer (PQ)
# ----------------------------

class ProductQuantizer:
    """
    Standard PQ: split d-dim vector into n_codebooks chunks of d/n_codebooks dims each,
    train independent KMeans (256 centroids) on each chunk.
    """
    def __init__(self, n_codebooks=8, n_centroids=256, batch_size=2048, random_state=42):
        self.n_codebooks = n_codebooks
        self.n_centroids = n_centroids
        self.batch_size = batch_size
        self.random_state = random_state
        self.codebooks = []
        self.sub_dim = None

    def fit(self, x):
        d = x.shape[1]
        if d % self.n_codebooks != 0:
            raise ValueError(f"dim {d} not divisible by n_codebooks {self.n_codebooks}")
        self.sub_dim = d // self.n_codebooks
        self.codebooks = []
        for i in range(self.n_codebooks):
            chunk = x[:, i * self.sub_dim : (i + 1) * self.sub_dim].astype(np.float32)
            km = MiniBatchKMeans(
                n_clusters=self.n_centroids,
                batch_size=self.batch_size,
                random_state=self.random_state + i,
                n_init="auto",
                max_iter=200,
            )
            km.fit(chunk)
            self.codebooks.append(km.cluster_centers_.astype(np.float32))
        return self

    def encode(self, x):
        codes = []
        for i, centers in enumerate(self.codebooks):
            chunk = x[:, i * self.sub_dim : (i + 1) * self.sub_dim].astype(np.float32)
            x_norm = np.sum(chunk ** 2, axis=1, keepdims=True)
            c_norm = np.sum(centers ** 2, axis=1)[None, :]
            dist = x_norm + c_norm - 2 * chunk @ centers.T
            labels = np.argmin(dist, axis=1).astype(np.uint16)
            codes.append(labels)
        return np.stack(codes, axis=1)

    def decode(self, codes):
        n = codes.shape[0]
        d = self.n_codebooks * self.sub_dim
        out = np.zeros((n, d), dtype=np.float32)
        for i, centers in enumerate(self.codebooks):
            out[:, i * self.sub_dim : (i + 1) * self.sub_dim] = centers[codes[:, i]]
        return out

    def codebook_bytes(self):
        return sum(c.nbytes for c in self.codebooks)


# ----------------------------
# Main experiment
# ----------------------------

SEEDS = [42, 123, 456]


def main():
    dataset_path = download_and_extract_scifact()

    corpus_path = dataset_path / "corpus.jsonl"
    queries_path = dataset_path / "queries.jsonl"
    qrels_path = dataset_path / "qrels" / "test.tsv"

    print("Loading SciFact...")
    corpus = load_beir_jsonl(corpus_path)
    queries = load_beir_jsonl(queries_path)
    qrels = load_qrels(qrels_path)

    docids = list(corpus.keys())
    qids = [qid for qid in queries.keys() if qid in qrels]

    docs = [
        (corpus[docid].get("title", "") + " " + corpus[docid].get("text", "")).strip()
        for docid in docids
    ]
    query_texts = [queries[qid]["text"] for qid in qids]

    print(f"Documents: {len(docs)}")
    print(f"Queries with qrels: {len(query_texts)}")

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    print("Encoding documents...")
    doc_emb = model.encode(
        docs,
        batch_size=64,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    print("Encoding queries...")
    query_emb = model.encode(
        query_texts,
        batch_size=64,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    raw_bytes = doc_emb.shape[1] * 4
    results = []

    def time_search(q, d, n_runs=3):
        ts = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            search_topk(q, d, topk=10)
            ts.append(time.perf_counter() - t0)
        return min(ts) / len(q) * 1000.0  # ms/query, best-of-n

    def eval_method(name, q, d, bytes_per_embedding, seed, codebook_bytes=0):
        q_n = normalize(q)
        d_n = normalize(d)

        ms_per_query = time_search(q_n, d_n)

        topk = search_topk(q_n, d_n, topk=10)
        r10 = recall_at_k(topk, qids, docids, qrels, k=10)
        n10 = ndcg_at_k(topk, qids, docids, qrels, k=10)

        comp = raw_bytes / bytes_per_embedding

        print(
            f"{name:38s} seed={seed:>4} bytes={bytes_per_embedding:7.2f} "
            f"comp={comp:7.2f}x R@10={r10:.4f} nDCG@10={n10:.4f} "
            f"ms/q={ms_per_query:.3f} cb_kB={codebook_bytes/1024:.1f}"
        )

        results.append({
            "method": name,
            "seed": seed,
            "bytes_per_embedding": bytes_per_embedding,
            "compression": comp,
            "recall_at_10": r10,
            "ndcg_at_10": n10,
            "ms_per_query": ms_per_query,
            "codebook_kb": codebook_bytes / 1024.0,
        })

    # Deterministic baselines (single "seed" since fixed)
    print("\n=== Deterministic baselines ===")
    eval_method("Raw float32", query_emb, doc_emb, raw_bytes, seed=0)

    for k in [32, 64, 96, 128, 192]:
        pca = PCA(n_components=k, random_state=42)
        pca.fit(doc_emb)

        d = pca.transform(doc_emb).astype(np.float32)
        q = pca.transform(query_emb).astype(np.float32)

        eval_method(f"PCA-{k} float32", q, d, k * 4, seed=0)

        dq, scale = quantize_int8(d)
        qq, _ = quantize_int8(q, scale=scale)
        eval_method(
            f"PCA-{k} int8",
            dequantize_int8(qq, scale),
            dequantize_int8(dq, scale),
            k,
            seed=0,
        )

    # Stochastic methods: SERA-VQ (PCA + RVQ) and PQ on PCA-projected vectors.
    # 3 seeds each.
    print("\n=== Stochastic methods (RVQ, PQ) ===")
    for k in [64, 96, 128]:
        pca = PCA(n_components=k, random_state=42)
        pca.fit(doc_emb)
        d_pca = pca.transform(doc_emb).astype(np.float32)
        q_pca = pca.transform(query_emb).astype(np.float32)

        for n_codebooks in [8, 16, 24, 32]:
            for seed in SEEDS:
                rvq = ResidualVectorQuantizer(
                    n_codebooks=n_codebooks,
                    n_centroids=256,
                    batch_size=2048,
                    random_state=seed,
                )
                rvq.fit(d_pca)
                eval_method(
                    f"SERA-VQ-{k}-{n_codebooks}B",
                    rvq.decode(rvq.encode(q_pca)),
                    rvq.decode(rvq.encode(d_pca)),
                    n_codebooks,
                    seed=seed,
                    codebook_bytes=rvq.codebook_bytes(),
                )

            # PQ requires dim divisibility.
            if k % n_codebooks != 0:
                continue
            for seed in SEEDS:
                pq = ProductQuantizer(
                    n_codebooks=n_codebooks,
                    n_centroids=256,
                    batch_size=2048,
                    random_state=seed,
                )
                pq.fit(d_pca)
                eval_method(
                    f"PQ-{k}-{n_codebooks}B",
                    pq.decode(pq.encode(q_pca)),
                    pq.decode(pq.encode(d_pca)),
                    n_codebooks,
                    seed=seed,
                    codebook_bytes=pq.codebook_bytes(),
                )

    csv_path = "results/sera_beir_scifact_results.csv"
    Path("results").mkdir(exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method", "seed", "bytes_per_embedding", "compression",
                "recall_at_10", "ndcg_at_10", "ms_per_query", "codebook_kb",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved {csv_path}")

    # Aggregate per (method, bytes): mean+std nDCG over seeds
    print("\n=== Aggregated nDCG@10 (mean ± std over seeds) ===")
    from collections import defaultdict
    agg = defaultdict(list)
    for r in results:
        agg[(r["method"], r["bytes_per_embedding"])].append(r["ndcg_at_10"])

    rows = []
    for (m, b), vals in agg.items():
        rows.append((m, b, float(np.mean(vals)), float(np.std(vals)), len(vals)))
    rows.sort(key=lambda r: (r[1], -r[2]))
    for m, b, mu, sd, n in rows:
        print(f"  {m:38s} bytes={b:6.0f} nDCG@10={mu:.4f}±{sd:.4f}  (n={n})")


if __name__ == "__main__":
    main()
