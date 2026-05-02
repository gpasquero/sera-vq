import os
import csv
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
                # old format: qid, _, docid, score
                qid, _, docid, score = parts
            elif len(parts) == 3:
                # new format: qid, docid, score
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
    """
    Brute-force cosine search.
    Assumes embeddings are already normalized.
    """
    results = {}

    doc_matrix = doc_emb.T

    for start in tqdm(range(0, len(query_emb), batch_size), desc="Searching"):
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
            print(f"  training codebook {i + 1}/{self.n_codebooks}")
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


# ----------------------------
# Main experiment
# ----------------------------

def main():
    dataset_path = download_and_extract_scifact()

    corpus_path = dataset_path / "corpus.jsonl"
    queries_path = dataset_path / "queries.jsonl"
    qrels_path = dataset_path / "qrels" / "test.tsv"

    print("Loading SciFact files...")
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

    def eval_method(name, q, d, bytes_per_embedding):
        q = normalize(q)
        d = normalize(d)

        topk = search_topk(q, d, topk=10)

        r10 = recall_at_k(topk, qids, docids, qrels, k=10)
        n10 = ndcg_at_k(topk, qids, docids, qrels, k=10)

        comp = raw_bytes / bytes_per_embedding

        print(
            f"{name:35s} "
            f"bytes={bytes_per_embedding:8.2f} "
            f"comp={comp:8.2f} "
            f"Recall@10={r10:.4f} "
            f"nDCG@10={n10:.4f}"
        )

        results.append({
            "method": name,
            "bytes_per_embedding": bytes_per_embedding,
            "compression": comp,
            "recall_at_10": r10,
            "ndcg_at_10": n10,
        })

    print("\nEvaluating raw...")
    eval_method(
        "Raw float32",
        query_emb,
        doc_emb,
        raw_bytes,
    )

    # PCA + int8 baselines
    for k in [32, 64, 96, 128, 192]:
        print(f"\nPCA baseline k={k}")
        pca = PCA(n_components=k, random_state=42)
        pca.fit(doc_emb)

        d = pca.transform(doc_emb).astype(np.float32)
        q = pca.transform(query_emb).astype(np.float32)

        eval_method(
            f"PCA-{k} float32",
            q,
            d,
            k * 4,
        )

        dq, scale = quantize_int8(d)
        qq, _ = quantize_int8(q, scale=scale)

        d_deq = dequantize_int8(dq, scale)
        q_deq = dequantize_int8(qq, scale)

        eval_method(
            f"PCA-{k} int8",
            q_deq,
            d_deq,
            k,
        )

    # SERA-VQ
    for k in [64, 96, 128]:
        print(f"\nSERA-VQ PCA space k={k}")
        pca = PCA(n_components=k, random_state=42)
        pca.fit(doc_emb)

        d_train = pca.transform(doc_emb).astype(np.float32)
        d = d_train
        q = pca.transform(query_emb).astype(np.float32)

        for n_codebooks in [8, 16, 24, 32]:
            print(f"\nTraining RVQ: PCA-{k}, codebooks={n_codebooks}")
            rvq = ResidualVectorQuantizer(
                n_codebooks=n_codebooks,
                n_centroids=256,
                batch_size=2048,
                random_state=42,
            )

            rvq.fit(d_train)

            d_codes = rvq.encode(d)
            q_codes = rvq.encode(q)

            d_dec = rvq.decode(d_codes)
            q_dec = rvq.decode(q_codes)

            eval_method(
                f"SERA-VQ-{k}-{n_codebooks}B",
                q_dec,
                d_dec,
                n_codebooks,
            )

    csv_path = "sera_beir_scifact_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "bytes_per_embedding",
                "compression",
                "recall_at_10",
                "ndcg_at_10",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved {csv_path}")

    print("\nBest under budgets")
    for budget in [16, 32, 64, 96, 128]:
        print(f"\nBest under {budget} bytes")
        print("=" * 90)
        filtered = [r for r in results if r["bytes_per_embedding"] <= budget]
        filtered = sorted(filtered, key=lambda r: r["ndcg_at_10"], reverse=True)

        for r in filtered[:10]:
            print(
                f"{r['method']:35s} "
                f"{r['bytes_per_embedding']:8.2f}B "
                f"{r['compression']:8.2f}x "
                f"R@10={r['recall_at_10']:.4f} "
                f"nDCG@10={r['ndcg_at_10']:.4f}"
            )


if __name__ == "__main__":
    main()
