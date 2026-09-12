"""
Extracted from old/team-upemba-rag-agronomy-all local.ipynb.
Notebook-only commands were omitted for script compatibility.
"""

import os
import re
import math
import pickle
from collections import Counter
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder

# ==========================================
# 1. CONFIGURATION & CACHE DIRECTORIES
# ==========================================
class Config:
    # Dataset Paths
    BASE_PATH = os.environ.get(
        "AGRONOMY_INPUT_PATH",
        "/kaggle/input/competitions/agricultural-extension-rag-smart-retrieval-for-farmers/",
    )
    OUTPUT_DIR = os.environ.get("AGRONOMY_OUTPUT_PATH", "/kaggle/working/")

    DOCS_PATH = os.path.join(BASE_PATH, "documents.csv")
    TRAIN_QUERIES_PATH = os.path.join(BASE_PATH, "train_queries.csv")
    QRELS_PATH = os.path.join(BASE_PATH, "qrels_train.csv")
    TEST_QUERIES_PATH = os.path.join(BASE_PATH, "test_queries.csv")
    
    SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "submission.csv")


    # Local Cache Folders & Files
    CACHE_DIR = os.path.join(OUTPUT_DIR, "pipeline_cache")
    MODEL_CACHE_DIR = os.path.join(CACHE_DIR, "models")
    EMBEDDINGS_CACHE_PATH = os.path.join(CACHE_DIR, "nomic_doc_embeddings.pt")
    DOC_IDS_CACHE_PATH = os.path.join(CACHE_DIR, "doc_ids.pkl")
    BM25_CACHE_PATH = os.path.join(CACHE_DIR, "bm25_index.pkl")

    # Models
    DENSE_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
    RERANKER_MODEL_NAME = "mixedbread-ai/mxbai-rerank-large-v1"

    # Retrieval Hyperparameters
    DENSE_TOP_K = 40
    BM25_TOP_K = 40
    RRF_K = 60
    CANDIDATES_POOL_SIZE = 50
    FINAL_TOP_K = 5

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(Config.CACHE_DIR, exist_ok=True)
os.makedirs(Config.MODEL_CACHE_DIR, exist_ok=True)
print(f"Device: {Config.DEVICE} | Cache directory: {Config.CACHE_DIR}")


# ==========================================
# 2. CACHED BM25 LEXICAL RETRIEVER
# ==========================================
class CachedBM25Retriever:
    """Memory-efficient BM25Okapi with disk caching."""
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avgdl = 0.0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.doc_ids = []

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return re.findall(r"\b\w+\b", str(text).lower())

    def fit(self, doc_ids: List[str], docs: List[str], force_recompute: bool = False):
        if not force_recompute and os.path.exists(Config.BM25_CACHE_PATH):
            print(f"📦 Loading cached BM25 index from: {Config.BM25_CACHE_PATH}")
            with open(Config.BM25_CACHE_PATH, "rb") as f:
                state = pickle.load(f)
                self.k1 = state["k1"]
                self.b = state["b"]
                self.corpus_size = state["corpus_size"]
                self.avgdl = state["avgdl"]
                self.doc_freqs = state["doc_freqs"]
                self.idf = state["idf"]
                self.doc_len = state["doc_len"]
                self.doc_ids = state["doc_ids"]
            return

        print("⚡ Fitting BM25 index from corpus...")
        self.doc_ids = doc_ids
        self.corpus_size = len(docs)
        total_len = 0
        df = Counter()

        for doc in docs:
            tokens = self.tokenize(doc)
            self.doc_len.append(len(tokens))
            total_len += len(tokens)
            frequencies = Counter(tokens)
            self.doc_freqs.append(frequencies)
            for word in frequencies.keys():
                df[word] += 1

        self.avgdl = total_len / self.corpus_size if self.corpus_size > 0 else 0.0

        for word, freq in df.items():
            self.idf[word] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)

        with open(Config.BM25_CACHE_PATH, "wb") as f:
            pickle.dump({
                "k1": self.k1,
                "b": self.b,
                "corpus_size": self.corpus_size,
                "avgdl": self.avgdl,
                "doc_freqs": self.doc_freqs,
                "idf": self.idf,
                "doc_len": self.doc_len,
                "doc_ids": self.doc_ids
            }, f)
        print(f"💾 BM25 index saved to: {Config.BM25_CACHE_PATH}")

    def search(self, query: str, top_k: int = 40) -> List[Tuple[str, float]]:
        tokens = self.tokenize(query)
        scores = np.zeros(self.corpus_size)

        for token in tokens:
            if token not in self.idf:
                continue
            idf_val = self.idf[token]
            for i, doc_freq in enumerate(self.doc_freqs):
                tf = doc_freq.get(token, 0)
                if tf > 0:
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * (self.doc_len[i] / self.avgdl))
                    scores[i] += idf_val * (numerator / denominator)

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.doc_ids[idx], float(scores[idx])) for idx in top_indices]


# ==========================================
# 3. CACHED NOMIC DENSE RETRIEVER
# ==========================================
class CachedNomicRetriever:
    """Dense retriever using nomic-embed-text-v1.5 with prefixing and disk caching."""
    def __init__(self, model_name: str, device: str = "cpu"):
        self.device = device
        print(f"📥 Loading dense model weights ({model_name})...")
        self.model = SentenceTransformer(
            model_name,
            device=device,
            trust_remote_code=True,
            cache_folder=Config.MODEL_CACHE_DIR
        )
        self.doc_ids = []
        self.doc_embeddings = None

    def fit(self, doc_ids: List[str], docs: List[str], batch_size: int = 32, force_recompute: bool = False):
        cache_exists = os.path.exists(Config.EMBEDDINGS_CACHE_PATH) and os.path.exists(Config.DOC_IDS_CACHE_PATH)

        if not force_recompute and cache_exists:
            print(f"📦 Loading cached embeddings from: {Config.EMBEDDINGS_CACHE_PATH}")
            self.doc_embeddings = torch.load(Config.EMBEDDINGS_CACHE_PATH, map_location=self.device)
            with open(Config.DOC_IDS_CACHE_PATH, "rb") as f:
                self.doc_ids = pickle.load(f)
            return

        print("⚡ Computing Nomic embeddings for document corpus...")
        # Nomic requires "search_document: " prefix for indexing
        prefixed_docs = [f"search_document: {doc}" for doc in docs]

        self.doc_ids = doc_ids
        self.doc_embeddings = self.model.encode(
            prefixed_docs,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_tensor=True,
            device=self.device
        )

        torch.save(self.doc_embeddings, Config.EMBEDDINGS_CACHE_PATH)
        with open(Config.DOC_IDS_CACHE_PATH, "wb") as f:
            pickle.dump(self.doc_ids, f)
        print(f"💾 Document embeddings saved to: {Config.EMBEDDINGS_CACHE_PATH}")

    def search(self, query: str, top_k: int = 40) -> List[Tuple[str, float]]:
        # Nomic requires "search_query: " prefix for retrieval
        prefixed_query = f"search_query: {query}"

        query_embedding = self.model.encode(
            [prefixed_query],
            normalize_embeddings=True,
            convert_to_tensor=True,
            device=self.device
        )

        sim_scores = torch.matmul(self.doc_embeddings, query_embedding.T).squeeze(-1)
        top_scores, top_indices = torch.topk(sim_scores, k=min(top_k, len(self.doc_ids)))

        return [(self.doc_ids[idx], float(score)) for idx, score in zip(top_indices.cpu().numpy(), top_scores.cpu().numpy())]


# ==========================================
# 4. RRF FUSION & RE-RANKING ENGINE
# ==========================================
class RAGRetrievalEngine:
    def __init__(self, 
                 bm25: CachedBM25Retriever, 
                 dense: CachedNomicRetriever, 
                 reranker: CrossEncoder, 
                 doc_lookup: Dict[str, str]):
        self.bm25 = bm25
        self.dense = dense
        self.reranker = reranker
        self.doc_lookup = doc_lookup

    def reciprocal_rank_fusion(self, 
                               bm25_results: List[Tuple[str, float]], 
                               dense_results: List[Tuple[str, float]], 
                               k: int = 60
                              ) -> List[str]:
        rrf_scores = Counter()

        for rank, (doc_id, _) in enumerate(bm25_results):
            rrf_scores[doc_id] += 1.0 / (k + (rank + 1))

        for rank, (doc_id, _) in enumerate(dense_results):
            rrf_scores[doc_id] += 1.0 / (k + (rank + 1))

        fused_sorted = [doc_id for doc_id, _ in rrf_scores.most_common(Config.CANDIDATES_POOL_SIZE)]
        return fused_sorted

    def retrieve_and_rerank(self, query: str) -> List[str]:
        # 1. Candidate Retrieval
        bm25_candidates = self.bm25.search(query, top_k=Config.BM25_TOP_K)
        dense_candidates = self.dense.search(query, top_k=Config.DENSE_TOP_K)

        # 2. Reciprocal Rank Fusion Pooling
        candidate_ids = self.reciprocal_rank_fusion(bm25_candidates, dense_candidates, k=Config.RRF_K)

        # 3. Cross-Encoder Re-ranking
        cross_inputs = [[query, self.doc_lookup[doc_id]] for doc_id in candidate_ids]
        rerank_scores = self.reranker.predict(cross_inputs, show_progress_bar=False)

        # 4. Top-5 Extraction
        ranked_pairs = sorted(zip(candidate_ids, rerank_scores), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in ranked_pairs[:Config.FINAL_TOP_K]]


# ==========================================
# 5. METRICS: GRADED nDCG@5 EVALUATOR
# ==========================================
def clean_id(raw_id) -> str:
    """Normalizes IDs to prevent float-to-string mismatches (e.g., '1.0' vs '1')."""
    return str(raw_id).strip().replace(".0", "")

def compute_ndcg_at_5(predicted_rankings: Dict[str, List[str]], qrels_df: pd.DataFrame) -> float:
    qrels_dict = {}
    for _, row in qrels_df.iterrows():
        qid = clean_id(row["query_id"])
        did = clean_id(row["document_id"])
        rel = float(row["relevance"])
        if qid not in qrels_dict:
            qrels_dict[qid] = {}
        qrels_dict[qid][did] = rel

    ndcg_scores = []
    discounts = [math.log2(i + 2) for i in range(5)]

    for qid, preds in predicted_rankings.items():
        if qid not in qrels_dict:
            continue
        
        # DCG@5
        dcg = 0.0
        for i, doc_id in enumerate(preds[:5]):
            rel = qrels_dict[qid].get(doc_id, 0.0)
            dcg += (2**rel - 1.0) / discounts[i]

        # Ideal DCG@5
        all_rels = sorted(qrels_dict[qid].values(), reverse=True)
        idcg = 0.0
        for i, rel in enumerate(all_rels[:5]):
            idcg += (2**rel - 1.0) / discounts[i]

        if idcg > 0:
            ndcg_scores.append(dcg / idcg)
        else:
            ndcg_scores.append(0.0)
            
    # Fail gracefully if still empty
    if not ndcg_scores:
        return 0.0

    return float(np.mean(ndcg_scores))

# ==========================================
# 6. MAIN EXECUTION PIPELINE
# ==========================================
def main():
    print("--- 1. Loading Corpus & Datasets ---")
    # Enforcing dtype=str prevents pandas from aggressively casting IDs to floats
    docs_df = pd.read_csv(Config.DOCS_PATH, dtype=str)
    train_queries_df = pd.read_csv(Config.TRAIN_QUERIES_PATH, dtype=str)
    qrels_df = pd.read_csv(Config.QRELS_PATH, dtype=str)
    test_queries_df = pd.read_csv(Config.TEST_QUERIES_PATH, dtype=str)

    docs_df["full_text"] = (
        "Title: " + docs_df["title"].fillna("") + 
        " | Crop: " + docs_df["crop"].fillna("(general)") + 
        " | Body: " + docs_df["text"].fillna("")
    )
    
    # Bulletproof ID extraction
    doc_ids = [clean_id(did) for did in docs_df["document_id"]]
    doc_texts = docs_df["full_text"].tolist()
    doc_lookup = dict(zip(doc_ids, doc_texts))
    print(f"Total documents: {len(docs_df)}")

    print("\n--- 2. Initializing Retrievers (with Cache Check) ---")
    bm25 = CachedBM25Retriever()
    bm25.fit(doc_ids, doc_texts)

    dense = CachedNomicRetriever(Config.DENSE_MODEL_NAME, device=Config.DEVICE)
    dense.fit(doc_ids, doc_texts)

    print("\n--- 3. Loading Cross-Encoder ---")
    reranker = CrossEncoder(
        Config.RERANKER_MODEL_NAME,
        device=Config.DEVICE,
        cache_folder=Config.MODEL_CACHE_DIR
    )
    
    pipeline = RAGRetrievalEngine(bm25, dense, reranker, doc_lookup)

    print("\n--- 4. Evaluating on Training Set ---")
    val_predictions = {}
    for _, row in train_queries_df.iterrows():
        qid = clean_id(row["query_id"])
        val_predictions[qid] = pipeline.retrieve_and_rerank(str(row["query"]))

    cv_score = compute_ndcg_at_5(val_predictions, qrels_df)
    print("==================================================")
    print(f"🎯 Validation nDCG@5 Score: {cv_score:.4f}")
    print(f"Baseline to Beat: ~0.5510")
    print("==================================================")

    print("\n--- 5. Generating Test Predictions ---")
    test_rows = []
    for _, row in test_queries_df.iterrows():
        qid = clean_id(row["query_id"])
        top_5_docs = pipeline.retrieve_and_rerank(str(row["query"]))
        for doc_id in top_5_docs:
            test_rows.append({"QueryId": qid, "DocumentId": doc_id})

    submission_df = pd.DataFrame(test_rows)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    
    assert len(submission_df) == len(test_queries_df) * 5, "Row count mismatch!"
    assert list(submission_df.columns) == ["QueryId", "DocumentId"], "Columns mismatch!"
    print(f"✅ Generated '{Config.SUBMISSION_PATH}' ({len(submission_df)} rows).")

if __name__ == "__main__":
    main()
