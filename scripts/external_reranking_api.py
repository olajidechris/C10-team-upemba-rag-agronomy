"""
Extracted from old/team-upemba-rag-agronomy-external reranking api.ipynb.
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
from sentence_transformers import SentenceTransformer
import voyageai

# Securely load API Key from Kaggle Secrets
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    VOYAGE_KEY = user_secrets.get_secret("VOYAGE_API_KEY")
except Exception:
    print("⚠️ Kaggle Secrets not found. Falling back to environment variable.")
    VOYAGE_KEY = os.environ.get("VOYAGE_API_KEY", "")

# ==========================================
# 1. CONFIGURATION & CACHE DIRECTORIES
# ==========================================
class Config:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    REPO_ROOT = os.path.dirname(SCRIPT_DIR)
    LOCAL_INPUT_DEFAULT = os.path.join(REPO_ROOT, "data", "input")
    KAGGLE_INPUT_DEFAULT = "/kaggle/input/competitions/agricultural-extension-rag-smart-retrieval-for-farmers/"

    # Dataset Paths
    BASE_PATH = os.environ.get("AGRONOMY_INPUT_PATH") or (
        LOCAL_INPUT_DEFAULT
        if os.path.exists(os.path.join(LOCAL_INPUT_DEFAULT, "documents.csv"))
        else KAGGLE_INPUT_DEFAULT
    )
    LOCAL_OUTPUT_DEFAULT = os.path.join(REPO_ROOT, "data", "output")
    KAGGLE_OUTPUT_DEFAULT = "/kaggle/working/"
    OUTPUT_DIR = os.environ.get("AGRONOMY_OUTPUT_PATH") or (
        KAGGLE_OUTPUT_DEFAULT if os.path.isdir("/kaggle/working") else LOCAL_OUTPUT_DEFAULT
    )

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
    VOYAGE_MODEL_NAME = "rerank-3-lite" # Ultra-fast, state-of-the-art accuracy

    # Retrieval Hyperparameters
    DENSE_TOP_K = 40
    BM25_TOP_K = 40
    RRF_K = 60
    CANDIDATES_POOL_SIZE = 50 # Send the top 50 hybrid candidates to Voyage
    FINAL_TOP_K = 5

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

os.makedirs(Config.CACHE_DIR, exist_ok=True)
os.makedirs(Config.MODEL_CACHE_DIR, exist_ok=True)


# ==========================================
# 2. CACHED BM25 LEXICAL RETRIEVER
# ==========================================
class CachedBM25Retriever:
    """Memory-efficient BM25Okapi with disk caching."""
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1; self.b = b
        self.corpus_size = 0; self.avgdl = 0.0
        self.doc_freqs = []; self.idf = {}; self.doc_len = []; self.doc_ids = []

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return re.findall(r"\b\w+\b", str(text).lower())

    def fit(self, doc_ids: List[str], docs: List[str], force_recompute: bool = False):
        if not force_recompute and os.path.exists(Config.BM25_CACHE_PATH):
            print(f"📦 Loading cached BM25 index from: {Config.BM25_CACHE_PATH}")
            with open(Config.BM25_CACHE_PATH, "rb") as f:
                state = pickle.load(f)
            cached_doc_ids = state.get("doc_ids", [])
            if cached_doc_ids == doc_ids:
                self.__dict__.update(state)
                return
            print("♻️ BM25 cache mismatch detected; rebuilding index for current corpus.")

        print("⚡ Fitting BM25 index from corpus...")
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
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
            pickle.dump({k: v for k, v in self.__dict__.items() if k != "tokenize"}, f)
        print(f"💾 BM25 index saved to: {Config.BM25_CACHE_PATH}")

    def search(self, query: str, top_k: int = 40) -> List[Tuple[str, float]]:
        tokens = self.tokenize(query)
        scores = np.zeros(self.corpus_size)

        for token in tokens:
            if token not in self.idf: continue
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
        self.model = SentenceTransformer(model_name, device=device, cache_folder=Config.MODEL_CACHE_DIR)
        self.doc_ids = []; self.doc_embeddings = None

    def fit(self, doc_ids: List[str], docs: List[str], force_recompute: bool = False):
        cache_exists = os.path.exists(Config.EMBEDDINGS_CACHE_PATH) and os.path.exists(Config.DOC_IDS_CACHE_PATH)
        if not force_recompute and cache_exists:
            with open(Config.DOC_IDS_CACHE_PATH, "rb") as f:
                cached_doc_ids = pickle.load(f)
            if cached_doc_ids == doc_ids:
                print(f"📦 Loading cached embeddings from: {Config.EMBEDDINGS_CACHE_PATH}")
                self.doc_embeddings = torch.load(Config.EMBEDDINGS_CACHE_PATH, map_location=self.device)
                self.doc_ids = cached_doc_ids
                return
            print("♻️ Embedding cache mismatch detected; recomputing for current corpus.")

        print("⚡ Computing Nomic embeddings for document corpus...")
        prefixed_docs = [f"search_document: {doc}" for doc in docs]
        self.doc_ids = doc_ids
        self.doc_embeddings = self.model.encode(prefixed_docs, show_progress_bar=True, normalize_embeddings=True, convert_to_tensor=True, device=self.device)

        torch.save(self.doc_embeddings, Config.EMBEDDINGS_CACHE_PATH)
        with open(Config.DOC_IDS_CACHE_PATH, "wb") as f:
            pickle.dump(self.doc_ids, f)
        print(f"💾 Document embeddings saved to: {Config.EMBEDDINGS_CACHE_PATH}")

    def search(self, query: str, top_k: int = 40) -> List[Tuple[str, float]]:
        query_embedding = self.model.encode([f"search_query: {query}"], normalize_embeddings=True, convert_to_tensor=True, device=self.device)
        sim_scores = torch.matmul(self.doc_embeddings, query_embedding.T).squeeze(-1)
        top_scores, top_indices = torch.topk(sim_scores, k=min(top_k, len(self.doc_ids)))
        return [(self.doc_ids[idx], float(score)) for idx, score in zip(top_indices.cpu().numpy(), top_scores.cpu().numpy())]


# ==========================================
# 4. RRF FUSION & VOYAGE API ENGINE
# ==========================================
class RAGRetrievalEngine:
    def __init__(self, bm25: CachedBM25Retriever, dense: CachedNomicRetriever, voyage_client: voyageai.Client, doc_lookup: Dict[str, str]):
        self.bm25 = bm25
        self.dense = dense
        self.voyage = voyage_client
        self.doc_lookup = doc_lookup

    def reciprocal_rank_fusion(self, bm25_results: List[Tuple[str, float]], dense_results: List[Tuple[str, float]], k: int = 60) -> List[str]:
        rrf_scores = Counter()
        for rank, (doc_id, _) in enumerate(bm25_results): rrf_scores[doc_id] += 1.0 / (k + (rank + 1))
        for rank, (doc_id, _) in enumerate(dense_results): rrf_scores[doc_id] += 1.0 / (k + (rank + 1))
        return [doc_id for doc_id, _ in rrf_scores.most_common(Config.CANDIDATES_POOL_SIZE)]

    def retrieve_and_rerank(self, query: str) -> List[str]:
        # 1. Candidate Retrieval
        bm25_candidates = self.bm25.search(query, top_k=Config.BM25_TOP_K)
        dense_candidates = self.dense.search(query, top_k=Config.DENSE_TOP_K)

        # 2. Reciprocal Rank Fusion Pooling (Top 50)
        candidate_ids = self.reciprocal_rank_fusion(bm25_candidates, dense_candidates, k=Config.RRF_K)

        # 3. Compile Candidate Texts for Voyage
        cross_docs = [self.doc_lookup[doc_id] for doc_id in candidate_ids]

        try:
            # 4. Voyage AI Re-Ranking Call
            response = self.voyage.rerank(
                query=query,
                documents=cross_docs,
                model=Config.VOYAGE_MODEL_NAME,
                top_k=Config.FINAL_TOP_K,
                truncation=True
            )
            
            # Map the returned Voyage indices back to our internal document IDs
            return [candidate_ids[r.index] for r in response.results]
            
        except Exception as e:
            print(f"⚠️ Voyage API Error on query: {e}")
            # Safe Fallback: If API fails/rate-limits, return RRF top 5
            return candidate_ids[:Config.FINAL_TOP_K]


# ==========================================
# 5. METRICS: GRADED nDCG@5 EVALUATOR
# ==========================================
def clean_id(raw_id) -> str:
    return str(raw_id).strip().replace(".0", "")

def compute_ndcg_at_5(predicted_rankings: Dict[str, List[str]], qrels_df: pd.DataFrame) -> float:
    qrels_dict = {}
    for _, row in qrels_df.iterrows():
        qid = clean_id(row["query_id"])
        did = clean_id(row["document_id"])
        if qid not in qrels_dict: qrels_dict[qid] = {}
        qrels_dict[qid][did] = float(row["relevance"])

    ndcg_scores = []; discounts = [math.log2(i + 2) for i in range(5)]
    for qid, preds in predicted_rankings.items():
        if qid not in qrels_dict: continue
        dcg = sum((2**qrels_dict[qid].get(doc_id, 0.0) - 1.0) / discounts[i] for i, doc_id in enumerate(preds[:5]))
        idcg = sum((2**rel - 1.0) / discounts[i] for i, rel in enumerate(sorted(qrels_dict[qid].values(), reverse=True)[:5]))
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0


# ==========================================
# 6. MAIN EXECUTION PIPELINE
# ==========================================
def main():
    if not VOYAGE_KEY:
        raise ValueError("VOYAGE_API_KEY is required for scripts/external_reranking_api.py")

    print(f"--- Initialization ---")
    print(f"Device: {Config.DEVICE} | Voyage API Client Status: Ready")
    
    print("\n--- 1. Loading Corpus & Datasets ---")
    docs_df = pd.read_csv(Config.DOCS_PATH, dtype=str)
    train_queries_df = pd.read_csv(Config.TRAIN_QUERIES_PATH, dtype=str)
    qrels_df = pd.read_csv(Config.QRELS_PATH, dtype=str)
    test_queries_df = pd.read_csv(Config.TEST_QUERIES_PATH, dtype=str)

    docs_df["full_text"] = (
        "Title: " + docs_df["title"].fillna("") + 
        " | Crop: " + docs_df["crop"].fillna("(general)") + 
        " | Body: " + docs_df["text"].fillna("")
    )
    
    doc_ids = [clean_id(did) for did in docs_df["document_id"]]
    doc_texts = docs_df["full_text"].tolist()
    doc_lookup = dict(zip(doc_ids, doc_texts))
    print(f"Total documents loaded: {len(docs_df)}")

    print("\n--- 2. Initializing Local Candidate Retrievers ---")
    bm25 = CachedBM25Retriever()
    bm25.fit(doc_ids, doc_texts, force_recompute=False)

    dense = CachedNomicRetriever(Config.DENSE_MODEL_NAME, device=Config.DEVICE)
    dense.fit(doc_ids, doc_texts, force_recompute=False)

    print("\n--- 3. Orchestrating Voyage AI API Integration ---")
    voyage_client = voyageai.Client(api_key=VOYAGE_KEY)
    pipeline = RAGRetrievalEngine(bm25, dense, voyage_client, doc_lookup)

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
    # Test queries iterate silently; API handles the batching over HTTP
    for _, row in test_queries_df.iterrows():
        qid = clean_id(row["query_id"])
        top_5_docs = pipeline.retrieve_and_rerank(str(row["query"]))
        for doc_id in top_5_docs:
            test_rows.append({"QueryId": qid, "DocumentId": doc_id})

    submission_df = pd.DataFrame(test_rows)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    
    assert len(submission_df) == len(test_queries_df) * 5, "Row count mismatch!"
    print(f"✅ Generated final Kaggle payload '{Config.SUBMISSION_PATH}' ({len(submission_df)} rows).")


if __name__ == "__main__":
    main()
