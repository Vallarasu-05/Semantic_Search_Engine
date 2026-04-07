"""
Semantic Search Engine — FastAPI Backend
Model: msmarco-bert-base-dot-v5 | FAISS + BM25 Hybrid | CrossEncoder Reranker
Artifacts loaded from Hugging Face Hub
"""

import os
import time
import pickle
import logging
import asyncio
from contextlib import asynccontextmanager
from typing import List, Optional

import numpy as np
import faiss
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from huggingface_hub import hf_hub_download
import nltk

# ─────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# Config  (override via environment variables on Render)
# ─────────────────────────────────────────────────────────────────
HF_REPO_ID      = os.getenv("HF_REPO_ID", "YOUR_HF_USERNAME/semantic-engine")
HF_TOKEN        = os.getenv("HF_TOKEN", None)          # set in Render env vars
ARTIFACTS_DIR   = os.getenv("ARTIFACTS_DIR", "/tmp/semantic_engine")
MODEL_NAME      = os.getenv("MODEL_NAME", "msmarco-bert-base-dot-v5")
RERANKER_MODEL  = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-12-v2")
DENSE_TOP_K     = int(os.getenv("DENSE_TOP_K", "50"))
BM25_TOP_K      = int(os.getenv("BM25_TOP_K", "30"))
RERANK_TOP_N    = int(os.getenv("RERANK_TOP_N", "20"))
TOP_K           = int(os.getenv("TOP_K", "10"))
HYBRID_ALPHA    = float(os.getenv("HYBRID_ALPHA", "0.65"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────
# Global state (loaded once at startup)
# ─────────────────────────────────────────────────────────────────
state: dict = {}


def tokenize(text: str, stopwords_set: set) -> List[str]:
    tokens = word_tokenize(text.lower())
    return [t for t in tokens if t.isalpha() and t not in stopwords_set and len(t) > 2]


def download_artifact(filename: str) -> str:
    """Download a file from HF Hub if not already cached locally."""
    local_path = os.path.join(ARTIFACTS_DIR, filename)
    if os.path.exists(local_path):
        log.info(f"✅ Using cached artifact: {filename}")
        return local_path
    log.info(f"⬇️  Downloading {filename} from Hugging Face Hub …")
    path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=filename,
        repo_type="dataset",
        token=HF_TOKEN,
        local_dir=ARTIFACTS_DIR,
    )
    log.info(f"✅ Downloaded: {filename}")
    return path


def load_all():
    """Download artifacts from HF Hub and load into memory."""
    nltk.download("punkt_tab", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    stop = set(stopwords.words("english"))

    # 1. Artifacts from HF Hub
    faiss_path  = download_artifact("faiss_index.bin")
    corpus_path = download_artifact("corpus.pkl")
    bm25_path   = download_artifact("bm25.pkl")

    log.info("📂 Loading FAISS index …")
    index = faiss.read_index(faiss_path)
    log.info(f"   ↳ {index.ntotal:,} vectors")

    log.info("📂 Loading corpus …")
    with open(corpus_path, "rb") as f:
        corpus: List[dict] = pickle.load(f)
    log.info(f"   ↳ {len(corpus):,} chunks")

    log.info("📂 Loading BM25 …")
    with open(bm25_path, "rb") as f:
        bd = pickle.load(f)
    bm25 = bd["bm25"]

    log.info(f"🤖 Loading dense model: {MODEL_NAME} …")
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)

    log.info(f"🤖 Loading reranker: {RERANKER_MODEL} …")
    reranker = CrossEncoder(RERANKER_MODEL, max_length=512, device=DEVICE)

    state.update({
        "index":    index,
        "corpus":   corpus,
        "bm25":     bm25,
        "model":    model,
        "reranker": reranker,
        "stop":     stop,
        "ready":    True,
    })
    log.info("🟢 Search engine ready!")


# ─────────────────────────────────────────────────────────────────
# Lifespan (replaces deprecated @app.on_event)
# ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, load_all)
    yield
    log.info("🔴 Shutting down …")


# ─────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Semantic Search Engine",
    description="Hybrid FAISS + BM25 + CrossEncoder reranker over Wikipedia",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


# ─────────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────────
class SearchResult(BaseModel):
    rank:  int
    score: float
    title: str
    url:   str
    snippet: str

class SearchResponse(BaseModel):
    query:    str
    results:  List[SearchResult]
    elapsed:  float
    total_candidates: int


# ─────────────────────────────────────────────────────────────────
# Core search logic
# ─────────────────────────────────────────────────────────────────
def run_search(query: str, k: int = TOP_K) -> SearchResponse:
    if not state.get("ready"):
        raise RuntimeError("Engine not ready yet")

    index    = state["index"]
    corpus   = state["corpus"]
    bm25     = state["bm25"]
    model    = state["model"]
    reranker = state["reranker"]
    stop     = state["stop"]

    t0 = time.time()

    # Stage 1a — Dense retrieval
    qvec = model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    ).astype(np.float32)
    _, dense_idx = index.search(qvec, DENSE_TOP_K)
    dense_ids = dense_idx[0].tolist()

    # Stage 1b — BM25
    q_tokens   = tokenize(query, stop)
    bm25_scores = bm25.get_scores(q_tokens)
    bm25_ids    = np.argsort(bm25_scores)[::-1][:BM25_TOP_K].tolist()

    # Stage 2 — Reciprocal Rank Fusion
    rrf: dict[int, float] = {}
    kk = 60
    for rank, idx in enumerate(dense_ids):
        rrf[idx] = rrf.get(idx, 0.0) + HYBRID_ALPHA * (1.0 / (kk + rank + 1))
    for rank, idx in enumerate(bm25_ids):
        rrf[idx] = rrf.get(idx, 0.0) + (1 - HYBRID_ALPHA) * (1.0 / (kk + rank + 1))

    candidates = [i for i, _ in sorted(rrf.items(), key=lambda x: x[1], reverse=True)]
    candidates = candidates[:RERANK_TOP_N]

    # Stage 3 — CrossEncoder reranker
    pairs  = [(query, corpus[i]["text"]) for i in candidates]
    scores = reranker.predict(pairs, show_progress_bar=False)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:k]

    elapsed = time.time() - t0

    results = [
        SearchResult(
            rank    = r + 1,
            score   = float(s),
            title   = corpus[i]["title"],
            url     = corpus[i]["url"],
            snippet = corpus[i]["text"][:400],
        )
        for r, (i, s) in enumerate(ranked)
    ]

    return SearchResponse(
        query=query,
        results=results,
        elapsed=round(elapsed, 3),
        total_candidates=len(candidates),
    )


# ─────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse("static/index.html")


@app.get("/health")
async def health():
    return {
        "status": "ready" if state.get("ready") else "loading",
        "device": str(DEVICE),
        "corpus_size": len(state.get("corpus", [])),
    }


@app.get("/search", response_model=SearchResponse)
async def search(
    q: str = Query(..., min_length=1, max_length=500, description="Search query"),
    k: int = Query(default=10, ge=1, le=50, description="Number of results"),
):
    if not state.get("ready"):
        raise HTTPException(status_code=503, detail="Engine is still loading, please wait …")
    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, run_search, q, k)
        return response
    except Exception as e:
        log.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/suggest")
async def suggest(q: str = Query(..., min_length=1)):
    """Simple title-based autocomplete from corpus."""
    if not state.get("ready") or not q:
        return {"suggestions": []}
    corpus = state["corpus"]
    q_lower = q.lower()
    seen, suggestions = set(), []
    for doc in corpus:
        t = doc["title"]
        if t not in seen and q_lower in t.lower():
            seen.add(t)
            suggestions.append(t)
        if len(suggestions) >= 8:
            break
    return {"suggestions": suggestions}
