# 🔍 Nexus Semantic Search Engine
### Production deployment on Render.com · Google-style UI · Wikipedia · 100k articles

**Stack**: `msmarco-bert-base-dot-v5` · FAISS · BM25 · CrossEncoder reranker · FastAPI · Render.com

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  FastAPI  (app/main.py)                                         │
│                                                                 │
│  Stage 1a  Dense Retrieval ──── FAISS (top-50)                 │
│  Stage 1b  BM25 Retrieval  ──── rank_bm25 (top-30)             │
│  Stage 2   Hybrid Fusion   ──── Reciprocal Rank Fusion (α=0.65)│
│  Stage 3   Reranking       ──── CrossEncoder (top-20 → top-10) │
│                                                                 │
│  Artifacts loaded from Hugging Face Hub on first boot           │
│  Cached on Render Disk (/tmp/semantic_engine) across deploys    │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
Google-style UI (static/index.html)
```

---

## Step-by-Step Deployment

### Step 1 — Upload artifacts to Hugging Face Hub

Your `semantic_engine/` folder (downloaded from Kaggle) contains:
- `faiss_index.bin` — FAISS vector index
- `corpus.pkl` — document chunks
- `bm25.pkl` — BM25 index

**Create a HF account** at https://huggingface.co if you don't have one.

**Get a write token**: https://huggingface.co/settings/tokens → New token → Write

**Run the upload script:**
```bash
pip install huggingface_hub
python scripts/upload_to_hf.py \
    --folder /path/to/semantic_engine \
    --repo YOUR_HF_USERNAME/semantic-engine \
    --token hf_xxxxxxxxxxxxxxxxxxxx
```

This creates a **private** HF dataset repo and uploads all 3 artifact files.  
Note the `repo_id` (e.g. `alice/semantic-engine`) — you'll need it in Step 3.

---

### Step 2 — Push this project to GitHub

```bash
git init
git add .
git commit -m "Initial deploy: Nexus Semantic Search"
git remote add origin https://github.com/YOUR_USERNAME/nexus-search.git
git push -u origin main
```

---

### Step 3 — Deploy on Render.com

1. Go to https://render.com → New → **Blueprint**
2. Connect your GitHub repo
3. Render detects `render.yaml` automatically

**Set these environment variables in the Render Dashboard:**

| Key | Value |
|-----|-------|
| `HF_REPO_ID` | `YOUR_HF_USERNAME/semantic-engine` |
| `HF_TOKEN` | `hf_xxxxxxxxxxxxxxxxxxxx` (your HF read token) |

> All other variables are already set in `render.yaml`.

4. Click **Apply** — Render builds the Docker image and deploys.

**First boot takes ~5–10 minutes** (downloads 5GB+ artifacts + loads models).  
Subsequent boots are fast because artifacts are cached on the persistent disk.

---

### Step 4 — Verify

```bash
# Health check
curl https://your-app.onrender.com/health

# Test search
curl "https://your-app.onrender.com/search?q=quantum+computing&k=5"

# Open the UI
open https://your-app.onrender.com
```

---

## Local Development

```bash
# Clone and install
pip install -r requirements.txt

# Set env vars
export HF_REPO_ID=YOUR_HF_USERNAME/semantic-engine
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx

# Run
uvicorn app.main:app --reload --port 8000

# Open
open http://localhost:8000
```

---

## API Reference

### `GET /search`
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `q`   | str  | required | Search query |
| `k`   | int  | 10 | Number of results (1–50) |

**Response:**
```json
{
  "query": "quantum computing",
  "elapsed": 0.423,
  "total_candidates": 20,
  "results": [
    {
      "rank": 1,
      "score": 9.812,
      "title": "Quantum computing",
      "url": "https://en.wikipedia.org/wiki/Quantum_computing",
      "snippet": "Quantum computing is a type of computation …"
    }
  ]
}
```

### `GET /suggest?q=<prefix>`
Returns up to 8 title autocomplete suggestions.

### `GET /health`
Returns engine status, device, and corpus size.

---

## Render Plan Recommendations

| Plan | RAM | Cost | Suitability |
|------|-----|------|-------------|
| Starter (free) | 512 MB | $0 | ⚠️ May OOM with 5GB model — use for testing |
| Starter+ | 1 GB | ~$7/mo | ✅ Works if corpus fits |
| Standard | 2 GB | ~$25/mo | ✅ Recommended for production |

> **Tip**: On the free tier, set `DENSE_TOP_K=20`, `BM25_TOP_K=15`, `RERANK_TOP_N=10` to reduce memory usage.

---

## Project Structure

```
semantic-search-deploy/
├── app/
│   └── main.py           ← FastAPI backend (search logic)
├── static/
│   └── index.html        ← Google-style frontend
├── scripts/
│   └── upload_to_hf.py   ← One-time artifact uploader
├── Dockerfile            ← Docker build config
├── render.yaml           ← Render Blueprint spec
├── requirements.txt      ← Python dependencies
└── README.md             ← This file
```
