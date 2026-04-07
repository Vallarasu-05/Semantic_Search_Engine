#!/usr/bin/env python3
"""
upload_to_hf.py — Upload your semantic_engine folder to Hugging Face Hub
Run this ONCE from your local machine after downloading from Kaggle.

Usage:
    pip install huggingface_hub
    python scripts/upload_to_hf.py \
        --folder /path/to/semantic_engine \
        --repo YOUR_HF_USERNAME/semantic-engine \
        --token hf_xxxxxxxxxxxxxxxxxxxx
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo

def upload(folder: str, repo_id: str, token: str):
    folder = Path(folder)
    api = HfApi(token=token)

    # Create dataset repo (private by default — change private=False to make public)
    print(f"📦 Creating/checking repo: {repo_id}")
    create_repo(
        repo_id=repo_id,
        token=token,
        repo_type="dataset",
        private=True,
        exist_ok=True,
    )

    files_to_upload = [
        "faiss_index.bin",
        "corpus.pkl",
        "bm25.pkl",
    ]

    for fname in files_to_upload:
        fpath = folder / fname
        if not fpath.exists():
            print(f"⚠️  Skipping missing file: {fpath}")
            continue

        size_mb = fpath.stat().st_size / (1024 ** 2)
        print(f"⬆️  Uploading {fname} ({size_mb:.1f} MB) …")

        api.upload_file(
            path_or_fileobj=str(fpath),
            path_in_repo=fname,
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
        )
        print(f"   ✅ Done: {fname}")

    # Upload a README
    readme = f"""# Semantic Engine Artifacts

Artifacts for the Nexus Semantic Search Engine.

| File | Description |
|------|-------------|
| `faiss_index.bin` | FAISS IndexFlatIP — 768-dim dense vectors |
| `corpus.pkl` | List of dicts with `text`, `title`, `url` |
| `bm25.pkl`   | BM25Okapi index + tokenized corpus |

**Model**: `msmarco-bert-base-dot-v5`  
**Reranker**: `cross-encoder/ms-marco-MiniLM-L-12-v2`  
**Dataset**: Wikipedia 20231101.en — 100k articles  
"""
    api.upload_file(
        path_or_fileobj=readme.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
    )

    print(f"\n🎉 All artifacts uploaded to https://huggingface.co/datasets/{repo_id}")
    print(f"   Set HF_REPO_ID={repo_id} in your Render environment variables.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True, help="Path to local semantic_engine folder")
    parser.add_argument("--repo",   required=True, help="HF repo id, e.g. alice/semantic-engine")
    parser.add_argument("--token",  default=os.getenv("HF_TOKEN"), help="HF write token")
    args = parser.parse_args()

    if not args.token:
        raise ValueError("Pass --token or set HF_TOKEN environment variable")

    upload(args.folder, args.repo, args.token)
