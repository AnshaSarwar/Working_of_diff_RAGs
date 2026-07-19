"""
Shared configuration for the RAG learning project.

Why a central config?
- Keeps model names, paths, and thresholds in one place so every stage
  (vanilla / graph / agentic / advanced) uses the same defaults.
- Secrets stay in the environment (.env) — never hardcode API keys.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OKF_BUNDLE_DIR = PROJECT_ROOT / "data" / "company_okf_bundle"
CHROMA_PERSIST_DIR = PROJECT_ROOT / "data" / "chroma"
CHROMA_COLLECTION_NAME = "okf_vanilla"

# ---------------------------------------------------------------------------
# Retrieval defaults (Phase 1)
# ---------------------------------------------------------------------------
# top_k=4 is a deliberate learning default: enough context for a short policy
# answer without stuffing the whole corpus into the prompt.
DEFAULT_TOP_K = 4

# ---------------------------------------------------------------------------
# Embeddings — local fastembed (ONNX MiniLM)
# ---------------------------------------------------------------------------
# Why local MiniLM instead of an API embedder?
# - Free / offline — good for a learning project
# - Fast enough on CPU for a small OKF bundle
# - Uses fastembed (ONNX) rather than sentence-transformers/torch so Windows
#   installs don't hit MAX_PATH failures under deep torch license trees
# - Codes like "PTO250" still won't embed perfectly (that's why Phase 4
#   adds hybrid BM25 search) — seeing that failure mode is educational
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# LLM — OpenAI via LangChain
# ---------------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
# Default matches .env / .env.example — override via OPENAI_MODEL if needed.
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ---------------------------------------------------------------------------
# Neo4j — Phase 2 GraphRAG
# ---------------------------------------------------------------------------
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
# Local search: seed entity + N hops of LINKS_TO neighbors.
# 2 hops covers Project Atlas → Dave → Alice for the Phase 2 demo question.
GRAPH_HOP_COUNT = int(os.getenv("GRAPH_HOP_COUNT", "2"))

# ---------------------------------------------------------------------------
# Agentic retry cap (Phase 3) — documented early so the constraint is visible
# ---------------------------------------------------------------------------
MAX_AGENTIC_RETRIES = 2
# Narrow first retrieves so multi-doc questions often miss a policy on
# attempt 1; retries accumulate chunks (see nodes.retrieve). top_k=1 makes
# the Phase 3 demo loop observable on this small OKF bundle.
AGENTIC_TOP_K = int(os.getenv("AGENTIC_TOP_K", "1"))

# ---------------------------------------------------------------------------
# Phase 4 — Hybrid search + reranking (router SIMPLE / agentic; not Stage 1)
# ---------------------------------------------------------------------------
# Fetch this many from each of BM25 and vector before RRF fusion.
HYBRID_CANDIDATE_K = int(os.getenv("HYBRID_CANDIDATE_K", "20"))
# RRF constant (classic; 60 is the Cormack et al. default).
RRF_K = int(os.getenv("RRF_K", "60"))
# Cross-encoder keeps this many after reranking hybrid candidates.
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "5"))
# ONNX cross-encoder via fastembed (no torch). Default is the small MS MARCO
# MiniLM — same role as bge-reranker, much smaller download. Override with
# RERANKER_MODEL_NAME=BAAI/bge-reranker-base when HF rate limits allow.
RERANKER_MODEL_NAME = os.getenv(
    "RERANKER_MODEL_NAME",
    "Xenova/ms-marco-MiniLM-L-6-v2",
)

# ---------------------------------------------------------------------------
# Phase 5 — Tiered routing + semantic cache
# ---------------------------------------------------------------------------
# Optional cheaper OpenAI model for single-label classification
# (falls back to OPENAI_MODEL).
OPENAI_CLASSIFIER_MODEL = os.getenv("OPENAI_CLASSIFIER_MODEL", "") or OPENAI_MODEL
# Cosine similarity threshold for semantic cache hits (0–1).
SEMANTIC_CACHE_THRESHOLD = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.92"))
SEMANTIC_CACHE_PATH = PROJECT_ROOT / "data" / "semantic_cache.json"
