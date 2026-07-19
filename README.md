# Multi-Stage RAG Learning System (OKF)

Educational Python backend that progresses through common RAG techniques on a small Open Knowledge Format (OKF) markdown bundle (`data/company_okf_bundle/`):

| Phase | What you learn | Entry points |
|-------|----------------|--------------|
| **1** Vanilla | Structure-aware chunking → Chroma (local MiniLM via fastembed) → stuff → generate | `scripts/demo_stage1.py` |
| **2** GraphRAG | OKF links → Neo4j → N-hop local search → generate | `scripts/build_graph.py`, `scripts/demo_stage2.py` |
| **3** Agentic | LangGraph: retrieve → grade → rewrite (max 2 retries) → generate | `scripts/demo_stage3.py` |
| **4** Advanced | BM25 + vector **RRF** → optional cross-encoder / lexical rerank | `scripts/demo_stage4.py` |
| **5** Routing + cache | Classify → Stage 1/2/3 path; semantic cache before routing | `scripts/demo_stage5.py`, `run_routed_rag` |
| **6** Eval | Trace + RAGAS faithfulness / answer relevancy → CSV | `scripts/run_eval.py` |

**Public app entry (Phases 5+):** `src.cache.semantic_cache.run_routed_rag(query)`

---

## Quick start

```powershell
# From the project root
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

copy .env.example .env
# Edit .env: set OPENAI_API_KEY (and NEO4J_PASSWORD for Phase 2+)
```

### Phase 1 (no Neo4j)

```powershell
python scripts/demo_stage1.py
python scripts/demo_stage1.py --rebuild   # re-ingest OKF into Chroma
```

### Phase 2 (Neo4j)

**Option A — Docker** (Docker Desktop required):

```powershell
# NEO4J_PASSWORD must be set in .env first
docker compose up -d
python scripts/build_graph.py
python scripts/demo_stage2.py
```

**Option B — local Neo4j** install: set `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` in `.env`, start the DB, then run the same `build_graph` / `demo_stage2` commands.

### Phases 3–5

```powershell
python scripts/demo_stage3.py
python scripts/demo_stage4.py
python scripts/demo_stage5.py
```

If HuggingFace rate-limits the reranker ONNX download:

```powershell
$env:RERANK_LEXICAL_ONLY = "1"
```

### Phase 6 (eval)

```powershell
python scripts/run_eval.py --limit 3 --skip-ragas
python scripts/run_eval.py --limit 3
```

Eval questions live in `data/evaluation_questions.json` (also accepts `data/eval_questions.json`).

---

## Layout

```text
├── .env.example          # template only — copy to .env (gitignored)
├── docker-compose.yml    # optional Neo4j; password from .env
├── requirements.txt
├── data/
│   ├── company_okf_bundle/     # knowledge base
│   └── evaluation_questions.json
├── src/
│   ├── config.py
│   ├── loaders/                # OKF frontmatter + links
│   ├── stage1_vanilla/
│   ├── stage2_graphrag/
│   ├── stage3_agentic/
│   ├── stage4_advanced/        # hybrid + rerank composition
│   ├── routing/                # classify + metadata filter + router
│   └── cache/                  # semantic cache + run_routed_rag
├── eval/                       # traces + RAGAS runner
└── scripts/                    # demos + build_graph + run_eval
```

**Layering note:** Stage 1 (`run_vanilla_rag`) is **vector-only**. Hybrid + rerank is Phase 4 (`run_advanced_rag`) and is what the router’s SIMPLE path uses.

---

## Environment

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | LLM generation / classification / RAGAS |
| `OPENAI_MODEL` | Default `gpt-4o-mini` |
| `OPENAI_CLASSIFIER_MODEL` | Optional cheaper router labeler |
| `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` | GraphRAG |
| `RERANKER_MODEL_NAME` | ONNX cross-encoder (optional) |
| `RERANK_LEXICAL_ONLY=1` | Skip HF download; lexical rerank fallback |

Never commit `.env`. Rotate any key that was ever pasted into chat or committed by mistake.

---

## GitHub push checklist

Before `git init` / first push:

1. **Secrets**
   - [ ] `.env` is **not** tracked (listed in `.gitignore`).
   - [ ] `docker-compose.yml` has **no** hard-coded passwords or API keys (uses `${NEO4J_PASSWORD}` from `.env`).
   - [ ] Only placeholders live in `.env.example`.
2. **Local-only artifacts** (already gitignored)
   - [ ] `.venv/`, `data/chroma/`, `data/semantic_cache.json`, `eval/results/`, `__pycache__/`
3. **Repo hygiene**
   - [ ] `README.md`, `.env.example`, `requirements.txt`, `src/`, `scripts/`, `eval/`, `data/company_okf_bundle/`, `data/evaluation_questions.json`
   - [ ] Optional: exclude personal notes under `ai_learning/` if you don’t want them public
4. **Sanity**
   - [ ] Fresh clone path: copy `.env.example` → `.env`, `pip install -r requirements.txt`, `demo_stage1.py` works
   - [ ] If your OpenAI key was exposed anywhere, **rotate it** in the OpenAI dashboard before making the repo public

This project is a **learning / research** codebase, not a production deployment template.
