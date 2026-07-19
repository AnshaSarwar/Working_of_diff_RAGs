"""
Phase 6 — RAGAS evaluation runner.

Runs the fixed eval set through `run_routed_rag` (with tracing) and scores:
  - faithfulness  — is the answer supported by retrieved context?
  - answer relevancy (reported as "relevance") — does the answer address the question?

Also writes the summary CSV:
  query | route_taken | latency_ms | faithfulness | relevance | techniques_fired
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import OPENAI_API_KEY, OPENAI_MODEL, PROJECT_ROOT
from eval.trace_logger import TraceLogger, default_results_dir


def load_eval_questions(path: Path | None = None) -> list[dict[str, Any]]:
    candidates = [
        path,
        PROJECT_ROOT / "data" / "eval_questions.json",
        PROJECT_ROOT / "data" / "evaluation_questions.json",
    ]
    for p in candidates:
        if p is None:
            continue
        if p.is_file() and p.stat().st_size > 2:
            data = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(data, list) or not data:
                raise ValueError(f"Eval file must be a non-empty JSON list: {p}")
            return data
    raise FileNotFoundError(
        "No eval questions found. Add data/eval_questions.json "
        "(or data/evaluation_questions.json)."
    )


def _patch_ragas_langchain_compat() -> None:
    """
    ragas 0.4.x imports langchain_community.chat_models.vertexai at import
    time, but newer langchain-community removed that module. Stub it so ragas
    ragas can load; we inject ChatOpenAI via LangchainLLMWrapper.
    """
    import sys
    import types

    name = "langchain_community.chat_models.vertexai"
    if name in sys.modules:
        return
    try:
        __import__(name)
        return
    except ModuleNotFoundError:
        pass

    stub = types.ModuleType(name)

    class ChatVertexAI:  # noqa: N801 — match upstream symbol
        pass

    stub.ChatVertexAI = ChatVertexAI
    sys.modules[name] = stub


def _build_ragas_llm():
    """RAGAS judge LLM — reuse OpenAI so we don't need a second provider."""
    _patch_ragas_langchain_compat()
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY required for RAGAS scoring.")
    chat = ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY, temperature=0)
    return LangchainLLMWrapper(chat)


def _build_ragas_embeddings():
    """Local embeddings for answer_relevancy (no OpenAI needed)."""
    _patch_ragas_langchain_compat()
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from src.stage1_vanilla.embed_store import get_embeddings

    return LangchainEmbeddingsWrapper(get_embeddings())


def score_with_ragas(
    traces: TraceLogger,
    *,
    skip_ragas: bool = False,
) -> tuple[list[float | None], list[float | None]]:
    """
    Return parallel lists of faithfulness and relevance scores.

    If ragas isn't installed or skip_ragas=True, return Nones (CSV still useful
    for latency / routing analysis).
    """
    n = len(traces.records)
    if skip_ragas or n == 0:
        return [None] * n, [None] * n

    try:
        _patch_ragas_langchain_compat()
        from ragas import EvaluationDataset, SingleTurnSample, evaluate

        # Prefer classic metrics that accept LangchainLLMWrapper + Groq.
        # ragas.metrics.collections requires InstructorLLM/OpenAI-style clients.
        from ragas.metrics import answer_relevancy, faithfulness
    except ImportError as exc:
        print(f"[ragas] Package not available ({exc}); writing traces without scores.")
        return [None] * n, [None] * n

    samples: list[SingleTurnSample] = []
    for rec in traces.records:
        samples.append(
            SingleTurnSample(
                user_input=rec.query,
                response=rec.answer or "",
                retrieved_contexts=rec.contexts or [""],
            )
        )

    dataset = EvaluationDataset(samples=samples)
    llm = _build_ragas_llm()
    embeddings = _build_ragas_embeddings()

    # Bind LLM/embeddings onto metric singletons used by evaluate().
    faithfulness.llm = llm
    answer_relevancy.llm = llm
    answer_relevancy.embeddings = embeddings

    print(f"Scoring {n} samples with RAGAS (faithfulness + answer_relevancy)…")
    try:
        from ragas.run_config import RunConfig

        # OpenAI can still rate-limit under parallel load; raise timeouts and
        # serialize jobs so relevancy scoring is more reliable.
        run_config = RunConfig(timeout=180, max_workers=1)
        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=llm,
            embeddings=embeddings,
            run_config=run_config,
        )
    except TypeError:
        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=llm,
            embeddings=embeddings,
        )

    # ragas returns an EvaluationResult; convert to pandas / dict rows.
    try:
        df = result.to_pandas()
    except Exception:
        # Older/newer API fallback
        df = result

    faith_scores: list[float | None] = []
    rel_scores: list[float | None] = []

    # Column names vary slightly across ragas versions.
    faith_col = next(
        (c for c in df.columns if "faith" in c.lower()),
        None,
    )
    rel_col = next(
        (c for c in df.columns if "relev" in c.lower()),
        None,
    )

    for i in range(n):
        f_val = None
        r_val = None
        if faith_col is not None:
            raw = df.iloc[i][faith_col]
            f_val = float(raw) if raw == raw else None  # NaN check
        if rel_col is not None:
            raw = df.iloc[i][rel_col]
            r_val = float(raw) if raw == raw else None
        faith_scores.append(round(f_val, 4) if f_val is not None else None)
        rel_scores.append(round(r_val, 4) if r_val is not None else None)

    return faith_scores, rel_scores


def run_evaluation(
    *,
    questions_path: Path | None = None,
    limit: int | None = None,
    use_cache: bool = False,
    skip_ragas: bool = False,
    results_dir: Path | None = None,
) -> Path:
    questions = load_eval_questions(questions_path)
    if limit is not None:
        questions = questions[:limit]

    results_dir = results_dir or default_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    csv_path = results_dir / f"eval_run_{stamp}.csv"
    jsonl_path = results_dir / f"eval_run_{stamp}.jsonl"

    tracer = TraceLogger()
    print(f"Running {len(questions)} eval questions through routed RAG…")
    for i, item in enumerate(questions, start=1):
        q = item["question"]
        print(f"  [{i}/{len(questions)}] {q[:70]}…")
        rec = tracer.run(q, use_cache=use_cache)
        expected = item.get("expected_route")
        if expected and rec.route_taken and expected != rec.route_taken:
            print(f"      route={rec.route_taken} (expected {expected})")
        else:
            print(f"      route={rec.route_taken}  latency={rec.latency_ms:.0f}ms")

    faith, relev = score_with_ragas(tracer, skip_ragas=skip_ragas)

    tracer.write_jsonl(jsonl_path)
    tracer.write_csv(
        csv_path,
        extra_fields={
            "faithfulness": faith,
            "relevance": relev,
        },
    )

    print(f"\nWrote CSV:   {csv_path}")
    print(f"Wrote JSONL: {jsonl_path}")
    if not skip_ragas and any(s is not None for s in faith):
        valid_f = [s for s in faith if s is not None]
        valid_r = [s for s in relev if s is not None]
        if valid_f:
            print(f"Mean faithfulness: {sum(valid_f)/len(valid_f):.3f}")
        if valid_r:
            print(f"Mean relevance:    {sum(valid_r)/len(valid_r):.3f}")
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 6 RAGAS + trace eval runner")
    parser.add_argument(
        "--questions",
        type=Path,
        default=None,
        help="Path to eval questions JSON (default: data/eval_questions.json)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run the first N questions (smoke test)",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Allow semantic cache during eval (default: off)",
    )
    parser.add_argument(
        "--skip-ragas",
        action="store_true",
        help="Only collect traces/latency (no RAGAS LLM scores)",
    )
    args = parser.parse_args()

    run_evaluation(
        questions_path=args.questions,
        limit=args.limit,
        use_cache=args.use_cache,
        skip_ragas=args.skip_ragas,
    )


if __name__ == "__main__":
    main()
