"""
GraphRAG local search: seed entity → N-hop neighborhood → generate.

Why local search (not pure vector RAG)?
"Who manages the person leading Project Atlas?" is a *path* question.
Similarity search may return Atlas or Dave in isolation, but miss the
manager link two hops away. Walking LINKS_TO edges assembles the chain
Project → Owner → Manager into one context window.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from langchain_core.prompts import ChatPromptTemplate
from neo4j import Driver

from src.config import GRAPH_HOP_COUNT, OPENAI_API_KEY
from src.routing.metadata_filter import MetadataFilters, filter_graph_nodes
from src.stage1_vanilla.retrieve_generate import get_llm
from src.stage2_graphrag.graph_builder import get_neo4j_driver, verify_connectivity

SYSTEM_PROMPT = """You are a helpful assistant answering questions about Acme Corp \
using ONLY the provided graph neighborhood context.

Rules:
- Base your answer strictly on the context below.
- Prefer relationship chains when the question asks who manages / owns / reports to.
- If the context is insufficient, say what is missing rather than inventing facts.
- Be concise and cite concept paths (e.g. team/alice.md).
"""

USER_PROMPT = """Graph neighborhood context:
{context}

Question: {question}

Answer:"""


@dataclass
class GraphNodeContext:
    path: str
    title: str
    doc_type: str
    body: str
    hops_from_seed: int


@dataclass
class GraphRAGResult:
    question: str
    answer: str
    seed_paths: list[str]
    neighborhood: list[GraphNodeContext] = field(default_factory=list)
    hop_count: int = GRAPH_HOP_COUNT


def _tokenize(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2}


def find_seed_entities(
    driver: Driver,
    query: str,
    *,
    limit: int = 3,
) -> list[str]:
    """
    Resolve the query to one or more Concept seeds.

    Strategy (cheap → robust):
    1. Prefer nodes whose *title* appears as a phrase in the query
       (e.g. "Project Atlas" inside the question) — best for OKF demos
    2. Neo4j full-text index over title/description/body/path
    3. Fallback: token-overlap ranking on title/path

    We deliberately avoid an extra LLM call here so Phase 2 stays about
    *graph traversal*, not agentic rewriting (that's Phase 3).
    """
    query_lower = query.lower()

    with driver.session() as session:
        nodes = session.run(
            """
            MATCH (c:Concept)
            RETURN c.path AS path, c.title AS title
            """
        ).data()

        # 1) Phrase match on titles (longest titles first to prefer
        #    "Project Atlas" over a shorter substring title if any).
        phrase_hits: list[tuple[int, str]] = []
        for node in nodes:
            title = (node.get("title") or "").strip()
            if len(title) < 3:
                continue
            if title.lower() in query_lower:
                phrase_hits.append((len(title), node["path"]))
        if phrase_hits:
            phrase_hits.sort(key=lambda x: x[0], reverse=True)
            return [path for _, path in phrase_hits[:limit]]

        # 2) Full-text search
        try:
            rows = session.run(
                """
                CALL db.index.fulltext.queryNodes('concept_fulltext', $q)
                YIELD node, score
                RETURN node.path AS path, score
                ORDER BY score DESC
                LIMIT $limit
                """,
                q=query,
                limit=limit,
            ).data()
            if rows:
                return [r["path"] for r in rows]
        except Exception:
            pass

    # 3) Token overlap fallback
    q_tokens = _tokenize(query)
    if not q_tokens:
        return []

    scored: list[tuple[float, str]] = []
    for node in nodes:
        path = node["path"]
        title = node.get("title") or ""
        blob_tokens = _tokenize(f"{title} {path}")
        overlap = len(q_tokens & blob_tokens)
        if overlap == 0:
            continue
        title_bonus = 0.5 if any(t in title.lower() for t in q_tokens) else 0.0
        scored.append((overlap + title_bonus, path))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [path for _, path in scored[:limit]]


def local_search(
    driver: Driver,
    seed_paths: list[str],
    *,
    hops: int = GRAPH_HOP_COUNT,
) -> list[GraphNodeContext]:
    """
    Collect the undirected N-hop neighborhood around the seed nodes.

    Why undirected?
    OKF links are one-way in the markdown, but the *relationship* is often
    useful both ways (Atlas→Dave and Dave→Atlas both matter). Undirected
    expansion is the usual "local search" neighborhood.
    """
    if not seed_paths:
        return []

    # Neo4j does not allow parameters for variable-length path bounds
    # (*0..$hops). Validate and inline a safe integer instead.
    hops = int(hops)
    if hops < 0 or hops > 10:
        raise ValueError(f"hops must be 0..10, got {hops}")

    cypher = f"""
        UNWIND $seeds AS seed_path
        MATCH (seed:Concept {{path: seed_path}})
        // 0..{hops} includes the seed itself (hop 0)
        MATCH path = (seed)-[:LINKS_TO*0..{hops}]-(nbr:Concept)
        WITH nbr, min(length(path)) AS dist
        RETURN nbr.path AS path,
               nbr.title AS title,
               nbr.type AS type,
               nbr.body AS body,
               dist AS hops_from_seed
        ORDER BY dist ASC, nbr.title ASC
        """

    with driver.session() as session:
        rows = session.run(cypher, seeds=seed_paths).data()

    # Deduplicate (same node reachable from multiple seeds).
    seen: set[str] = set()
    neighborhood: list[GraphNodeContext] = []
    for row in rows:
        path = row["path"]
        if path in seen:
            continue
        seen.add(path)
        neighborhood.append(
            GraphNodeContext(
                path=path,
                title=row.get("title") or path,
                doc_type=row.get("type") or "",
                body=row.get("body") or "",
                hops_from_seed=int(row.get("hops_from_seed") or 0),
            )
        )
    return neighborhood


def format_graph_context(nodes: list[GraphNodeContext]) -> str:
    parts: list[str] = []
    for i, node in enumerate(nodes, start=1):
        parts.append(
            f"[Node {i} | {node.path} | {node.title} | type={node.doc_type} "
            f"| hops={node.hops_from_seed}]\n{node.body}"
        )
    return "\n\n---\n\n".join(parts)


def run_graph_rag(
    question: str,
    *,
    hops: int = GRAPH_HOP_COUNT,
    seed_limit: int = 3,
    driver: Driver | None = None,
    metadata_filters: MetadataFilters | None = None,
) -> GraphRAGResult:
    """
    End-to-end Phase 2 pipeline: seed → traverse → (optional type filter) → generate.

    Metadata filters run *before* generation so the LLM only sees the filtered
    neighborhood (sources/context in the router then match what was used).
    """
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )

    own_driver = driver is None
    driver = driver or get_neo4j_driver()
    try:
        verify_connectivity(driver)
        seeds = find_seed_entities(driver, question, limit=seed_limit)
        neighborhood = local_search(driver, seeds, hops=hops)
        if metadata_filters is not None and not metadata_filters.is_empty():
            neighborhood = filter_graph_nodes(neighborhood, metadata_filters)

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", USER_PROMPT),
            ]
        )
        llm = get_llm()
        messages = prompt.format_messages(
            context=format_graph_context(neighborhood)
            or "(no graph neighborhood found — is the graph built?)",
            question=question,
        )
        response = llm.invoke(messages)
        answer = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )

        return GraphRAGResult(
            question=question,
            answer=answer.strip(),
            seed_paths=seeds,
            neighborhood=neighborhood,
            hop_count=hops,
        )
    finally:
        if own_driver:
            driver.close()
