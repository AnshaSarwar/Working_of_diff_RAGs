"""
Build a Neo4j knowledge graph from OKF markdown links.

Why a graph (not just vectors)?
Vector search finds *similar text*. Multi-hop questions like
"Who manages the person leading Project Atlas?" need *relationships*:
  Project Atlas --LINKS_TO--> Dave Kowalski --LINKS_TO--> Alice Chen
OKF already encodes those relationships as markdown links, so we don't need
NLP entity extraction — we just materialize the explicit link graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from neo4j import Driver, GraphDatabase

from src.config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER, OKF_BUNDLE_DIR
from src.loaders.okf_loader import OKFDocument, load_okf_bundle


@dataclass
class GraphBuildStats:
    nodes: int
    edges: int
    skipped_dangling_links: int


def get_neo4j_driver(
    uri: str | None = None,
    user: str | None = None,
    password: str | None = None,
) -> Driver:
    """Create a Neo4j driver from env/.env credentials."""
    uri = uri or NEO4J_URI
    user = user or NEO4J_USER
    password = password if password is not None else NEO4J_PASSWORD
    if not password:
        raise RuntimeError(
            "NEO4J_PASSWORD is not set. Add it to .env (see .env.example) "
            "and start Neo4j (docker compose up -d)."
        )
    return GraphDatabase.driver(uri, auth=(user, password))


def verify_connectivity(driver: Driver) -> None:
    """Fail fast with a clear message if Neo4j isn't reachable."""
    try:
        driver.verify_connectivity()
    except Exception as exc:  # noqa: BLE001 — surface driver errors as setup help
        raise RuntimeError(
            "Cannot connect to Neo4j at "
            f"{NEO4J_URI}. Start it with `docker compose up -d` "
            "(Docker Desktop) or Neo4j Desktop, then check NEO4J_* in .env.\n"
            f"Underlying error: {exc}"
        ) from exc


def clear_okf_graph(driver: Driver) -> None:
    """Remove previous Concept nodes/edges so rebuilds stay deterministic."""
    with driver.session() as session:
        session.run("MATCH (n:Concept) DETACH DELETE n")


def _upsert_node(tx, doc: OKFDocument) -> None:
    # path is the stable ID (bundle-relative). Labels stay generic (:Concept)
    # plus a type property so we can filter later without dynamic Cypher labels.
    tx.run(
        """
        MERGE (c:Concept {path: $path})
        SET c.title       = $title,
            c.type        = $type,
            c.description = $description,
            c.tags        = $tags,
            c.body        = $body,
            c.resource    = $resource
        """,
        path=doc.bundle_path,
        title=doc.title or doc.bundle_path,
        type=doc.doc_type,
        description=doc.description or "",
        tags=doc.tags,
        body=doc.body,
        resource=doc.resource or "",
    )


def _upsert_edge(tx, source_path: str, target_path: str) -> None:
    tx.run(
        """
        MATCH (a:Concept {path: $source})
        MATCH (b:Concept {path: $target})
        MERGE (a)-[:LINKS_TO]->(b)
        """,
        source=source_path,
        target=target_path,
    )


def build_okf_graph(
    *,
    bundle_dir: Path | None = None,
    driver: Driver | None = None,
    clear: bool = True,
) -> GraphBuildStats:
    """
    Load the OKF bundle and write one node per file + one edge per markdown link.

    Dangling links (targets not in the bundle) are skipped — they can't become
    traversable Concept nodes.
    """
    bundle_dir = bundle_dir or OKF_BUNDLE_DIR
    own_driver = driver is None
    driver = driver or get_neo4j_driver()
    verify_connectivity(driver)

    docs = load_okf_bundle(bundle_dir, skip_index_type=True)
    known_paths = {d.bundle_path for d in docs}

    try:
        if clear:
            clear_okf_graph(driver)

        with driver.session() as session:
            for doc in docs:
                session.execute_write(_upsert_node, doc)

            skipped = 0
            edge_count = 0
            for doc in docs:
                for target in doc.outgoing_links:
                    if target not in known_paths:
                        skipped += 1
                        continue
                    session.execute_write(_upsert_edge, doc.bundle_path, target)
                    edge_count += 1

            # Full-text index speeds seed-entity lookup for local search.
            session.run(
                """
                CREATE FULLTEXT INDEX concept_fulltext IF NOT EXISTS
                FOR (c:Concept) ON EACH [c.title, c.description, c.body, c.path]
                """
            )

        return GraphBuildStats(
            nodes=len(docs),
            edges=edge_count,
            skipped_dangling_links=skipped,
        )
    finally:
        if own_driver:
            driver.close()
