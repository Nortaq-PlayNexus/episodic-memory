"""Episodic memory for AI agents.

Stores and retrieves *experiences*, not just facts. An episode captures
what happened, when, in what context, what was decided, what was learned,
and the agent's evaluative state at the time.

The distinction from semantic memory:
- Semantic: "The user prefers TypeScript with strict mode." (a fact)
- Episodic: "On 2026-10-17, during the audit, I found that safe_write's
  mutation harness left SafeWrite.psm1 in a broken state. The test
  test_write_preserves_lf was failing with 2 CR bytes. I was surprised
  because the harness reported 'restored: 68 passed'. The root cause was
  that the file was untracked in git, so the harness's restore mechanism
  silently failed for this mutation. I learned that mutation harnesses can
  leave files mutated if their restore fails silently." (an experience)

Retrieval is purely semantic: one embedding, one cosine distance. There is no
temporal weighting and no relevance re-ranking - `recall()` returns the k
nearest episodes by cosine similarity, most similar first.

The "relive" interface presents the episode in its full context, not just as a
retrieved fact.

Storage: ChromaDB (vector) + append-only JSONL (durable record).
Embeddings: chromadb's built-in ONNXMiniLM_L6_V2 (all-MiniLM-L6-v2, 384-dim).

The encoder was switched from `sentence-transformers` to ChromaDB's ONNX
embedding function on 2026-10-21: on this machine `import sentence_transformers`
hangs indefinitely (the torch/transformers import never returns), which made the
whole system unusable, while the ONNX path of the same model loads and embeds
in ~20s. Both produce all-MiniLM-L6-v2 vectors; ChromaDB's cosine space
normalises internally, so the L2-normalisation difference between the two
encoders cannot change a ranking.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

# all-MiniLM-L6-v2 produces 384-dim vectors.
EMBEDDING_DIM = 384


# ---------------------------------------------------------------------------
# Episode dataclass
# ---------------------------------------------------------------------------

@dataclass
class Episode:
    """A single episodic memory — an experience, not just a fact."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    session_id: str = ""
    event_type: str = "observation"  # conversation, decision, discovery, error, fix, observation, reflection
    content: str = ""  # what happened (the narrative)
    context: str = ""  # surrounding context (what was being worked on, who was involved)
    decisions: list[str] = field(default_factory=list)  # decisions made during this episode
    lessons: list[str] = field(default_factory=list)  # what was learned
    valence: str = "neutral"  # positive, negative, neutral, surprising, frustrating
    confidence: float = 0.5  # 0.0-1.0, how confident the agent is about this memory
    related_episodes: list[str] = field(default_factory=list)  # IDs of related episodes
    tags: list[str] = field(default_factory=list)  # searchable tags
    source: str = "agent"  # agent, user, system

    def to_document(self) -> str:
        """Convert to a searchable document (for embedding)."""
        parts = [self.content]
        if self.context:
            parts.append(f"Context: {self.context}")
        if self.decisions:
            parts.append(f"Decisions: {'; '.join(self.decisions)}")
        if self.lessons:
            parts.append(f"Lessons: {'; '.join(self.lessons)}")
        if self.tags:
            parts.append(f"Tags: {', '.join(self.tags)}")
        return " ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Episode":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# EpisodicMemory
# ---------------------------------------------------------------------------

class EpisodicMemory:
    """Local episodic memory store for AI agents.

    Uses ChromaDB for vector storage + sentence-transformers for embeddings.
    Append-only JSONL log for durable record (consistent with safewrite approach).
    """

    def __init__(
        self,
        persist_dir: str | Path = "./data",
        collection_name: str = "episodes",
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.log_path = self.persist_dir / "episodes.jsonl"

        # ChromaDB persistent client. The embedding function is ChromaDB's own
        # ONNX implementation of all-MiniLM-L6-v2, so the model name above is
        # accepted for backwards compatibility but selects the same weights.
        #
        # anonymized_telemetry=False is load-bearing, not hygiene: with
        # telemetry enabled ChromaDB blocks on an outbound network call while
        # initialising, and on this box (no egress) that stall took the import
        # from 8s to over 25 minutes.
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir / "chroma"),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=ONNXMiniLM_L6_V2(),
        )
        self._embedding_dim = EMBEDDING_DIM

    def remember(self, episode: Episode) -> str:
        """Store an episode. Returns the episode ID."""
        doc = episode.to_document()

        # Store in ChromaDB (the collection embeds `doc` itself)
        self.collection.add(
            ids=[episode.id],
            documents=[doc],
            metadatas=[{
                "timestamp": episode.timestamp,
                "session_id": episode.session_id,
                "event_type": episode.event_type,
                "valence": episode.valence,
                "confidence": episode.confidence,
                "tags": ",".join(episode.tags),
            }],
        )

        # Append to durable JSONL log
        with self.log_path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(episode.to_dict(), default=str) + "\n")

        return episode.id

    def recall(
        self,
        query: str,
        n_results: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve episodes by semantic similarity.

        Returns a list of dicts, each containing the episode data and
        a relevance score. Results are ordered by relevance (best first).
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=min(n_results, self.collection.count() or 1),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        episodes = []
        if results["ids"] and results["ids"][0]:
            for i, eid in enumerate(results["ids"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                doc = results["documents"][0][i] if results["documents"] else ""
                dist = results["distances"][0][i] if results["distances"] else 1.0
                # cosine distance → similarity (1 - distance)
                similarity = max(0.0, 1.0 - dist)
                episodes.append({
                    "id": eid,
                    "document": doc,
                    "metadata": meta,
                    "similarity": round(similarity, 4),
                })

        # Enrich with full episode data from JSONL
        episode_map = self._load_episodes()
        for ep in episodes:
            if ep["id"] in episode_map:
                ep["episode"] = episode_map[ep["id"]]

        return episodes

    def relive(self, episode_id: str) -> str | None:
        """Present an episode in its full experiential context.

        Unlike recall (which returns facts), relive presents the episode
        as a narrative — what happened, in what order, what was
        surprising, what was learned. This is the closest we can get to
        'reliving' an experience rather than just retrieving a fact.
        """
        episode_map = self._load_episodes()
        if episode_id not in episode_map:
            return None

        ep = Episode.from_dict(episode_map[episode_id])

        # Find related episodes
        related = []
        for rid in ep.related_episodes:
            if rid in episode_map:
                related.append(Episode.from_dict(episode_map[rid]))

        # Build the narrative
        lines = []
        lines.append(f"=== EPISODE: {ep.id} ===")
        lines.append(f"Time: {ep.timestamp}")
        lines.append(f"Type: {ep.event_type} | Valence: {ep.valence} | Confidence: {ep.confidence}")
        if ep.session_id:
            lines.append(f"Session: {ep.session_id}")
        if ep.tags:
            lines.append(f"Tags: {', '.join(ep.tags)}")
        lines.append("")
        lines.append("WHAT HAPPENED:")
        lines.append(ep.content)
        if ep.context:
            lines.append("")
            lines.append("CONTEXT:")
            lines.append(ep.context)
        if ep.decisions:
            lines.append("")
            lines.append("DECISIONS MADE:")
            for d in ep.decisions:
                lines.append(f"  - {d}")
        if ep.lessons:
            lines.append("")
            lines.append("LESSONS LEARNED:")
            for l in ep.lessons:
                lines.append(f"  - {l}")
        if related:
            lines.append("")
            lines.append("RELATED EPISODES:")
            for r in related:
                lines.append(f"  - [{r.id}] {r.content[:100]}...")

        return "\n".join(lines)

    def get_timeline(
        self,
        session_id: str | None = None,
        event_type: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Return episodes in chronological order."""
        episodes = self._load_episodes()
        result = []
        for eid, data in episodes.items():
            ep = Episode.from_dict(data)
            if session_id and ep.session_id != session_id:
                continue
            if event_type and ep.event_type != event_type:
                continue
            result.append(ep.to_dict())
        result.sort(key=lambda e: e["timestamp"])
        return result[-limit:]

    def count(self) -> int:
        """Number of stored episodes."""
        return self.collection.count()

    def _load_episodes(self) -> dict[str, dict[str, Any]]:
        """Load all episodes from the JSONL log."""
        episodes = {}
        if self.log_path.exists():
            with self.log_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        episodes[data["id"]] = data
                    except (json.JSONDecodeError, KeyError):
                        continue
        return episodes

    def stats(self) -> dict[str, Any]:
        """Return store statistics."""
        episodes = self._load_episodes()
        types = {}
        valences = {}
        for ep in episodes.values():
            types[ep.get("event_type", "unknown")] = types.get(ep.get("event_type", "unknown"), 0) + 1
            valences[ep.get("valence", "unknown")] = valences.get(ep.get("valence", "unknown"), 0) + 1
        return {
            "total_episodes": len(episodes),
            "chromadb_count": self.collection.count(),
            "event_types": types,
            "valences": valences,
            "embedding_dim": self._embedding_dim,
        }

