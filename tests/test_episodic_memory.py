"""Tests for episodic memory system."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.episodic_memory import EpisodicMemory, Episode


@pytest.fixture
def mem(tmp_path):
    return EpisodicMemory(persist_dir=str(tmp_path / "store"))


def test_remember_and_recall(mem):
    ep = Episode(
        content="Found that safe_write left SafeWrite.psm1 in a mutated state.",
        context="Auditing mem-integrity tooling",
        event_type="discovery",
        decisions=["Restore the mutated line"],
        lessons=["Mutation harnesses can leave files mutated"],
        valence="surprising",
        confidence=0.9,
        tags=["mutation-harness", "audit"],
        session_id="test-session",
    )
    eid = mem.remember(ep)
    assert eid == ep.id
    assert mem.count() == 1

    results = mem.recall("mutation harness left file broken", n_results=3)
    assert len(results) >= 1
    assert results[0]["similarity"] > 0.3
    assert "safe_write" in results[0]["document"] or "SafeWrite" in results[0]["document"]


def test_recall_returns_relevant_not_irrelevant(mem):
    ep1 = Episode(content="Fixed the safe_write mutation leftover", tags=["fix"])
    ep2 = Episode(content="The weather is sunny today", tags=["weather"])
    mem.remember(ep1)
    mem.remember(ep2)

    results = mem.recall("safe_write mutation", n_results=2)
    assert len(results) == 2
    # The safe_write episode should be more relevant
    assert results[0]["similarity"] > results[1]["similarity"]


def test_relive(mem):
    ep = Episode(
        content="Discovered the safe_write gap",
        context="Audit of mem-integrity",
        event_type="discovery",
        decisions=["Document the gap"],
        lessons=["Always check audit logs after failed writes"],
        valence="surprising",
        confidence=0.8,
    )
    eid = mem.remember(ep)
    narrative = mem.relive(eid)
    assert narrative is not None
    assert "EPISODE:" in narrative
    assert "WHAT HAPPENED:" in narrative
    assert "Discovered the safe_write gap" in narrative
    assert "CONTEXT:" in narrative
    assert "DECISIONS MADE:" in narrative
    assert "LESSONS LEARNED:" in narrative


def test_relive_unknown_returns_none(mem):
    assert mem.relive("nonexistent-id") is None


def test_timeline(mem):
    ep1 = Episode(content="First event", timestamp="2026-10-17T10:00:00Z")
    ep2 = Episode(content="Second event", timestamp="2026-10-17T11:00:00Z")
    mem.remember(ep1)
    mem.remember(ep2)

    timeline = mem.get_timeline(limit=10)
    assert len(timeline) == 2
    assert timeline[0]["content"] == "First event"
    assert timeline[1]["content"] == "Second event"


def test_timeline_filtered_by_event_type(mem):
    ep1 = Episode(content="A discovery", event_type="discovery")
    ep2 = Episode(content="A decision", event_type="decision")
    mem.remember(ep1)
    mem.remember(ep2)

    discoveries = mem.get_timeline(event_type="discovery")
    assert len(discoveries) == 1
    assert discoveries[0]["content"] == "A discovery"


def test_stats(mem):
    ep = Episode(content="test", event_type="observation", valence="neutral")
    mem.remember(ep)
    stats = mem.stats()
    assert stats["total_episodes"] == 1
    assert stats["chromadb_count"] == 1
    assert "observation" in stats["event_types"]
    assert "neutral" in stats["valences"]


def test_persistence_across_instances(tmp_path):
    store_dir = str(tmp_path / "store")
    mem1 = EpisodicMemory(persist_dir=store_dir)
    ep = Episode(content="Persistent episode", tags=["test"])
    eid = mem1.remember(ep)

    # New instance, same store
    mem2 = EpisodicMemory(persist_dir=store_dir)
    assert mem2.count() == 1
    results = mem2.recall("persistent episode", n_results=1)
    assert len(results) == 1
    assert results[0]["id"] == eid


def test_episode_serialization_roundtrip():
    ep = Episode(
        content="test content",
        context="test context",
        event_type="discovery",
        decisions=["d1", "d2"],
        lessons=["l1"],
        valence="positive",
        confidence=0.85,
        tags=["t1", "t2"],
        session_id="s1",
    )
    data = ep.to_dict()
    restored = Episode.from_dict(data)
    assert restored.content == ep.content
    assert restored.decisions == ep.decisions
    assert restored.lessons == ep.lessons
    assert restored.valence == ep.valence
    assert restored.confidence == ep.confidence
    assert restored.tags == ep.tags


def test_relive_with_related_episodes(mem):
    ep1 = Episode(content="First episode")
    ep2 = Episode(content="Second episode", related_episodes=[ep1.id])
    mem.remember(ep1)
    mem.remember(ep2)

    narrative = mem.relive(ep2.id)
    assert narrative is not None
    assert "RELATED EPISODES" in narrative
    assert ep1.id in narrative