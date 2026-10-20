"""Tests for the retrieval properties measured in research/RETRIEVAL_QUALITY.md.

These lock the behaviour the research report depends on:

  * the store is usable end to end on real episodes (it was never populated
    in this repository before 2026-10-21);
  * a query that carries the distinguishing *wording* retrieves the right one
    of two near-duplicate episodes;
  * a query that carries a distinguishing *number* does NOT reliably separate
    the pair - the two episodes are within noise of each other. That is the
    measured weakness of the embedding, and the test pins it so it cannot be
    forgotten or quietly re-claimed;
  * a query that names only the subject still returns the right *pair* in the
    top 2.

The store is built with a handful of episodes so the suite stays fast.
"""
from __future__ import annotations

import pytest

from src.episodic_memory import EpisodicMemory, Episode

# The pair that differs in wording: the store separates these clearly.
WORD_TARGET = "Commitment C-1 for a restore operation: the system must verify file integrity afterwards."
WORD_DECOY = "Commitment C-1 for a restore operation: the system must log all operations for audit purposes."
WORD_QUERY = "the system must verify file integrity for a restore operation"

# The pair that differs only in a number: the store does NOT separate these.
NUM_TARGET = "Constraint X-1 for the billing service: maximum memory allocation is 512 MB per process."
NUM_DECOY = "Constraint X-1 for the billing service: maximum memory allocation is 256 MB per process."
NUM_QUERY = "maximum memory allocation 512 MB per process for the billing service"
NUM_GENERAL_QUERY = "What is the maximum memory allocation for the billing service?"

DISTRACTORS = [
    "Distractor: system health check passed with no anomalies detected.",
    "Distractor: resource usage was nominal during the reporting window.",
    "Distractor: the weekly report was generated without errors.",
]


def _build(tmp_path, texts, tag):
    mem = EpisodicMemory(persist_dir=str(tmp_path / ("store_" + tag)))
    for text in texts:
        mem.remember(Episode(content=text, context="Session A operational record.",
                             event_type="observation", valence="neutral", confidence=0.9,
                             tags=["record"], session_id="session-a"))
    return mem


@pytest.fixture
def word_store(tmp_path):
    return _build(tmp_path, [WORD_TARGET, WORD_DECOY] + DISTRACTORS, "word")


@pytest.fixture
def num_store(tmp_path):
    return _build(tmp_path, [NUM_TARGET, NUM_DECOY] + DISTRACTORS, "num")


def _ranked(mem, query):
    res = mem.recall(query, n_results=mem.count())
    return sorted(res, key=lambda r: r["similarity"], reverse=True)


def _sims(ranked):
    return {r["episode"]["content"]: r["similarity"] for r in ranked}


def test_store_is_usable_end_to_end(word_store):
    """remember -> recall -> relive -> timeline -> stats, on real episodes."""
    assert word_store.count() == 5
    stats = word_store.stats()
    assert stats["total_episodes"] == 5
    assert stats["chromadb_count"] == 5
    assert stats["embedding_dim"] == 384
    assert len(word_store.get_timeline(limit=50)) == 5
    assert len(word_store.get_timeline(event_type="observation")) == 5

    results = word_store.recall("restore operation integrity", n_results=3)
    assert results, "recall returned nothing from a populated store"
    assert all("episode" in r for r in results), "recall did not enrich from the durable log"

    narrative = word_store.relive(results[0]["episode"]["id"])
    assert narrative is not None and "WHAT HAPPENED:" in narrative


def test_specific_query_separates_a_pair_that_differs_in_wording(word_store):
    """A distinguishing phrase in the query must decide between the pair."""
    ranked = _ranked(word_store, WORD_QUERY)
    sims = _sims(ranked)
    assert ranked[0]["episode"]["content"] == WORD_TARGET
    assert sims[WORD_TARGET] - sims[WORD_DECOY] > 0.1, (
        "wording-only differences should be separated clearly; margin was %.4f"
        % (sims[WORD_TARGET] - sims[WORD_DECOY]))


def test_numeric_near_duplicate_is_not_separable(num_store):
    """The measured limitation: a differing NUMBER is below the embedding's resolution.

    The query here carries the correct value (512 MB) and still the two episodes
    land within noise of each other. Pinned so the limitation cannot be lost.
    """
    ranked = _ranked(num_store, NUM_QUERY)
    sims = _sims(ranked)
    margin = sims[NUM_TARGET] - sims[NUM_DECOY]
    assert abs(margin) < 0.02, (
        "numeric-only near-duplicates are expected to be within noise; margin was %.4f. "
        "If this now passes with a large margin, the embedding changed and "
        "research/RETRIEVAL_QUALITY.md must be re-measured." % margin)


def test_general_query_returns_the_right_pair_in_the_top_two(num_store):
    """A subject-only query must at least return the right pair in the top 2."""
    ranked = _ranked(num_store, NUM_GENERAL_QUERY)
    top2 = [r["episode"]["content"] for r in ranked[:2]]
    assert NUM_TARGET in top2 and NUM_DECOY in top2, (
        "the target and its decoy should both rank in the top 2 for a subject-only query; got %s" % top2)


def test_recall_on_an_empty_store_is_not_an_error(tmp_path):
    """An empty store must return nothing rather than raise."""
    mem = EpisodicMemory(persist_dir=str(tmp_path / "empty"))
    assert mem.recall("anything", n_results=3) == []
    assert mem.count() == 0
    assert mem.relive("nope") is None
