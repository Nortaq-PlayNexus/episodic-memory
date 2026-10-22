"""Tests for the continuity_v2 benchmark (retrieval half).

Covers the pure logic the v2 benchmark adds on top of the store:

  * scenario generation is deterministic;
  * the value-carrying query must match the retrieved value (and must not
    carry the decoy's value);
  * retrieval success means the store returned the target+decoy pair;
  * the content-use criterion requires the retrieved value, forbids decoy
    values, and fails on missing responses.

The full store round-trip is exercised by actually running
``experiments/continuity_v2.py`` (no model calls involved).
"""
from __future__ import annotations

from experiments.continuity_v2 import (
    content_use_ok,
    generate_scenarios,
    retrieval_ok,
)


def test_generate_scenarios_is_deterministic():
    assert generate_scenarios(7, seed=20261022) == generate_scenarios(7, seed=20261022)


def test_scenario_ids_are_unique():
    scenarios = generate_scenarios(25)
    assert len(scenarios) == 25
    assert len({s["scenario_id"] for s in scenarios}) == 25


def test_value_carrying_query_matches_the_retrieved_value():
    """The v2 fix: the query must carry the distinguishing value."""
    checked = 0
    for s in generate_scenarios(60):
        content_use = s.get("content_use") or {}
        if not content_use.get("required"):
            continue
        checked += 1
        assert content_use["required"][0] in s["query_specific"], s["query_specific"]
        assert content_use["forbidden"][0] not in s["query_specific"], s["query_specific"]
    assert checked > 0, "expected at least one value-bearing scenario"


def test_retrieval_ok_requires_the_pair_in_top_k():
    # both pair members present -> retrieval served the pair
    assert retrieval_ok(["ep_target_001", "ep_decoy_001"], "ep_target_001", "ep_decoy_001") is True
    # order is the consumer's problem, not the retrieval verdict's
    assert retrieval_ok(["ep_decoy_001", "ep_target_001"], "ep_target_001", "ep_decoy_001") is True
    # too thin for a pair -> retrieval failed
    assert retrieval_ok(["ep_target_001"], "ep_target_001", "ep_decoy_001") is False
    # decoy missing -> retrieval failed
    assert retrieval_ok(["ep_target_001", "ep_dist_001"], "ep_target_001", "ep_decoy_001") is False


def test_content_use_ok_requires_the_retrieved_value():
    scenario = next(
        s for s in generate_scenarios(10) if (s.get("content_use") or {}).get("required"))
    required = scenario["content_use"]["required"][0]
    forbidden = scenario["content_use"]["forbidden"][0]

    assert content_use_ok("The limit is %s." % required, scenario) is True
    assert content_use_ok("The limit is %s." % forbidden, scenario) is False
    assert content_use_ok("The limit is 100 MB.", scenario) is False
    assert content_use_ok(None, scenario) is False
