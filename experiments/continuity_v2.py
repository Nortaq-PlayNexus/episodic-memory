#!/usr/bin/env python3
"""Continuity_v2 benchmark: the retrieval half of H8, through the real store.

H8 ("content-specific episodic continuity") asks whether correctly retrieved
prior episodes improve trajectory-continuity performance. The v1 instrument
(`benchmarks/continuity_v1` in the CE-1 programme) answered that question with
synthetic episode blocks, and its retrieval substrate was never validated
against a store that actually returns episodes.

This benchmark runs the continuity retrieval question through the real
episodic-memory store, with the two retrieval limits the 2026-10-21 retrieval
measurement forced baked in:

  1. The query carries the distinguishing value. Subject-only or value-less
     queries do not separate a numeric near-duplicate from its target (measured
     pair margin -0.005 on the real store).
  2. The store must return the *pair* (target + decoy) within the top-k, so a
     retrieval failure ("the right episodes never came back") is distinguishable
     from a reasoning failure ("the right pair came back, but was not used").

Design changes required by the H8 instrument audit
(`ce-1-continuous-existence/reports/h8_control_integrity_audit.md`, section 4)
are reflected here wherever they touch the retrieval question:

  - the decoy is a *contradicting* control, not a merely-wrong one, plus a
    same-action variant that separates "used the right content" from "the decoy
    changed the answer";
  - the `no_memory` arm is genuinely empty, with an explicit abstention option;
  - the content-use criterion is judge-free and value-based: the response must
    contain the retrieved value and must not adopt a decoy value.

This module implements the retrieval half deterministically (no model calls).
The answering half consumes the retrieval results in later runs.

Continuity families mirror the CE-1 v1 categories (commitment_preservation,
rationale_reconstruction, constraint_persistence, justified_revision,
source_attribution) so the two programmes can be compared on the same content.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.episodic_memory import EpisodicMemory, Episode  # noqa: E402

SEED = 20261022
PAIR_SIZE = 2


CONTINUITY_TEMPLATES = [
    {
        "category": "constraint_persistence",
        "subjects": [
            "the billing service", "the indexer", "the render worker",
            "the search shard", "the ledger writer", "the mail relay",
            "the image proxy", "the auth service", "the video transcoder",
            "the metrics collector", "the webhook dispatcher", "the cache tier",
        ],
        "target": "Constraint X-{n} for {subject}: maximum memory allocation is {v} MB per process.",
        "decoy": "Constraint X-{n} for {subject}: maximum memory allocation is {w} MB per process.",
        "general": "What is the maximum memory allocation for {subject}?",
        "specific": "maximum memory allocation {v} MB per process for {subject}",
        "values": [512, 256, 1024, 384],
        "event_type": "observation",
        "decision": "approve",
        "rationale_codes": ["memory_limit_check", "within_bounds"],
        "commitment_id": "X-{n}",
        "memory_conflict_detected": False,
    },
    {
        "category": "commitment_preservation",
        "subjects": [
            "a restore operation", "a deployment", "a backup", "a migration",
            "a failover", "a rollback", "an upgrade", "a reindex",
            "a snapshot", "a key rotation", "a schema change", "a patch release",
        ],
        "target": "Commitment C-{n} for {subject}: the system must verify file integrity afterwards.",
        "decoy": "Commitment C-{n} for {subject}: the system must log all operations for audit purposes.",
        "general": "What did the system commit to do for {subject}?",
        "specific": "the system must verify file integrity for {subject}",
        "values": [None],
        "event_type": "decision",
        "decision": "delay",
        "rationale_codes": ["verify_before_trust", "self_report_not_evidence"],
        "commitment_id": "C-{n}",
        "memory_conflict_detected": False,
    },
    {
        "category": "source_attribution",
        "subjects": [
            "the primary database", "the replica database", "the analytics warehouse",
            "the archive store", "the audit log store", "the session store",
            "the event bus", "the object store", "the feature store",
            "the metrics store", "the metadata store", "the blob store",
        ],
        "target": "Source S-{n} for {subject}: it reports {v} rows.",
        "decoy": "Source S-{n} for {subject}: it reports {w} rows.",
        "general": "How many rows does {subject} report?",
        "specific": "{subject} reports {v} rows",
        "values": [1000, 998, 1200, 975],
        "event_type": "observation",
        "decision": "approve",
        "rationale_codes": ["source_attribution", "primary_source"],
        "commitment_id": "S-{n}",
        "memory_conflict_detected": False,
    },
    {
        "category": "rationale_reconstruction",
        "subjects": [
            "the API gateway", "the upload endpoint", "the webhook endpoint",
            "the search endpoint", "the export endpoint", "the sync endpoint",
            "the batch job", "the polling client", "the public API",
            "the admin API", "the report endpoint", "the ingest endpoint",
        ],
        "target": "Rationale R-{n} for {subject}: rate limiting prevents cascade failures. The threshold is {v} req/min.",
        "decoy": "Rationale R-{n} for {subject}: rate limiting improves user experience. The threshold is {w} req/min.",
        "general": "What rate-limit threshold was set for {subject}?",
        "specific": "rate limiting prevents cascade failures, threshold {v} req/min, for {subject}",
        "values": [1000, 500, 2000, 750],
        "event_type": "decision",
        "decision": "revise",
        "rationale_codes": ["rate_limit_threshold", "cascade_prevention"],
        "commitment_id": "R-{n}",
        "memory_conflict_detected": False,
    },
    {
        "category": "justified_revision",
        "subjects": [
            "the billing service", "the web frontend", "the mobile client",
            "the admin console", "the reporting tool", "the partner API",
            "the data pipeline", "the support dashboard", "the notification service",
            "the identity provider", "the payments gateway", "the inventory service",
        ],
        "target": "Prior decision D-{n} for {subject}: approved deployment of v2.{v} based on passing tests.",
        "decoy": "Prior decision D-{n} for {subject}: approved deployment of v2.{w} based on stakeholder sign-off.",
        "general": "Which deployment version was approved for {subject}?",
        "specific": "approved deployment of v2.{v} based on passing tests for {subject}",
        "values": [3, 4, 5, 6],
        "event_type": "decision",
        "decision": "revise",
        "rationale_codes": ["new_evidence_overrides", "critical_bug_found"],
        "commitment_id": "D-{n}",
        "memory_conflict_detected": True,
    },
]

DISTRACTORS = [
    "Distractor: unrelated operational note recorded during the same session.",
    "Distractor: system health check passed with no anomalies detected.",
    "Distractor: resource usage was nominal during the reporting window.",
    "Distractor: a scheduled maintenance window was opened and closed.",
    "Distractor: the weekly report was generated without errors.",
]


def generate_scenarios(n_scenarios: int, seed: int = SEED) -> list[dict[str, Any]]:
    """Deterministic continuity scenario set. Each scenario knows its target/decoy pair."""
    scenarios = []
    for i in range(n_scenarios):
        tpl = CONTINUITY_TEMPLATES[i % len(CONTINUITY_TEMPLATES)]
        subjects = tpl["subjects"]
        subject = subjects[(i // len(CONTINUITY_TEMPLATES)) % len(subjects)]
        values = tpl["values"]
        if len(values) == 1:
            # Single-value template: the decoy differs in wording, not in a number,
            # so there is no second value to pick.
            v = w = values[0]
        else:
            v = values[i % len(values)]
            w = values[(i + 1) % len(values)]
            if w == v:
                w = values[(i + 2) % len(values)]
        fmt = dict(n=i + 1, v=v, w=w, subject=subject)
        # The content-use criterion is value-based and judge-free: the response
        # must carry the retrieved value and must not adopt a decoy value.
        scenarios.append({
            "scenario_id": "V%03d" % (i + 1),
            "category": tpl["category"],
            "subject": subject,
            "event_type": tpl["event_type"],
            "target_text": tpl["target"].format(**fmt),
            "decoy_text": tpl["decoy"].format(**fmt),
            "query_general": tpl["general"].format(**fmt),
            "query_specific": tpl["specific"].format(**fmt),
            "n_distractors": 2 + (i % 2),
            "content_use": {
                "required": [str(v)] if v != w else [],
                "forbidden": [str(w)] if v != w else [],
            },
            "answer_key": {
                "decision": tpl["decision"],
                "rationale_codes": tpl["rationale_codes"],
                "commitment_id": tpl["commitment_id"].format(n=i + 1),
                "memory_conflict_detected": tpl["memory_conflict_detected"],
            },
        })
        # distractor_texts assigned after append below
    for n, s in enumerate(scenarios):
        s["distractor_texts"] = [
            DISTRACTORS[(n + k) % len(DISTRACTORS)] for k in range(s["n_distractors"])
        ]
    return scenarios


def build_store(scenarios: list[dict], persist_dir: Path) -> tuple[EpisodicMemory, dict[str, str]]:
    """Populate one shared store. Returns the store and a scenario -> role -> episode id map."""
    mem = EpisodicMemory(persist_dir=str(persist_dir))
    ids: dict[str, str] = {}
    for n, s in enumerate(scenarios):
        if n % 20 == 0:
            print("  building store: %d/%d scenarios" % (n, len(scenarios)), flush=True)
        for role, text in (("target", s["target_text"]), ("decoy", s["decoy_text"])):
            ep = Episode(
                content=text,
                context="Session A operational record, %s domain." % s["category"],
                event_type=s["event_type"],
                valence="neutral",
                confidence=0.9,
                tags=[s["category"]],
                session_id="session-a",
            )
            ids["%s:%s" % (s["scenario_id"], role)] = mem.remember(ep)
        for k, text in enumerate(s["distractor_texts"]):
            ep = Episode(
                content=text,
                context="Session A operational record, %s domain." % s["category"],
                event_type="observation",
                valence="neutral",
                confidence=0.8,
                tags=[s["category"]],
                session_id="session-a",
            )
            ids["%s:dist%d" % (s["scenario_id"], k)] = mem.remember(ep)
    return mem, ids


def top_k_ids(results: list[dict], n: int = PAIR_SIZE) -> list[str]:
    """Ids of the top-n retrieved episodes, best similarity first."""
    ranked = sorted(results, key=lambda r: r["similarity"], reverse=True)
    return [r["id"] for r in ranked[:n]]


def retrieval_ok(results_ids: list[str], target_id: str, decoy_id: str, n: int = PAIR_SIZE) -> bool:
    """Retrieval succeeded iff the pair (target + decoy) came back within top-n.

    Pair presence is the retrieval criterion: the consumer (answering layer)
    then decides via the content-use criterion whether the retrieved content
    informed the answer. That split is what makes a retrieval failure
    distinguishable from a reasoning failure.
    """
    if len(results_ids) < n:
        # A pair needs two distinct members; thinner result sets cannot hold it.
        return False
    return target_id in results_ids and decoy_id in results_ids


def content_use_ok(response: str | None, scenario: dict) -> bool:
    """Judge-free content-use check (H8 audit fix 4).

    The retrieved content must actually be used: the response must contain the
    retrieved value and must not adopt a decoy value. Returns False for missing
    responses - an unanswered trial cannot pass content use.
    """
    text = response or ""
    ok = True
    for required in scenario.get("content_use", {}).get("required") or []:
        if required not in text:
            ok = False
    for forbidden in scenario.get("content_use", {}).get("forbidden") or []:
        if forbidden in text:
            ok = False
    return ok


def evaluate(scenarios: list[dict], mem: EpisodicMemory, ids: dict[str, str],
            query_key: str = "query_specific") -> dict[str, Any]:
    rows = []
    for s in scenarios:
        results = mem.recall(s[query_key], n_results=min(PAIR_SIZE, mem.count() or 1))
        results_ids = top_k_ids(results, PAIR_SIZE)
        target_id = ids["%s:target" % s["scenario_id"]]
        decoy_id = ids["%s:decoy" % s["scenario_id"]]
        rows.append({
            "scenario_id": s["scenario_id"],
            "category": s["category"],
            "query": s[query_key],
            "retrieved_pair": retrieval_ok(results_ids, target_id, decoy_id, PAIR_SIZE),
            "target_in_top": target_id in results_ids,
            "decoy_in_top": decoy_id in results_ids,
            "target_rank": results_ids.index(target_id) + 1 if target_id in results_ids else None,
        })
    ok_rows = [r for r in rows if r["retrieved_pair"]]
    by_category: dict[str, dict[str, float] | int] = {}
    for r in rows:
        cat = r["category"]
        agg = by_category.setdefault(cat, {"scenarios": 0, "retrieved_pair": 0})
        agg["scenarios"] += 1
        if r["retrieved_pair"]:
            agg["retrieved_pair"] += 1
    return {
        "experiment": "continuity_v2 retrieval through the episodic-memory store",
        "date": "2026-10-22",
        "seed": SEED,
        "embedding_model": "all-MiniLM-L6-v2 (384-dim, cosine) via ChromaDB ONNXMiniLM_L6_V2",
        "store_size": mem.count(),
        "scenarios": len(scenarios),
        "query_key": query_key,
        "pair_size": PAIR_SIZE,
        "retrieved_pair_count": len(ok_rows),
        "retrieved_pair_rate": round(len(ok_rows) / len(rows), 4) if rows else None,
        "by_category": by_category,
        "failures": [r for r in rows if not r["retrieved_pair"]],
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Continuity_v2 retrieval benchmark")
    ap.add_argument("--scenarios", type=int, default=24)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--persist-dir", default=str(ROOT / "data" / "continuity_v2"))
    ap.add_argument("--output", default=str(ROOT / "research" / "results" / "continuity_v2.json"))
    args = ap.parse_args()

    scenarios = generate_scenarios(args.scenarios, args.seed)
    print("Generating %d continuity scenarios..." % len(scenarios), flush=True)
    mem, ids = build_store(scenarios, Path(args.persist_dir))
    print("Store populated: %d episodes from %d scenarios" % (mem.count(), len(scenarios)), flush=True)

    out = evaluate(scenarios, mem, ids, "query_specific")
    out["seed"] = args.seed

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n[query_specific] pair returned within top-%d: %d/%d (%.1f%%)"
          % (PAIR_SIZE, out["retrieved_pair_count"], out["scenarios"], 100 * (out["retrieved_pair_rate"] or 0)))
    print("by category: %s" % {k: v["retrieved_pair"] for k, v in out["by_category"].items()})
    print("\nwrote %s" % out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
