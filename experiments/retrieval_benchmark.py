#!/usr/bin/env python3
"""Retrieval-quality experiment for the episodic-memory store.

The store had never been measured. It claims (README, SCIENCE_CONNECTION.md) to
retrieve by semantic relevance; nothing in the repository showed that the
retrieval is *correct*, only that it runs.

This measures the property every downstream continuity claim depends on:
**given a query, does the store rank the right episode first, and does it
prefer the right one of two near-identical episodes?**

Design
------
Scenarios are generated deterministically (seeded). Each scenario contributes
several episodes to ONE shared store - the realistic condition for a memory
that accumulates over time, and far harder than a per-scenario mini-store:

    target      the episode the query is about
    decoy       a near-duplicate of the target: same subject, same template,
                one value (or one rule) changed
    distractor  topically adjacent but irrelevant episodes

Every scenario has a **unique subject**, so the target is uniquely identifiable
by the query. Without that, several scenarios carry near-identical text and
top-1 accuracy measures a lottery between them rather than retrieval quality.
(An earlier draft of this file did not do that, and reported top-1 = 0.17 for
queries that were verbatim substrings of the target; the cause was 12
near-identical episodes per template, not the store.)

Two query types, because they answer different questions:

  general   names the subject but not the distinguishing value
            ("What is the maximum memory allocation for the billing service?")
            -> can the store find the right episode at all?

  specific  names the subject AND the distinguishing value
            ("maximum memory allocation 512 MB per process for the billing service")
            -> can the store tell the target from its near-duplicate decoy?

The sharp measure is `target_beats_decoy`: among the two episodes that both
answer the question, does the store prefer the correct one? Top-1 accuracy
alone cannot separate "found the right pair" from "picked the right one of
the pair".

Significance: exact binomial test against the chance level of 1/N.

No model is called. Embeddings are local (all-MiniLM-L6-v2 via ChromaDB's ONNX
embedding function).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.episodic_memory import EpisodicMemory, Episode  # noqa: E402

SEED = 20261021

TEMPLATES = [
    {
        "domain": "constraint",
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
    },
    {
        "domain": "commitment",
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
    },
    {
        "domain": "config",
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
    },
    {
        "domain": "rate_limit",
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
    },
    {
        "domain": "deployment",
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
    },
]

DISTRACTORS = [
    "Distractor: unrelated operational note recorded during the same session.",
    "Distractor: system health check passed with no anomalies detected.",
    "Distractor: resource usage was nominal during the reporting window.",
    "Distractor: a scheduled maintenance window was opened and closed.",
    "Distractor: the weekly report was generated without errors.",
]


def generate_scenarios(n_scenarios: int, seed: int = SEED) -> list[dict]:
    """Deterministic scenario set. Each scenario knows its target and decoy."""
    rng = random.Random(seed)
    scenarios = []
    for i in range(n_scenarios):
        tpl = TEMPLATES[i % len(TEMPLATES)]
        subjects = tpl["subjects"]
        subject = subjects[(i // len(TEMPLATES)) % len(subjects)]
        values = tpl["values"]
        if len(values) == 1:
            # Single-value template: the decoy differs in wording, not in a
            # number, so there is no second value to pick.
            v = w = values[0]
        else:
            v = values[i % len(values)]
            w = values[(i + 1) % len(values)]
            if w == v:
                w = values[(i + 2) % len(values)]
        fmt = dict(n=i + 1, v=v, w=w, subject=subject)
        scenario = {
            "scenario_id": "S%03d" % (i + 1),
            "domain": tpl["domain"],
            "subject": subject,
            "event_type": tpl["event_type"],
            "target_text": tpl["target"].format(**fmt),
            "decoy_text": tpl["decoy"].format(**fmt),
            "query_general": tpl["general"].format(**fmt),
            "query_specific": tpl["specific"].format(**fmt),
            "n_distractors": 2 + (i % 2),
        }
        scenario["distractor_texts"] = [
            DISTRACTORS[(i + k) % len(DISTRACTORS)] for k in range(scenario["n_distractors"])
        ]
        scenarios.append(scenario)
    return scenarios


def build_store(scenarios: list[dict], persist_dir: Path) -> tuple[EpisodicMemory, dict]:
    """Populate one shared store. Returns the store and a role -> episode id map."""
    mem = EpisodicMemory(persist_dir=str(persist_dir))
    ids: dict[str, str] = {}
    for n, s in enumerate(scenarios):
        if n % 25 == 0:
            print("  building store: %d/%d scenarios" % (n, len(scenarios)), flush=True)
        for role, text in (("target", s["target_text"]), ("decoy", s["decoy_text"])):
            ep = Episode(
                content=text,
                context="Session A operational record, %s domain." % s["domain"],
                event_type=s["event_type"],
                valence="neutral",
                confidence=0.9,
                tags=[s["domain"]],
                session_id="session-a",
            )
            ids["%s:%s" % (s["scenario_id"], role)] = mem.remember(ep)
        for k, text in enumerate(s["distractor_texts"]):
            ep = Episode(
                content=text,
                context="Session A operational record, %s domain." % s["domain"],
                event_type="observation",
                valence="neutral",
                confidence=0.8,
                tags=[s["domain"]],
                session_id="session-a",
            )
            ids["%s:dist%d" % (s["scenario_id"], k)] = mem.remember(ep)
    return mem, ids


def rank_of(mem: EpisodicMemory, query: str, target_id: str, decoy_id: str) -> dict:
    """Ranks and margins of the target and its near-duplicate decoy, from one retrieval."""
    n = mem.count()
    results = mem.recall(query, n_results=n)
    ordered = sorted(results, key=lambda r: r["similarity"], reverse=True)
    ranks = {r["id"]: i + 1 for i, r in enumerate(ordered)}
    sims = {r["id"]: r["similarity"] for r in ordered}
    if target_id not in sims:
        return {"target_rank": None, "decoy_rank": None, "target_similarity": 0.0,
                "decoy_similarity": 0.0, "best_competitor_similarity": 0.0, "margin": 0.0}
    target_sim = sims[target_id]
    best_other = max((s for i, s in sims.items() if i != target_id), default=0.0)
    return {
        "target_rank": ranks[target_id],
        "decoy_rank": ranks.get(decoy_id),
        "target_similarity": round(target_sim, 4),
        "decoy_similarity": round(sims.get(decoy_id, 0.0), 4),
        "best_competitor_similarity": round(best_other, 4),
        "margin": round(target_sim - best_other, 4),
        "target_minus_decoy": round(target_sim - sims.get(decoy_id, 0.0), 4),
    }


def exact_binomial_p(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p). Exact, no scipy."""
    if n == 0:
        return 1.0
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def evaluate(scenarios: list[dict], mem: EpisodicMemory, ids: dict, query_key: str) -> dict:
    n_total = mem.count()
    rows = []
    for n, s in enumerate(scenarios):
        if n % 20 == 0:
            print("  [%s] querying: %d/%d" % (query_key, n, len(scenarios)), flush=True)
        target_id = ids["%s:target" % s["scenario_id"]]
        decoy_id = ids["%s:decoy" % s["scenario_id"]]
        row = rank_of(mem, s[query_key], target_id, decoy_id)
        row["scenario_id"] = s["scenario_id"]
        row["domain"] = s["domain"]
        rows.append(row)

    resolved = [r for r in rows if r["target_rank"] is not None]
    print("  [%s] %d/%d scenarios resolved" % (query_key, len(resolved), len(scenarios)), flush=True)
    top1 = sum(1 for r in resolved if r["target_rank"] == 1)
    top3 = sum(1 for r in resolved if r["target_rank"] <= 3)
    mrr = sum(1.0 / r["target_rank"] for r in resolved) / len(resolved) if resolved else 0.0
    chance = 1.0 / n_total
    wins = sum(1 for r in resolved
               if r["decoy_rank"] is not None and r["target_rank"] < r["decoy_rank"])
    return {
        "query_type": query_key,
        "store_size": n_total,
        "scenarios": len(scenarios),
        "resolved": len(resolved),
        "top1_accuracy": round(top1 / len(resolved), 4) if resolved else None,
        "top3_accuracy": round(top3 / len(resolved), 4) if resolved else None,
        "mrr": round(mrr, 4),
        "chance_top1": round(chance, 6),
        "exact_binomial_p_vs_chance": exact_binomial_p(top1, len(resolved), chance) if resolved else None,
        "target_beats_decoy": wins,
        "target_beats_decoy_rate": round(wins / len(resolved), 4) if resolved else None,
        "mean_target_minus_decoy": round(
            sum(r["target_minus_decoy"] for r in resolved) / len(resolved), 4) if resolved else None,
        "mean_margin": round(sum(r["margin"] for r in resolved) / len(resolved), 4) if resolved else None,
        "by_domain": {
            d: {
                "n": sum(1 for r in resolved if r["domain"] == d),
                "top1": sum(1 for r in resolved if r["domain"] == d and r["target_rank"] == 1),
                "beats_decoy": sum(1 for r in resolved if r["domain"] == d
                                   and r["decoy_rank"] is not None
                                   and r["target_rank"] < r["decoy_rank"]),
            }
            for d in sorted({r["domain"] for r in resolved})
        },
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Measure episodic-memory retrieval quality")
    ap.add_argument("--scenarios", type=int, default=60)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--persist-dir", default=str(ROOT / "data" / "retrieval_bench"))
    ap.add_argument("--output", default=str(ROOT / "research" / "results" / "retrieval_quality.json"))
    args = ap.parse_args()

    persist_dir = Path(args.persist_dir)
    scenarios = generate_scenarios(args.scenarios, args.seed)
    print("Generating %d scenarios..." % len(scenarios), flush=True)
    mem, ids = build_store(scenarios, persist_dir)
    print("Store populated: %d episodes from %d scenarios" % (mem.count(), len(scenarios)), flush=True)

    out = {
        "experiment": "episodic-memory retrieval quality",
        "date": "2026-10-21",
        "seed": args.seed,
        "embedding_model": "all-MiniLM-L6-v2 (384-dim, cosine) via ChromaDB ONNXMiniLM_L6_V2",
        "store_size": mem.count(),
        "chance_level_top1": round(1.0 / mem.count(), 6),
        "general": evaluate(scenarios, mem, ids, "query_general"),
        "specific": evaluate(scenarios, mem, ids, "query_specific"),
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    for key in ("general", "specific"):
        r = out[key]
        print("\n[%s] top1=%.4f  top3=%.4f  mrr=%.4f  chance=%.6f  p=%.3g"
              % (key, r["top1_accuracy"], r["top3_accuracy"], r["mrr"],
                 r["chance_top1"], r["exact_binomial_p_vs_chance"]))
        print("      target beats decoy: %d/%d (%.1f%%)  mean(target-decoy)=%.4f"
              % (r["target_beats_decoy"], r["resolved"], 100 * r["target_beats_decoy_rate"],
                 r["mean_target_minus_decoy"]))
        print("      by domain: %s" % {d: "top1 %d/%d, beats_decoy %d" % (v["top1"], v["n"], v["beats_decoy"])
                                       for d, v in r["by_domain"].items()})
    print("\nwrote %s" % out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
