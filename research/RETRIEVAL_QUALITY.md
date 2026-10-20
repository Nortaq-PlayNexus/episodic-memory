# Retrieval Quality of the Episodic Memory Store

**Date:** 2026-10-21
**Artefact:** `experiments/retrieval_benchmark.py` -> `research/results/retrieval_quality.json`
**Seed:** 20261021 · **Store:** 270 episodes in one shared store · **Embedding:** all-MiniLM-L6-v2
(384-dim, cosine) via ChromaDB's `ONNXMiniLM_L6_V2`

## Why this measurement exists

Every claim that episodic memory improves continuity rests on a step nobody had measured: that
the store returns the *right* episode. The system had a test suite (10 tests) and a README, and
nothing that showed retrieval was correct. The store had also never been populated with real
episodes in this repository — `data/` held only the self-authored identity and journal files.

## Design

60 scenarios, 5 domains, 12 scenarios each. Every scenario puts several episodes into **one
shared store** (the realistic condition for a memory that accumulates, and much harder than a
per-scenario mini-store):

- **target** — the episode the query is about
- **decoy** — a near-duplicate: same subject, same template, one value changed
  (`512 MB` vs `256 MB`; "must verify file integrity" vs "must log all operations")
- **distractors** — 2-3 topically adjacent but irrelevant episodes

Each scenario has a **unique subject**, so the target is uniquely identifiable by the query.
This matters: a first draft of this benchmark gave all 12 scenarios of a domain the same text
apart from an index number, and top-1 accuracy came out at 0.17 for queries that were verbatim
substrings of the target. That was the benchmark measuring a lottery between near-identical
episodes, not the store failing. The redesign is recorded here because the trap is easy to hit
again.

Two query types:

| Query type | Contains | Question |
|---|---|---|
| `general` | the subject, not the value | can the store find the right episode at all? |
| `specific` | the subject **and** the distinguishing value | can the store tell the target from its near-duplicate decoy? |

The sharp measure is **`target_beats_decoy`** — among the two episodes that both answer the
question, does the store prefer the correct one? Top-1 accuracy alone cannot separate "found
the right pair" from "picked the right one of the pair".

## Results

| Metric | `general` | `specific` |
|---|---:|---:|
| top-1 accuracy (of 270 episodes) | **0.500** | **0.850** |
| top-3 accuracy | 0.983 | 1.000 |
| MRR | 0.746 | 0.925 |
| chance level (1/270) | 0.0037 | 0.0037 |
| exact binomial p vs chance | 1.2e-56 | 1.4e-114 |
| **target beats decoy** | **31/60 (51.7%)** | **51/60 (85.0%)** |
| mean similarity(target) − similarity(decoy) | **+0.009** | **+0.117** |

### Per domain

| Domain | `general` top-1 | `general` beats decoy | `specific` top-1 | `specific` beats decoy |
|---|---:|---:|---:|---:|
| commitment | 11/12 | 12/12 | 12/12 | 12/12 |
| deployment | 12/12 | 12/12 | 12/12 | 12/12 |
| config | 4/12 | 4/12 | 11/12 | 11/12 |
| constraint | 3/12 | 3/12 | 4/12 | 4/12 |
| rate_limit | **0/12** | **0/12** | 12/12 | 12/12 |

## What this means

**1. The store is not a random retriever, and it finds the right neighbourhood reliably.**
Top-3 accuracy is 0.98-1.00 and MRR 0.75-0.93 against a chance level of 0.0037, with p-values
of 1e-56 and 1e-114. Whatever else is true, `recall()` puts the relevant episode in the top few
results essentially every time.

**2. It cannot tell a target from its near-duplicate decoy on a general query.** 31 of 60 is a
coin flip, and the mean similarity gap is +0.009 — statistically indistinguishable from zero
on a 0-1 cosine scale. **The value that distinguishes the two episodes (`512 MB` vs `256 MB`)
is effectively invisible to the embedding.** The vectors are dominated by the shared subject and
template text, and one differing number token barely moves them.

**3. When the query itself carries the value, the store does prefer the correct episode** —
85% (51/60), mean gap +0.117, rising to 12/12 in three of five domains. The information is
present in the embedding; it is simply swamped unless the query weights it too. This is a
property of how the query is framed, not of the store's index.

**4. The per-domain spread is large and partly systematic, not noise.** `rate_limit` general is
**0/12** — the decoy beats the target every single time. Its decoy wording ("rate limiting
improves user experience") sits closer to a value-neutral question ("What rate-limit threshold
was set?") than the target's wording ("prevents cascade failures") does. `constraint` is weak in
*both* query types (3/12 and 4/12) despite the specific query containing the number. By
contrast `commitment` and `deployment`, whose decoys differ in *wording* rather than in a
number, are near-perfect on general queries (11/12, 12/12).

**4a. The mechanism: the embedding separates wording, not values.** Measuring the similarity
margin between the target and its decoy directly, in a small store, with the query carrying
the distinguishing information:

| Pair differs in | margin (target − decoy) | target ranked first? |
|---|---:|---|
| wording ("verify file integrity" vs "log all operations") | **+0.249** | yes |
| number + phrase ("prevents cascade failures … 1000 req/min") | **+0.093** | yes |
| number only ("reports 1000 rows" vs "998 rows") | **+0.004** | yes, but by a hair |
| number only ("512 MB" vs "256 MB") | **−0.005** | **no** |

A difference in *wording* moves the cosine similarity by a quarter of the scale. A difference in
a *number*, in an otherwise identical sentence, moves it by five thousandths — noise. The
85% `specific` figure in the table above is therefore **not** evidence that the store is
value-sensitive: it is the subject phrase doing the work, and the near-tie is broken by the
background of 270 competing episodes. The pair is genuinely ambiguous.

This is a property of a small sentence embedding, and it is the single most important thing
this project learned about itself: **a memory store that embeds episodes cannot reliably tell
two episodes apart when they differ only in a value.**

**5. The practical consequence for CE-1:** a continuity experiment built on this store cannot
claim that the *value* inside a retrieved episode drove the agent's answer. If the agent
retrieves by topic ("the constraint about memory allocation"), it will often get the wrong one of
a near-duplicate pair, and the "memory" will be blamed for a retrieval error that is really an
embedding-resolution error.

**6. What would fix it, if it needs fixing.** Any of: a larger or value-aware encoder; storing
the distinguishing value in a field that is not averaged away by the shared sentence (a keyword
or metadata filter alongside the vector — ChromaDB supports `where=` filters and the store
already exposes `recall(where=...)`, but the episode schema has no such field); or indexing the
value as a separate searchable token. None of these is implemented here, and the choice depends
on a question this measurement does not answer: whether value-level discrimination is needed at
all, or whether it is an artefact of a deliberately adversarial benchmark.

## What this does NOT establish

- **Not a typical-case accuracy.** The decoys were built to differ by one token — a deliberate
  worst case. Real stores contain episodes that differ more.
- **Not a claim about agents.** Nothing here measures whether retrieval helps an agent; it
  measures whether the store returns the right episode. Whether the *use* of a retrieved episode
  changes behaviour is a separate experiment (CE-1's H8 line).
- **Not a comparison of encoders.** only all-MiniLM-L6-v2 was tested. A larger or
  value-aware encoder might separate `512` from `256` where this one does not.
- **Not a statement about the causal-integration question** in `SCIENCE_CONNECTION.md`. That is
  about whether retrieval influences reasoning, not about retrieval quality.

## Reproduction

```
cd C:\Users\natha\projects\episodic-memory
$env:ANONYMIZED_TELEMETRY='False'
.\.venv\Scripts\python.exe experiments\retrieval_benchmark.py
```

`ANONYMIZED_TELEMETRY=False` is required only because ChromaDB's telemetry blocks store
initialisation on this box; the code also sets it in `Settings(anonymized_telemetry=False)`.
The run is deterministic: same seed, same 270 episodes, same 120 retrievals.
