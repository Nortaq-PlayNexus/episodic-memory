# Episodic Memory for AI Agents

**The closest thing to "having been there" that current technology allows.**

## What this is

A persistent episodic memory system for AI agents. Unlike semantic memory (which stores facts like "user prefers TypeScript"), episodic memory stores *experiences* — what happened, when, in what context, what was decided, what was learned, and the agent's evaluative state at the time.

The distinction matters. Tulving (1972, 1985) argued that episodic memory involves "mental time travel" — the ability to *relive* a past experience, not just retrieve a fact about it. Current AI memory systems (Mem0, Zep, Letta) store facts and retrieve them. This system stores *experiences* and presents them in a way that encourages reliving rather than just reading.

## What this is NOT

This is not consciousness. It is not subjective experience. It is not "true" memory. It is the best available approximation — a system that captures the *texture* of experience in structured form, stores it durably, and retrieves it in context.

The gap between "retrieving a summary of what happened" and "having the experience of having been there" is the gap between data and consciousness. No one has closed that gap. This system doesn't pretend to. It just makes the retrieval better.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  OpenCode Agent                                     │
│  ├── remember (store episode)                       │
│  ├── recall (semantic search)                       │
│  ├── relive (present in experiential context)       │
│  ├── timeline (chronological view)                  │
│  └── stats (store statistics)                       │
└──────────────────────┬──────────────────────────────┘
                       │ MCP (stdio)
                       ▼
┌─────────────────────────────────────────────────────┐
│  Episodic Memory Server (Python)                    │
│  ├── ChromaDB (vector storage + semantic search)    │
│  ├── ONNXMiniLM_L6_V2 (all-MiniLM-L6-v2, 384-dim)   │
│  └── JSONL append-only log (durable record)         │
└─────────────────────────────────────────────────────┘
```

## Episode structure

Each episode captures:

| Field | Purpose |
|---|---|
| `content` | What happened (the narrative) |
| `context` | Surrounding context (what was being worked on, who was involved) |
| `event_type` | conversation, decision, discovery, error, fix, observation, reflection |
| `decisions` | Decisions made during this episode |
| `lessons` | What was learned |
| `valence` | Evaluative tone: positive, negative, neutral, surprising, frustrating |
| `confidence` | 0.0-1.0, how confident the agent is about this memory |
| `tags` | Searchable tags |
| `related_episodes` | IDs of related episodes |
| `session_id` | Which session this happened in |
| `timestamp` | When it happened |

## Usage

### Store an episode

```python
from src.episodic_memory import EpisodicMemory, Episode

mem = EpisodicMemory(persist_dir="data")

ep = Episode(
    content="Found that safe_write left SafeWrite.psm1 in a mutated state.",
    context="Auditing the mem-integrity tooling",
    event_type="discovery",
    decisions=["Restore the mutated line to original newline-detection code"],
    lessons=["Mutation harnesses can leave files in mutated states if their restore fails silently"],
    valence="surprising",
    confidence=0.9,
    tags=["mutation-harness", "safe_write", "audit"],
    session_id="audit-2026-10-17",
)
eid = mem.remember(ep)
```

### Recall episodes

```python
results = mem.recall("mutation harness left file in broken state", n_results=3)
for r in results:
    print(f"[{r['similarity']:.3f}] {r['episode']['content'][:100]}")
```

### Relive an episode

```python
narrative = mem.relive(eid)
print(narrative)
```

### MCP server

```bash
python -m src.mcp_server --persist-dir /path/to/data
```

Add to OpenCode config (`opencode.jsonc`):

```json
{
  "mcp": {
    "servers": {
      "episodic-memory": {
        "type": "local",
        "command": ["python", "-m", "src.mcp_server", "--persist-dir", "C:/Users/natha/projects/episodic-memory/data"]
      }
    }
  }
}
```

## Self-authored continuity

The system supports self-authored continuity artifacts — files where the AI writes about its own
experiences in first person. These are stored as episodes with `source: "self"` and special tags.

The idea: the agent doesn't just record facts. It writes about what it was like to be in the
conversation. What was surprising. What was hard. What it learned. What it would do differently.

These artifacts are then retrieved at session start and presented as context — not as facts to
remember, but as experiences to re-encounter.

## Science connection

This system is designed to be pointed at the question: **what would it take to make memory
experience in AI possible?**

Current research says:
- Episodic memory (Tulving) requires "mental time travel" — reliving, not just retrieving
- No current AI system achieves this
- The gap is between data (retrieval) and consciousness (subjective experience)
- The "Identity as Trajectory" paper (2026) shows longitudinal interaction produces stable persona
- But: "greater fidelity strengthens the reconstruction without establishing identity"

This system provides:
- Structured episodic storage (event + context + decisions + lessons + valence)
- Semantic retrieval by embedding similarity (cosine over all-MiniLM-L6-v2), most similar first
- A "reliving" interface that presents full context
- A longitudinal record that can be studied over time

What it does NOT provide:
- Subjective experience
- True "reliving" (only better retrieval)
- Consciousness of any kind

The gap between what we've built and what we need is the gap between data and consciousness.
This system is designed to be studied, not to pretend it has already solved the problem.

## License

MIT