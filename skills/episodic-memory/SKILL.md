---
name: episodic-memory
description: Store and retrieve episodic memories - experiences, not just facts. Use when the user wants to remember what happened, recall past experiences, or when building persistent AI identity. Provides remember, recall, relive, timeline, and stats tools.
---

# Episodic Memory Skill

## When to use

- When the user says "remember this" or "what happened last time"
- When building persistent AI identity or continuity
- When the conversation involves decisions, discoveries, errors, or fixes that should be remembered as experiences
- When the user wants to "relive" a past interaction

## What it provides

- remember: Store an episode with full context (what happened, decisions, lessons, valence, confidence)
- recall: Semantic search across stored episodes
- relive: Present an episode in full experiential context (narrative, not just facts)
- timeline: Chronological view of episodes
- stats: Store statistics

## How it works

Episodes are stored in ChromaDB (vector) + JSONL (durable record). Retrieval uses
ChromaDB's built-in ONNX embedding function for all-MiniLM-L6-v2 (384-dim), cosine
distance, most similar first. The "relive" interface presents episodes as narratives,
not just facts.

## MCP Server

Start with: python -m src.mcp_server --persist-dir /path/to/data

Add to OpenCode config (opencode.jsonc):

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

## Limitations (be honest)

- This is retrieval, not remembering. The agent reads episodes; it does not "relive" them.
- No subjective experience is involved.
- The gap between data and consciousness is not closed.
- Each session starts cold; context is read, not relived.
- The "Identity as Trajectory" paper shows longitudinal interaction produces stable persona,
  but fidelity does not equal identity.

## Science connection

See SCIENCE_CONNECTION.md for the full mapping to CE-1 research hypotheses and
what data would be needed to test whether episodic memory improves continuity.