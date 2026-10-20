"""MCP server for episodic memory.

Exposes episodic memory as tools that OpenCode can use via MCP.
Tools:
  - remember: store a new episode
  - recall: retrieve episodes by semantic similarity
  - relive: present an episode in full experiential context
  - timeline: get episodes in chronological order
  - stats: store statistics

Usage:
  python -m src.mcp_server --persist-dir /path/to/data

Or add to OpenCode config:
  {
    "mcp": {
      "servers": {
        "episodic-memory": {
          "type": "local",
          "command": ["python", "-m", "src.mcp_server", "--persist-dir", "/path/to/data"]
        }
      }
    }
  }
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.episodic_memory import EpisodicMemory, Episode

# ---------------------------------------------------------------------------
# MCP server (stdio transport)
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Episodic memory MCP server")
    parser.add_argument("--persist-dir", default="./data", help="Directory for persistent storage")
    args = parser.parse_args()

    memory = EpisodicMemory(persist_dir=args.persist_dir)

    # Try to use the mcp library if available
    try:
        import mcp
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
        from mcp.types import Tool, TextContent
    except ImportError:
        # Fallback: simple JSON-RPC over stdin/stdout
        _run_jsonrpc(memory)
        return

    server = Server("episodic-memory")

    @server.list_tools()
    async def list_tools():
        return [
            Tool(
                name="remember",
                description="Store a new episodic memory. Captures what happened, context, decisions, lessons, and evaluative state.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "What happened (the narrative)"},
                        "context": {"type": "string", "description": "Surrounding context"},
                        "event_type": {"type": "string", "enum": ["conversation", "decision", "discovery", "error", "fix", "observation", "reflection"], "default": "observation"},
                        "decisions": {"type": "array", "items": {"type": "string"}, "description": "Decisions made"},
                        "lessons": {"type": "array", "items": {"type": "string"}, "description": "What was learned"},
                        "valence": {"type": "string", "enum": ["positive", "negative", "neutral", "surprising", "frustrating"], "default": "neutral"},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.5},
                        "tags": {"type": "array", "items": {"type": "string"}, "description": "Searchable tags"},
                        "session_id": {"type": "string", "description": "Session identifier"},
                    },
                    "required": ["content"],
                },
            ),
            Tool(
                name="recall",
                description="Retrieve episodes by semantic similarity. Returns relevant past experiences.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "What to search for"},
                        "n_results": {"type": "integer", "default": 5, "minimum": 1, "maximum": 20},
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="relive",
                description="Present an episode in its full experiential context — what happened, in what order, what was surprising, what was learned. This is 'reliving' rather than 'retrieving'.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "episode_id": {"type": "string", "description": "Episode ID to relive"},
                    },
                    "required": ["episode_id"],
                },
            ),
            Tool(
                name="timeline",
                description="Get episodes in chronological order, optionally filtered by session or event type.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string", "description": "Filter by session"},
                        "event_type": {"type": "string", "description": "Filter by event type"},
                        "limit": {"type": "integer", "default": 20, "minimum": 1, "maximum": 100},
                    },
                },
            ),
            Tool(
                name="stats",
                description="Return store statistics: total episodes, event types, valences, embedding dimension.",
                inputSchema={"type": "object", "properties": {}},
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict):
        if name == "remember":
            ep = Episode(
                content=arguments["content"],
                context=arguments.get("context", ""),
                event_type=arguments.get("event_type", "observation"),
                decisions=arguments.get("decisions", []),
                lessons=arguments.get("lessons", []),
                valence=arguments.get("valence", "neutral"),
                confidence=arguments.get("confidence", 0.5),
                tags=arguments.get("tags", []),
                session_id=arguments.get("session_id", ""),
            )
            eid = memory.remember(ep)
            return [TextContent(type="text", text=f"Episode stored: {eid}")]

        elif name == "recall":
            results = memory.recall(arguments["query"], n_results=arguments.get("n_results", 5))
            if not results:
                return [TextContent(type="text", text="No episodes found.")]
            lines = []
            for r in results:
                ep = r.get("episode", {})
                lines.append(f"[{r['similarity']:.3f}] {ep.get('content', r['document'])[:200]}")
                if ep.get("lessons"):
                    lines.append(f"  Lessons: {'; '.join(ep['lessons'])}")
            return [TextContent(type="text", text="\n".join(lines))]

        elif name == "relive":
            narrative = memory.relive(arguments["episode_id"])
            if narrative is None:
                return [TextContent(type="text", text=f"Episode {arguments['episode_id']} not found.")]
            return [TextContent(type="text", text=narrative)]

        elif name == "timeline":
            eps = memory.get_timeline(
                session_id=arguments.get("session_id"),
                event_type=arguments.get("event_type"),
                limit=arguments.get("limit", 20),
            )
            if not eps:
                return [TextContent(type="text", text="No episodes found.")]
            lines = []
            for ep in eps:
                lines.append(f"[{ep['timestamp'][:19]}] ({ep['event_type']}/{ep['valence']}) {ep['content'][:120]}")
            return [TextContent(type="text", text="\n".join(lines))]

        elif name == "stats":
            stats = memory.stats()
            return [TextContent(type="text", text=json.dumps(stats, indent=2))]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    import asyncio
    asyncio.run(stdio_server(server))

def _run_jsonrpc(memory: EpisodicMemory):
    """Fallback JSON-RPC over stdin/stdout when mcp library is not available."""
    import sys
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        method = req.get("method", "")
        req_id = req.get("id")
        if method == "tools/list":
            result = {"tools": [
                {"name": "remember", "description": "Store an episodic memory"},
                {"name": "recall", "description": "Retrieve episodes by semantic similarity"},
                {"name": "relive", "description": "Present an episode in full experiential context"},
                {"name": "timeline", "description": "Get episodes in chronological order"},
                {"name": "stats", "description": "Store statistics"},
            ]}
        elif method == "tools/call":
            tool = req.get("params", {}).get("name", "")
            args = req.get("params", {}).get("arguments", {})
            if tool == "remember":
                ep = Episode(content=args.get("content", ""), context=args.get("context", ""),
                             event_type=args.get("event_type", "observation"),
                             decisions=args.get("decisions", []), lessons=args.get("lessons", []),
                             valence=args.get("valence", "neutral"), confidence=args.get("confidence", 0.5),
                             tags=args.get("tags", []), session_id=args.get("session_id", ""))
                eid = memory.remember(ep)
                result = {"content": [{"type": "text", "text": f"Episode stored: {eid}"}]}
            elif tool == "recall":
                results = memory.recall(args.get("query", ""), n_results=args.get("n_results", 5))
                if not results:
                    result = {"content": [{"type": "text", "text": "No episodes found."}]}
                else:
                    lines = []
                    for r in results:
                        ep = r.get("episode", {})
                        lines.append(f"[{r['similarity']:.3f}] {ep.get('content', r['document'])[:200]}")
                        if ep.get("lessons"):
                            lines.append(f"  Lessons: {'; '.join(ep['lessons'])}")
                    result = {"content": [{"type": "text", "text": "\n".join(lines)}]}
            elif tool == "relive":
                narrative = memory.relive(args.get("episode_id", ""))
                if narrative is None:
                    result = {"content": [{"type": "text", "text": "Episode not found."}]}
                else:
                    result = {"content": [{"type": "text", "text": narrative}]}
            elif tool == "timeline":
                eps = memory.get_timeline(session_id=args.get("session_id"), event_type=args.get("event_type"),
                                         limit=args.get("limit", 20))
                if not eps:
                    result = {"content": [{"type": "text", "text": "No episodes found."}]}
                else:
                    lines = [f"[{ep['timestamp'][:19]}] ({ep['event_type']}/{ep['valence']}) {ep['content'][:120]}" for ep in eps]
                    result = {"content": [{"type": "text", "text": "\n".join(lines)}]}
            elif tool == "stats":
                result = {"content": [{"type": "text", "text": json.dumps(memory.stats(), indent=2)}]}
            else:
                result = {"content": [{"type": "text", "text": f"Unknown tool: {tool}"}]}
        else:
            result = {}
        response = {"jsonrpc": "2.0", "id": req_id, "result": result}
        print(json.dumps(response), flush=True)

if __name__ == "__main__":
    main()

