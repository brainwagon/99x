# agent99x

A streamlined rewrite of [agent99](../99) — a small, local-LLM coding agent.
Same core loop, fewer moving parts. See `ASCETIC.md` (in the original repo) for
the guiding principles and `PROPOSAL.md` for the reasoning behind the cuts.

## What it is

A classic agent loop over an OpenAI-compatible chat API:

```
build system prompt (AGENT.md + memory + skills)
loop:
    stream a completion
    if tool_calls: dispatch each, append results, continue
    else: return text
```

Two modes:

| Command | Mode | Description |
|---|---|---|
| `99xsh <prompt>` | One-shot | Single query, prints reply, exits. Stateless. |
| `99x` | TUI | Interactive Textual UI with a persistent session. |

## Setup

Requires a running [ollama](https://ollama.com) (default, `localhost:11434`)
with a tool-capable model pulled, or an OpenRouter key for remote models.

```
pip install -e .          # installs deps; provides 99x / 99xsh
99x init                  # seed ~/.99x from agent99x/_defaults
99xsh --model gemma4:e4b-it-qat "what time is it?"
99x                       # interactive TUI
```

Optional MCP support: `pip install -e .[mcp]`, then drop an `mcp-config.json`
in the project or `~/.99x`.

## Layout

Files live flat in `agent99x/`. Global state lives in `~/.99x/`
(override with `AGENT_HOME`); per-project state in `.99x/`.

## What differs from agent99

Cut: handoff/contract/evaluator subsystem, async/background subagents,
blackboard, heartbeat, and the weather/web_search/fetch_rss built-in tools
(now skills). MCP is gated behind a config file. Providers trimmed to ollama
and openrouter.
