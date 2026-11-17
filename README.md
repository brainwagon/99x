# agent99x

A second implementation of a simple AI-agent harness, building on the original
[agent99](../99). The goal: a smaller, tighter codebase that's easier for others
to pick up, extend, and run against their own local models. Same core loop,
fewer moving parts — about half the source of the original.

## Quick Start

Requires a running [ollama](https://ollama.com) (default, `localhost:11434`) with
a tool-capable model pulled, or an [OpenRouter](https://openrouter.ai) API key.

```bash
pip install -e .              # installs deps; provides 99x / 99xsh
99x init                      # seed ~/.99x with defaults
99xsh --model gemma3:12b "what time is it?"
99x                           # interactive TUI
```

Optional MCP support: `pip install -e .[mcp]`, then drop an `mcp-config.json` in
the project root or `~/.99x`.

## How It Works

A classic agent loop over an OpenAI-compatible chat API:

```
build system prompt (AGENT.md + memory + skills/agent catalogs)
loop:
    stream a completion
    if tool_calls: dispatch each (parallel when >1), append results, continue
    else: return text
```

Two entry points:

| Command | Mode | Description |
|---|---|---|
| `99xsh <prompt>` | One-shot | Single query, prints reply, exits. Stateless. |
| `99x` | TUI | Interactive [Textual](https://textual.textualize.io/) TUI with persistent session. |

Pass `--provider`, `--model`, `--host`, `--effort` (`low`, `medium`,
`high`, or `none` to clear it), `--plan`, or `-c`/`--context` to either
command. In the TUI, use slash commands to change these at runtime.
At startup, known model-specific advisories and (for OpenRouter models)
pricing may be shown.

`--provider` restores the model and host last used with that provider;
an explicit `--model`/`--host` overrides them.

**Stateless one-shot runs.** `99xsh` requires a prompt (without one it
prints usage and exits with status 2). A one-shot run (`99xsh <prompt>`,
or `99x <prompt>`) uses its flags for that run only: it does not write
`config.json`, and it writes no context file unless `-c` names a context
to continue, in which case the new turn is appended to that context. The
interactive `99x` TUI, by contrast, remembers flag and slash-command
choices in `config.json`.

Add `--trace-hooks` to trace metadata-only lifecycle events (see
[Lifecycle Hook Trace](#lifecycle-hook-trace)).

## Three Kinds of Capability

The agent is taught a simple decision tree for choosing how to act:

| Kind | What | Invoke via | Runs code? |
|---|---|---|---|
| **Tool** | A function implemented in Python. | Native function call. | Yes — the tool *is* the code. |
| **Skill** | Written instructions (Markdown) for a procedure. | `load_skill(name)`, then act in the current context. | No — prose only; may point at scripts the model runs with `run_bash`. |
| **Agent** | A fresh specialist worker with its own loop. | `spawn_agent(task, name)`; runs in its own context and returns a compressed result. | Indirectly — it runs its own full tool loop. |

**Decision rule: do it yourself → tool; need the recipe → skill; hand it off → agent.**

Every system prompt carries a `# YOUR CAPABILITIES` block with a prescriptive
primer plus catalogs of all three kinds. Three symmetric discovery tools
(`list_tools`, `list_skills`, `list_agents`) exist for runtime enumeration.
Skills and agents use the [agentskills.io](https://agentskills.io) convention
for directory layout and `SKILL.md`/`AGENT.md` frontmatter.

See `docs/three-kinds.md` for the full design rationale.

## Built-in Tools

### Files
| Tool | Description |
|---|---|
| `read_file` | Read a file with optional offset, limit, and line numbers. |
| `write_file` | Overwrite a file. Prefer `edit_file` for targeted changes. |
| `edit_file` | Replace exact-match text. Tolerates CRLF/LF and trailing-whitespace drift. Supports `replace_all`. |
| `replace_lines` | Replace a contiguous line range by number (1-based, end inclusive). Use `end=start-1` to insert without deleting. |
| `patch` | Apply a unified diff with strict context matching. |

### Shell
| Tool | Description |
|---|---|
| `run_bash` | Run a command under `/bin/bash` (30s timeout, 1 MB output cap per stream, stdin is `/dev/null`). Returns when the shell exits; background jobs keep running. Cancel-aware via per-thread event. |

### Search
| Tool | Description |
|---|---|
| `grep` | Regex search with optional path and glob filter. Uses `ripgrep` if available; falls back to Python. Up to 200 matches. |
| `glob` | File discovery by glob pattern. `**/` matches zero or more directories, so `**/*.py` includes top-level files; `*.py` matches only the top level. Up to 500 matches. |

### Planning
| Tool | Description |
|---|---|
| `write_todos` | Replace the todo list. Supports `pending` / `in_progress` / `done` statuses. |
| `read_todos` | Read the current todo list. |

### Info
| Tool | Description |
|---|---|
| `current_datetime` | Current date/time with optional IANA timezone. Returns date, time, weekday, epoch, UTC offset. |

### Network
| Tool | Description |
|---|---|
| `http_request` | HTTP client (GET/POST/PUT/PATCH/DELETE/HEAD) with configurable timeout and response body cap (default 100 KB, set with `max_bytes`; the body is streamed and reading stops at the cap). |

### Meta (bridge tools — hidden from catalog, covered by primer)
| Tool | Description |
|---|---|
| `list_tools` | Exhaustive list of every registered tool (including MCP tools). |
| `list_skills` | Available skills by name and description. |
| `list_agents` | Available agents by name and description. |
| `load_skill` | Load a skill's full instructions and its directory path. |
| `spawn_agent` | Delegate a task to a specialist agent; returns its compressed result. |

## Skills

Skills are pure Markdown instructions that drop into `~/.99x/skills/<name>/` (or
`.99x/skills/<name>/` for project-local overrides). They follow the
[agentskills.io](https://agentskills.io) `SKILL.md` convention with YAML-ish
frontmatter (`name`, `description`, `license`, `compatibility`, `metadata`).
The parser handles a dependency-free YAML subset: plain and quoted
scalars, indented continuation lines, `[a, b]` and `- item` lists,
`>`/`|` block scalars (with `-`/`+` chomping), and nested maps.

**Skills do not run code.** They tell the model what to do with existing tools.
Scripts bundled in `scripts/` are run via `run_bash` — `load_skill` returns the
skill's `dir` so relative paths resolve.

Project-local skills shadow global ones (same precedence as agents).

### Bundled skills

| Skill | Description |
|---|---|
| `weather` | Current conditions + 5-day forecast via open-meteo.com (no API key). |
| `web_search` | Search via DuckDuckGo HTML endpoint (no API key). |
| `fetch_recent_rss` | Fetch the N most recent RSS/Atom items (default 5) via a bundled `scripts/fetch_rss.py`, keeping raw XML out of context; `--full` pulls full item content for reasoning. |
| `grill-me` | Interview the user relentlessly about a plan or design. |

## Agents (Subagents)

Agents are specialist models invoked via `spawn_agent(task, name)`. Each agent:

- Gets its own system prompt built from its `AGENT.md`.
- Runs a full `agent_loop` to completion and returns one compressed text result.
- May carry `allowed_tools` (whitelist), `model`, `provider`, and `effort`
  overrides specified in its frontmatter.
- Respects a global call budget shared across the agent tree.
- Limited to `MAX_AGENT_DEPTH` (3) to prevent runaway recursion.

Agent `AGENT.md` files live in `~/.99x/agents/<name>/` (or `.99x/agents/<name>/`
for project-local overrides). The frontmatter supports:

| Field | Description |
|---|---|
| `description` | What the agent does — appears in `list_agents` and the catalog. |
| `allowed_tools` | List of tool names the agent may call (enforced by the loop). |
| `model` | Override the model for this agent. |
| `provider` | Override the provider for this agent. |
| `host` | Override the host for this agent. |
| `effort` | Override reasoning effort for this agent. |
| `inherit_memory` | If truthy, the agent's system prompt includes the global `~/.99x/MEMORY.md`. |

## Providers

Two providers ship with the agent:

| Provider | Default URL | API Key | Notes |
|---|---|---|---|
| `ollama` | `http://localhost:11434/v1` | `ollama` | Local inference. `call_budget` = unlimited. |
| `openrouter` | `https://openrouter.ai/api/v1` | `$OPENROUTER_API_KEY` | Remote. `call_budget` = 100. |

Provider/model selection is persisted in `~/.99x/config.json`. The per-provider
host override (`--host` / `/host`) is also persisted, so you can save a remote
ollama address (e.g. `10.0.0.5:11434`) and switch back to it later.

Context-window size is lazily fetched from the provider on first use and cached
for the session. Ollama queries `/api/show`; OpenRouter queries `/models`.

## MCP (Model Context Protocol)

Gated behind an `mcp-config.json` file (project root or `~/.99x`). Requires
`pip install -e .[mcp]`.

When a config file is present, the agent starts an MCP event loop on a
background thread, connects to each configured server over stdio, and registers
their tools in the OpenAI function-calling format (prefixed `mcp__<server>__`;
names are sanitized to `[A-Za-z0-9_-]`, max 64 chars). MCP tools are blocked
in plan mode unless the server marks them `readOnlyHint`. If a server dies,
its tools return a "disconnected" error; there is no automatic reconnect.

Global MCP tools (`mcp_list_resources`, `mcp_read_resource`, `mcp_list_prompts`,
`mcp_get_prompt`) are always registered when MCP is active.

Example `mcp-config.json`:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    }
  }
}
```

## TUI Slash Commands

Available in the interactive TUI (`99x`):

| Command | Description |
|---|---|
| `/help` | List all commands and available skills. |
| `/provider <name>` | Switch provider (interactive model picker). |
| `/model` | Pick a different model on the current provider. |
| `/host [<provider> [<addr>\|clear]]` | Manage per-provider host overrides. |
| `/effort [low\|medium\|high\|none]` | Set reasoning effort; with no argument, show options and current setting. |
| `/thinking [off\|terse\|full]` | Show available modes, or set the thinking display mode. |
| `/plan` | Toggle plan mode (blocks writes, restricts shell to read-only). |
| `/compact` | Summarize conversation history to free context. |
| `/budget [<n>\|unlimited\|default]` | Show or set the global call budget. |
| `/todos` | List current todos. |
| `/todo [add\|done\|rm] <text>` | Manage todos inline. |
| `/clear` or `/new` | Clear conversation history and screen. |
| `/exit` or `/quit` | Exit the agent. |
| `/cancel` | Cancel a pending interactive prompt. |

Skills also appear as pseudo slash-commands. `/weather Paris` sends the
model a message that includes the `weather` skill's instructions (as
`load_skill` would return them), tells it to carry them out now, and
passes `Paris` as the arguments.

While an agent run (or `/compact`) is in progress, commands that would
replace the conversation history or the model client (`/clear`, `/new`,
`/compact`, `/provider <name>`, `/model`, `/host <provider> ...`, and
answering a model picker) are refused with a message; press Esc to cancel
the run first. Read-only commands such as `/help` and `/budget` still
work. Model listing for `/provider` and `/model` runs in the background,
and Esc abandons it.

## Plan Mode

Toggle with `--plan` on launch or `/plan` in the TUI. When active:

- `write_file`, `edit_file`, `replace_lines`, and `patch` are blocked.
- `run_bash` is restricted to read-only commands (checked against an
  allowlist of read-only commands).
- `http_request` is restricted to GET only.
- MCP tools are blocked unless annotated `readOnlyHint: true`.
- The system prompt instructs the model to describe rather than execute.
  The blocked-tool list in that prompt and in `/plan`'s message comes from
  the tool registry (every tool registered with `plan_safe=False`).
- Subagents inherit plan mode, including its prompt and tool blocks.

Useful for exploring a codebase safely or reviewing a plan before committing.

## Context Management

Conversations are saved as JSON files under `.99x/contexts/`. Each run writes to
one context file.

| Flag | Behavior |
|---|---|
| *(none)* | Fresh timestamped context (one-shot runs: no context file). |
| `-c` | Restore the most recently modified context for this project. |
| `-c <name>` | Restore (or create) the named context. |

The system prompt is rebuilt fresh on every turn (so AGENT.md edits take effect
immediately), but the conversation history persists.

## Compaction

When token usage exceeds `autocompact_threshold` (default 0.85) of the context
window, the agent automatically summarizes the conversation into a compact
context block, preserving decisions, file changes, discovered facts, and
work-in-progress state.

Manual compaction is available via `/compact`.

## Call Budget

A global call budget limits the total number of LLM API calls across the main
agent and all subagents. Provider defaults:

- `ollama`: unlimited
- `openrouter`: 100

There is no CLI flag for it. Use `/budget` in the TUI: `/budget <n>` sets
a limit, `/budget unlimited` removes the limit for every provider, and
`/budget default` returns to the provider default. `/budget` alone shows
calls used against the limit in effect. The choice is saved as
`call_budget` in `config.json`: `null` (provider default), a positive
integer, or `"unlimited"`.

## Validation

After every turn where file-mutating tools run, modified files are
syntax-checked. Python, JSON, YAML, INI, TOML, and XML are checked
in-process; the external checkers run with a timeout, and a missing
checker skips the check:

| Extension | Check |
|---|---|
| `.py` | `python -m py_compile` |
| `.sh`, `.bash` | `bash -n` |
| `.c` | `gcc -fsyntax-only` |
| `.cpp`, `.cc`, `.cxx`, `.h`, `.hpp` | `g++ -fsyntax-only` |
| `.js` | `node --check` |
| `.json` | `python -m json.tool` |
| `.yaml`, `.yml` | `yaml.safe_load` |
| `.ini` | `configparser` |
| `.toml` | `tomllib` (3.11+) |
| `.xml`, `.rss` | `xml.etree.ElementTree` |

HTML and CSS are not checked: the available tools report lint and
missing-config problems rather than syntax errors.

If validation fails, the modified files are rolled back to their pre-edit
snapshots. The agent is given up to 3 retries. On the 3rd failure, it escalates
to the user.

## Guardrails

- **Home-dir guard**: Refuses to run if CWD's `.99x/` would collide with
  `AGENT_HOME` (`~/.99x`), preventing project state from mixing with global config.
- **Handback detection**: If the model stops with question-shaped text (e.g.
  "Would you like me to..."), the loop nudges it once to continue working or
  declare completion.
- **Repeat-detection**: 3 consecutive identical tool calls trigger an early-stop
  escalation to the user.
- **Length cutoff**: If the model response is truncated (finish_reason=length),
  the loop automatically prompts it to continue.
- **Reasoning-only stalls**: If the model produces reasoning content but no text
  or tool calls, the loop prompts it to produce output.

## Reasoning / Thinking Display

Models that emit `reasoning_content` (e.g. DeepSeek-R1, Qwen QwQ via
OpenRouter's `include_reasoning` param, or Gemma's ` thinking`/` response`
tags) have that content parsed and displayed separately.

The TUI renders thinking in dimmed text. Three display modes:

| Mode | Behavior |
|---|---|
| `off` | Thinking is discarded. |
| `terse` | First and last ~100 chars shown. |
| `full` | All thinking shown. |

Use `/thinking` to show the available modes, or `/thinking off|terse|full` to
set one. The setting is persisted in `config.json`.

In `full` mode the thinking streams as it arrives; once the reply is done
it collapses to a one-line `▶ Thinking (N words)` summary. Ctrl+T expands
or collapses the most recent one.

## Model Compatibility

The harness supports common reasoning formats and a few model-specific
workarounds:

- **Gemma 4:** Setting `--effort` or `/effort` prepends Gemma 4's `<|think|>`
  control token to the system prompt. Reasoning is separated from the final
  answer, including when a backend returns Gemma's channel tags or streams
  reasoning in a `reasoning` field. The control token is added only to the
  request, not saved in conversation history.
- **Gemma 3:** A streaming workaround removes the family's occasional
  mid-word newlines. It is not applied to other models, so ordinary line breaks
  (such as numbered lists) are preserved.
- **MiniCPM:** Tool schemas are also included in the prompt, and XML-style
  `<function>` responses are recognized as tool calls for models or backends
  that do not return native tool-call objects.

These adaptations are selected from the model name; other models continue to
use the normal OpenAI-compatible tool-calling and reasoning paths.

## Usage and Cost Display

The TUI uses reported prompt and completion token counts to track context use.
For OpenRouter, it also fetches the selected model's published pricing and
shows an estimated cumulative cost in the header during the session. This
estimate depends on usage and pricing data being available; it is not shown for
local Ollama models.

## Lifecycle Hook Trace

Hooks are opt-in. Run `99x --trace-hooks` for the interactive TUI or
`99xsh --trace-hooks "your prompt"` for a one-shot trace. Events include
run start/finish/cancel, model request timing and token counts, tool
start/finish, and errors. Nested runs include their parent run ID. The
trace omits prompts, completions, tool arguments, and tool results.

Where the trace goes:

- **One-shot** (`99xsh`, or `99x <prompt>`): stderr, so the response on
  stdout stays separate.
- **TUI** (`99x`): appended to `.99x/hook-trace.log` in the project,
  because Textual owns the terminal (stderr included) while it runs. The
  TUI prints the file's path at startup; follow it from another terminal
  with `tail -f .99x/hook-trace.log`.

Programmatic callers can still pass their own `AgentHooks` handlers; enabling
`--trace-hooks` adds the built-in trace handler without replacing them. See
`examples/hooks/` for reusable trace and usage-summary handlers.

## File Layout

```
agent99x/                   # Python package (flat)
  __init__.py
  cli.py                    # CLI parsing, entry points, one-shot runner
  core.py                   # Agent loop and tool dispatch
  prompt.py                 # System prompt assembly, AGENT.md parsing
  tools.py                  # Tool registry + all built-in tool primitives
  skills.py                 # Skill/agent discovery tools + spawn_agent
  session.py                # SessionConfig dataclass and path constants
  config.py                 # Central runtime constants (limits)
  config_codec.py           # Config serialize/deserialize (one field list)
  config_io.py              # Config and session file I/O
  providers.py              # Ollama + OpenRouter providers
  llm.py                    # Streaming completion with cancel support
  commands.py               # Slash command routing
  conversation.py           # Session-turn lifecycle (history management)
  compaction.py             # Conversation summarization
  contexts.py               # Context file naming and resolution
  reasoning.py              # Reasoning/thinking content parser + renderer
  validation.py             # Post-edit syntax validation + rollback
  scopes.py                 # Skill/agent scope resolution (project shadows global)
  todos.py                  # Todo list persistence (todos.md format)
  mcp.py                    # MCP client, lifecycle, tool registration
  tui.py                    # Textual TUI application
  logs.py                   # Structured logging (tool calls, results)
  _defaults/                # Seed files for `99x init`
    AGENT.md                # Default agent persona
    skills/
      weather/SKILL.md
      web_search/SKILL.md
      fetch_recent_rss/         # SKILL.md + scripts/fetch_rss.py
      grill-me/SKILL.md
```

### State directories

| Path | Purpose |
|---|---|
| `~/.99x/` (`$AGENT_HOME`) | Global config, persona, memory, skills, agents |
| `~/.99x/config.json` | Persisted provider/model/host/effort/threshold |
| `~/.99x/AGENT.md` | Agent persona and base instructions |
| `~/.99x/USER.md` | Facts about the user (loaded into system prompt) |
| `~/.99x/MEMORY.md` | User-preference facts (write-on-demand by agent) |
| `~/.99x/diary/` | Daily notes (`YYYY-MM-DD.md`) |
| `~/.99x/skills/` | Global skills |
| `~/.99x/agents/` | Global agents |
| `.99x/` (project) | Project-local state |
| `.99x/AGENT.md` | Project context (loaded into system prompt) |
| `.99x/MEMORY.md` | Project-specific facts (write-on-demand by agent) |
| `.99x/todos.md` | Todo list |
| `.99x/skills/` | Project-local skills (shadow global) |
| `.99x/agents/` | Project-local agents (shadow global) |
| `.99x/contexts/` | Saved conversation histories |
| `.99x/hook-trace.log` | `99x --trace-hooks` output (TUI mode) |

**Memory writing rule:** facts about the *user* go to `~/.99x/MEMORY.md`; facts
about the *project* go to `.99x/MEMORY.md`.

## Configuration Reference

`~/.99x/config.json` (persisted automatically by the TUI; one-shot runs
never write it). A field with the wrong type is ignored with a warning:

```json
{
  "provider": "ollama",
  "model": "gemma3:12b",
  "host": null,
  "effort": null,
  "autocompact": 0.85,
  "call_budget": null,
  "show_thinking": "full",
  "provider_models": {
    "ollama": "gemma3:12b",
    "openrouter": "google/gemini-2.5-flash-preview"
  },
  "provider_hosts": {
    "ollama": null
  }
}
```

Environment variables:

| Variable | Purpose |
|---|---|
| `AGENT_HOME` | Override global state directory (default `~/.99x`) |
| `OPENROUTER_API_KEY` | API key for the openrouter provider |

## Differences from agent99

Cut from the original:

- Handoff/contract/evaluator subsystem
- Async/background subagents
- Blackboard
- Heartbeat
- Built-in `weather`, `web_search`, `fetch_rss` tools → now skills
- Providers other than ollama and openrouter

Added or refined:

- Three-kind capability model (tools/skills/agents) with symmetric discovery
- Agentskills.io-compatible skill/agent frontmatter parsing
- `replace_lines` tool (line-number-based editing)
- `patch` tool (unified diff application)
- Plan mode with read-only enforcement
- Automatic syntax validation + rollback on file edits
- Context management with named/timestamped save files
- MCP support (optional, gated behind config file)
- Per-provider host overrides
- Call budget with shared state across subagent tree
- Handback detection and repeat-detection loop guards
- Gemma 4 thinking support, Gemma 3 streaming workaround, and MiniCPM tool-call fallback
- OpenRouter pricing and session cost estimate in the TUI

## License

See the original [agent99](../99) repository.
