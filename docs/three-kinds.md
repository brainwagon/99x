# Three kinds: tools, skills, agents

99x gives the model three distinct kinds of capability. Earlier versions blurred
them — most visibly, asking the agent to "list tools" returned *nothing*, because
tools (unlike skills and agents) had no way to be enumerated. This document fixes
the model so the three kinds are **conceptually distinct** and **discovered the
same way**.

## The three kinds

| Kind | What it is | How the model invokes it | Runs code? |
|---|---|---|---|
| **Tool** | A verb the harness implements in Python. | Native function call. | Yes — the tool *is* the code. |
| **Skill** | Written instructions (Markdown) for a procedure. | `load_skill(name)`, then act in the current context. | No — prose only; it may *point at* scripts the model runs via `run_bash`. |
| **Agent** | A fresh specialist worker. | `spawn_agent(task, name)`; it works in its own context and returns a result. | Indirectly — it runs its own tool loop. |

The decision rule the model is given: **do it yourself → tool; need the recipe →
skill; hand it off → agent.**

## Why an agent is not "just a skill with a clean context"

A skill is *passive prose* loaded into the current model's head. An agent is a
*separate invocation* of a model that:

- runs its own full `agent_loop` to completion and returns **one compressed result**,
- may carry a restricted toolset (`allowed_tools`) and `model`/`provider`/`effort`
  overrides.

Isolation is the mechanism; delegation + result-compression is the point.

## Discovery: eager catalogs + `list_*`

Every turn the system prompt carries a **"YOUR CAPABILITIES"** block: a short
prescriptive primer plus three catalogs.

- **Tools** — rendered **terse**, grouped by family (`Files:`, `Shell:`, …).
  Descriptions are *not* repeated here because the full JSON schemas are already
  sent to the model's function-calling API; the catalog only needs to anchor
  *which* tools exist and what families they fall in.
- **Skills** / **Agents** — rendered **full** (`name — description`), because that
  prose appears nowhere else in context.

In addition, three symmetric tools exist for on-demand / runtime enumeration:
`list_tools`, `list_skills`, `list_agents`. The eager catalog is *curated*
(meta/bridge tools are hidden); `list_tools` is *exhaustive* (it returns every
registered tool, including runtime-added MCP tools and the bridge tools).

### Bridge / meta tools

`list_tools`, `list_skills`, `list_agents`, `load_skill`, and `spawn_agent` are
the plumbing of the three-kind system. They are registered with `group="meta"`,
which **hides them from the eager catalog** (the primer already explains them) but
keeps them in `list_tools`.

### Tool groups

`@tool(..., group="files")` tags each tool with a family. The catalog renderer
orders groups (`files, shell, search, net, plan, info`) and buckets anything else
(e.g. MCP tools) under `Other`. `group="meta"` is the sentinel for "don't show in
the catalog."

## Skills are pure prose (agentskills.io compatible)

There is **no dedicated script runner**. A skill that bundles helpers in
`scripts/` just tells the model — in its Markdown — to run them, and the model
uses the ordinary `run_bash` tool:

```bash
uv run scripts/extract.py --input data.csv
```

This matches the [agentskills.io](https://agentskills.io/skill-creation/using-scripts)
convention exactly (it, too, runs scripts via plain shell, not a special tool).
For relative paths like `scripts/extract.py` to resolve, `load_skill(name)`
returns the skill's absolute **`dir`** alongside its `content`; the primer tells
the model to run bundled scripts from there.

The net effect: **only tools run code.** `run_bash` is the single code-runner; a
skill never gets a privileged execution path.

### Drop-in `SKILL.md` compatibility

99x parses the full agentskills.io `SKILL.md` frontmatter so community skills drop
into `~/.99x/skills/<name>/` and work unmodified:

| Field | Required | Notes |
|---|---|---|
| `name` | yes | lowercase, must match the directory name |
| `description` | yes | what it does + when to use it |
| `license` | no | parsed, informational |
| `compatibility` | no | parsed, informational |
| `metadata` | no | nested key→value map (parser supports nested maps) |
| `allowed-tools` | no | **parsed but not enforced** — see below |

Directory layout: `SKILL.md` (required), plus optional `scripts/`, `references/`,
`assets/`. Relative paths resolve from the skill root.

**`allowed-tools` is parsed but not enforced for skills.** A skill is loaded
*inline* into the current context — there is no separate loop at which to restrict
its toolset (unlike an agent, whose `allowed_tools` *is* enforced via its
sub-session). The field is "experimental" in the spec anyway. Note the naming
split: skills use `allowed-tools` (hyphen, agentskills.io); 99x agents use
`allowed_tools` (underscore) in their `AGENT.md`.
