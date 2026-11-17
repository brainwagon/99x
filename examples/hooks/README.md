# Hook examples

Hooks are synchronous, metadata-only observers registered per run. Pass an
`AgentHooks` object when 99x code calls `agent_loop`:

```python
from examples.hooks.metadata_trace import run_with_trace
from examples.hooks.usage_totals import run_with_usage_summary

# `session` is an already configured SessionConfig.
run_with_trace(session)
run_with_usage_summary(session)
```

- `metadata_trace.py` prints lifecycle events, including linked subagent runs.
- `usage_totals.py` aggregates model token usage and elapsed time per run.

`AgentHooks` can also be passed to `cli.main_tui(hooks=...)` or
`cli.main_oneshot(hooks=...)`; the TUI carries the same hook set into each turn.

## Event reference

Every immutable `HookEvent` has `name`, `run_id`, optional `parent_run_id`,
Unix-seconds `timestamp`, and immutable `metadata`.

| Event | Metadata |
| --- | --- |
| `run.started` | `provider`, `model` |
| `run.finished` | `status`, `duration_seconds` |
| `run.cancelled` | — |
| `model_request.started` | `provider`, `model` |
| `model_request.finished` | `status`, `duration_seconds`, `input_tokens`, `output_tokens`, optional `error_type` |
| `tool.started` | `tool` |
| `tool.finished` | `tool`, `status`, `duration_seconds` |
| `error` | `source`, `error_type`, optional `tool` |

Run statuses are `completed`, `stopped`, `cancelled`, and `failed`. Model
request statuses are `completed`, `cancelled`, and `failed`; tool statuses are
`completed` and `failed`. Prompts, completions, tool arguments, and tool
results are never included.

Handlers run inline, in registration order, and are serialized across worker
threads. Parallel tool events may arrive in nondeterministic order. Keep
handlers quick; exceptions are logged and isolated from the agent run. Async
handlers are not supported.
