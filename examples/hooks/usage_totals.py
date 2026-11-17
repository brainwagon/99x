"""Summarize token usage and elapsed time for each run."""

from agent99x.core import agent_loop
from agent99x.hooks import AgentHooks, HookEvent
from agent99x.session import SessionConfig


class UsageTotals:
    def __init__(self) -> None:
        self._totals = {}

    def __call__(self, event: HookEvent) -> None:
        if event.name == "run.started":
            self._totals[event.run_id] = [0, 0]
        elif event.name == "model_request.finished":
            totals = self._totals.setdefault(event.run_id, [0, 0])
            totals[0] += event.metadata.get("input_tokens") or 0
            totals[1] += event.metadata.get("output_tokens") or 0
        elif event.name == "run.finished":
            input_tokens, output_tokens = self._totals.pop(event.run_id, [0, 0])
            print(
                f"run {event.run_id[:8]}: {event.metadata['status']}, "
                f"{input_tokens} input + {output_tokens} output tokens, "
                f"{event.metadata['duration_seconds']:.2f}s"
            )


def run_with_usage_summary(session: SessionConfig):
    return agent_loop(session, hooks=AgentHooks([UsageTotals()]))
