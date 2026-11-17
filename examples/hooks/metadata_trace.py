"""Print a safe lifecycle trace for one run.

Call ``run_with_trace(session)`` with a configured SessionConfig. The trace
contains event metadata only; prompts, completions, and tool data are omitted.
"""

from agent99x.core import agent_loop
from agent99x.hooks import AgentHooks, HookEvent
from agent99x.session import SessionConfig


def print_event(event: HookEvent) -> None:
    parent = f" parent={event.parent_run_id[:8]}" if event.parent_run_id else ""
    details = " ".join(
        f"{key}={value}"
        for key, value in event.metadata.items()
        if value is not None
    )
    print(f"{event.timestamp:.3f} {event.name} run={event.run_id[:8]}{parent} {details}")


def run_with_trace(session: SessionConfig):
    return agent_loop(session, hooks=AgentHooks([print_event]))
