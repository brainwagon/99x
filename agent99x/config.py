"""Central runtime-tuning constants for agent99x.

Import from here rather than defining limits inline per-module.
"""

# ── tool output limits ─────────────────────────────────────────────
MAX_BASH_OUTPUT: int = 1_000_000   # bytes per stream; excess dropped
MAX_GREP_RESULTS: int = 200
MAX_GLOB_MATCHES: int = 500

# ── agent loop limits ──────────────────────────────────────────────
MAX_AGENT_ITERATIONS: int = 25
MAX_AGENT_DEPTH: int = 3
