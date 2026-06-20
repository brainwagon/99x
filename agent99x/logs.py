import logging
import sys
from typing import Any, Dict

# The main logger for the agent
logger = logging.getLogger("agent99x")


def setup_logging(level: int = logging.INFO, stream=sys.stderr):
    """Configure the agent logger with basic defaults."""
    logger.setLevel(level)
    for h in logger.handlers[:]:
        logger.removeHandler(h)
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(handler)


def log_tool_call(name: str, args: Dict[str, Any]):
    logger.info(f"tool call: {name}", extra={"type": "tool_call", "tool_name": name, "tool_args": args})


def log_tool_result(name: str, result: str, elapsed: float):
    logger.info(f"tool result: {name} ({elapsed:.2f}s)",
                extra={"type": "tool_result", "tool_name": name, "result": result, "elapsed": elapsed})


def log_warning(msg: str):
    logger.warning(msg)
