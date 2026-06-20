"""Skill and agent discovery tools, plus synchronous subagent dispatch."""

import os
import subprocess
from typing import Any, Dict, List, Optional

from agent99x import providers
from agent99x import scopes
from agent99x import tools
from agent99x.config import MAX_AGENT_DEPTH
from agent99x.prompt import _skill_meta, build_agent_system, parse_frontmatter
from agent99x.session import AGENT_HOME, SessionConfig
from agent99x.tools import tool


# ── skill discovery ────────────────────────────────────────────────

@tool("list_skills", "List available skills by name and description.",
      doc="Call when you need a capability you don't know how to perform.")
def list_skills() -> Dict[str, Any]:
    results: List[Dict[str, str]] = []
    for name, path in scopes.discover("skills"):
        desc, _body = _skill_meta(path)
        results.append({"name": name, "description": desc})
    return {"skills": results}


@tool("load_skill", "Load full instructions for a named skill.",
      params={"name": {"type": "string", "description": "Skill name (from list_skills), e.g. 'weather'"}})
def load_skill(name: str) -> Dict[str, Any]:
    path = scopes.resolve("skills", name)
    if not path:
        return {"error": f"Unknown skill: {name}. Call list_skills to see available skills."}
    _desc, body = _skill_meta(path)
    return {"skill": name, "content": body}


@tool(
    "run_skill_script",
    "Run a helper script bundled with a skill.",
    params={
        "skill": {"type": "string", "description": "Skill name (same as the skills/ subdirectory name)."},
        "script": {"type": "string", "description": "Script filename inside the skill's scripts/ directory, e.g. 'post.py'."},
        "args": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Command-line arguments to pass to the script.",
        },
    },
)
def run_skill_script(skill: str, script: str,
                     args: Optional[List[str]] = None) -> Dict[str, Any]:
    skill_md = scopes.resolve("skills", skill)
    scripts_dir = (os.path.join(os.path.dirname(skill_md), "scripts") if skill_md
                   else os.path.join(AGENT_HOME, "skills", skill, "scripts"))
    script_path = os.path.join(scripts_dir, script)
    real_scripts_dir = os.path.realpath(scripts_dir)
    try:
        real_script = os.path.realpath(script_path)
    except OSError as e:
        return {"error": str(e)}
    if not real_script.startswith(real_scripts_dir + os.sep):
        return {"error": f"Script path escapes skill directory: {script!r}"}
    if not os.path.isfile(real_script):
        return {"error": f"Script not found: {script!r} in skill {skill!r}"}
    try:
        result = subprocess.run(
            [real_script] + (args or []),
            capture_output=True, text=True, timeout=60,
        )
        return {"returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr}
    except subprocess.TimeoutExpired:
        return {"error": "script timed out (60s)"}
    except OSError as e:
        return {"error": str(e)}


@tool("list_agents", "List available agents by name and description.",
      doc="Call before spawn_agent if you don't know which agent to use.")
def list_agents() -> Dict[str, Any]:
    results: List[Dict[str, str]] = []
    for name, path in scopes.discover("agents"):
        try:
            with open(path, encoding="utf-8") as f:
                text = f.read()
        except OSError:
            continue
        meta, _body = parse_frontmatter(text)
        desc = meta.get("description") or ""
        results.append({"name": name, "description": str(desc)})
    return {"agents": results}


# ── subagents (synchronous) ────────────────────────────────────────

@tool(
    "spawn_agent",
    "Delegate a task to a named specialist agent and return its response.",
    params={
        "task": {"type": "string", "description": "The task or question for the subagent."},
        "name": {"type": "string", "description": "Agent name (subdirectory of agents/). Call list_agents to see options. Omit for default."},
    },
    required=["task"],
    needs_session=True,
)
def spawn_agent(task: str, name: Optional[str] = None,
                session: Optional[SessionConfig] = None) -> Dict[str, Any]:
    """Run a subagent synchronously."""
    from agent99x.core import agent_loop  # lazy: core imports skills at module load
    if session is None:
        return {"error": "Internal error: session missing"}
    depth = getattr(tools._thread_local, 'depth', 0)
    if depth >= MAX_AGENT_DEPTH:
        return {"error": f"Max agent depth {MAX_AGENT_DEPTH} reached; refusing to spawn."}
    built = build_agent_system(session, name)
    if built is None:
        return {"error": f"Unknown agent: {name}"}
    system, meta = built
    allowed = meta.get("allowed_tools")
    if allowed is not None and not isinstance(allowed, list):
        return {"error": f"Agent '{name}' has invalid allowed_tools (expected list)."}

    sub_session = SessionConfig(
        provider=session.provider,
        model=session.model,
        host=session.host,
        effort=session.effort,
        context_window=session.context_window,
        model_timeout=session.model_timeout,
        client=session.client,
        shared_state=session.shared_state,
        allowed_tools=allowed,
        history=[system, {"role": "user", "content": task}],
    )

    ov_provider = meta.get("provider") or None
    ov_host = meta.get("host") or None
    ov_model = meta.get("model") or None
    ov_effort = meta.get("effort") or None
    if ov_provider or ov_host:
        try:
            providers.setup_provider(
                sub_session,
                provider=ov_provider or (session.provider.name if session.provider else None),
                model=ov_model or session.model,
                host=ov_host,
            )
        except ValueError as e:
            return {"error": f"Agent '{name}' provider override failed: {e}"}
    elif ov_model:
        sub_session.model = ov_model
    if ov_effort:
        sub_session.effort = ov_effort

    prev = depth
    tools._thread_local.depth = depth + 1
    try:
        return {"result": agent_loop(sub_session)}
    finally:
        tools._thread_local.depth = prev
