"""Regression tests for code-review fixes in tools.py, mcp.py and fetch_rss.py."""

import asyncio
import importlib.util
import os
import pathlib
import sys
import textwrap
import time
import xml.etree.ElementTree as ET

import httpx
import pytest

from agent99x import tools


def _write(p, s):
    with open(p, "w", encoding="utf-8", newline="") as f:
        f.write(s)


def _read(p):
    with open(p, encoding="utf-8", newline="") as f:
        return f.read()


# ── run_bash ───────────────────────────────────────────────────────

def test_run_bash_background_job_does_not_hang(in_tmp_cwd):
    t0 = time.monotonic()
    r = tools.run_bash("sleep 60 & echo started")
    assert time.monotonic() - t0 < 5
    assert r["stdout"].strip() == "started"
    assert r["returncode"] == 0
    assert "error" not in r


def test_run_bash_timeout_kills_group(in_tmp_cwd, monkeypatch):
    monkeypatch.setattr(tools, "BASH_TIMEOUT", 0.5)
    t0 = time.monotonic()
    r = tools.run_bash("echo before; sleep 30")
    assert time.monotonic() - t0 < 5
    assert r["error"] == "timeout"
    assert "before" in r["stdout"]


def test_run_bash_timeout_when_shell_closes_its_pipes(in_tmp_cwd, monkeypatch):
    monkeypatch.setattr(tools, "BASH_TIMEOUT", 0.5)
    t0 = time.monotonic()
    r = tools.run_bash("exec >&- 2>&-; sleep 30")
    assert time.monotonic() - t0 < 5
    assert r["error"] == "timeout"


def test_run_bash_stdin_is_devnull_and_shell_is_bash(in_tmp_cwd):
    t0 = time.monotonic()
    r = tools.run_bash("cat; [[ 1 == 1 ]] && echo bash")
    assert time.monotonic() - t0 < 5
    assert r["stdout"].strip() == "bash"


# ── line splitting / replace_lines / edit_file ─────────────────────

def test_replace_lines_counts_only_real_newlines(in_tmp_cwd):
    _write("c.txt", "l1\nl2 \x0c x\nl3\nl4\n")
    numbered = tools.read_file("c.txt", with_line_numbers=True)["content"]
    assert "3\tl3\n" in numbered
    tools.replace_lines("c.txt", 3, 3, "L3")
    assert _read("c.txt") == "l1\nl2 \x0c x\nL3\nl4\n"


def test_replace_lines_append_after_last_line_without_newline(in_tmp_cwd):
    _write("a.txt", "a")
    r = tools.replace_lines("a.txt", 2, 1, "b")
    assert r["ok"]
    assert _read("a.txt") == "a\nb"


def test_replace_lines_append_crlf(in_tmp_cwd):
    _write("a.txt", "a\r\nb")
    tools.replace_lines("a.txt", 3, 2, "c")
    assert _read("a.txt") == "a\r\nb\r\nc"


def test_edit_file_empty_old_string_rejected(in_tmp_cwd):
    _write("d.txt", "abc")
    r = tools.edit_file("d.txt", "", "X", replace_all=True)
    assert "error" in r and "empty" in r["error"]
    assert _read("d.txt") == "abc"


def test_edit_file_empty_old_string_fills_empty_file(in_tmp_cwd):
    _write("e.txt", "")
    r = tools.edit_file("e.txt", "", "hello\n")
    assert r["ok"]
    assert _read("e.txt") == "hello\n"


def test_edit_file_fallback_preserves_new_string_exactly(in_tmp_cwd):
    _write("i.txt", "a  \nb\nc\n")
    r = tools.edit_file("i.txt", "a\nb\n", "X\n\n")
    assert "rstripping" in r["note"]
    assert _read("i.txt") == "X\n\nc\n"


def test_edit_file_fallback_without_trailing_newline_keeps_line_break(in_tmp_cwd):
    _write("i.txt", "a  \nb\nc\n")
    tools.edit_file("i.txt", "a\nb", "X")
    assert _read("i.txt") == "X\nc\n"


def test_edit_file_fallback_crlf(in_tmp_cwd):
    _write("i.txt", "a  \r\nb\r\nc\r\n")
    tools.edit_file("i.txt", "a\nb\n", "X\nY\n")
    assert _read("i.txt") == "X\r\nY\r\nc\r\n"


# ── patch ──────────────────────────────────────────────────────────

def test_patch_large_hunk_applied_in_full(in_tmp_cwd):
    _write("f.txt", "x\n")
    diff = "@@ -1,1 +1,151 @@\n x\n" + "".join(f"+L{i}\n" for i in range(150))
    r = tools.apply_patch("f.txt", diff)
    assert r["ok"] and "warnings" not in r
    assert len(_read("f.txt").splitlines()) == 151


def test_patch_miscounted_header_applies_with_warning(in_tmp_cwd):
    _write("f.txt", "a\nb\nc\n")
    r = tools.apply_patch("f.txt", "@@ -1,2 +1,2 @@\n a\n-b\n+B\n c\n")
    assert r["ok"]
    assert r["warnings"]
    assert _read("f.txt") == "a\nB\nc\n"


def test_patch_pure_insertion_goes_after_line(in_tmp_cwd):
    _write("e.txt", "1\n2\n3\n")
    r = tools.apply_patch("e.txt", "@@ -2,0 +3 @@\n+NEW\n")
    assert r["ok"]
    assert _read("e.txt") == "1\n2\nNEW\n3\n"


def test_patch_insertion_at_top_of_file(in_tmp_cwd):
    _write("e.txt", "1\n2\n")
    tools.apply_patch("e.txt", "@@ -0,0 +1 @@\n+TOP\n")
    assert _read("e.txt") == "TOP\n1\n2\n"


def test_patch_append_after_last_line_without_newline(in_tmp_cwd):
    _write("b.txt", "a")
    r = tools.apply_patch("b.txt", "@@ -1,1 +1,2 @@\n a\n+b\n")
    assert r["ok"]
    assert _read("b.txt") == "a\nb\n"


def test_patch_honours_no_newline_marker(in_tmp_cwd):
    _write("b.txt", "a\n")
    diff = "@@ -1 +1,2 @@\n a\n+b\n\\ No newline at end of file\n"
    tools.apply_patch("b.txt", diff)
    assert _read("b.txt") == "a\nb"


def test_patch_blank_context_line_without_space(in_tmp_cwd):
    _write("g.txt", "a\n\nb\n")
    r = tools.apply_patch("g.txt", "@@ -1,3 +1,3 @@\n a\n\n-b\n+B\n")
    assert r["ok"]
    assert _read("g.txt") == "a\n\nB\n"


def test_patch_line_numbers_ignore_form_feed(in_tmp_cwd):
    _write("c.txt", "l1\nl2\x0cx\nl3\n")
    r = tools.apply_patch("c.txt", "@@ -3 +3 @@\n-l3\n+L3\n")
    assert r["ok"]
    assert _read("c.txt") == "l1\nl2\x0cx\nL3\n"


def test_patch_rejects_sibling_dir_sharing_cwd_prefix(tmp_path, monkeypatch):
    work = tmp_path / "w"
    evil = tmp_path / "w-evil"
    work.mkdir()
    evil.mkdir()
    _write(evil / "x.txt", "a\n")
    monkeypatch.chdir(work)
    r = tools.apply_patch("../w-evil/x.txt", "@@ -1 +1 @@\n-a\n+PWN\n")
    assert "Security violation" in r["error"]
    assert _read(evil / "x.txt") == "a\n"


# ── grep / glob ────────────────────────────────────────────────────

def _grep_backends():
    backends = ["python"]
    import shutil
    if shutil.which("rg"):
        backends.insert(0, "rg")
    return backends


@pytest.fixture(params=_grep_backends())
def grep_backend(request, monkeypatch):
    if request.param == "python":
        real_which = tools.shutil.which
        monkeypatch.setattr(tools.shutil, "which",
                            lambda name, *a, **k: None if name == "rg" else real_which(name, *a, **k))
    return request.param


def test_grep_single_file(in_tmp_cwd, grep_backend):
    _write("h.py", "foo = 1\nbar: foo\n")
    r = tools.grep("foo", "h.py")
    assert r["tool"] == grep_backend
    assert [(m["path"], m["line"]) for m in r["results"]] == [("h.py", 1), ("h.py", 2)]
    assert r["results"][1]["text"] == "bar: foo"


def test_grep_path_with_colon(in_tmp_cwd, grep_backend):
    _write("a:b.txt", "x: foo\n")
    r = tools.grep("foo", ".")
    assert [(m["path"], m["line"], m["text"]) for m in r["results"]] == [("./a:b.txt", 1, "x: foo")]


def test_grep_missing_path_is_error(in_tmp_cwd, grep_backend):
    r = tools.grep("foo", "nope")
    assert "error" in r


def test_grep_invalid_regex_is_error(in_tmp_cwd, grep_backend):
    _write("h.py", "foo\n")
    r = tools.grep("foo(", ".")
    assert "error" in r


def test_grep_glob_filters_by_name_at_any_depth(in_tmp_cwd, grep_backend):
    os.makedirs("sub")
    _write("a.txt", "foo\n")
    _write("sub/b.py", "foo\n")
    r = tools.grep("foo", ".", "*.py")
    assert [m["path"] for m in r["results"]] == ["./sub/b.py"]


def test_glob_double_star_matches_top_level(in_tmp_cwd):
    os.makedirs("pkg/sub")
    for p in ("top.py", "pkg/a.py", "pkg/sub/b.py", "readme.md"):
        _write(p, "")
    assert sorted(tools.glob_files("**/*.py")["matches"]) == ["pkg/a.py", "pkg/sub/b.py", "top.py"]
    assert tools.glob_files("*.py")["matches"] == ["top.py"]
    assert tools.glob_files("pkg/**/*.py", ".")["matches"] == ["pkg/a.py", "pkg/sub/b.py"]
    assert tools.glob_files("*.py", "pkg")["matches"] == ["pkg/a.py"]


def test_glob_skips_ignored_dirs(in_tmp_cwd):
    os.makedirs(".git")
    _write(".git/x.py", "")
    _write("y.py", "")
    assert tools.glob_files("**/*.py")["matches"] == ["y.py"]


# ── http_request ───────────────────────────────────────────────────

def _mock_client(monkeypatch, handler):
    real = httpx.Client

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real(*args, **kwargs)

    monkeypatch.setattr(tools.httpx, "Client", factory)


def test_http_request_streams_and_stops_at_cap(monkeypatch):
    pulled = []

    def body():
        for i in range(1000):
            pulled.append(i)
            yield b"x" * 1000

    _mock_client(monkeypatch, lambda req: httpx.Response(200, content=body()))
    r = tools.http_request("http://example.invalid/", max_bytes=5500)
    assert r["status"] == 200
    assert r["truncated"] is True
    assert r["body"] == "x" * 5500
    assert len(pulled) < 20  # did not download the whole 1MB


def test_http_request_default_cap_is_context_sized(monkeypatch):
    _mock_client(monkeypatch, lambda req: httpx.Response(200, content=b"y" * 300_000))
    r = tools.http_request("http://example.invalid/")
    assert r["truncated"] is True
    assert len(r["body"]) == tools._HTTP_BODY_CAP == 100_000


def test_http_request_small_body_and_utf8_boundary(monkeypatch):
    _mock_client(monkeypatch, lambda req: httpx.Response(
        200, content="é".encode() * 3, headers={"content-type": "text/plain; charset=utf-8"}))
    assert tools.http_request("http://example.invalid/")["body"] == "ééé"
    r = tools.http_request("http://example.invalid/", max_bytes=3)
    assert r["truncated"] and r["body"] == "é"  # half a char dropped, not mangled


# ── MCP ────────────────────────────────────────────────────────────

mcp_mod = pytest.importorskip("agent99x.mcp")


@pytest.fixture
def fresh_names(monkeypatch):
    monkeypatch.setattr(mcp_mod, "MCP_TOOL_NAMES", {})


def test_mcp_tool_names_are_sanitized(fresh_names):
    name = mcp_mod.openai_tool_name("my.server", "do thing/now")
    assert name == "mcp__my_server__do_thing_now"


def test_mcp_tool_names_truncated_and_unique(fresh_names):
    import re
    long1 = mcp_mod.openai_tool_name("srv", "a" * 100)
    long2 = mcp_mod.openai_tool_name("srv", "a" * 99 + "b")
    for n in (long1, long2):
        assert re.fullmatch(r"[A-Za-z0-9_-]{1,64}", n)
    assert long1 != long2
    # stable for the same tool
    assert mcp_mod.openai_tool_name("srv", "a" * 100) == long1


def test_mcp_tool_names_avoid_sanitization_collisions(fresh_names):
    a = mcp_mod.openai_tool_name("s", "x.y")
    b = mcp_mod.openai_tool_name("s", "x_y")
    assert a != b
    assert mcp_mod.MCP_TOOL_NAMES[a] == ("s", "x.y")
    assert mcp_mod.MCP_TOOL_NAMES[b] == ("s", "x_y")


def test_mcp_tool_traits_follow_read_only_hint():
    from mcp.types import Tool, ToolAnnotations
    schema = {"type": "object", "properties": {}}
    plain = Tool(name="t", inputSchema=schema)
    ro = Tool(name="t", inputSchema=schema, annotations=ToolAnnotations(readOnlyHint=True))
    rw = Tool(name="t", inputSchema=schema, annotations=ToolAnnotations(readOnlyHint=False))
    assert mcp_mod.mcp_tool_traits(plain).plan_safe is False
    assert mcp_mod.mcp_tool_traits(rw).plan_safe is False
    assert mcp_mod.mcp_tool_traits(ro).plan_safe is True
    assert mcp_mod.mcp_tool_traits(ro).mutates is False


def test_run_in_mcp_loop_from_loop_thread_raises():
    async def inner():
        return 1

    async def outer():
        try:
            mcp_mod.run_in_mcp_loop(inner())
        except RuntimeError as e:
            return str(e)
        return "no error"

    msg = mcp_mod.run_in_mcp_loop(outer())
    assert "MCP event loop thread" in msg


class _FakeClient:
    def __init__(self, name, resources=(), read=None, prompts=(), get_prompt=None):
        self.name = name
        self.connected = True
        self.disconnect_reason = None
        self._resources = list(resources)
        self._read = read
        self._prompts = list(prompts)
        self._get_prompt = get_prompt

    def mark_disconnected(self, reason):
        self.connected = False
        self.disconnect_reason = reason

    async def list_resources(self):
        return self._resources

    async def read_resource(self, uri):
        return self._read(uri)

    async def list_prompts(self):
        return self._prompts

    async def get_prompt(self, name, arguments):
        return self._get_prompt(name, arguments)

    async def call_tool(self, name, arguments):
        raise RuntimeError("boom")


@pytest.fixture
def mcp_session(clean_registry):
    from types import SimpleNamespace
    s = SimpleNamespace(mcp_clients=[])
    mcp_mod._register_global_mcp_tools(s)
    return s


def _handler(name):
    return tools.TOOL_HANDLERS[name]


def test_mcp_list_prompts_with_no_arguments(mcp_session):
    from mcp.types import Prompt, PromptArgument
    mcp_session.mcp_clients.append(_FakeClient("s", prompts=[
        Prompt(name="bare"),
        Prompt(name="argful", arguments=[PromptArgument(name="who", required=True)]),
    ]))
    out = _handler("mcp_list_prompts")()
    assert [p["name"] for p in out] == ["bare", "argful"]
    assert out[0]["arguments"] == []
    assert out[1]["arguments"][0]["name"] == "who"


def test_mcp_read_resource_reports_real_error(mcp_session):
    from types import SimpleNamespace

    def broken(uri):
        raise RuntimeError("disk on fire")

    mcp_session.mcp_clients.append(_FakeClient(
        "s", resources=[SimpleNamespace(uri="file:///x", name="x")], read=broken))
    out = _handler("mcp_read_resource")(uri="file:///x")
    assert "disk on fire" in out["error"]
    assert "not found" not in out["error"]


def test_mcp_read_resource_not_found(mcp_session):
    from mcp.shared.exceptions import McpError
    from mcp.types import ErrorData

    def missing(uri):
        raise McpError(ErrorData(code=-32002, message="Resource not found"))

    mcp_session.mcp_clients.append(_FakeClient("s", read=missing))
    out = _handler("mcp_read_resource")(uri="file:///nope")
    assert "not found" in out["error"]


def test_mcp_read_resource_success(mcp_session):
    from types import SimpleNamespace
    mcp_session.mcp_clients.append(_FakeClient(
        "s", read=lambda uri: SimpleNamespace(contents=[SimpleNamespace(text="hello")])))
    assert _handler("mcp_read_resource")(uri="file:///x") == "hello"


def test_mcp_get_prompt_reports_real_error(mcp_session):
    from mcp.types import Prompt

    def broken(name, args):
        raise RuntimeError("template exploded")

    mcp_session.mcp_clients.append(_FakeClient("s", prompts=[Prompt(name="p")],
                                               get_prompt=broken))
    out = _handler("mcp_get_prompt")(name="p")
    assert "template exploded" in out["error"]
    out = _handler("mcp_get_prompt")(name="missing")
    assert "not found" in out["error"]


def test_mcp_tool_handler_disconnected_client():
    c = _FakeClient("gone")
    c.mark_disconnected("server exited")
    out = mcp_mod.make_tool_handler(c, "t")()
    assert "disconnected" in out["error"] and "gone" in out["error"]


def test_mcp_tool_handler_reports_call_errors():
    out = mcp_mod.make_tool_handler(_FakeClient("s"), "t")()
    assert "boom" in out["error"]


_SERVER = textwrap.dedent("""
    import os
    from mcp.server.fastmcp import FastMCP
    app = FastMCP("t")

    @app.tool()
    def echo(text: str) -> str:
        return text

    @app.tool()
    def die() -> str:
        os._exit(3)

    app.run()
""")


def test_mcp_server_death_gives_clear_error(tmp_path):
    pytest.importorskip("mcp.server.fastmcp")
    script = tmp_path / "server.py"
    script.write_text(_SERVER)
    client = mcp_mod.MCPClient("dying", {"command": sys.executable, "args": [str(script)],
                                         "timeout": 10})
    mcp_mod.run_in_mcp_loop(client.start())
    try:
        assert mcp_mod.make_tool_handler(client, "echo")(text="hi") == "hi"
        out = mcp_mod.make_tool_handler(client, "die")()
        assert "disconnected" in out["error"]
        t0 = time.monotonic()
        out = mcp_mod.make_tool_handler(client, "echo")(text="hi")
        assert "disconnected" in out["error"]
        assert time.monotonic() - t0 < 2
    finally:
        mcp_mod.run_in_mcp_loop(client.stop())


# ── fetch_rss.py ───────────────────────────────────────────────────

_RSS_SCRIPT = (pathlib.Path(tools.__file__).parent / "_defaults" / "skills"
               / "fetch_recent_rss" / "scripts" / "fetch_rss.py")


@pytest.fixture(scope="module")
def fetch_rss():
    spec = importlib.util.spec_from_file_location("fetch_rss_under_test", _RSS_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_fetch_rss_atom_xhtml_content(fetch_rss):
    feed = """<feed xmlns="http://www.w3.org/2005/Atom">
      <title>F</title>
      <entry>
        <title type="xhtml"><div xmlns="http://www.w3.org/1999/xhtml">Hello <b>World</b></div></title>
        <link href="http://e/1"/>
        <updated>2024-01-01T00:00:00Z</updated>
        <content type="xhtml"><div xmlns="http://www.w3.org/1999/xhtml"><p>First para.</p><p>Second &amp; last.</p></div></content>
      </entry>
    </feed>"""
    title, items = fetch_rss.parse_items(ET.fromstring(feed), full=True)
    assert title == "F"
    assert items[0]["title"] == "Hello World"
    assert items[0]["summary"] == "First para. Second & last."


def test_fetch_rss_rss1_rdf(fetch_rss):
    feed = """<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
                 xmlns="http://purl.org/rss/1.0/" xmlns:dc="http://purl.org/dc/elements/1.1/">
      <channel rdf:about="http://e/"><title>RDF feed</title><link>http://e/</link></channel>
      <item rdf:about="http://e/1"><title>One</title><link>http://e/1</link>
        <description>&lt;p&gt;Body one&lt;/p&gt;</description><dc:date>2024-01-02</dc:date></item>
      <item rdf:about="http://e/2"><title>Two</title><description>Body two</description></item>
    </rdf:RDF>"""
    title, items = fetch_rss.parse_items(ET.fromstring(feed), full=False)
    assert title == "RDF feed"
    assert [i["title"] for i in items] == ["One", "Two"]
    assert items[0]["date"] == "2024-01-02"
    assert items[0]["summary"] == "Body one"
    assert items[1]["link"] == "http://e/2"  # falls back to rdf:about


def test_fetch_rss_plain_title_untouched(fetch_rss):
    feed = "<rss><channel><title>a &lt; b</title><item><title>x &lt; y</title></item></channel></rss>"
    title, items = fetch_rss.parse_items(ET.fromstring(feed), full=False)
    assert title == "a < b"
    assert items[0]["title"] == "x < y"
