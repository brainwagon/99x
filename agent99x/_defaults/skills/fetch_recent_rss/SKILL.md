---
name: fetch_recent_rss
description: Fetch the N most recent items from an RSS 2.0 or Atom feed (default N=5) via a bundled script, returning a compact summary. Add --full to pull full item content when you need to reason over the feed. Use when the user gives a feed URL or asks for the latest items from a blog/news feed.
---

# Fetch recent RSS / Atom items

This skill fetches and parses the feed with a bundled script, so the raw
XML never enters context — only an already-parsed, plain-text list comes
back. It supersedes the older `fetch_rss` skill (which parsed XML inline).

## Run

From this skill's directory (the `dir` returned by `load_skill`), run the
script with `run_bash`:

```
run_bash(command="python3 scripts/fetch_rss.py '<FEED URL>' [-n N] [--full]")
```

- `<FEED URL>` — the feed's URL (RSS 2.0 or Atom; both are handled).
- `-n N` — optional item count. Omit to use the default of 5.
- `--full` — emit each item's full, untruncated content instead of a short
  blurb. Use this only when the user wants to **reason over** the feed
  ("which of these mention X?", "summarise the themes"); it costs more
  context. Leave it off for a plain "show me the latest" listing.

The script depends only on the Python standard library.

## Output

stdout is the only thing you need to read back to the user:

```
Feed: <feed title>
Showing <k> of most recent items

1. <item title>
   date: <pubDate / published>
   link: <url>
   <plain-text summary — ~280 chars, or full content under --full>
...
```

On failure the script exits non-zero and prints a single `fetch failed:`
or `parse failed:` line to stderr — relay that and stop.

## Present

Show the list to the user roughly as printed, tightening formatting as
needed. Do not paste raw XML; if they want more detail on one item, fetch
that item's `link` separately.
