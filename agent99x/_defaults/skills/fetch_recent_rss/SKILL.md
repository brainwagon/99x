---
name: fetch_recent_rss
description: Fetch the N most recent items from an RSS 2.0 or Atom feed (default N=5) via a bundled script, returning a compact summary. Use when the user gives a feed URL or asks for the latest items from a blog/news feed.
---

# Fetch recent RSS / Atom items

This skill fetches and parses the feed with a bundled script, so the raw
XML never enters context — only a short, already-summarised list comes back.

## Run

From this skill's directory (the `dir` returned by `load_skill`), run the
script with `run_bash`:

```
run_bash(command="python3 scripts/fetch_rss.py '<FEED URL>' [N]")
```

- `<FEED URL>` — the feed's URL (RSS 2.0 or Atom; both are handled).
- `[N]` — optional item count. Omit it to use the default of 5.

The script depends only on the Python standard library.

## Output

stdout is the only thing you need to read back to the user:

```
Feed: <feed title>
Showing <k> of most recent items

1. <item title>
   date: <pubDate / published>
   link: <url>
   <short plain-text summary>
...
```

On failure the script exits non-zero and prints a single `fetch failed:`
or `parse failed:` line to stderr — relay that and stop.

## Present

Show the list to the user roughly as printed, tightening formatting as
needed. Do not paste raw XML; if they want more detail on one item, fetch
that item's `link` separately.
