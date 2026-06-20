#!/usr/bin/env python3
"""Fetch the N most recent items from an RSS 2.0 or Atom feed.

Prints a compact, context-friendly summary to stdout: feed title followed
by one block per item (index, title, date, link, short summary). All the
bulky XML is parsed here so it never reaches the agent's context.

Usage:
    fetch_rss.py <FEED_URL> [N]

N defaults to 5. Exit code is non-zero on fetch/parse failure, with a
one-line error on stderr.
"""

import html
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET

DEFAULT_N = 5
SUMMARY_CHARS = 280
USER_AGENT = "agent99x-fetch_rss/1.0 (+https://example.invalid)"


def strip_html(text: str) -> str:
    """Collapse an HTML/escaped blurb into a short single line of plain text."""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > SUMMARY_CHARS:
        text = text[:SUMMARY_CHARS].rstrip() + "…"
    return text


def localname(tag: str) -> str:
    """Drop the XML namespace, e.g. '{http://www.w3.org/2005/Atom}entry' -> 'entry'."""
    return tag.rsplit("}", 1)[-1]


def child_text(elem, name: str) -> str:
    for c in elem:
        if localname(c.tag) == name:
            return (c.text or "").strip()
    return ""


def atom_link(entry) -> str:
    # Prefer rel="alternate" (the human page); fall back to first link with href.
    fallback = ""
    for c in entry:
        if localname(c.tag) != "link":
            continue
        href = c.attrib.get("href", "")
        if not href:
            continue
        rel = c.attrib.get("rel", "alternate")
        if rel == "alternate":
            return href
        fallback = fallback or href
    return fallback


def parse_items(root):
    """Return a list of {title, link, date, summary} for RSS or Atom."""
    tag = localname(root.tag)
    items = []

    if tag == "rss" or tag == "channel":
        channel = root if tag == "channel" else next(
            (c for c in root if localname(c.tag) == "channel"), root
        )
        feed_title = child_text(channel, "title")
        for it in channel:
            if localname(it.tag) != "item":
                continue
            items.append({
                "title": child_text(it, "title"),
                "link": child_text(it, "link"),
                "date": child_text(it, "pubDate"),
                "summary": strip_html(child_text(it, "description")),
            })
    elif tag == "feed":  # Atom
        feed_title = child_text(root, "title")
        for it in root:
            if localname(it.tag) != "entry":
                continue
            items.append({
                "title": child_text(it, "title"),
                "link": atom_link(it),
                "date": child_text(it, "published") or child_text(it, "updated"),
                "summary": strip_html(
                    child_text(it, "summary") or child_text(it, "content")
                ),
            })
    else:
        raise ValueError(f"Unrecognised feed root <{tag}> (expected rss/feed)")

    return feed_title, items


def main(argv):
    if len(argv) < 2:
        print("usage: fetch_rss.py <FEED_URL> [N]", file=sys.stderr)
        return 2
    url = argv[1]
    n = DEFAULT_N
    if len(argv) > 2:
        try:
            n = max(1, int(argv[2]))
        except ValueError:
            print(f"N must be an integer, got {argv[2]!r}", file=sys.stderr)
            return 2

    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read()
    except Exception as e:  # noqa: BLE001 - report any fetch failure cleanly
        print(f"fetch failed: {e}", file=sys.stderr)
        return 1

    try:
        root = ET.fromstring(raw)
        feed_title, items = parse_items(root)
    except Exception as e:  # noqa: BLE001
        print(f"parse failed: {e}", file=sys.stderr)
        return 1

    items = items[:n]
    print(f"Feed: {feed_title or '(untitled)'}")
    print(f"Showing {len(items)} of most recent items\n")
    for i, it in enumerate(items, 1):
        print(f"{i}. {it['title'] or '(untitled)'}")
        if it["date"]:
            print(f"   date: {it['date']}")
        if it["link"]:
            print(f"   link: {it['link']}")
        if it["summary"]:
            print(f"   {it['summary']}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
