#!/usr/bin/env python3
"""Fetch the N most recent items from an RSS 2.0, RSS 1.0 (RDF) or Atom feed.

Two modes, both parsing the feed here so the bulky XML never reaches the
agent's context:

* default (compact) — one short block per item (index, title, date, link,
  ~280-char plain-text summary). Best for "show me the latest N items".
* ``--full`` — full untruncated plain-text content per item, so the model
  can reason over the feed (themes, which items mention X, etc.).

Usage:
    fetch_rss.py <FEED_URL> [-n N] [--full]

N defaults to 5. Exit code is non-zero on fetch/parse failure, with a
one-line error on stderr.
"""

import argparse
import html
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET

DEFAULT_N = 5
SUMMARY_CHARS = 280
USER_AGENT = "agent99x-fetch_rss/1.0 (+https://example.invalid)"


def strip_html(text: str, limit: int | None = SUMMARY_CHARS) -> str:
    """Collapse an HTML/escaped blurb into plain text, optionally truncated."""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    if limit is not None and len(text) > limit:
        text = text[:limit].rstrip() + "…"
    return text


def localname(tag: str) -> str:
    """Drop the XML namespace, e.g. '{http://www.w3.org/2005/Atom}entry' -> 'entry'."""
    return tag.rsplit("}", 1)[-1]


def inner_text(elem) -> str:
    """Element content as text. Child elements (Atom type="xhtml" content is
    a <div> of XHTML) are serialized as markup for strip_html to flatten."""
    parts = [elem.text or ""]
    for child in elem:
        parts.append(ET.tostring(child, encoding="unicode"))  # includes its tail
    return "".join(parts).strip()


def child_text(elem, name: str) -> str:
    for c in elem:
        if localname(c.tag) == name:
            return inner_text(c)
    return ""


def child_plain(elem, name: str) -> str:
    """Like child_text, but flatten XHTML markup (Atom type="xhtml") to text.
    Plain text is returned untouched, so a title like "a < b" survives."""
    for c in elem:
        if localname(c.tag) == name:
            return strip_html(inner_text(c), None) if len(c) else (c.text or "").strip()
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


def parse_items(root, full: bool):
    """Return (feed_title, items) for RSS 2.0, RSS 1.0 or Atom. items carry
    plain-text summaries."""
    tag = localname(root.tag)
    limit = None if full else SUMMARY_CHARS
    items = []

    if tag in ("rss", "channel", "RDF"):
        channel = root if tag == "channel" else next(
            (c for c in root if localname(c.tag) == "channel"), root
        )
        feed_title = child_plain(channel, "title")
        # RSS 2.0 nests <item>s in <channel>; RSS 1.0 (RDF) makes them
        # siblings of <channel> under <rdf:RDF>.
        parent = root if tag == "RDF" else channel
        for it in parent:
            if localname(it.tag) != "item":
                continue
            # In full mode prefer content:encoded (full article) over description.
            body = child_text(it, "encoded") if full else ""
            body = body or child_text(it, "description")
            items.append({
                "title": child_plain(it, "title"),
                "link": child_text(it, "link") or it.attrib.get(
                    "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about", ""),
                "date": child_text(it, "pubDate") or child_text(it, "date"),
                "summary": strip_html(body, limit),
            })
    elif tag == "feed":  # Atom
        feed_title = child_plain(root, "title")
        for it in root:
            if localname(it.tag) != "entry":
                continue
            # In full mode prefer content (full entry) over summary.
            body = child_text(it, "content") if full else ""
            body = body or child_text(it, "summary") or child_text(it, "content")
            items.append({
                "title": child_plain(it, "title"),
                "link": atom_link(it),
                "date": child_text(it, "published") or child_text(it, "updated"),
                "summary": strip_html(body, limit),
            })
    else:
        raise ValueError(f"Unrecognised feed root <{tag}> (expected rss/RDF/feed)")

    return feed_title, items


def main(argv):
    parser = argparse.ArgumentParser(
        prog="fetch_rss.py",
        description="Fetch the N most recent items from an RSS/Atom feed.",
    )
    parser.add_argument("url", help="Feed URL (RSS 2.0, RSS 1.0 or Atom).")
    parser.add_argument("-n", "--num", type=int, default=DEFAULT_N,
                        help=f"Number of items (default {DEFAULT_N}).")
    parser.add_argument("--full", action="store_true",
                        help="Emit full untruncated item content for reasoning.")
    args = parser.parse_args(argv[1:])
    n = max(1, args.num)

    try:
        req = urllib.request.Request(args.url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read()
    except Exception as e:  # noqa: BLE001 - report any fetch failure cleanly
        print(f"fetch failed: {e}", file=sys.stderr)
        return 1

    try:
        root = ET.fromstring(raw)
        feed_title, items = parse_items(root, args.full)
    except Exception as e:  # noqa: BLE001
        print(f"parse failed: {e}", file=sys.stderr)
        return 1

    items = items[:n]
    print(f"Feed: {feed_title or '(untitled)'}")
    print(f"Showing {len(items)} of most recent items"
          f"{' (full content)' if args.full else ''}\n")
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
