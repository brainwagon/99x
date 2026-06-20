---
name: fetch_rss
description: Fetch and summarise an RSS or Atom feed. Use when the user gives a feed URL or asks for the latest items from a blog/news feed.
---

# Fetch RSS / Atom

Use the `http_request` tool to GET the feed URL, then parse the returned XML.

```
http_request(url="<FEED URL>")
```

The `body` is XML in one of two shapes:

- **RSS 2.0**: `<rss><channel>` with `<title>` and repeated `<item>`
  elements, each holding `<title>`, `<link>`, `<description>`, `<pubDate>`.
- **Atom**: `<feed>` with `<title>` and repeated `<entry>` elements, each
  holding `<title>`, `<link href="...">`, `<summary>`, `<published>`.

Extract the feed title and the most recent ~10 items (title, link, short
summary, date) and present them as a list. If the body is large, you may
write it to a temp file and use `run_bash` with `grep`/`xmllint` to slice it.
