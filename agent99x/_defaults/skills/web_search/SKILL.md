---
name: web_search
description: Search the web for current information using DuckDuckGo. Use when the user asks something that needs up-to-date or external information you don't already know.
---

# Web search

Use the `http_request` tool against DuckDuckGo's HTML endpoint (no API key).

```
http_request(
  url="https://html.duckduckgo.com/html/",
  method="POST",
  headers={"Content-Type": "application/x-www-form-urlencoded",
           "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
  body="q=<URL-ENCODED QUERY>",
)
```

The response `body` is HTML. Each result is an `<a class="result__a" href="...">`
(title + link) followed by an `<a class="result__snippet">` (description).
Extract the first handful of title/URL/snippet triples and summarise them for
the user, citing the URLs. To read a specific page in full, call
`http_request` on its URL and parse the returned HTML.
