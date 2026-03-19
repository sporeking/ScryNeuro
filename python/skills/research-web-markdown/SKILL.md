---
name: research-web-markdown
description: Use when the task asks to gather web information (news, website summaries, trend checks) and produce a markdown deliverable. Trigger on requests mentioning web fetch, news search, source citation, or markdown report generation.
license: MIT
compatibility: Requires built-in tools web_fetch, write_file, read_file, and optional grep_text.
triggers: web fetch, news, search, source, markdown, summary, report
requires_tools: web_fetch, write_file, read_file
category: research
priority: 2
max_injection_chars: 1200
---

# Research Web Markdown Skill

## Overview
Collect information from one or more web pages and produce a concise, source-cited markdown document.

## When to Use
- User asks for latest news summary.
- User asks to browse/fetch websites and synthesize findings.
- User asks for a markdown report/output file.

## Workflow
1. Fetch at least one relevant source URL with `web_fetch`.
2. Extract the most important points; avoid copying excessive raw HTML.
3. Draft markdown with this structure:
   - `# Title`
   - `## Sources`
   - `## Summary`
   - `## Key Points`
4. Write output via `write_file` to user-specified path (or `reports/latest_news.md` by default).
5. Read back with `read_file` (optional) and provide a short completion message.

## Output Requirements
- Keep summary concise and factual.
- Include source URL(s).
- Use markdown headings and bullet lists.

## Anti-Patterns
- Do not claim you visited pages without running `web_fetch`.
- Do not fabricate source URLs.
- Do not output only raw HTML when user asked for a digest.
