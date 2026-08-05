#!/usr/bin/env python3
"""Parse one /completion response into  prompt_n#content_len#preview

⚠ THIS LIVES IN A FILE ON PURPOSE. It was inlined in the gate as `python3 -c '...'` inside shell
single quotes, and produced TWO defects in a row from escaping alone:
  1. the empty-content default was written as the two-character string "" -- so a MISSING field
     arrived as non-empty text and the empty-counter scored it as content. The gate PASSED on a run
     that visibly printed None#"" twice.
  2. rewriting it with \" escapes made Python receive literal backslash-quotes: every run PARSE_FAIL.
A verdict that cannot represent "nothing" cannot detect nothing, and a parser I cannot quote
correctly is a parser I cannot trust. Both problems disappear once it stops being a shell string.

content_len is NUMERIC: -1 absent, 0 empty, >0 real. No string can accidentally look like content.
"""
import json
import sys

try:
    d = json.load(sys.stdin)
except Exception:
    print("PARSE_FAIL#-1#")
    sys.exit(0)

t = d.get("timings") or {}
c = d.get("content")
n = len(c) if isinstance(c, str) else -1
print(f'{t.get("prompt_n")}#{n}#{(c or "")[:20]!r}')
