#!/usr/bin/env python3
"""Render harvested K3 traces (JSONL) into K3-native XTML calibration text.

Input:  JSONL of {"prompt", "reasoning", "content"} from harvest_v3_traces.py
Output: native-format corpus on stdout (same structure as make_native_calib.py):
  <|open|>message role="system" type="thinking-effort"<|sep|>...<|end_of_msg|>
  <|open|>message role="user"<|sep|>PROMPT<|close|>message<|sep|><|end_of_msg|>
  <|open|>message role="assistant"<|sep|>
  <|open|>think<|sep|>REASONING<|close|>think<|sep|>
  <|open|>response<|sep|>CONTENT<|close|>response<|sep|>
  <|close|>message<|sep|><|end_of_msg|>

Skips traces with empty reasoning (no think block to teach) or errors.
Traces with empty CONTENT are kept (think-only examples) but counted.
"""
import json, sys

THINK_EFFORT_SYS = ('<|open|>message role="system" type="thinking-effort"<|sep|>'
                    "`thinking_effort` guides on how much to think in the response."
                    "<|close|>message<|sep|><|end_of_msg|>\n")

def msg(role, content):
    return (f'<|open|>message role="{role}"<|sep|>{content}'
            f'<|close|>message<|sep|><|end_of_msg|>\n')

def assistant(reasoning, content):
    out = '<|open|>message role="assistant"<|sep|>\n'
    out += f'<|open|>think<|sep|>{reasoning.strip()}<|close|>think<|sep|>\n'
    if content.strip():
        out += f'<|open|>response<|sep|>{content.strip()}<|close|>response<|sep|>\n'
    out += '<|close|>message<|sep|><|end_of_msg|>\n'
    return out

def main():
    n = n_skip = n_think_only = 0
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            n_skip += 1
            continue
        if "error" in r or not r.get("reasoning", "").strip():
            n_skip += 1
            continue
        sys.stdout.write(THINK_EFFORT_SYS)
        sys.stdout.write(msg("user", r["prompt"]))
        sys.stdout.write(assistant(r["reasoning"], r.get("content", "")))
        if not r.get("content", "").strip():
            n_think_only += 1
        n += 1
    print(f"rendered={n} think_only={n_think_only} skipped={n_skip}", file=sys.stderr)

if __name__ == "__main__":
    main()
