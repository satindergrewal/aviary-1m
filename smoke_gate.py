#!/usr/bin/env python3
"""Coherence gate v2: serve the GGUF briefly, judge ONLY the model's message content.

Usage: smoke_gate.py <model.gguf> [<llama-server-or-cli-path>]
FAIL on repetition-loop collapse (the 1-bit failure signature) or empty output.
"""
import collections
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

model = sys.argv[1]
server = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser(
    "~/Documents/GitHub/llama.cpp/build/bin/llama-server")
if server.endswith("llama-cli"):
    server = server.replace("llama-cli", "llama-server")
PORT = 8199

proc = subprocess.Popen(
    [server, "-m", model, "-c", "4096", "-np", "1", "-ngl", "99",
     "--jinja", "--port", str(PORT), "--host", "127.0.0.1"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    stdin=subprocess.DEVNULL, start_new_session=True)

def cleanup():
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except Exception:
        pass

try:
    deadline = time.time() + 900  # big models load slowly from disk
    up = False
    while time.time() < deadline:
        if proc.poll() is not None:
            print(f"FAIL: server exited rc={proc.returncode} during load")
            sys.exit(1)
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3)
            up = True
            break
        except Exception:
            time.sleep(3)
    if not up:
        print("FAIL: server never became healthy")
        cleanup(); sys.exit(1)

    body = json.dumps({
        "messages": [{"role": "user", "content": "Say hello and name three colors."}],
        "temperature": 0.0, "max_tokens": 220,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        content = json.load(r)["choices"][0]["message"]["content"] or ""
finally:
    cleanup()

toks = content.split()
if len(toks) < 8:
    print(f"FAIL: only {len(toks)} tokens in content: {content[:120]!r}")
    sys.exit(1)
grams = collections.Counter(tuple(toks[i:i+6]) for i in range(len(toks) - 5))
worst = grams.most_common(1)[0][1] if grams else 0
uniq = len(set(toks)) / len(toks)
verdict = "FAIL" if (worst >= 4 or uniq < 0.25) else "PASS"
print(f"{verdict}: max6gram={worst} uniq={uniq:.2f} content: {content[:160]!r}")
sys.exit(0 if verdict == "PASS" else 1)
