#!/usr/bin/env python3
"""Crude coherence gate: FAIL if generation shows loop collapse (IQ1_S signature)."""
import subprocess, sys, collections
model = sys.argv[1]
cli = sys.argv[2] if len(sys.argv) > 2 else "build/bin/llama-cli"
r = subprocess.run([cli, "-m", model, "--jinja", "-p",
                    "Say hello and name three colors.", "-n", "150", "--temp", "0", "-no-cnv"],
                   capture_output=True, text=True, timeout=600)
text = r.stdout.strip()
toks = text.split()
if len(toks) < 20:
    print(f"FAIL: only {len(toks)} tokens generated"); sys.exit(1)
grams = collections.Counter(tuple(toks[i:i+6]) for i in range(len(toks)-5))
worst = grams.most_common(1)[0][1] if grams else 0
uniq = len(set(toks)) / len(toks)
verdict = "FAIL" if (worst >= 4 or uniq < 0.25) else "PASS"
print(f"{verdict}: max 6-gram repeats={worst}, unique-token ratio={uniq:.2f}, sample: {' '.join(toks[:25])!r}")
sys.exit(0 if verdict == "PASS" else 1)
