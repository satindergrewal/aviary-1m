#!/usr/bin/env python3
# D1 measurement: pooled draft acceptance AND the generated text, per arm.
# Text is captured so spec-on output can be compared against a no-spec baseline:
# llama.cpp verifies drafts against the target, so greedy output must be identical
# no matter how badly the MTP block is quantized. If it is not, that is a real finding.
import json, sys, urllib.request, hashlib

PORT = sys.argv[1]; TAG = sys.argv[2]; OUT = sys.argv[3]
BASE = f"http://127.0.0.1:{PORT}"

PROMPTS = [
    "Explain how a lighthouse works, in detail.",
    "Write a short story about a cartographer who maps a city that does not exist.",
    "Describe the process of photosynthesis step by step.",
    "List and explain five common algorithms for sorting an array.",
    "Summarise the causes of the fall of the Western Roman Empire.",
    "Write a technical explanation of how TCP congestion control works.",
]

def post(o, t=300):
    r = urllib.request.Request(BASE + "/completion", data=json.dumps(o).encode(),
                               headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=t))

rows, texts = [], []
tot_dn = tot_da = 0
for i, p in enumerate(PROMPTS):
    d = post({"prompt": p, "n_predict": 256, "temperature": 0, "cache_prompt": False})
    t = d.get("timings", {})
    dn = t.get("draft_n", 0); da = t.get("draft_n_accepted", 0)
    tot_dn += dn; tot_da += da
    txt = d.get("content", "")
    texts.append(txt)
    rows.append({"i": i, "draft_n": dn, "accepted": da,
                 "ar": (da/dn) if dn else None,
                 "tps": t.get("predicted_per_second"),
                 "sha": hashlib.sha256(txt.encode()).hexdigest()[:16]})
    print(f"  p{i}: draft_n {dn:>5} accepted {da:>5} "
          f"AR {(da/dn if dn else float('nan')):.4f}  {t.get('predicted_per_second',0):.1f} tok/s  sha {rows[-1]['sha']}")

pooled = (tot_da/tot_dn) if tot_dn else None
mean_tps = sum(r["tps"] or 0 for r in rows)/len(rows)
print(f"POOLED {TAG}: {tot_da}/{tot_dn} = {pooled if pooled is None else round(pooled,4)}   mean {mean_tps:.1f} tok/s")

json.dump({"tag": TAG, "pooled_ar": pooled, "mean_tps": mean_tps,
           "rows": rows, "texts": texts}, open(OUT, "w"))
