#!/usr/bin/env python3
# F3 real axis: does the server's -c VALUE change draft acceptance?
# Issue #23658 claims -c 12032 => AR 16%, -c 12288 => AR 71% (same generations).
# Fixed prompt set, temp 0, 256 tokens each, pooled AR. Identical everything but -c.
import json, sys, urllib.request
PORT = sys.argv[1]; TAG = sys.argv[2]
BASE = f"http://127.0.0.1:{PORT}"

PROMPTS = [
    "Explain how a lighthouse works, in detail.",
    "Write a short story about a cartographer who maps a city that does not exist.",
    "Describe the process of photosynthesis step by step.",
    "List and explain five common algorithms for sorting an array.",
    "Summarise the causes of the fall of the Western Roman Empire.",
    "Write a technical explanation of how TCP congestion control works.",
]

def post(o, t=240):
    r = urllib.request.Request(BASE + "/completion", data=json.dumps(o).encode(),
                               headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=t))

tot_dn = tot_da = 0
print(f"--- {TAG} (port {PORT}) ---")
for i, p in enumerate(PROMPTS):
    d = post({"prompt": p, "n_predict": 256, "temperature": 0, "cache_prompt": False})
    t = d.get("timings", {})
    dn = t.get("draft_n", 0); da = t.get("draft_n_accepted", 0)
    tot_dn += dn; tot_da += da
    acc = (da / dn) if dn else float("nan")
    print(f"  p{i}: draft_n {dn:>5}  accepted {da:>5}  AR {acc:.4f}  tp {d.get('tokens_predicted')}")
pooled = (tot_da / tot_dn) if tot_dn else float("nan")
print(f"POOLED {TAG}: {tot_da}/{tot_dn} = {pooled:.4f}")
