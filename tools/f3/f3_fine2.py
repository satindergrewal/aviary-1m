#!/usr/bin/env python3
# F3 #23658 — fine boundary sweep v2.
# Fixes v1's saturation confound: high-entropy prose at fixed temp (not a formulaic list),
# 128-token windows for ~2x the draft sample per cell, 2048 an EXACT window edge.
# Cache reuse proven per-window via timings.prompt_n (==1 means pure KV reuse).
import json, sys, urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "8650"
UB   = sys.argv[2] if len(sys.argv) > 2 else "512"
TEMP = float(sys.argv[3]) if len(sys.argv) > 3 else 0.7
SEED_N = int(sys.argv[4]) if len(sys.argv) > 4 else 1234
BASE = f"http://127.0.0.1:{PORT}"

SEED = ("Continue this novel. Keep inventing new scenes, new characters and new "
        "descriptions; never summarise and never repeat a sentence.\n\n"
        "The lighthouse keeper had not spoken to another living soul in nine months, "
        "and so when the boat appeared on the grey line of the horizon, he")

def post(path, obj, t=240):
    r = urllib.request.Request(BASE + path, data=json.dumps(obj).encode(),
                               headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=t))

def gen(toks, n):
    d = post("/completion", {"prompt": toks, "n_predict": n, "temperature": TEMP,
                             "seed": SEED_N, "cache_prompt": True, "return_tokens": True})
    t = d.get("timings", {})
    dn = t.get("draft_n", 0); da = t.get("draft_n_accepted", 0)
    return {"toks": d.get("tokens", []), "dn": dn, "da": da,
            "acc": (da / dn) if dn else float("nan"),
            "pn": t.get("prompt_n"), "tp": d.get("tokens_predicted", 0)}

ctx = list(post("/tokenize", {"content": SEED})["tokens"])
w = gen(ctx, 1792 - len(ctx)); ctx += w["toks"]
print(f"warm: depth {len(ctx)}  (seed acc {w['acc']:.4f})")

WIN = 128
rows = []
for _ in range(5):                      # 1792 -> 2432, edge at 2048
    s = len(ctx)
    r = gen(ctx, WIN); ctx += r["toks"]
    rows.append((s, len(ctx), r["dn"], r["da"], r["acc"], r["pn"]))

print(f"\n=== F3 #23658 sweep v2 | ub={UB} | temp={TEMP} | 2048 = exact edge ===")
print(f"{'window':<14} {'draft_n':>8} {'acc_n':>6} {'acceptance':>11} {'prompt_n':>9}  pos")
for s, e, dn, da, acc, pn in rows:
    pos = "crosses 2048" if s < 2048 <= e else ("post" if s >= 2048 else "pre")
    print(f"{str(s)+'-'+str(e):<14} {dn:>8} {da:>6} {acc:>11.4f} {pn:>9}  {pos}")

pre  = [r for r in rows if r[1] <= 2048]
post_= [r for r in rows if r[0] >= 2048]
if pre and post_:
    ap = sum(r[3] for r in pre)/max(1,sum(r[2] for r in pre))
    aq = sum(r[3] for r in post_)/max(1,sum(r[2] for r in post_))
    print(f"\npooled pre-2048 {ap:.4f} (n={sum(r[2] for r in pre)})  |  "
          f"pooled post-2048 {aq:.4f} (n={sum(r[2] for r in post_)})  |  delta {aq-ap:+.4f}")
saturated = all(abs(r[4]-1.0) < 1e-9 for r in rows)
print("cache reuse:", "OK (prompt_n==1)" if all(r[5] == 1 for r in rows) else f"CHECK prompt_n={[r[5] for r in rows]}")
print("saturation :", "STILL SATURATED (acc=1.0) - result uninformative" if saturated else "OK (acc<1, informative)")
