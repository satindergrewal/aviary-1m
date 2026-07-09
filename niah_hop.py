#!/usr/bin/env python3
"""Hop tier: multi-hop chains and variable tracking in a haystack.

Goes beyond parallel retrieval (niah_test.py): the model must COMBINE facts
spread across the context, not just fetch them.

Test 1, chained needles (2-hop): "the courier code for <city> is <code>" at one
depth, "parcels under courier code <code> contain <item>" far away. Question
asks what <city>'s parcels contain; answering requires resolving city -> code
-> item across the window.

Test 2, variable tracking: a named register is initialised once and mutated
several times at spread depths ("the register <name> is doubled", "... increases
by 17"). Question asks the final value; answering requires integrating every
mutation in document order.

Usage: python3 niah_hop.py --tokens 262144 [--chains 5] [--vars 3] [--url ...]
Appends to results_hop.jsonl in the working directory.
"""
import argparse, json, random, time, urllib.request

CITIES = ["Wellington", "Auckland", "Christchurch", "Dunedin", "Hamilton",
          "Tauranga", "Napier", "Nelson", "Queenstown", "Rotorua",
          "Invercargill", "Whangarei", "Gisborne", "Timaru", "Blenheim"]
ITEMS = ["woolen blankets", "ceramic tiles", "solar panels", "violin strings",
         "espresso machines", "hiking boots", "beeswax candles", "copper wire",
         "kayak paddles", "telescope lenses"]
REGISTERS = ["kauri", "totara", "rimu", "manuka", "kowhai"]

FILLER = [
    "The morning fog rolled slowly across the harbour while fishing boats prepared their nets for the day ahead.",
    "Economists continue to debate whether interest rate adjustments have any measurable effect on regional housing markets.",
    "The old library on Cuba Street holds thousands of maps that nobody has catalogued since the nineteen seventies.",
    "Migration patterns of coastal birds shift subtly each decade in response to changing ocean temperatures.",
    "A good sourdough starter requires patience, consistent feeding, and a kitchen that stays reasonably warm overnight.",
    "The tramway commission rejected three separate proposals before settling on the current route through the valley.",
    "Volcanic soil in the region produces wines with a mineral character that critics struggle to describe precisely.",
    "Software projects tend to accumulate complexity gradually until someone insists on deleting half of everything.",
    "The lighthouse keeper's journal records forty years of storms, shipwrecks, and the occasional visiting whale.",
    "Local rugby clubs have merged twice in the past decade as rural populations drift toward the cities.",
]

MUTATIONS = [
    ("is doubled", lambda v: v * 2),
    ("is halved and rounded down", lambda v: v // 2),
    ("increases by 17", lambda v: v + 17),
    ("decreases by 8", lambda v: v - 8),
    ("increases by 40", lambda v: v + 40),
    ("is tripled", lambda v: v * 3),
]

def tokenize_len(url, text):
    req = urllib.request.Request(f"{url}/tokenize",
        data=json.dumps({"content": text}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return len(json.load(r)["tokens"])

def build_haystack(target_tokens, planted, chars_per_token, seed):
    # planted: list of (depth_fraction, sentence); order within same slot preserved
    target_chars = int(target_tokens * chars_per_token)
    rng = random.Random(seed)
    chunks, total = [], 0
    while total < target_chars:
        s = rng.choice(FILLER)
        chunks.append(s)
        total += len(s) + 1
    slots = {}
    for f, s in planted:
        slots.setdefault(min(int(f * len(chunks)), len(chunks) - 1), []).append(s)
    out = []
    for i, c in enumerate(chunks):
        if i in slots:
            out.extend(slots[i])
        out.append(c)
    return " ".join(out)

def ask(url, prompt, thinking=True):
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0, "max_tokens": 16384 if thinking else 4096,
        "chat_template_kwargs": {"enable_thinking": bool(thinking)},
    }).encode()
    t0 = time.time()
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=100000) as r:
        resp = json.load(r)
    dt = time.time() - t0
    return resp["choices"][0]["message"]["content"], resp.get("usage", {}), dt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, required=True)
    ap.add_argument("--chains", type=int, default=5)
    ap.add_argument("--vars", type=int, default=3)
    ap.add_argument("--mutations", type=int, default=5, help="mutations per register")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--chars-per-token", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--no-thinking", action="store_true",
                    help="disable thinking (default ON: hop tier tests reasoning, not just recall)")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    # ---- chains: key at one depth, value far away (alternating direction) ----
    cities = rng.sample(CITIES, args.chains)
    items = rng.sample(ITEMS, args.chains)
    chains, planted = [], []
    for i, (city, item) in enumerate(zip(cities, items)):
        code = rng.randint(100000, 999999)
        d1 = round((i + 0.3) / args.chains, 3)          # key depth
        d2 = round(((i + args.chains // 2) % args.chains + 0.8) / args.chains, 3)  # value far away
        planted.append((d1, f"Note for the ledger: the courier code for {city} is {code}."))
        planted.append((d2, f"Warehouse manifest: parcels under courier code {code} contain {item}."))
        chains.append({"city": city, "code": code, "item": item,
                       "key_depth": d1, "value_depth": d2})

    # ---- variable tracking: init + mutations in document order ----
    registers = rng.sample(REGISTERS, args.vars)
    var_specs = []
    for r_i, name in enumerate(registers):
        v0 = rng.randint(10, 60)
        value = v0
        base = r_i / args.vars
        span = 1.0 / args.vars
        planted.append((round(base + 0.05 * span, 3),
                        f"Audit trail: the register named {name} is initialised to {v0}."))
        steps = rng.sample(MUTATIONS, args.mutations)
        for m_i, (desc, fn) in enumerate(steps):
            d = round(base + (0.15 + 0.8 * (m_i + 1) / (args.mutations + 1)) * span, 3)
            planted.append((d, f"Audit trail: the register named {name} {desc}."))
            value = fn(value)
        var_specs.append({"name": name, "init": v0, "final": value,
                          "mutations": args.mutations})

    cpt = args.chars_per_token
    if not cpt:
        sample = " ".join(random.Random(7).choices(FILLER, k=300))
        cpt = len(sample) / tokenize_len(args.url, sample)
        print(f"calibrated chars/token = {cpt:.3f}")

    hay = build_haystack(args.tokens - 800, planted, cpt, seed=args.seed * 31 + 42)
    question = (
        "Now answer using only the text above, in two parts.\n"
        "Part 1: for each city below, find its courier code, then find what parcels "
        "under that courier code contain. Answer one per line as 'City: item'. Cities: "
        + ", ".join(sorted(c["city"] for c in chains)) + ".\n"
        "Part 2: each named register was initialised once and then modified several times, "
        "in the order the audit trail states. Give the final value of each register, one per "
        "line as 'name: value'. Registers: " + ", ".join(sorted(r["name"] for r in var_specs)) + "."
    )
    prompt = hay + "\n\n" + question
    actual = tokenize_len(args.url, prompt)
    print(f"prompt tokens: {actual}")

    answer, usage, dt = ask(args.url, prompt, thinking=not args.no_thinking)
    print(f"\n--- response ({dt:.0f}s, prompt={usage.get('prompt_tokens')}, "
          f"completion={usage.get('completion_tokens')}) ---\n{answer}\n---")

    low = answer.lower()
    chain_ok = 0
    chain_rows = []
    for c in chains:
        hit = c["item"].lower() in low
        chain_ok += hit
        chain_rows.append({**c, "hit": bool(hit)})
        print(f"chain  {c['city']:14s} key@{int(c['key_depth']*100):3d}% val@{int(c['value_depth']*100):3d}%  "
              f"{c['item']:18s} {'PASS' if hit else 'FAIL'}")
    var_ok = 0
    var_rows = []
    for v in var_specs:
        import re
        m = re.search(rf"{v['name']}\s*[:=]?\s*(-?\d+)", low)
        got = int(m.group(1)) if m else None
        hit = got == v["final"]
        var_ok += hit
        var_rows.append({**v, "got": got, "hit": bool(hit)})
        print(f"track  {v['name']:14s} init={v['init']:3d} x{v['mutations']} steps -> expect {v['final']:5d} "
              f"got {str(got):>6s}  {'PASS' if hit else 'FAIL'}")

    print(f"\nchains: {chain_ok}/{args.chains}  tracking: {var_ok}/{args.vars} at ~{actual} tokens")
    with open("results_hop.jsonl", "a") as f:
        f.write(json.dumps({
            "target_tokens": args.tokens, "prompt_tokens": actual,
            "chains_score": chain_ok, "chains_total": args.chains,
            "tracking_score": var_ok, "tracking_total": args.vars,
            "wall_seconds": round(dt, 1), "seed": args.seed, "thinking": not args.no_thinking,
            "chains": chain_rows, "tracking": var_rows,
        }) + "\n")

if __name__ == "__main__":
    main()
