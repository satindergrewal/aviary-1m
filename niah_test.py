#!/usr/bin/env python3
"""Multi-needle in a haystack test against a llama-server (OpenAI-compatible API).

Plants N needles ("The secret code for <city> is <code>") at spread depths in
filler text, asks for all of them in one pass, scores retrieval per depth.

Usage: python3 niah_test.py --tokens 262144 [--needles 10] [--url http://127.0.0.1:8000]
"""
import argparse, json, random, time, urllib.request

CITIES = ["Wellington", "Auckland", "Christchurch", "Dunedin", "Hamilton",
          "Tauranga", "Napier", "Nelson", "Queenstown", "Rotorua",
          "Invercargill", "Whangarei", "Gisborne", "Timaru", "Blenheim"]

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

def tokenize_len(url, text):
    req = urllib.request.Request(f"{url}/tokenize",
        data=json.dumps({"content": text}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return len(json.load(r)["tokens"])

def build_haystack(target_tokens, needles, chars_per_token, seed=42):
    # needles: list of (depth_fraction, sentence)
    target_chars = int(target_tokens * chars_per_token)
    rng = random.Random(seed)
    chunks, total = [], 0
    while total < target_chars:
        s = rng.choice(FILLER)
        chunks.append(s)
        total += len(s) + 1
    positions = {int(f * len(chunks)): s for f, s in needles}
    out = []
    for i, c in enumerate(chunks):
        if i in positions:
            out.append(positions[i])
        out.append(c)
    return " ".join(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, required=True)
    ap.add_argument("--needles", type=int, default=10)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--chars-per-token", type=float, default=0.0, help="override calibration")
    ap.add_argument("--seed", type=int, default=1337, help="vary to bust server prompt cache for cold timing")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    cities = rng.sample(CITIES, args.needles)
    codes = {c: f"{rng.randint(1000000, 9999999)}" for c in cities}
    depths = [round((i + 0.5) / args.needles, 3) for i in range(args.needles)]
    needles = [(d, f"Remember this fact: the secret code for {c} is {codes[c]}.")
               for d, c in zip(depths, cities)]

    cpt = args.chars_per_token
    if not cpt:
        sample = " ".join(random.Random(7).choices(FILLER, k=300))
        cpt = len(sample) / tokenize_len(args.url, sample)
        print(f"calibrated chars/token = {cpt:.3f}")

    # leave headroom for question + answer + template
    hay = build_haystack(args.tokens - 600, needles, cpt, seed=args.seed * 31 + 42)
    question = ("Now answer using only the text above. List the secret code for each of "
                "these cities, one per line in the format 'City: code'. Cities: "
                + ", ".join(sorted(cities)) + ".")
    prompt = hay + "\n\n" + question

    actual = tokenize_len(args.url, prompt)
    print(f"prompt tokens: {actual}")

    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0, "max_tokens": 4096,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    t0 = time.time()
    req = urllib.request.Request(f"{args.url}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=100000) as r:
        resp = json.load(r)
    dt = time.time() - t0
    answer = resp["choices"][0]["message"]["content"]
    usage = resp.get("usage", {})
    print(f"\n--- response ({dt:.0f}s, prompt={usage.get('prompt_tokens')}, "
          f"completion={usage.get('completion_tokens')}) ---\n{answer}\n---")

    ok = 0
    results = []
    for d, c in sorted(zip(depths, cities)):
        hit = codes[c] in answer
        ok += hit
        results.append({"depth": d, "city": c, "hit": bool(hit)})
        print(f"depth {int(d*100):3d}%  {c:14s} {codes[c]}  {'PASS' if hit else 'FAIL'}")
    print(f"\nscore: {ok}/{args.needles} at ~{actual} tokens")

    with open("results.jsonl", "a") as f:
        f.write(json.dumps({
            "target_tokens": args.tokens, "prompt_tokens": actual,
            "score": ok, "needles": args.needles, "wall_seconds": round(dt, 1),
            "completion_tokens": usage.get("completion_tokens"),
            "depths": results,
        }) + "\n")

if __name__ == "__main__":
    main()
