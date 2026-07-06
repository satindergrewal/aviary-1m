#!/usr/bin/env python3
# RULER runner via chat completions. RULER owns data-gen + scoring; this owns only the calling.
# Default: thinking ON (models tested as actually used). --no-thinking for the ablation condition.
import sys, json, os, argparse, urllib.request

def chat(url, prompt, max_tokens, thinking):
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0, "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": bool(thinking)},
    }).encode()
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=100000) as r:
        msg = json.load(r)["choices"][0]["message"]
    content = (msg.get("content") or "").strip()
    reasoning = (msg.get("reasoning_content") or "")
    return content, len(reasoning)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("infile"); ap.add_argument("outfile"); ap.add_argument("url")
    ap.add_argument("--no-thinking", action="store_true")
    args = ap.parse_args()
    thinking = not args.no_thinking
    max_tokens = 4096 if thinking else 256
    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    lines = [json.loads(l) for l in open(args.infile) if l.strip()]
    with open(args.outfile, "w") as f:
        for d in lines:
            prompt = d["input"] + d.get("answer_prefix", "")
            try:
                pred, rlen = chat(args.url, prompt, max_tokens, thinking)
            except Exception as e:
                pred, rlen = "", 0
                sys.stderr.write(f"call failed idx {d.get('index')}: {str(e)[:100]}\n")
            rec = {"index": d["index"], "pred": pred, "input": d["input"],
                   "outputs": d.get("outputs", [d.get("output","")]),
                   "others": d.get("others", {}), "reasoning_chars": rlen,
                   "thinking": thinking}
            f.write(json.dumps(rec) + "\n")
    print(f"WROTE {len(lines)} preds (thinking={thinking}) -> {args.outfile}")

if __name__ == "__main__":
    main()
