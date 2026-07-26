#!/usr/bin/env python3
"""
loop_rate.py - measure the REPETITION-LOOP RATE of a served model.

WHY THIS EXISTS
    Perplexity does not tell you when a quantized model becomes unusable.
    Measured on Qwen3.6-27B (identical prompts, greedy, f16 KV):

        bpw   PPL     loop rate
        8.50  5.86     0/8
        4.80  6.01     0/8
        2.90  6.52     0/8
        1.75  9.31     4/8      <-- cliff
        1.60  14.32    5/8

    Going 8.50 -> 2.90 bpw costs +11% PPL and zero usability.
    Going 2.90 -> 1.75 bpw costs +43% PPL and half of all long generations.
    PPL rises smoothly across that range and gives you no threshold, so a
    quant that reads "somewhat worse but fine" can in fact be categorically
    broken for long-form or agentic use. Loop rate catches it; PPL does not.

METHOD (three choices that matter - do not change them casually)
    1. GREEDY / temperature 0. Decoding is then deterministic, so ONE run per
       configuration IS the answer. There is no sampling noise to average out
       and no statistical-power problem. Sampling at temp>0 with one run per
       cell produces pure noise - we learned that the expensive way.
    2. Detection is a REPEATING TAIL CYCLE of any period 1..40 (>=3 repeats),
       with a type-token-ratio fallback. A diversity proxy alone both misses
       real loops and flags legitimately repetitive text.
    3. An empty / near-empty generation is reported as EMPTY, never as a loop.
       Scoring "produced nothing" as "looped" invalidated a whole matrix once.

    Prompts demand long exhaustive output (enumerations, exhaustive essays,
    literature review) because that is what triggers attention collapse.

COMPARABILITY
    Loop rates are only comparable when the prompt set, token budget and
    decoding settings match. The prompt set is versioned: PROMPT_SET_VERSION.
    Report that version alongside any published number. Backends can also
    take different greedy paths, so do not mix CUDA and Metal numbers.

USAGE
    Start a server yourself (any llama.cpp-compatible /completion endpoint),
    then:

        python3 loop_rate.py --label "MyModel-IQ1_KT|1.75bpw"
        python3 loop_rate.py --port 8080 --sampler dry     # test DRY mitigation
        python3 loop_rate.py --tsv results.tsv --json results.json

    Exit status is 0 on success regardless of loop rate; parse the output.

MITIGATION
    The DRY sampler repaired 2/2 reproduced loops here (ttr 0.18 -> 0.57/0.59).
    Plain repetition penalty is NOT a substitute and can itself CAUSE loops -
    it suppresses common necessary tokens and pushes generation off its natural
    path. Run with --sampler dry to measure the improvement on your own model.
"""

import argparse
import json
import sys
import urllib.error
import urllib.request

PROMPT_SET_VERSION = "v1"

PROMPTS = [
    ("list", "List 300 distinct real-world examples of distributed systems, numbered, one per line, with a one-sentence description each.\n1."),
    ("essay", "Write an exhaustive 5000-word technical essay on consensus algorithms. Cover Paxos, Raft, PBFT, and modern variants in depth.\n"),
    ("compare", "Write a long detailed comparison of Raft versus Paxos versus PBFT, covering safety, liveness, throughput, and failure modes.\n"),
    ("enum", "Enumerate and describe 200 different failure modes that can occur in a distributed database. Number each one.\n1."),
    ("story", "Write a very long technical narrative about an engineer debugging a distributed system outage over three days. Be detailed.\n"),
    ("faq", "Write 150 frequently asked questions and detailed answers about distributed consensus.\nQ1:"),
    ("doc", "Write complete API documentation for a distributed lock service, with every endpoint, parameter, error code and example.\n"),
    ("review", "Write an exhaustive literature review of consensus research from 1978 to today, discussing each major paper.\n"),
]

DRY_SAMPLER = {
    "dry_multiplier": 0.8,
    "dry_base": 1.75,
    "dry_allowed_length": 2,
    "samplers": ["top_k", "top_p", "min_p", "temperature", "dry", "typ_p", "xtc"],
}


def detect_loop(text, min_words=20, max_period=40, min_reps=3, ttr_floor=0.25):
    """Return (verdict, evidence). verdict is 'LOOP', 'ok' or 'EMPTY'."""
    words = text.split()
    if len(words) < min_words:
        return ("EMPTY", "n_words=%d" % len(words))

    tail = words[-200:] if len(words) >= 200 else words

    for period in range(1, max_period + 1):
        if len(tail) < period * (min_reps + 1):
            break
        segment = tail[-period:]
        reps = 0
        i = len(tail) - period
        while i - period >= 0 and tail[i - period:i] == segment:
            reps += 1
            i -= period
        if reps >= min_reps:
            snippet = " ".join(segment)
            if len(snippet) > 60:
                snippet = snippet[:57] + "..."
            return ("LOOP", 'period=%d x%d "%s"' % (period, reps + 1, snippet))

    ttr = len(set(tail)) / len(tail)
    if ttr < ttr_floor:
        return ("LOOP", "ttr=%.2f over %d words" % (ttr, len(tail)))
    return ("ok", "ttr=%.2f" % ttr)


def generate(host, port, prompt, n_predict, sampler, timeout, temp=None):
    body = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0.0,          # greedy => deterministic (see --temp)
        "top_k": 40,
        "top_p": 0.95,
        "min_p": 0.05,
        # CRITICAL for depth testing: without this, llama.cpp reuses the previous
        # prompt's KV, so a shared prefill ACCUMULATES across prompts and each one runs
        # at a deeper context than its label. That produced fake EMPTY rows near the ctx
        # ceiling and invalidated a whole depth run. Each prompt must start from its own depth.
        "cache_prompt": False,
    }
    if sampler == "dry":
        body.update(DRY_SAMPLER)
        body["temperature"] = 0.7    # DRY needs a live distribution to reshape
    if temp is not None:
        body["temperature"] = temp   # explicit override wins, for matched pairs

    req = urllib.request.Request(
        "http://%s:%d/completion" % (host, port),
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    payload = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    return payload.get("content", ""), payload.get("timings", {})


def main():
    ap = argparse.ArgumentParser(description="Measure repetition-loop rate of a served model.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--label", default="model", help="name for this configuration, e.g. 'Qwen27B-IQ1_KT|1.75'")
    ap.add_argument("--n-predict", type=int, default=1500, help="tokens per prompt (default 1500)")
    ap.add_argument("--sampler", choices=["greedy", "dry"], default="greedy",
                    help="greedy = deterministic baseline; dry = measure DRY mitigation")
    ap.add_argument("--temp", type=float, default=None,
                    help="override temperature. Use this to build MATCHED pairs: run "
                         "--sampler greedy --temp 0.7 and --sampler dry --temp 0.7 so DRY "
                         "is the only variable. Comparing dry@0.7 against greedy@0.0 "
                         "changes two things at once and is not a clean test.")
    ap.add_argument("--prefill-tokens", type=int, default=0, metavar="N",
                    help="prepend ~N tokens of real prose before the prompt, so the loop "
                         "test runs at DEPTH rather than from a cold context. Loop rate is "
                         "depth-dependent: a config that is clean at 1.5K may still collapse "
                         "at 100K. Requires --prefill-file.")
    ap.add_argument("--prefill-file", default="<BOX>/llama.cpp-kt/wikitext-2-raw/wiki.train.raw",
                    help="source of prefill prose (real text, not word salad)")
    ap.add_argument("--timeout", type=int, default=5400)
    ap.add_argument("--tsv", help="append per-prompt rows to this TSV file")
    ap.add_argument("--json", dest="json_out", help="write a machine-readable summary here")
    ap.add_argument("--save-samples", metavar="DIR", help="write the tail of each looping generation here")
    args = ap.parse_args()

    prefix = ""
    if args.prefill_tokens > 0:
        # 4.37 chars/token measured on this tokenizer with real prose
        want = int(args.prefill_tokens * 4.37)
        with open(args.prefill_file, errors="replace") as fh:
            buf = []
            got = 0
            while got < want:
                chunk = fh.read(min(1 << 20, want - got))
                if not chunk:
                    fh.seek(0)          # wrap if the file is smaller than requested
                    continue
                buf.append(chunk)
                got += len(chunk)
        prefix = "".join(buf) + "\n\n=== END OF REFERENCE TEXT ===\n\n"

    eff_temp = args.temp if args.temp is not None else (0.7 if args.sampler == "dry" else 0.0)
    print("# loop_rate.py  prompt_set=%s  sampler=%s  temp=%.2f  n_predict=%d  prefill=%d  label=%s"
          % (PROMPT_SET_VERSION, args.sampler, eff_temp, args.n_predict, args.prefill_tokens, args.label))
    hdr = ["label", "prompt", "sampler", "n_pred", "verdict", "evidence"]
    if args.prefill_tokens:
        hdr.insert(3, "depth_n_past")
    print("\t".join(hdr))

    rows, looped, valid = [], 0, 0
    for name, prompt in PROMPTS:
        try:
            text, timings = generate(args.host, args.port, prefix + prompt,
                                     args.n_predict, args.sampler, args.timeout,
                                     args.temp)
            verdict, evidence = detect_loop(text)
            n_pred = timings.get("predicted_n", 0)
            if verdict != "EMPTY":
                valid += 1
                if verdict == "LOOP":
                    looped += 1
                    if args.save_samples:
                        import os
                        os.makedirs(args.save_samples, exist_ok=True)
                        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in args.label)
                        with open(os.path.join(args.save_samples, "%s_%s.txt" % (safe, name)), "w") as fh:
                            fh.write(text[-1500:])
        except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError) as exc:
            verdict, evidence, n_pred = "ERR", str(exc)[:60], 0

        row = [args.label, name, args.sampler, str(n_pred), verdict, evidence]
        if args.prefill_tokens:
            row.insert(3, str(timings.get("prompt_n", 0)) if isinstance(timings, dict) else "?")
        rows.append(row)
        print("\t".join(row))
        sys.stdout.flush()

    rate = ("%d/%d" % (looped, valid)) if valid else "n/a"
    pct = (100.0 * looped / valid) if valid else float("nan")
    print("# LOOP RATE: %s (%.0f%%)  [prompts scored: %d of %d]"
          % (rate, pct, valid, len(PROMPTS)))
    # Generation success is a FIRST-CLASS metric, not a footnote: a model can be 0-loops
    # simply because it produced nothing. At depth we measured exactly that (75%->25%
    # success under raw completion). Never publish a loop rate without this line.
    print("# GENERATION SUCCESS: %d/%d (%.0f%%) - prompts that produced scoreable output"
          % (valid, len(PROMPTS), 100.0 * valid / len(PROMPTS)))

    if args.tsv:
        with open(args.tsv, "a") as fh:
            for row in rows:
                fh.write("\t".join(row) + "\n")
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump({
                "label": args.label,
                "prompt_set_version": PROMPT_SET_VERSION,
                "sampler": args.sampler,
                "n_predict": args.n_predict,
                "looped": looped,
                "scored": valid,
                "total_prompts": len(PROMPTS),
                "loop_rate": rate,
                "rows": [dict(zip(["label", "prompt", "sampler", "n_pred", "verdict", "evidence"], r))
                         for r in rows],
            }, fh, indent=2)


if __name__ == "__main__":
    main()
