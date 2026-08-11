#!/usr/bin/env python3
"""Harvest K3-native thinking traces for the v3 calibration corpus.

Why: the compound verdict (2026-08-01) showed the remaining failures are CLASS-pure:
explanatory construction confabulates, computation echoes the problem and never computes.
The v2 corpus contained recall-flavored thinking only. This harvests REAL K3 reasoning
(code / math / explanation classes) via the local ollama cloud endpoint, capturing
reasoning_content AND content as pairs — the response regime needs pairs, not bare think.

Endpoint: local ollama (127.0.0.1:11434), model kimi-k3:cloud. No credentials read —
ollama holds them. Output: JSONL of {prompt, reasoning, content} to stdout.
"""
import json, sys, time, urllib.request

EP = "http://127.0.0.1:11434/v1/chat/completions"
MODEL = "kimi-k3:cloud"

CODE = [
    "What is the time complexity of binary search and why?",
    "Explain what this code does: `def f(x): return x if x > 0 else -x`",
    "Find the bug in this code: `for i in range(len(a)+1): print(a[i])`",
    "Why is quicksort's worst case O(n^2) and how do you avoid it?",
    "Explain the difference between a stack and a queue with one example each.",
    "What does this regex match: `^[a-z]+_[0-9]{2,4}$`?",
    "When would you choose a hash map over a balanced tree?",
    "Explain why this deadlocks: two threads each holding one lock and requesting the other's.",
    "What is the difference between `shallow copy` and `deep copy` in Python?",
    "Improve this function: `def dup(l): r=[]\nfor x in l:\n  if x not in r: r.append(x)\nreturn r`",
    "Explain what a context manager does in Python with a short example.",
    "Why does `float('0.1') + float('0.2') != 0.3` in most languages?",
    "What is the purpose of an index in a database and when does it hurt?",
    "Explain the difference between processes and threads for memory sharing.",
    "What does `tail -f` do and how is it different from `tail`?",
    "Explain why this recursion is inefficient: `def fib(n): return n if n<2 else fib(n-1)+fib(n-2)`",
    "What is the difference between HTTP 301 and 302 redirects for SEO?",
    "Explain what a JOIN does in SQL and the difference between INNER and LEFT.",
    "Why is `git rebase` considered dangerous on shared branches?",
    "What does the `volatile` keyword mean in C and when do you need it?",
]

MATH = [
    "What is 127 times 43? Show your reasoning.",
    "Compute 236 times 58 step by step.",
    "What is 15% of 840?",
    "A train travels 240 km in 3 hours. What is its average speed in m/s?",
    "What is the next number in the sequence 3, 7, 15, 31, 63?",
    "If a shirt costs $45 after a 25% discount, what was the original price?",
    "What is 2^14? Reason it out without a calculator.",
    "Convert 72 km/h to meters per second.",
    "What is the sum of the first 100 positive integers?",
    "If 3x + 7 = 25, what is x?",
    "What is the area of a circle with radius 5 cm? Use 3.14 for pi.",
    "Estimate: is 499 x 51 closer to 25,000 or 30,000? Show why.",
    "What is the greatest common divisor of 84 and 120?",
    "A rectangle is 12 cm by 7 cm. What is its perimeter and area?",
    "What is 1000 divided by 7 to two decimal places?",
    "If inflation is 4% per year, what does $100 become after 2 years?",
    "What is the least common multiple of 6, 8, and 10?",
    "How many seconds are in 2.5 hours?",
    "What is the median of 4, 9, 2, 7, 5?",
    "If a recipe for 4 people needs 300g flour, how much for 10 people?",
]

EXPLAIN = [
    "Explain in two sentences why the sky is blue.",
    "Explain in two sentences why metals conduct electricity.",
    "Explain in two sentences what an API is.",
    "Explain in two sentences why the moon has phases.",
    "Explain in two sentences what DNS does.",
    "Explain in two sentences why ice floats on water.",
    "Explain in two sentences what a hash function is.",
    "Explain in two sentences why planes can fly.",
    "Explain in two sentences what a checksum is for.",
    "Explain in two sentences why the sea is salty.",
    "Explain in two sentences what cache memory is.",
    "Explain in two sentences why leaves change color in autumn.",
    "Explain in two sentences what a load balancer does.",
    "Explain in two sentences why the speed of light is a limit.",
    "Explain in two sentences what garbage collection does in a language runtime.",
]

def ask(prompt):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2500,
        "temperature": 0.6,
    }).encode()
    req = urllib.request.Request(EP, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.load(r)
    m = d["choices"][0]["message"]
    return {
        "prompt": prompt,
        "reasoning": m.get("reasoning_content") or m.get("reasoning") or "",
        "content": m.get("content") or "",
    }

def main():
    n_ok = n_empty_reasoning = n_empty_content = 0
    for i, p in enumerate(CODE + MATH + EXPLAIN):
        for attempt in range(3):
            try:
                r = ask(p)
                break
            except Exception as e:
                if attempt == 2:
                    print(json.dumps({"prompt": p, "error": str(e)}), flush=True)
                    r = None
                else:
                    time.sleep(10 * (attempt + 1))
        if r is None:
            continue
        if not r["reasoning"].strip():
            n_empty_reasoning += 1
        if not r["content"].strip():
            n_empty_content += 1
        n_ok += 1
        print(json.dumps(r), flush=True)
        time.sleep(1)
    print(f"harvested={n_ok} empty_reasoning={n_empty_reasoning} empty_content={n_empty_content}", file=sys.stderr)

if __name__ == "__main__":
    main()
