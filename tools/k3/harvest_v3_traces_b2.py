#!/usr/bin/env python3
"""Batch 2 of K3-native thinking-trace harvest — fresh prompts, same engine.
Output: JSONL {prompt, reasoning, content} to stdout."""
import json, sys, time, urllib.request

EP = "http://127.0.0.1:11434/v1/chat/completions"
MODEL = "kimi-k3:cloud"

CODE = [
    "Explain what tail call optimization is and which languages support it.",
    "What is the difference between TCP and UDP, and when would you pick each?",
    "Why does this Python code print 4: `def f(x=[]): x.append(1); return len(x); f(); f(); print(f())`?",
    "Explain the difference between a mutex and a semaphore.",
    "What does a foreign key cascade delete do and when is it dangerous?",
    "Explain the difference between a left fold and a right fold in functional programming.",
    "What is a race condition in the context of file writes, and how do you prevent it?",
    "Explain what consistent hashing is used for in distributed systems.",
    "Why is `SELECT *` often discouraged in production SQL?",
    "Explain the difference between compile-time and run-time polymorphism.",
    "What does the CAP theorem say, and what trade-off does it force?",
    "Explain what memoization is and rewrite naive fibonacci using it.",
    "What is the difference between a library and a framework?",
    "Explain why `i++` is not atomic in most languages and what that implies.",
    "What is the purpose of a circuit breaker pattern in microservices?",
    "Explain the difference between breadth-first and depth-first search with a use case for each.",
    "What does idempotency mean for an HTTP endpoint and why does it matter?",
    "Explain what a bloom filter is and what it cannot tell you.",
    "Why do floating point comparisons need an epsilon?",
    "Explain the difference between optimistic and pessimistic locking.",
]

MATH = [
    "What is 512 divided by 16? Show the long division.",
    "Compute 89 x 67 step by step.",
    "What is 7/8 as a decimal and a percentage?",
    "If a car uses 8.5 litres per 100 km, how much fuel for 350 km?",
    "What is the cube root of 2744? Reason it out.",
    "Simplify: (3x + 2)(x - 4). Show the expansion.",
    "What is 17 squared minus 15 squared? Use the difference of squares.",
    "A tank fills at 12 litres/min and drains at 5 litres/min. Net rate and time to fill 210 litres?",
    "What is the 8th Fibonacci number? List the sequence to get there.",
    "If log base 2 of x equals 6, what is x?",
    "What is the probability of rolling a total of 7 with two dice?",
    "Compute 1001 x 1001 using (1000+1)^2.",
    "What is 0.75 of 360?",
    "How many prime numbers are there between 20 and 40?",
    "If y = 3x^2 - 2x + 1, what is y when x = 4?",
    "What is the average of 14, 22, 9, 31, 18?",
    "Convert 2.4 kilometres to centimetres.",
    "A shop sells 3 pens for $5. What do 11 pens cost?",
    "What is 144 divided by 0.12?",
    "If a number is doubled and then increased by 9 to get 31, what was the number?",
]

EXPLAIN = [
    "Explain in two sentences what a virtual machine is.",
    "Explain in two sentences why vaccines work.",
    "Explain in two sentences what a rainbow is.",
    "Explain in two sentences why seasons happen.",
    "Explain in two sentences what a firewall does.",
    "Explain in two sentences why hot air rises.",
    "Explain in two sentences what inflation is.",
    "Explain in two sentences why we sleep.",
    "Explain in two sentences what a compiler does.",
    "Explain in two sentences why steel rusts.",
    "Explain in two sentences what an operating system kernel is.",
    "Explain in two sentences why mirrors reflect.",
    "Explain in two sentences what a stock exchange is.",
    "Explain in two sentences why deserts form.",
    "Explain in two sentences what a proxy server does.",
]

def ask(prompt):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2500,
        "temperature": 0.7,
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
    n_ok = 0
    for p in CODE + MATH + EXPLAIN:
        r = None
        for attempt in range(3):
            try:
                r = ask(p)
                break
            except Exception as e:
                if attempt == 2:
                    print(json.dumps({"prompt": p, "error": str(e)}), flush=True)
                else:
                    time.sleep(10 * (attempt + 1))
        if r and r["reasoning"].strip():
            print(json.dumps(r), flush=True)
            n_ok += 1
        time.sleep(0.8)
    print(f"batch2_harvested={n_ok}", file=sys.stderr)

if __name__ == "__main__":
    main()
