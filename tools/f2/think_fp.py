#!/usr/bin/env python3
# Does detect_loop false-positive on healthy chain-of-thought?
# Control by construction: Qwen3.6-27B-Q4_K_M is ~4.5 bpw, far ABOVE the 2.30 loop knee,
# so it should not be looping. Anything detect_loop flags in its think blocks is a
# FALSE POSITIVE caused by applying a prose-tuned threshold to reasoning text.
import json, sys, urllib.request
sys.path.insert(0, "$HOME/Documents/GitHub/ornith-1m/tools")
from loop_rate import detect_loop

PORT = sys.argv[1] if len(sys.argv) > 1 else "8777"
NPRED = int(sys.argv[2]) if len(sys.argv) > 2 else 400

# prompts chosen to provoke the reasoning behaviours most likely to look repetitive:
# re-checking, enumerating, restating constraints, counting.
PROMPTS = [
    "A farmer has 17 sheep. All but 9 run away. How many are left? Check your answer carefully.",
    "List the first 12 prime numbers, then verify each one is actually prime.",
    "If a train leaves at 3:15pm travelling 80km/h and another at 4:00pm at 100km/h, when does the second catch the first? Show your working and double-check it.",
    "Sort these by population, largest first: Peru, Nepal, Ghana, Chile, Zambia. Reconsider before answering.",
    "What is 17 * 24? Compute it two different ways and confirm they agree.",
    "Name the days of the week that contain the letter 's', then re-read your list to be sure.",
]

def gen(p):
    body = {"messages":[{"role":"user","content":p}], "max_tokens":NPRED,
            "temperature":0.0, "cache_prompt":False}
    r = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                               data=json.dumps(body).encode(),
                               headers={"Content-Type":"application/json"})
    d = json.load(urllib.request.urlopen(r, timeout=600))
    m = (d.get("choices") or [{}])[0].get("message", {}) or {}
    return (m.get("reasoning_content") or m.get("reasoning") or ""), (m.get("content") or "")

print(f"{'#':<3} {'think words':>11} {'think verdict':<28} {'content verdict':<20}")
fp_think = fp_content = n = 0
for i, p in enumerate(PROMPTS):
    think, content = gen(p)
    tw = len(think.split())
    tv, tev = detect_loop(think)
    cv, cev = detect_loop(content)
    n += 1
    fp_think += (tv == "LOOP")
    fp_content += (cv == "LOOP")
    print(f"{i:<3} {tw:>11} {tv+' ('+tev+')':<28} {cv+' ('+cev+')':<20}")

print()
print(f"think blocks flagged as LOOP  : {fp_think}/{n}   <-- every one is a FALSE POSITIVE")
print(f"content flagged as LOOP       : {fp_content}/{n}")
print()
print("READING:", "prose-tuned threshold DOES misfire on reasoning traces"
      if fp_think > fp_content else "no evidence of a think-specific false-positive problem")
