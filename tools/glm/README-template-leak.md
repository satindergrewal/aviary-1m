# GLM-5.2 chat template: the minja `{% set %}` leak, and a two-line fix

## What breaks

llama.cpp's minja template engine **leaks `{% set %}` assignments across loop iterations**.
Real Jinja2 scopes them to the iteration. Minimal proof, template:

```jinja
{% for m in messages %}{% if m.role=="user" %}{% set X="SET" %}{% endif %}[{{X if X is defined else "UNDEF"}}]{% endfor %}
```

```
minja  : [user:SET][assistant:SET][user:SET][assistant:SET][user:SET]
jinja2 : [user:SET][assistant:UNDEF][user:SET][assistant:UNDEF][user:SET]
```

GLM-5.2's shipped template sets `reasoning_content` inside its message loop and then gates
emission on `reasoning_content is defined`. Under minja a value set on an early assistant turn
survives, so the gate stays open and **the same `<think>` block is re-emitted on every assistant
turn after the last user message**. Those are exactly the tool-calling turns of an agentic loop.

## Measured on our own serving path

`llama-server --jinja --chat-template-file <tmpl>`, `/apply-template`, a realistic coding-agent
conversation (user request, then alternating tool result / assistant), reasoning block ~330 chars:

| tool steps | as-shipped | fixed | think blocks (both) | leaked blocks: shipped -> fixed |
|---|---|---|---|---|
| 1 | 548 chars | 548 | 2 | 1 -> 1 (correct, it is the real one) |
| 4 | 1,784 | 869 | 5 | 4 -> 1 |
| 8 | 3,432 | 1,297 | 9 | 8 -> 1 |
| 16 | 6,740 | **2,165** | 17 | **16 -> 1** |

Leaked copies scale one-for-one with agentic depth on the shipped template and stay pinned at the
single genuine block once fixed. The think-block *count* is identical in both, so structure is
preserved; only duplicated content disappears. The 1-step case is byte-identical, which is the
regression check.

An independent audit of 3000 randomised conversations (replaying llama-server's exact chat path
against real Jinja2 with the same template text) measured the same defect in token terms: with a
660-token reasoning block, 16 tool steps gave 11,143 tokens against a correct 583, i.e. **19.1x**.

## Why it matters beyond prompt size

1. **It burns context.** The measured ceiling on this box is roughly 221K tokens at F16 KV; this
   consumes it far faster than the conversation warrants.
2. **It slows prefill**, proportional to the bloat.
3. **It fills the window with verbatim repeated text.** Note carefully: this does NOT invalidate
   the `loop-rate-quality-axis` result (weight bit-width causes looping). That harness defaults to
   raw `/completion` with a string prompt so minja never runs, uses a single user message so the
   leak has no later assistant turn to land in, and was a *paired* comparison where a template is
   a constant across arms - a constant cannot produce a between-arm difference. The plausible
   interaction is **amplification**, not confounding: repeated context times low bit-width may be
   multiplicative, which would make real agentic sessions worse than a clean-prompt reproducer
   predicts, and makes this fix more valuable rather than less.

## The fix (two lines, template-level, no interpreter change)

```diff
  {%- elif m.role == 'assistant' -%}
  <|assistant|>
+ {%- set reasoning_content = none %}
  {%- set content = visible_text(m.content) %}
...
- {%- if (... ) and reasoning_content is defined -%}
+ {%- if (... ) and reasoning_content is not none -%}
```

Reset the variable at the top of the assistant branch, then test the **value** rather than
definedness (after the reset, `is defined` can no longer discriminate). Both changes are correct
under real Jinja2 as well, so the fixed template cannot diverge from reference behaviour.

Patching minja's scoping would fix every affected template at once but is an interpreter change
with regression surface across all of them. The template-level fix has none, so it is the one to
run with first.

## Files

| file | what |
|---|---|
| `glm52_chat_template_asshipped.jinja` | extracted verbatim from our served GGUF |
| `glm52_chat_template_noleak.jinja` | the two-line fix |
| `fixture_agentic_{1,4,16}.json` | agentic-loop fixtures for `/apply-template` |

## Reproducing

```bash
# any tiny model; only the template renders, -ngl 0, no GPU needed
LD_LIBRARY_PATH=<build>/bin nohup <build>/bin/llama-server \
  -m <any-small>.gguf -ngl 0 -c 4096 --jinja \
  --chat-template-file tools/glm/glm52_chat_template_noleak.jinja \
  --host 127.0.0.1 --port 8462 &

curl -s -X POST 127.0.0.1:8462/apply-template -H 'Content-Type: application/json' \
  -d @tools/glm/fixture_agentic_16.json | python3 -c \
  'import json,sys; p=json.load(sys.stdin)["prompt"]; print(len(p), p.count("<think>"))'
```

Use `/apply-template`, not `tests/test-chat-template`: the latter hardcodes
`inputs.add_generation_prompt = true` (`tests/test-chat-template.cpp:244`) and therefore cannot
exercise prefill for any model. Its default fixture also has a tool *call* but no tool *result*,
so it misses that branch too.
