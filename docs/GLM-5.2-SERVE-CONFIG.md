# GLM 5.2 (KT IQ1_S) serve config: validated 2026-07-28

The exact configuration that ran well for hands-on and agentic use. Captured live
from `/proc/<pid>/cmdline` before shutdown, not reconstructed from memory.

## Command

```bash
cd <BOX>/llama.cpp-fleet
./build/bin/llama-server \
  -m <BOX>/bigmodels/glm52-ours/GLM-5.2-ours-IQ1_S-prot.gguf \
  -ngl 99 \
  --tensor-split 49,51 \
  -c 65536 \
  -b 1024 -ub 256 \
  -ctk q4_0 -ctv q4_0 \
  -fa on \
  --jinja \
  --samplers 'top_k;top_p;min_p;temperature;dry;typ_p;xtc' \
  --temp 0.7 \
  --dry-multiplier 0.8 --dry-base 1.75 --dry-allowed-length 1 \
  --host 0.0.0.0 --port 8090
```

Build: `ffee9f47e9cae25031a87fb0f6c7e0271c28c0d6` (branch `fleet-sync`,
"glm-dsa: load the NextN/MTP tensors instead of skipping them").
Working directory matters: the binary is invoked as `./build/bin/llama-server`
from `<BOX>/llama.cpp-fleet`.

## Measured behaviour

| Metric | Value |
|---|---|
| Slots | 4, each with the full 65,536 context |
| VRAM GPU0 (RTX PRO 6000 Workstation) | 81,576 MiB |
| VRAM GPU1 (RTX PRO 6000 Max-Q) | 91,522 MiB |
| VRAM total | 173,098 MiB |
| Prefill, 28,812-token prompt | 512.8 tok/s |
| Prefill at 20K+ depth | ~230 tok/s |
| Generate | 35.0 tok/s measured, 41-48 tok/s in interactive use |
| Largest prompt served clean | 45,798 tokens, `truncated = 0` |

## Why each flag

- **`-ub 256` is the flag that matters.** The compute buffer scales with
  *ubatch*, not with context length. At `-ub 512` the model OOM'd at 16K; at
  `-ub 256` it went 8K to 20K to 64K on the same cards. Anything that raises
  ubatch costs context.
- `--tensor-split 49,51` gives the Max-Q card slightly more, because GPU0 also
  hosts the `llama-imatrix` job (15,064 MiB).
- `-ctk q4_0 -ctv q4_0`: MLA compressed KV, 78 layers x 576 (kv_lora 512 +
  rope 64), roughly 25-47 MiB per 1K tokens.
- DRY sampler (`0.8 / 1.75 / allowed_length 1`) is the validated mitigation for
  the loop behaviour this bit-width shows: 1/8 to 0/8 at ~15K depth. It is
  **not** sufficient at extreme depth, see Known limits.
- `--jinja` is required for tool calling and the web UI.

## Untested headroom

4 slots x 65,536 means KV for **262,144 tokens is already allocated and fits**.
Repartitioning with `-np 1 -c 262144` should give a single 256K-context slot at
the same VRAM cost. Not yet verified with a real long generation.

## Gotchas

1. `--samplers` contains semicolons. Quote it or the shell eats the string.
2. **Do not point Claude Code's plumbing tiers at this model.** In Spock, keep
   `haiku` / `sonnet` / `opus` and `[advisor]` off-LAN. The Bash safety
   classifier in Auto mode calls the small/fast tier; a 1.90bpw model answering
   slowly under prefill contention surfaces as
   `claude-opus-5 is temporarily unavailable` and every Bash call is refused.
   Route only the tier under test (`fable`) at the LAN model.
3. On the box, `llama-imatrix` and the `geth` / `lighthouse` processes are
   unrelated long-running jobs. Never blanket-`pkill`.

## Known limits at this bit-width

Reasoning can run to the context wall without terminating. Observed in the
`GLM 5.2 - Claude_Code - Test v02` run (2026-07-21/22): a single response hit
`n_tokens = 65535, truncated = 1`, containing the same line repeated 726 times,
*after* the model had already completed and verified the task. Thinking to
visible output ran 416:1. DRY was active with full-context lookback throughout.
