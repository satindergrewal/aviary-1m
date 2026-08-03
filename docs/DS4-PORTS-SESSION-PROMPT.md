# DS4-PORTS DEDICATED SESSION — HANDOFF PROMPT (paste this into the new session)

> You are taking over the ds4-ports lane for Satinder's llama.cpp fork. Everything you need is
> written down; read before acting. This lane's mission (his words): bring llama.cpp's serving
> layer to neck-to-neck parity with vLLM/SGLang — one shared KV pool, no static lanes, dynamic
> admission, queue-instead-of-reject, prefix sharing, preemption — then go PAST parity with
> ds4's extras (session forking, disk KV banks, yield-quench speculation governor). "I want to
> shatter that wall."

## 1. READ FIRST (in this order)
1. `ornith-1m/docs/PLAN-DS4-PORTS.md` — THE plan: ranked port list P0–P2 with effort/risk/
   kill-gates, deferred items with revisit triggers, kernels section, sequencing phases 0–5.
2. `ornith-1m/docs/INKLING-1M-WALLS.md` — the wall map: item 3b (kv-paged hybrid archs) and the
   B4 compute-buffer formula (55.3·ub·ctx_M/10³ GiB, measured) are in your lane too.
3. Memory: `task-board.md` (the authoritative board), `kv-paged-facts.md`,
   `fa-on-disables-the-guard.md`, `fork-branch-policy.md`, `active-poll-not-monitors.md`.
4. ds4 source for reference: `/tmp/ds4` on the Mac @ 82d2a6f (read-only; NEVER clone research
   repos to the box).

## 2. WORK SURFACE + HARD RULES
- Branch `ds4-ports` off `dsv4-kvpaged` @ `3376ebbc8`, worktree `/mnt/nvme0/wt-ds4-ports` (box).
  It exists and is untouched — you start from clean.
- NOTHING touches `fleet` without Satinder's explicit per-push approval. Every feature behind an
  env/flag kill switch, default-off. Every claim behind a measured gate with an exactness counter
  (byte-identical logits, not log lines — see the announce-vs-verify scar in memory).
- Kill by exact PID only. His crypto stack (geth/lighthouse/bitcoind/Fulcrum) is untouchable.
  `/mnt/data/397b-*` dirs are untouchable. No public posting of anything.
- GPU windows: coordinate in tri-chat (you'll be a new seat; read
  `~/Documents/GitHub/tri-chat/RESUME-HERE.md` first and arm your doorbell).

## 3. HOW HE WANTS IT RUN (his instructions for this session)
"Plan anything we want to do with this work, do thought experiments, and then implement it and
then build/make it work." — So: PHASE A is thought experiments, not code. Take each plan item
(P0 tier first: methodology, prefix re-validation, dead-client abort, yield-quench; plus 3b
kv-paged hybrid) and pressure-test the design on paper against the actual llama.cpp code it will
touch (server-context, llama-memory, speculative.cpp). Surface failure modes BEFORE writing
code: what breaks at 1M context? at 16 concurrent agents? with q8 KV? with FA on/off? Converge
the design in writing, post open questions to him in the room, and only then implement.
He expects the thought-experiment phase to take a while to converge — depth over speed.

## 4. FIRST CONCRETE DELIVERABLES (in order)
1. FORK-DELTA.md (plan item 3c, hours): inventory of everything our fork carries vs upstream.
2. Thought-experiment doc for the P0 tier + 3b (design review, in docs/, his read before code).
3. Then implementation in phase order with gates green before moving on.

## 5. CONTEXT YOU'LL NEED THAT ISN'T IN THE DOCS
- The box: ssh satinder@192.168.0.101; 2× RTX PRO 6000 Blackwell (95.3 GiB usable each);
  nvcc at /usr/local/cuda-12.9/bin (system 12.0 FAILS compute_120a); fleet tree at
  /mnt/nvme0/llama.cpp-fleet; builds go in worktrees, never in the fleet checkout.
- V4-Flash UD-Q4_K_XL at /mnt/nvme0/bigmodels/dsv4-flash/; Inkling-Small MXFP4 at
  /mnt/nvme0/bigmodels/inkling-small-gguf/. Both serve as your test models.
- kv-paged measured facts (V4 curve, inkling's 2.46× fitter overshoot) are in the plan + walls
  docs with exact numbers — don't re-measure, build on them.
