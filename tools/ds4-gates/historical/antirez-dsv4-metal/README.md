# Rescued: two DSV4 Metal commits that existed on one disk and nowhere else

**2026-08-09.** These two patches were the entire unique content of
`~/Documents/GitHub/llama-dsv4-antirez`, a clone of
`github.com/antirez/llama.cpp-deepseek-v4-flash`. That clone has been deleted; this directory is
what survives it.

| | |
|---|---|
| author | the repo owner |
| written | 2026-07-10 |
| on top of | `antirez/main` @ `2f2d44052` (2026-04-26, "Speed up DeepSeek V4 prompt replay") |
| pushed anywhere | **no** — `git branch -r --contains` returned empty for both |

`origin` was antirez's repository, which we cannot push to, so the commits were never mirrored.
`rm -rf` on that clone would have destroyed them with no recovery.

## ⚠⚠ THE PATCHES DO NOT APPLY TO OUR FORK, AND THAT IS NOT A FILE-DRIFT PROBLEM

`ds4-ports` is based on **mainline**, which later shipped its **own, independent** DeepSeek V4
implementation. antirez's `deepseek4.cpp` and mainline's `deepseek4.cpp` are two different programs
for the same model, not two revisions of one file:

| | antirez's tree (these patches) | ours (mainline) |
|---|---|---|
| top-k call | `ggml_argsort_top_k(...)` → patched to `ggml_top_k` | **already `ggml_top_k`**, `deepseek4.cpp:699` |
| mask helper | `dsv4_build_compressed_mask_from_topk()` | **does not exist** (grep count 0) |
| graph budget | `max(524288u, n_tokens*192 + 64u*n_tensors())` → `*768` | `max(n_tokens*40, 32u*n_tensors())` — **different shape, no floor** |

⇒ **A cherry-pick has nothing to land on.** Kept as history and as evidence, never as something to
apply. The value is in the findings below.

## Findings triage — what is dead, what is banked, what is still live

**1. `ggml_argsort_top_k` → `ggml_top_k`. DEAD (already true upstream).**
Mainline reached the same conclusion independently. Nothing to do.

**2. Graph node budget past `-ub 512`. ⚠ STILL LIVE, AND UNMEASURED ON OUR CODE.**
The commit message records the failure precisely:

> 192 objs/token undercounts the replay compressor's per-token cluster: the 524288 floor happens to
> cover it up to n_ubatch 512, then ggml.c `GGML_ASSERT(obj_new)` fires during context reserve at
> 1024.

Our formula is `max(n_tokens*40, 32u*model.n_tensors())` — a different shape **with no floor at
all** — and `LLM_ARCH_DEEPSEEK4` is in the arch list that uses it. So the same class of failure is
**not excluded, merely untested** on our fork.

⚠ This lane already carries a note that `n_tokens*40` is a graph **node** budget rather than a
compute-buffer size. **That note and this commit message are two halves of one diagnosis, and
neither states it alone.** If DSV4 aborts during reserve at `-ub 1024`, read both.

**3. "Gather top-k keys at decode; fuse indexer scorer tail". UNCHECKED.**
Metal optimisations written against antirez's graph structure. Whether mainline's implementation
does the equivalent has not been looked at. If DSV4 decode is slow on Metal, `0002` is the prior art
for what to try — as a description of an approach, not as a patch.

## ⚠ How the export itself nearly went wrong

The first attempt ran `git format-patch -2 b4f1d8856^`, which means *"two patches ending one commit
BEFORE b4f1d8856"* — the wrong direction. It produced a single **111 MB, 1.67-million-line** patch of
antirez's own commit, and it would have been committed as "the rescue" by anyone who did not look at
the output. **The post-condition is the byte count and the hunk list, never the exit code.**
Corrected to `git format-patch origin/main..main`; the real content is 3.5 KB and 6.5 KB.
