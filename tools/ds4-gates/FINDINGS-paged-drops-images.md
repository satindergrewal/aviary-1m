# The paged request path silently DROPS image content from multimodal prompts

**Status: CONFIRMED IN SOURCE, 2026-08-07. Found by `arch_serve_gate.sh` in image mode on
`hunyuan_vl`. Same structural class as `FINDINGS-paged-no-speculation.md`: the paged scheduler owns
the request path and only understands plain token arrays.**

## What was measured

HunyuanOCR Q8_0 + its mmproj, shown a locally generated 320×96 PNG containing the word **PARIS**,
asked *"What word is written in this image? Answer with the word only."*, greedy:

| arm | answer |
|---|---|
| static | **`PARIS`** — correct |
| paged | **`BUTTERFISH`** |

`DS4P-CONSUME = 288`, zero fallbacks, no warning, no error. The wrong answer is a plausible English
word delivered with the same confidence as the right one — not gibberish, which is what makes it
dangerous. In a document-processing workload nobody would notice.

## Root cause

`tools/server/server-context.cpp:2018`, the paged request path:

```cpp
if (paged_sched) {
    const llama_tokens toks = slot.task->tokens.get_text_tokens();
    ...
    llama_paged_scheduler_add_request(paged_sched, toks.data(), (int32_t) toks.size(), slot.id, n_warm)
```

and `tools/server/server-common.cpp:411`:

```cpp
llama_tokens server_tokens::get_text_tokens() const {
    llama_tokens res;
    for (llama_token t : tokens) {
        if (t != LLAMA_TOKEN_NULL) {       // <-- media placeholders are DROPPED
            res.push_back(t);
        }
    }
    return res;
}
```

`LLAMA_TOKEN_NULL` is precisely the placeholder that marks media-chunk positions
(`server-common.cpp:370` emplaces it for each media token). So for a multimodal prompt the paged path:

1. flattens the prompt to text only, silently removing every image position,
2. hands the scheduler a **shorter** token array with no image in it,
3. and the model answers from the surrounding text alone.

`BUTTERFISH` is exactly what a confident OCR model produces when asked to read a picture it was never
given.

There is exactly **one** `has_media()` check in `server-context.cpp` (line 2293,
`check_slot_no_media`), and it guards slot save/restore/erase — **not** the paged request path.

## Why it was invisible

`hunyuan_vl` was recorded UNANSWERED earlier the same day for a *correct* reason: HunyuanOCR is an
image model, text-only prompts return `''` or `' $ $ $ $'`, and the gate VOIDed on a degenerate
reference. That was the right call — but *"the arbiter is broken"* is not *"the arch is fine"*, and
the row stayed open rather than clean.

Giving the model a question it can actually answer turned an unanswered row into a defect in one run.

## The arbiter, and why it is generated rather than downloaded

`AG_MMPROJ` + `AG_IMAGE` on `arch_serve_gate.sh`, with the degenerate-reference guard unchanged.

The test image is **generated locally** — 320×96 PNG, the word PARIS drawn from a 5×7 bitmap font in
about a dozen lines of Python, 268 bytes, byte-identical on every run. A fixture pulled off the
internet can change, disappear, or carry text nobody read. The arbiter has to be the most trusted
object in the room.

Verified before wiring it in: the model answers `PARIS` to that image. Text-mode regression after
wiring: `ernie4_5` still passes, so the existing path is untouched.

## What would close it

The paged request path needs to carry media chunks rather than flatten them away — either by
admitting `server_tokens` (which knows about `map_idx_to_media`) instead of a raw `llama_tokens`, or
by refusing multimodal prompts under `--kv-paged` with a designed refusal until it does.

⚠ **A refusal would be strictly better than today's behaviour**, and is far cheaper: right now the
answer is silently wrong. `check_slot_no_media` already exists and shows the pattern.

Not estimated. An estimate with no measurement behind it becomes a schedule.
