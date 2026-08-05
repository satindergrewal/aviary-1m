# Draft patches — UNAPPLIED, for review

Nothing here has been applied. The fork stayed at `53ed79e2`, dirty 0, for the entire session that
produced them. `git apply` if accepted, delete if not.

## `DRAFT-defect2-raii-leak.patch`

Fixes FINDINGS defect **#2** — `llama_kv_cache_paged::init` allocates eight handles and drops all of
them: two pushed to `gpu_ctxs`/`gpu_bufs` as raw pointers on a class with **no destructor**, plus six
locals (`ctx_cpu`/`buf_cpu` pool A, `ctx_gpu`/`buf_gpu`, `ctx_cpu`/`buf_cpu` pool B) never stored at
all.

20 additions, 4 deletions, across two files.

**Verified:**
- `git apply --check` passes.
- reverse-apply correctly fails (i.e. it is genuinely not yet applied).
- the include path is **proven, not assumed**: `src/llama-adapter.h:5` and `src/llama-context.h:11`
  already `#include "ggml-cpp.h"`, and `llama-adapter.h` declares
  `std::vector<ggml_context_ptr> ctxs;` / `std::vector<ggml_backend_buffer_ptr> bufs;` — the
  identical pattern this patch introduces.

**NOT verified — say so out loud:**
- **not compile-tested.** No build was run against it.
- **not run against `rset_leak_probe.sh`.** The probe is the acceptance test: arm 3 (paged, no
  request) must stop reporting `rset_assert=1` while arms 1–2 (static) stay at 0.

**Ownership order, checked rather than assumed.** `kv_gpu_layers`/`kv_cpu_layers` are raw
`ggml_tensor*` views into these contexts. Members destruct in reverse declaration order, so the
owning vectors — declared after them — are released first. That is safe here only because those
vectors hold plain pointers with trivial destruction and nothing dereferences them during teardown.
If either ever gains a non-trivial destructor, this ordering becomes load-bearing.

**Acceptance:** apply → build → `./rset_leak_probe.sh` → arm 3 `rset_assert=0`, arms 1–2 unchanged.

## `DRAFT-defect3-slotsave-refuse.patch`

Fixes FINDINGS defect **#3** — `/slots/N?action=save` answers **200** with `n_saved=27,
n_written=716` for a save that saved nothing (expected ~864 KB).

18 additions, 0 deletions, one file.

**Why a refusal and not a repair.** The paged path runs `finish() -> free_blocks()` and returns a
sequence's blocks to the pool the moment it completes — *"holding a finished sequence's blocks would
fight the entire point of paging"* (`server-context.cpp:3229`). This endpoint runs **after** the
completion returns, so there is no KV left to serialise. The caller is wired and `state_write` is
correct; the endpoint is the thing making a false claim.

Gated on `paged_sched`, which is non-null exactly when the server owns a paged scheduler — a
**structural** test, not a heuristic on the byte count. This lane has a scar from a guard that rested
on a heuristic and failed where it was tested.

**Verified:** `git apply --check` passes; anchor asserted unique before editing.
**NOT verified:** not compile-tested; not run against `state_serdes_gate.sh`.

**Acceptance:** apply → build → `./state_serdes_gate.sh` → arm A must report a **refusal**, not
`n_written=716`. ⚠ The gate currently expects a *successful* save, so it will need its arm-A bar
inverted to "refuses cleanly" — the same correction already applied to the eviction gate's
single-sequence arm when the design intent was established.
