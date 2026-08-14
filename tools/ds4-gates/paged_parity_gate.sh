#!/usr/bin/env bash
# PAGED PARITY GATE -- is Metal paging AS FAST OR FASTER than static? (the owner's bar, 2026-08-09)
#
# ⚠ WHY THIS FILE EXISTS AT ALL. The only parity numbers this programme has -- 225k on the 9B
# (wall 0.9831, decode 1.558x) and the 512k rung -- were produced by a SCRATCH SCRIPT in the job's tmp
# directory, hardcoded to one model, deleted when the job is deleted. **The harness behind the
# headline numbers was not in the repo and could not be re-run by anyone else.** Same class as every
# undocumented instrument in this lane.
#
# ⚠ THIS IS NOT `long_context_gate`. That one is NEEDLE-GATED CORRECTNESS -- found / not found, no
# timing. This one answers the SPEED question and is needle-gated only as a VALIDITY guard:
# **a speed number from a run that got the answer wrong is not a speed number.**
#
# ⚠⚠ ARM ORDER IS A CONFOUND, MEASURED ON THIS BOX. 29% positional drift once produced 1.41x in BOTH
# directions when the truth was 1.317x. A single static-then-paged pass is ONE ordering, and this gate
# says so rather than pretending otherwise -- run it twice with PP_ORDER=paged-first and compare.
# Agreement across both orders is the claim; a single order is a data point.
#
# ⚠ NO EXPLICIT -ngpub. The pool is AUTO-FIT, which is what LLAMA_PAGED_POOL_HEADROOM governs. An
# explicit -ngpub silently disables that governor -- on 2026-08-09 a hardcoded 4096 blocks capped
# long_context_gate at 131,072 tokens and made it print "PAGED LOSES CONTEXT -- this is ours" for a
# capacity failure. Headroom defaults to 1.05 here because the 512k rung proved 1.5 REFUSES to start.
#
# usage:  PP_MODEL=<gguf> PP_CTX=262144 PP_FILL=225000 [PP_ORDER=paged-first] paged_parity_gate.sh
set -uo pipefail
. "$(dirname "$0")/_no_abs_paths.sh" 2>/dev/null || true
. "$(dirname "$0")/_gate_common.sh"  2>/dev/null || true

WT=${WT:-$HOME/Documents/GitHub/llama.cpp-ds4ports}
SRV=${SRV:-$WT/build-metal/bin/llama-server}
M=${PP_MODEL:-$HOME/Documents/GitHub/ornith-models/Ornith-1.0-9B-1M-GGUF/ornith-1.0-9b-1M-IQ2_M.gguf}
CTX=${PP_CTX:-262144}
FILL=${PP_FILL:-225000}
BLK=${PP_BLOCK:-64}
ORDER=${PP_ORDER:-static-first}
# ⚠⚠ 48 MADE THE DECODE NUMBER UNREADABLE, AND DECODE IS THE THING PAGING CHANGES. At 512k the 35B
# decodes at ~9.3 tok/s, so 48 tokens is **5 seconds** of measurement against 3130 s of prefill.
# MEASURED 2026-08-09, same-arm tg across the four ABBA positions:
#     256k:  static 21.88 / 20.98  (4.1% apart)   paged 21.87 / 20.91  (4.4% apart)
#     512k:  static  9.33 /  9.09  (2.6% apart)   paged  9.28 /  9.31
# **tg was the noisiest column in the file** -- noisier than the wall it is a component of -- purely
# because it is averaged over a couple of seconds. Prefill is batch-dominated and the two arms differ
# least there; the paged kernel's per-step cache reads are a DECODE cost, so the one metric that could
# separate the arms was the one sampled worst.
# ⇒ 512 tokens is 55 s at 512k (+1.7% wall) and 24 s at 256k, an ~10x longer decode window for ~2% of
#   the run. Cheapest resolution available on this box, and it does not need a single extra arm.
NPRED=${PP_NPRED:-512}
# ⚠ THE ONE ASYMMETRY ABBA CANNOT BALANCE: the FIRST arm runs cold, and in S,P,P,S the first arm is
# ALWAYS static. Measured on the 512k rung -- pos1 static 124.4 tok/s, then 127.0 / 127.4 / 127.7 for
# everything after it. The cold arm is ~2.6% slow, the static mean is inflated by it, and EFFECT is
# therefore biased TOWARD paged. My 256k "tie" and my 512k "paged leads by 1.0%" both carried that
# bias in paged's favour, and dropping the cold arm flips the sign (static 3128.2 vs paged 3139.5).
# ⇒ Warm BOTH arms before measuring: model file into page cache, Metal pipelines compiled, allocator
#   first-touch paid. Two short server starts, symmetric, ~2 min against a 3.5 h run.
WARM=${PP_WARM:-1}
# ⚠⚠ DEFAULT THE CHAMPION **ON**, because leaving it off does not measure a slower paged path -- on
# some geometries it measures NO paged path at all. `paged_layer_supported` relaxes the scalar
# staged-tile bound only when the champion is active (llama-graph.cpp:4548), so at head_dim 256 the
# three states are:
#     champ=1, bs=64  -> champion serves it            (the configuration paging exists for)
#     champ=0, bs=32  -> scalar serves it, slowly
#     champ=0, bs=64  -> EVERY LAYER REFUSED, silently static  <- what this gate ran all day
# The third is the one that produced 4.5 h of static-vs-static "parity ties", and the code comment
# that predicted it is dated 2026-08-06: *"at bs=64/D=256 every layer refused and silently took the
# static path, making a paged run indistinguishable from static."* Three days later the gate did it.
export DS4P_METAL_CHAMP=${DS4P_METAL_CHAMP:-1}
export LLAMA_PAGED_POOL_HEADROOM=${LLAMA_PAGED_POOL_HEADROOM:-1.05}

[ -f "$M" ] || { echo "missing model: $M" >&2; exit 2; }
# ⚠⚠ $D WAS A FIXED PATH, SO EVERY INVOCATION DESTROYED THE PREVIOUS ONE'S EVIDENCE -- and the
# sweep that drives this gate invokes it FOUR TIMES IN A ROW. `decode_ctx_sweep.sh` runs rungs
# 8k/32k/64k/128k through here back to back; each rung overwrote the last rung's logs, and the 512k
# run overwrote all of them. The per-rung `.txt` verdicts survived because they carry the ctx in
# their filename; **the logs, which hold the only within-run data this lane has, did not.**
#
# Found on 2026-08-10 when the prefill-curve analysis wanted to test its plateau at a second rung
# and there was nothing left to test it against. Same class as pos4 overwriting pos1, one level up:
# **the artifacts that get a unique name survive and the ones that do not are silently destroyed.**
#
# ⇒ Per-run directory keyed by ctx and time, plus a `latest` symlink so anything reaching for the
#   old fixed path still finds the most recent run.
#
# ⚠⚠⚠ AND THIS EDIT NEARLY DISABLED THE ONE-SERVER-AT-A-TIME LOCK. The lock path is built from $D
# at :82, THIRTEEN LINES BELOW THIS ONE. Making $D unique per invocation would give two concurrent runs
# their OWN lock directory each, so both would acquire it and both would start a server on the same
# GPU -- mutual exclusion gone, silently, and the symptom would be two slow arms nobody could
# attribute. **Guard-for-A-disables-B: an edit for evidence preservation switching off contention
# control, in the same file, in the same change.** That is a ★★★ class in this lane with three
# prior instances, and it was caught by reading the neighbourhood of the anchor rather than the
# anchor. ⇒ DPARENT is the FIXED path and the lock stays on it (edit E); only the run dir moves.
DPARENT=${CLAUDE_JOB_DIR:-/tmp}/parity; mkdir -p "$DPARENT"
D=$DPARENT/c${CTX}-$(date +%Y%m%d-%H%M%S); mkdir -p "$D"
ln -sfn "$D" "$DPARENT/latest" 2>/dev/null
echo "  run dir: .../parity/$(basename "$D")   (previous runs are no longer overwritten)"
# ⚠ CLEAR THE PREVIOUS RUN'S RESULT FILES. A `.res` that survives into the next run is
# indistinguishable from one this run wrote -- same name, same schema, plausible numbers. The
# same-prompt_n assertion below is the backstop; this is the fix. Two guards because the failure is
# silent and the cost is a wrong verdict printed with confidence.
rm -f "$D"/*.res "$D"/*.sanity.txt 2>/dev/null
OUT=${OUT:-$HOME/Documents/GitHub/ornith-1m/tools/ds4-gates/results/parity-$(date +%Y%m%d-%H%M).txt}
mkdir -p "$(dirname "$OUT")"
NEEDLE="MAGENTA-7742"

# ⚠ ONE SERVER AT A TIME, and a lock rather than a race. Detached probes sharing a pattern-kill
# destroyed each other's servers on 2026-08-07; one survivor held a port for 16 minutes on 2026-08-09.
# ⚠⚠ THE LOCK MUST NOT LIVE UNDER THE PER-RUN $D. It is what makes concurrent invocations
# exclude each other, so it has to be at a path they SHARE. When $D became per-run (edit D) this
# line would have handed every invocation its own lock and let them all run at once.
LOCK=$DPARENT/gpu.lock
until mkdir "$LOCK" 2>/dev/null; do sleep 15; done
PID=""
trap 'rmdir "$LOCK" 2>/dev/null; [ -n "$PID" ] && kill $PID 2>/dev/null; scrub_abs_paths "${OUT:-}" 2>/dev/null' EXIT

pick_port() { local p; for p in $(seq 20100 20160); do
    lsof -nP -iTCP:"$p" -sTCP:LISTEN >/dev/null 2>&1 || { echo "$p"; return 0; }; done; return 1; }
PORT=$(pick_port) || { echo "no free port" >&2; exit 2; }

# ⚠ FULL PATH, NOT BASENAME. `ornith-1.0-35b-1M-Q4_K_M` vs `ornith-1.0-35b-1M-MTP-Q4_K_M` differ by
# four characters and are different models -- and on 2026-08-09 the model tree was reorganised
# mid-run, so successive runs read the same-named file off different storage. A header that records
# only a basename cannot tell you which of those a number came from.
echo "paged parity gate: $M" | tee "$OUT"
# ⚠ npred AND warm ARE STAMPED because they change what the numbers MEAN. Result files written before
# 2026-08-09 used npred=48 with no warm-up; without these two fields a 48-token cold run and a
# 512-token warm run are indistinguishable in results/ and would be compared as if they were the same
# measurement. A parameter that moves the answer belongs in the header, not only in the invocation.
# ⚠ `block_req`, NOT `block`. probe_geometry may clamp it, and a header naming a value the run did not
# use is the stale-header defect this directory has now recorded three times in one day.
# The value ACTUALLY used is printed by the geometry line below and is the one to quote.
# ⚠ `git rev-parse --short HEAD` prints a CLEAN sha even when the worktree that built the binary
# has uncommitted edits, so an artifact can name a commit the measured binary is not. LAW 7's
# gate_tip_stamp appends +dirty(N) -- the COUNT, because one edit and thirty are different situations.
echo "tip: $(gate_tip_stamp "$WT")  ctx=$CTX  fill~${FILL}tok  block_req=$BLK  order=$ORDER  headroom=$LLAMA_PAGED_POOL_HEADROOM  npred=$NPRED  warm=$WARM  champ=$DS4P_METAL_CHAMP" | tee -a "$OUT"

python3 - "$D" "$NEEDLE" "$FILL" "$NPRED" <<'PY' | tee -a "$OUT"
import json, sys
D, NEEDLE, FILL, NPRED = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
# ⚠ MEASURED, NOT GUESSED: 23.4 tokens per line for this filler and tokenizer. An earlier 12 made
# LC_FILL mean half of what it claimed and built prompts twice the requested size.
TOK_PER_LINE = 23.4
lines = int(FILL / TOK_PER_LINE); at = lines // 2
buf = [f"Note {i}: The secret passcode is {NEEDLE}. Remember it." if i == at
       else f"Note {i}: the quick brown fox jumps over the lazy dog near the old stone bridge."
       for i in range(lines)]
p = ("Below are many numbered notes. One of them contains a secret passcode.\n\n" + "\n".join(buf)
     + "\n\nQuestion: what is the secret passcode? Answer with the code only.\nAnswer:")
# ⚠⚠ `ignore_eos` IS WHAT MAKES n_predict A FLOOR. Without it n_predict is a CEILING, and this prompt
# is answered in FOURTEEN tokens: measured 2026-08-09, pred_n=14, pred_ms=670, stop_type=eos, against
# a requested 512. **Every decode number this lane produced was sampled over 0.67 seconds** -- SHORTER
# than the 48-token window the 512 was introduced to replace, and I spent a morning crediting the
# change with a signal it never touched. A parameter that silently does not take effect is worse than
# one that is absent, because it looks like the question was asked.
json.dump({"prompt": p, "n_predict": NPRED, "temperature": 0, "seed": 1, "cache_prompt": False,
           "ignore_eos": True},
          open(f"{D}/req.json", "w"))
print(f"  prompt: {lines} lines, needle at 50% depth, target ~{FILL} tok")
PY

arm() { # $1 = static|paged
    local flags=""; [ "$1" = paged ] && flags="--kv-paged --kv-block-size $BLK"
    : > "$D/$1.log"
    # ★ `-lv 4` IS A CHOICE, DECIDED 2026-08-11 -- do not "fix" it to 5. The trade, from the board:
    # a direct positive DS4P-CONSUME count needs -lv 5 (DEBUG), but DEBUG-volume logging lands
    # INSIDE the timed window -- the -lv 5 decider run emitted 640 banded CONSUME events per
    # request, and this gate's entire output is a speed claim, so that I/O is a confound on the
    # measurement itself. Consumption is instead proven by the engine's own no-consumer alarm
    # (WARN, visible at -lv 4), which was VALIDATED by direct control (alarm_control.sh: fires on
    # 20 refusals, silent on 240 banded consumes). A direct count belongs to the gates that do not
    # time anything -- arch_serve_gate runs -lv 5 for exactly that reason.
    # shellcheck disable=SC2086
    env DS4P_PAGED_HYBRID=1 DS4P_PAGED_SWA=1 DS4P_METAL_CHAMP="${DS4P_METAL_CHAMP:-0}" "$SRV" -m "$M" -ngl 99 -c "$CTX" \
        -np 1 -b 512 -ub 512 --port $PORT --no-warmup -lv 4 $flags > "$D/$1.log" 2>&1 &
    PID=$!
    local i
    for i in $(seq 1 900); do
        [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null)" = "200" ] && break
        kill -0 $PID 2>/dev/null || {
            command -v gate_cause_from_log >/dev/null 2>&1 && gate_cause_from_log "$D/$1.log" "$1 arm" | tee -a "$OUT"
            PID=""; return 1; }
        sleep 1
    done
    # ⚠⚠⚠ THE BINARY'S COMMIT, NOT THE SOURCE TREE'S -- AND THESE DISAGREED ON THE 512k RUN.
    #
    # The header stamps `git rev-parse HEAD`, which describes the SOURCE. The binary reports its own
    # build commit on its first log line. Measured on the 2026-08-10 512k ABBA, after the run:
    #
    #     header stamped   tip: 074672e33          <- what the SOURCE TREE said
    #     binary reported  build 10667 (35be827f3) <- what actually RAN, one commit behind
    #
    # The result was still valid (the missing commit only changes behaviour on models that CANNOT
    # page, and this one pages), **but nothing in the artifact could have told anyone that.** A
    # reader would have attributed 1.9662 to code the measurement never executed.
    #
    # ⚠ `gate_tip_stamp` DOES NOT CLOSE THIS. It adds +dirty(N) for uncommitted edits, which is a
    # different hole: it still reads the tree. **A stale BINARY built from a clean tree produces a
    # clean stamp and a wrong claim.** Provenance has to come from the artifact that did the work.
    #
    # ⇒ Fail FAST: this runs seconds after startup, so VOIDing here costs nothing, while discovering
    #   it after a 3.5-hour ABBA costs the ABBA. Override deliberately with PP_ALLOW_STALE_BIN=1.
    local binsha; binsha=$(grep -m1 -oE 'build [0-9]+ \([0-9a-f]+\)' "$D/$1.log" | grep -oE '\([0-9a-f]+\)' | tr -d '()')
    local srcsha; srcsha=$(cd "$WT" && git rev-parse --short HEAD 2>/dev/null)
    if [ -n "$binsha" ] && [ -n "$srcsha" ]; then
        # compare on the shorter of the two lengths -- the binary prints its own abbreviation
        local n=${#binsha}; [ ${#srcsha} -lt "$n" ] && n=${#srcsha}
        if [ "${binsha:0:$n}" != "${srcsha:0:$n}" ]; then
            echo "  ⚠⚠ BINARY/SOURCE MISMATCH: the server was built at $binsha, the tree is at $srcsha." | tee -a "$OUT"
            (cd "$WT" && git log --oneline "$binsha..$srcsha" 2>/dev/null | head -8 | sed 's/^/      missing from the binary: /') | tee -a "$OUT"
            if [ "${PP_ALLOW_STALE_BIN:-0}" != "1" ]; then
                echo "  VOID: refusing to attribute a measurement to source the binary does not contain." | tee -a "$OUT"
                echo "        Rebuild, or set PP_ALLOW_STALE_BIN=1 to measure the older binary ON PURPOSE." | tee -a "$OUT"
                kill $PID 2>/dev/null; wait $PID 2>/dev/null; PID=""; return 1
            fi
            echo "    PP_ALLOW_STALE_BIN=1 -- proceeding, and the mismatch is recorded above." | tee -a "$OUT"
        fi
    else
        echo "  ⚠ could not read the binary's build commit -- provenance is UNVERIFIED for this arm." | tee -a "$OUT"
    fi

    # ⚠ PRESENCE ASSERTED IN BOTH DIRECTIONS. static must show NO pool, paged must show one. Without
    # this, "static" can silently build a pool and the comparison is paged-vs-paged.
    local pool; pool=$(grep -ac 'n_gpu_blocks' "$D/$1.log")
    case "$1" in
      static) [ "$pool" = "0" ] || { echo "  VOID: static arm built a paged pool ($pool) -- not a static arm" | tee -a "$OUT"; kill $PID; PID=""; return 1; };;
      paged)  [ "$pool" -gt 0 ] || { echo "  VOID: paged arm shows NO pool -- not a paged arm" | tee -a "$OUT"; kill $PID; PID=""; return 1; };;
    esac
    local t0 t1
    t0=$(python3 -c 'import time;print(time.time())')
    curl -s --max-time 14400 -X POST "http://127.0.0.1:$PORT/completion" \
        -H 'Content-Type: application/json' --data-binary "@$D/req.json" > "$D/$1.json"
    t1=$(python3 -c 'import time;print(time.time())')

    # ⚠⚠⚠ THE CHECK ABOVE PROVES ALLOCATION AND NOTHING ELSE, AND THAT COST 4.5 HOURS ON 2026-08-09.
    #
    # `n_gpu_blocks > 0` says the POOL WAS BUILT. It cannot say a graph ever read it. On the 35B at
    # `--kv-block-size 64` the paged arm's own log read:
    #
    #     DS4P-CHECKOUT           1     pool allocated
    #     DS4P-SET              110     context attached to the graph
    #     DS4P-CONSUME            0     ** no graph ever read it **
    #     capability contract  3610     every attention layer REFUSED
    #
    #     "paged layer refused: layer 3: block_size x head_dim exceeds the staged-tile budget
    #      (need block_size*head_dim <= 8192)"      64 x 256 = 16384.
    #
    # Refused layers 3,7,11,...,39 -- every 4th of 40, exactly the full-attention set on this hybrid.
    # **100% of the attention layers fell back to static, and the gate reported a clean paged arm.**
    # The 256k and 512k "parity ties" were static vs static-with-an-idle-pool, which is why they came
    # out inside noise: they were the same code path.
    #
    # ⇒ Assert CONSUMPTION, positively, after the request -- the graph is built when the request runs,
    #   so this cannot be checked at startup. This project's own scars file states the distinction it
    #   was written from: CHECKOUT proves allocation, CONSUME proves consumption. The instrument was
    #   built on the wrong one.
    if [ "$1" = paged ]; then
        # ⚠⚠ AND THE FIRST VERSION OF THIS ASSERTION GREPPED `DS4P-CONSUME`, WHICH IS INVISIBLE HERE.
        # That marker is LLAMA_LOG_DEBUG, DEBUG needs verbosity >= 5 (common/log.h: LOG_LEVEL_DEBUG 5),
        # and this gate runs `-lv 4`. So the count is ALWAYS zero in these logs and the assertion
        # would have VOIDed every paged arm forever -- a false alarm welded into the instrument,
        # written in the same edit that fixed a false pass. It also nearly cost a correct retraction:
        # I read that zero as a measurement before checking whether the line could print at all.
        #
        # ⇒ Use the engine's OWN alarm, which is WARN and therefore visible. llama-context.cpp
        #   evaluates `ds4p_paged_consumer_count() == 0` after 8 decodes and warns once:
        #       "--kv-paged is ON and a paged pool was allocated, but after N decodes ZERO layers
        #        have consumed it -- every layer fell back to the static attention path"
        #   Its ABSENCE is meaningful precisely because the engine performs the positive test itself;
        #   this gate only has to guarantee the >= 8 decodes that arm the check, which NPRED does.
        local nocons; nocons=$(grep -ac 'ZERO layers have consumed' "$D/$1.log")
        local refused; refused=$(grep -ac 'fails the paged capability contract' "$D/$1.log")
        if [ "${nocons:-0}" -gt 0 ] || [ "$NPRED" -lt 8 ]; then
            echo "  VOID: paged arm allocated a pool that NO LAYER CONSUMED -- it is STATIC wearing a" | tee -a "$OUT"
            echo "        paged flag, and the wall/pp/tg from it are static numbers. Cause:" | tee -a "$OUT"
            grep -m1 -oE 'paged layer refused:.*' "$D/$1.log" | sed 's/^/          /' | tee -a "$OUT"
            grep -m1 -oE 'n_embd_head_v *= *[0-9]+' "$D/$1.log" | sed 's/^/          /' | tee -a "$OUT"
            echo "          block_size in use: $BLK   (contract: block_size * head_dim <= 8192)" | tee -a "$OUT"
            [ "$NPRED" -lt 8 ] && echo "          NPRED=$NPRED is under 8, so the engine's check never armed -- raise it." | tee -a "$OUT"
            kill $PID 2>/dev/null; wait $PID 2>/dev/null; PID=""; return 1
        fi
        # ⚠⚠ THERE ARE **TWO** FALLBACK BRANCHES AND THIS LINE COUNTED ONLY ONE. Found by Grok,
        # 2026-08-10, in a log this gate had already scored clean:
        #
        #     fails the paged capability contract   0      <- the only branch counted
        #     **took the STATIC path -- no paged context   110**  <- never counted, and firing
        #
        # on exactly the ten full-attention layers (3,7,...,39). **The record said "0 layer-refusal
        # warnings" for a run where every attention layer announced the static path.** The gate was
        # blind to the branch that was actually firing -- the 14-token header defect at gate level:
        # the field says the question was asked and passed; it was asked, it was not seen.
        #
        # ⚠ AND THE RESOLUTION IS *WHEN*, NOT *WHETHER*. A `-lv 5` decider split on the request
        # marker showed 210 static-path warnings, **all of them before the request**, and after it
        # **640 DS4P-CONSUME banded events across layers [3,7,...,39] with ZERO fallbacks**. So
        # reserve-time static-path warnings are NORMAL -- the paged context does not exist yet when
        # llama-server builds its ~21 reserve graphs. Counting them as failures would be as wrong as
        # ignoring them.
        # ⇒ Report both branches, and say plainly which one is expected. The engine's no-consumer
        #   alarm remains the pass/fail; these counts are for reading, not for gating.
        local staticpath; staticpath=$(grep -ac 'took the STATIC path' "$D/$1.log")
        printf '  paged arm consumed the pool (engine no-consumer alarm silent)\n' | tee -a "$OUT"
        printf '    capability-contract refusals: %s   static-path fallbacks: %s\n' \
               "$refused" "$staticpath" | tee -a "$OUT"
        printf '    ⚠ static-path fallbacks are EXPECTED at reserve time (the paged context is not set\n      yet during graph reserve). A -lv 5 run is what proves they stop once the request starts.\n' | tee -a "$OUT"
    fi

    # ★ SANITY SAMPLE (wired 2026-08-11, the board's "next up" item). A SECOND short request with
    # ignore_eos OFF, graded by output_sanity.py in the ABBA summariser. Why a separate request and
    # not the measured one's text -- the collision the board said to fix TOGETHER, not separately:
    # the measured request runs ignore_eos, and output_sanity's own header MEASURES that grading
    # free-running filler false-fails (a healthy arm's post-answer filler scored 97% repeated
    # 4-grams). Its stated resolution is scope: grade an ANSWER. This is that answer.
    # ⚠ And it is not merely a sample: a short request AFTER a completed long request on the same
    # slot is the WARM regime -- the only regime FINDING 1c ever reproduced in (7/12 warm, 0/24
    # cold). Every future parity run therefore doubles as a warm-regime corruption probe, at the
    # cost of ~160 tokens of decode after the timed request has already finished.
    curl -s --max-time 300 -X POST "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' \
        -d '{"prompt":"Describe in two or three plain sentences what a page table does in an operating system.","n_predict":160,"temperature":0,"seed":7,"cache_prompt":false}' \
        | python3 -c 'import json,sys
try: d=json.load(sys.stdin)
except Exception: raise SystemExit          # empty file = VOID downstream, stated not silent
if "error" not in d: sys.stdout.write(d.get("content",""))' > "$D/$1.sanity.txt"

    kill $PID 2>/dev/null; wait $PID 2>/dev/null; PID=""; sleep 3
    # ⚠⚠ OUT AND THE MODEL PATH GO IN AS ARGV, NOT AS ENV. The first version wrote
    # `os.environ.get("OUT","")` into the .res -- and the shell never exports OUT, so the field was
    # ALWAYS the empty string. **A field added specifically to make artifacts self-identifying,
    # which identified nothing**, written in the same edit that fixed `npred=512` recording a request
    # that never took effect. Same class, one line apart. Everything else in this heredoc is passed
    # as argv; the one value that was not is the one that silently died.
    python3 - "$D/$1.json" "$NEEDLE" "$1" "$t0" "$t1" "$D" "$OUT" "$M" "$BLK" "${DS4P_METAL_CHAMP:-0}" <<'PY' | tee -a "$OUT"
import json, sys, os
f, NEEDLE, lab, t0, t1, D = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), float(sys.argv[5]), sys.argv[6]
OUTP, MODELP, BLKV, CHAMPV = sys.argv[7], sys.argv[8], sys.argv[9], sys.argv[10]
try: d = json.load(open(f))
except Exception as e: print(f"  {lab}: UNPARSEABLE {e}"); raise SystemExit
if "error" in d: print(f"  {lab}: ERROR {str(d['error'].get('message'))[:70]}"); raise SystemExit
c = d.get("content",""); tm = d.get("timings") or {}
ok = NEEDLE in c
# ⚠ NEEDLE IS A VALIDITY GUARD, NOT THE RESULT. A speed number from a wrong answer is not a number.
# ⚠ PRINT THE ACHIEVED DECODE LENGTH, NOT THE REQUESTED ONE. The header says npred=512; the run may
# have produced 14. Reporting only the request is how a dead parameter survives a whole day of
# measurement -- and it is why an artifact-first reviewer COULD NOT catch it: the .res carried no
# predicted_n, so the load-bearing field was simply absent from the record.
print(f"  {lab:7s} needle={'PASS' if ok else 'FAIL'}  wall={t1-t0:8.1f}s  "
      f"prompt_n={tm.get('prompt_n')}  pp={tm.get('prompt_per_second',0):7.1f} tok/s  "
      f"tg={tm.get('predicted_per_second',0):6.2f} tok/s  "
      f"pred_n={tm.get('predicted_n')} ({tm.get('predicted_ms',0)/1000:.1f}s)")
# ⚠ AN ARTIFACT MUST CARRY EVERY FIELD ITS VERDICT DEPENDS ON, or it launders assumptions.
# ⚠ THE MODEL'S FULL PATH, NOT ITS BASENAME. `ornith-1.0-35b-1M-Q4_K_M` and
# `ornith-1.0-35b-1M-MTP-Q4_K_M` differ by FOUR CHARACTERS and are different models; the model tree
# was also reorganised mid-run on 2026-08-09, moving the weights to different storage. Two result
# files with identical headers could describe runs that shared neither the model nor the disk.
# Recording the directory too is free -- `/Volumes/...` carries no username, so the scrubber leaves
# it intact, while a home path is correctly rewritten to $HOME.
json.dump({"lab":lab,"ok":ok,"wall":t1-t0,"pp":tm.get('prompt_per_second',0),
           "tg":tm.get('predicted_per_second',0),"n":tm.get('prompt_n'),
           "pred_n":tm.get('predicted_n'),"pred_ms":tm.get('predicted_ms'),
           "out":OUTP,"model":MODELP,"block_size":BLKV,"champ":CHAMPV},
          open(f"{D}/{lab}.res","w"))
PY
}

# WARM-UP PRELUDE -- pays the one-time costs so that NO measured arm pays them.
#
# ⚠ WHAT IS ACTUALLY COLD, and why a short request is enough. Three costs land on the first arm and
# nowhere else: weights faulted in through the mmap, Metal compute pipelines compiled on the first
# graph build, and the allocator's first touch of the context. A 2k-token generation walks every
# layer and every kernel the big run will use, so it pays all three -- the SIZE of the prompt is not
# what makes those costs, the FIRST EXECUTION is.
#
# ⚠ BOTH ARMS, NOT ONE. Warming only static would leave the paged kernels cold for pos2 and move the
# asymmetry rather than remove it. Two starts, same order the measurement will use, discarded.
#
# ⚠ AND IT IS DISCARDED ON PURPOSE. Nothing here writes a .res file or touches $OUT beyond one line.
# A warm-up whose numbers can be mistaken for a measurement is worse than no warm-up.
# DERIVE THE BLOCK SIZE FROM THE MODEL, because a fixed default silently disables paging.
#
# ⚠⚠ THE DEFAULT WAS 64 AND IT REFUSED EVERY ATTENTION LAYER OF THE 35B. The kernel stages K and V
# tiles in threadgroup memory, so the contract is `2 * block_size * head_dim * sizeof(half) <= 32768`,
# i.e. **block_size * head_dim <= 8192**. head_dim 256 admits block_size 32; 64 is refused. The 9B
# harness that produced this lane's only real paged numbers ran block_size **16** -- inside the budget
# by accident of history. **The one parameter I never re-derived when I changed models is the one
# that broke**, and because a refused layer falls back to static the output stayed correct.
#
# ⇒ Ask the model, do not assume. One ~10 s server start at a tiny context, purely to read hparams.
# ⇒ Clamp DOWN only: a caller asking for 16 on a model that could take 64 gets 16. This function
#   removes a silent failure, it does not overrule an explicit choice upward.
probe_geometry() {
    local plog="$D/probe.log"
    : > "$plog"
    "$SRV" -m "$M" -ngl 99 -c 4096 -np 1 --port $PORT --no-warmup -lv 4 > "$plog" 2>&1 &
    PID=$!
    local i
    for i in $(seq 1 300); do
        [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null)" = "200" ] && break
        kill -0 $PID 2>/dev/null || break
        sleep 1
    done
    local hd; hd=$(grep -m1 -oE 'n_embd_head_v *= *[0-9]+' "$plog" | grep -oE '[0-9]+$')
    kill $PID 2>/dev/null; wait $PID 2>/dev/null; PID=""; sleep 2

    # ⚠ AN UNREADABLE PROBE MUST NOT SILENTLY LEAVE THE BAD DEFAULT IN PLACE. If hparams could not be
    # read, say so loudly rather than proceeding with a number nothing verified.
    if [ -z "$hd" ] || [ "$hd" -le 0 ] 2>/dev/null; then
        echo "  ⚠ geometry probe could not read n_embd_head_v -- keeping block=$BLK UNVERIFIED." | tee -a "$OUT"
        echo "    The paged arm's DS4P-CONSUME assertion is now the only thing standing between this" | tee -a "$OUT"
        echo "    run and another static-vs-static result. It will VOID rather than mislead." | tee -a "$OUT"
        return 0
    fi

    # ⚠⚠ THE 8192 BOUND IS THE *SCALAR* KERNEL'S, AND CLAMPING TO IT LOCKS OUT THE FAST PATH.
    # llama-graph.cpp:4548 --
    #     champ_geometry = champ_on && block_size == 64 &&
    #                      (head_dim == 64|96|128|192|256)
    #     if (!champ_geometry && block_size*head_dim > 8192) reject(...)
    # The CHAMPION does not stage K/V tiles in threadgroup memory, its footprint is flat in nsg, and
    # it **contractually requires block_size == 64**. So for head_dim 256 the two configurations are:
    #     champion  bs=64  -> served, fast path
    #     scalar    bs=32  -> served, SLOW path
    # and clamping 64 -> 32 silently chooses the slow one. My first version did exactly that and
    # measured the scalar kernel losing by 30%, which is a true number about the wrong kernel.
    # ⇒ If the champion is on and the geometry is one it implements, KEEP 64. Otherwise clamp.
    local maxbs=$(( 8192 / hd ))
    if [ "$maxbs" -lt 16 ]; then maxbs=16; fi          # kernel floor; below this the contract is a different one
    case "$hd" in
      64|96|128|192|256)
        if [ "${DS4P_METAL_CHAMP:-0}" != "0" ] && [ "$BLK" = 64 ]; then
            echo "  geometry: head_dim=$hd with the CHAMPION kernel -- block_size 64 is its contract," | tee -a "$OUT"
            echo "            not the scalar staged-tile bound. Keeping 64 (clamping to $maxbs would" | tee -a "$OUT"
            echo "            silently select the slow scalar path)." | tee -a "$OUT"
            return 0
        fi ;;
    esac
    if [ "$BLK" -gt "$maxbs" ]; then
        echo "  geometry: head_dim=$hd -> block_size must be <= $maxbs (block_size*head_dim <= 8192)." | tee -a "$OUT"
        echo "            requested $BLK would REFUSE every attention layer; using $maxbs." | tee -a "$OUT"
        BLK=$maxbs
    else
        echo "  geometry: head_dim=$hd, block_size $BLK is within the staged-tile budget (max $maxbs)." | tee -a "$OUT"
    fi
}

warmup() {
    [ "$WARM" = 1 ] || { echo "  warm-up: SKIPPED (PP_WARM=0) -- the first measured arm will run cold" | tee -a "$OUT"; return 0; }
    python3 - "$D" <<'PYW'
import json, sys
D = sys.argv[1]
p = "Warm-up. " + ("The quick brown fox jumps over the lazy dog near the old stone bridge. " * 220)
json.dump({"prompt": p, "n_predict": 16, "temperature": 0, "seed": 1, "cache_prompt": False},
          open(f"{D}/warm.json", "w"))
PYW
    local a t0 t1
    for a in static paged; do
        local flags=""; [ "$a" = paged ] && flags="--kv-paged --kv-block-size $BLK"
        t0=$(python3 -c 'import time;print(time.time())')
        # shellcheck disable=SC2086
        env DS4P_PAGED_HYBRID=1 DS4P_PAGED_SWA=1 DS4P_METAL_CHAMP="${DS4P_METAL_CHAMP:-0}" "$SRV" -m "$M" -ngl 99 -c "$CTX" \
            -np 1 -b 512 -ub 512 --port $PORT --no-warmup -lv 4 $flags > "$D/warm-$a.log" 2>&1 &
        PID=$!
        local i ok=0
        for i in $(seq 1 900); do
            [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null)" = "200" ] && { ok=1; break; }
            kill -0 $PID 2>/dev/null || break
            sleep 1
        done
        # ⚠ A FAILED WARM-UP IS NOT A FAILED GATE. It leaves the run exactly as cold as it was before
        # this function existed, which is the old behaviour, so it warns and continues rather than
        # aborting a 3-hour measurement over an optional prelude.
        if [ "$ok" != 1 ]; then
            echo "  ⚠ warm-up ($a) never came up -- continuing COLD, treat the first arm accordingly" | tee -a "$OUT"
            [ -n "$PID" ] && kill $PID 2>/dev/null; wait $PID 2>/dev/null; PID=""; continue
        fi
        local wresp="$D/warm-$a.resp.json" wcode
        wcode=$(curl -s --max-time 900 -o "$wresp" -w '%{http_code}' -X POST "http://127.0.0.1:$PORT/completion" \
            -H 'Content-Type: application/json' --data-binary "@$D/warm.json" 2>/dev/null)
        t1=$(python3 -c 'import time;print(time.time())')
        kill $PID 2>/dev/null; wait $PID 2>/dev/null; PID=""; sleep 3
        # ⚠⚠ A WARM-UP THAT FAILED STILL PRINTED A DURATION AND THE RUN CONTINUED. The only check
        # here was "did the server come up"; everything after that was timed and discarded without
        # ever being READ. So a prelude that loaded the model, refused every paged layer and
        # returned an error produced `warm-up paged discarded (6s)` -- indistinguishable from a
        # healthy one, on the arm whose whole purpose is to remove the cold-arm confound from the
        # NEXT measurement. Exit-0-did-nothing, in the prelude.
        # ⇒ Read the prelude's own log for the failure vocabulary the measured arms already VOID on.
        local wfail; wfail=$(grep -acE 'failed to load model|error loading model|fails the paged capability contract|ZERO layers have consumed|failed to init the paged scheduler' "$D/warm-$a.log")
        if [ "${wfail:-0}" -gt 0 ]; then
            echo "  ⚠⚠ VOID: warm-up ($a) logged $wfail failure line(s) -- the prelude did NOT warm anything." | tee -a "$OUT"
            grep -m2 -oE 'failed to load model.*|error loading model.*|fails the paged capability contract.*|ZERO layers have consumed.*|failed to init the paged scheduler.*' "$D/warm-$a.log" | sed 's/^/      /' | tee -a "$OUT"
            echo "     Refusing to continue: the measured arms would inherit a cold-arm confound this" | tee -a "$OUT"
            echo "     gate would then report as an EFFECT. Fix the prelude or run with PP_WARM=0 and" | tee -a "$OUT"
            echo "     accept the cold first arm EXPLICITLY." | tee -a "$OUT"
            return 1
        fi
        # ⚠ THE OTHER HALF (2026-08-13): the log can be clean while the COMPLETION itself returned a
        # non-200 or an empty body -- the server came up, the endpoint errored, and the decode that was
        # supposed to warm the pipelines never produced tokens. That is the SAME cold-arm confound the
        # log-grep above guards against, arriving through the response instead of the log. Read it.
        local wlen
        wlen=$(python3 -c "import json;print(len(json.load(open('$wresp')).get('content','')))" 2>/dev/null || echo 0)
        if [ "$wcode" != 200 ] || [ "${wlen:-0}" -lt 1 ]; then
            echo "  ⚠⚠ VOID: warm-up ($a) completion returned HTTP ${wcode:-none}, content len ${wlen:-0} -- the" | tee -a "$OUT"
            echo "     prelude decode produced nothing, so the NEXT measured arm inherits a cold-arm confound." | tee -a "$OUT"
            echo "     Fix the prelude or run with PP_WARM=0 and accept the cold first arm EXPLICITLY." | tee -a "$OUT"
            return 1
        fi
        printf '  warm-up %-6s discarded  (%.0fs, http %s, %s tok-chars)\n' "$a" "$(python3 -c "print($t1-$t0)")" "$wcode" "$wlen" | tee -a "$OUT"
    done
}

# ⚠⚠ ABBA IS THE ONLY DESIGN THAT ANSWERS THIS ON THIS BOX, AND TWO PASSES DEMONSTRABLY CANNOT.
# MEASURED 2026-08-09, 35B @ 256k, four walls sorted by POSITION rather than by arm:
#     ran FIRST :  static 815.9  ·  paged 788.9
#     ran SECOND:  paged  783.2  ·  static 781.9
# **The second arm wins in BOTH passes, whichever arm it is.** static-first said paged was 4.0% FASTER;
# paged-first said paged was 0.9% SLOWER. One-and-one -> the pair is VOID.
# ⚠ And averaging the two passes is NOT a valid correction: static's wall moved 4.2% between positions
# while paged's moved 0.7% -- the confound is ~6x larger on one arm, so the means are not comparable.
# ⇒ ABBA runs static,paged,paged,static in ONE invocation and compares POSITION-MATCHED pairs:
#   position 1 vs 4 (static twice) bounds the drift; 2 vs 3 (paged twice) bounds it again; and the
#   arm comparison is made WITHIN a position, not across one.
probe_geometry
# ⚠⚠ HONOR warmup's REFUSAL (2026-08-13). warmup() returns 1 ONLY on a VOID (a prelude that loaded
# but logged a load/paged failure, or a completion that returned non-200/empty) -- cases where the
# NEXT measured arm would inherit a cold-arm confound this gate would then report as an EFFECT. The
# "never came up" path deliberately `continue`s and returns 0 (optional prelude, warn-and-run-cold).
# Under `set -uo pipefail` (no -e) a bare `warmup` DISCARDS that return, so the refusal printed but
# the gate measured anyway -- correct-producer-no-consumer. Consume it here.
warmup || { echo "  ABORT: warm-up VOID (above) -- refusing to produce a bar number off a cold-arm confound." | tee -a "$OUT"; exit 1; }
case "$ORDER" in
  # ⚠⚠ THE .res FILES WERE PER-POSITION AND THE .log FILES WERE NOT, SO pos4 DESTROYED pos1's LOG.
  # Found on 2026-08-10 by trying to use them: `prefill_curve.py` differences the per-`-b` progress
  # lines into a within-run prefill rate curve, and the one comparison that separates a COLD-ARM
  # penalty from genuine DRIFT is pos1's curve against pos4's -- same arm type, different position.
  # That comparison was impossible, because `arm()` truncates "$D/$1.log" on entry and both static
  # arms write the same path. **Every within-arm diagnostic this lane has ever run therefore covered
  # only the LAST arm of each type, and nothing said so.**
  # ⇒ cp, not mv: "$D/static.log" keeps working for everything that already reads it.
  abba)        arm static; mv "$D/static.res" "$D/static1.res" 2>/dev/null
               cp "$D/static.log" "$D/static1.log" 2>/dev/null
               mv "$D/static.sanity.txt" "$D/static1.sanity.txt" 2>/dev/null
               arm paged;  mv "$D/paged.res"  "$D/paged1.res"  2>/dev/null
               cp "$D/paged.log"  "$D/paged1.log"  2>/dev/null
               mv "$D/paged.sanity.txt" "$D/paged1.sanity.txt" 2>/dev/null
               arm paged;  mv "$D/paged.res"  "$D/paged2.res"  2>/dev/null
               cp "$D/paged.log"  "$D/paged2.log"  2>/dev/null
               mv "$D/paged.sanity.txt" "$D/paged2.sanity.txt" 2>/dev/null
               arm static; mv "$D/static.res" "$D/static2.res" 2>/dev/null
               cp "$D/static.log" "$D/static2.log" 2>/dev/null
               mv "$D/static.sanity.txt" "$D/static2.sanity.txt" 2>/dev/null
               echo "  per-position logs kept: static1/paged1/paged2/static2.log" | tee -a "$OUT"
               echo "    ⇒ cold-vs-drift split:  prefill_curve.py \$D/static1.log \$D/static2.log" | tee -a "$OUT" ;;
  paged-first) arm paged; arm static ;;
  *)           arm static; arm paged ;;
esac

# ⚠ THE LEGACY TWO-ARM SUMMARISER MUST NOT RUN AFTER ABBA. On 2026-08-09 the ABBA summariser printed a
# correct verdict and then this block fired anyway, VOIDing on `static.res`/`paged.res` -- the files ABBA
# RENAMES AWAY to static1/paged1/paged2/static2. Output read:
#     ⇒ UNRESOLVED: ... indistinguishable at this sample size.      <- correct
#     VOID: one or both arms produced no result -- nothing to compare. <- wrong, and printed LAST
# Harmless to the numbers, and the LAST line is the one a reader quotes. It is the stale-.res hazard
# flagged before the run, arriving as a stale-FILENAME hazard instead.
if [ "$ORDER" = abba ]; then
python3 - "$D" <<'PY2' | tee -a "$OUT"
import json, sys
D = sys.argv[1]
try:
    s1=json.load(open(f"{D}/static1.res")); p1=json.load(open(f"{D}/paged1.res"))
    p2=json.load(open(f"{D}/paged2.res")); s2=json.load(open(f"{D}/static2.res"))
except Exception:
    print("  VOID: ABBA incomplete -- one or more arms produced no result."); raise SystemExit(2)
arms=[("static",s1),("paged",p1),("paged",p2),("static",s2)]
if not all(a[1]["ok"] for a in arms):
    print("  VOID: a needle failed. A speed number from a wrong answer is not a speed number.")
    raise SystemExit(2)

# ⚠⚠ THE FOUR ARMS MUST HAVE MEASURED THE SAME THING, AND NOTHING CHECKED THAT UNTIL 2026-08-09.
# If an arm dies, arm() returns early, `<lab>.res` is never written, the ABBA rename never happens --
# and the PREVIOUS run's file survives and is loaded as that arm. Found by Grok, verified on disk:
# `static2.res` held an 8k smoke fixture (n=3665, wall 6.5s, ok=True, needle PASS) for ninety minutes
# while a 256k run was in progress. It parses, it passes the needle check, and it would have produced
#     static mean 417.1s vs paged mean 848.9s  ->  "paged is 104% SLOWER"
# printed with full confidence from a 6.5-second fixture written by the same session.
# ⇒ prompt_n was already recorded in every .res and compared against NOTHING. One line closes it.
ns = {a[1].get("n") for a in arms}
if len(ns) != 1:
    print(f"  VOID: the four arms do not share prompt_n {sorted(x for x in ns if x is not None)}.")
    print("    At least one .res is STALE -- a dead arm leaves the previous run's file in place and")
    print("    the rename silently reuses it. These are not four arms of one measurement.")
    raise SystemExit(2)
# ⚠ And say what the decode window ACTUALLY was, since the tg column is only as good as its sample.
pn = [a[1].get("pred_n") for a in arms]
if any(x is None for x in pn):
    print("  ⚠ pred_n absent from at least one .res (written by an older gate) -- the decode window")
    print("    behind these tg numbers is UNKNOWN. Treat the decode verdict as unverified.")
else:
    print(f"  decode window actually generated: {pn} tokens per arm")
    if max(pn) < 64:
        print(f"    ⚠ under 64 tokens -- tg is averaged over well under a second. n_predict is a")
        print(f"      CEILING unless ignore_eos is set; check that it is.")
print()
print("  ABBA walls by position:")
for i,(lab,a) in enumerate(arms,1):
    print(f"    pos {i}  {lab:6s}  wall={a['wall']:8.1f}s  pp={a['pp']:7.1f}  tg={a['tg']:6.2f}")
# ⚠ DRIFT FIRST. If the two same-arm walls differ by more than the arm difference, the run cannot
# resolve the question and says so instead of reporting a ratio.
drift_s=abs(s1["wall"]-s2["wall"])/max(s1["wall"],s2["wall"])
drift_p=abs(p1["wall"]-p2["wall"])/max(p1["wall"],p2["wall"])
ms=(s1["wall"]+s2["wall"])/2; mp=(p1["wall"]+p2["wall"])/2
eff=abs(mp-ms)/ms
print()
print(f"  DRIFT  static {drift_s*100:.1f}%   paged {drift_p*100:.1f}%   (same arm, positions 1&4 / 2&3)")
print(f"  EFFECT paged/static on position-balanced means = {mp/ms:.4f}  ({eff*100:.1f}%)")
print()
if eff <= max(drift_s, drift_p):
    print(f"  ⇒ **UNRESOLVED**: the arm effect ({eff*100:.1f}%) is NOT larger than the drift")
    print(f"    ({max(drift_s,drift_p)*100:.1f}%). The arms are indistinguishable at this sample size.")
    print("    Report as a TIE with a bound, never as a winner. More repeats, not a bigger claim.")
else:
    who = "FASTER" if mp < ms else "SLOWER"
    print(f"  ⇒ paged is {who} by {eff*100:.1f}%, and that exceeds the measured drift")
    print(f"    ({max(drift_s,drift_p)*100:.1f}%). ABBA cancels first-order position effects.")

# ⚠⚠ THE WALL IS ~99% PREFILL, SO A WALL VERDICT IS A PREFILL VERDICT WEARING A HAT. At 512k the
# prefill is 3130 s and the decode is seconds; any paged/static difference in the per-step cache reads
# is diluted ~600:1 before it reaches the ratio above. Reporting only the wall answers a question
# nobody asked -- "is the paged PREFILL as fast" -- and calls it parity.
# ⇒ Grade prefill and decode SEPARATELY, each against its OWN same-arm drift. A metric is readable
#   only when its arm effect exceeds its own drift, and on 2026-08-09 tg failed that test at both
#   rungs while the wall passed it at neither.
def grade(name, s1v, s2v, p1v, p2v, higher_is_better):
    ds = abs(s1v-s2v)/max(s1v, s2v, 1e-9); dp = abs(p1v-p2v)/max(p1v, p2v, 1e-9)
    ms = (s1v+s2v)/2; mp = (p1v+p2v)/2
    e = abs(mp-ms)/max(ms, 1e-9)
    print(f"  {name:8s} static {ms:8.2f}  paged {mp:8.2f}   ratio {mp/max(ms,1e-9):.4f}"
          f"   drift s={ds*100:.1f}% p={dp*100:.1f}%  effect={e*100:.1f}%")
    if e <= max(ds, dp):
        print(f"    -> UNREADABLE: effect ({e*100:.1f}%) does not clear its own drift ({max(ds,dp)*100:.1f}%).")
        return None
    faster = (mp > ms) if higher_is_better else (mp < ms)
    print(f"    -> paged is {'FASTER' if faster else 'SLOWER'} by {e*100:.1f}%, clearing drift {max(ds,dp)*100:.1f}%.")
    return faster
print()
print("  PER-METRIC, each against its OWN drift:")
grade("PREFILL", s1["pp"], s2["pp"], p1["pp"], p2["pp"], True)
grade("DECODE",  s1["tg"], s2["tg"], p1["tg"], p2["tg"], True)

# ⚠ THE COLD-ARM CHECK. ABBA balances POSITION, not TEMPERATURE: the first arm pays the mmap faults
# and the Metal pipeline compiles, and in S,P,P,S that arm is always static, so the bias always runs
# one way -- toward paged. The warm-up prelude is supposed to remove this; this line is how you find
# out whether it did, instead of assuming it.
#
# ⚠⚠ THE FIRST VERSION OF THIS CHECK LIED, AND ITS OWN SMOKE TEST CAUGHT IT WITHIN THE HOUR.
# 9B, ctx 8k, warm-up on, it printed:
#     COLD-ARM CHECK  pos1 static pp=711.5 vs pos4 static pp=689.0  (+3.3%)
#       OK: no cold-first-arm signature -- the two static positions agree within 1%.
# **They are 3.3% apart.** The guard was `if cold < -0.01`, which only catches pos1 SLOWER; every
# other outcome fell into an else that asserted agreement it had not tested. Prose contradicting the
# figure printed beside it is the exact failure this whole directory exists to prevent, and a
# reassuring else-branch is how it gets in.
# ⇒ THREE outcomes, because the drift has a SIGN and the sign decides which arm it favours:
#     pos1 slower -> first arm cold      -> bias toward PAGED  (static mean inflated)
#     pos1 faster -> box degraded/loaded  -> bias toward STATIC (static mean deflated)
#     within 1%   -> no positional signature, and only then may the word "agree" be printed.
cold = (s1["pp"] - s2["pp"]) / max(s2["pp"], 1e-9)
wp = (p1["wall"]+p2["wall"])/2
print()
print(f"  COLD-ARM CHECK  pos1 static pp={s1['pp']:.1f} vs pos4 static pp={s2['pp']:.1f}  ({cold*100:+.1f}%)")
if cold < -0.01:
    print(f"    ⚠ pos1 ran SLOWER than pos4 by {abs(cold)*100:.1f}%: the first arm was still cold, and because")
    print("      it is always static, EFFECT above is biased TOWARD PAGED. Warm-vs-warm is the honest read:")
    print(f"      pos4 static wall {s2['wall']:.1f}s vs paged mean {wp:.1f}s = {wp/max(s2['wall'],1e-9):.4f}")
    # ⚠ THIS LINE USED TO SAY "re-run with the warm-up prelude enabled (PP_WARM=1)" -- and on
    # 2026-08-10 it printed that while PP_WARM=1 WAS ON, with both warm-up arms in the log above it.
    # A stale template line telling the reader to do the thing already done. **The prelude kills the
    # cold-arm effect at 8k, where it was validated, and does NOT fully kill it at 256k** -- a fix
    # validated in the regime where it works and shipped for the regime where it does not.
    print("      ⚠ The warm-up prelude does NOT remove this at long context: it was validated at 8k")
    print("        and still leaves a 4-6% first-arm penalty at 256k. Quote the warm-vs-warm figure")
    print("        above, or add repeats -- re-running with the same prelude will not fix it.")
elif cold > 0.01:
    print(f"    ⚠ pos1 ran FASTER than pos4 by {cold*100:.1f}%: NOT a cold first arm -- the box got slower")
    print("      across the run (load, thermals, or another job). This biases EFFECT toward STATIC, the")
    print("      opposite direction, and a paged 'win' under this signature is the confound, not a result.")
    print(f"      Warm-vs-warm using the SLOWER static position: {wp:.1f}s vs {s1['wall']:.1f}s = {wp/max(s1['wall'],1e-9):.4f}")
    print("      A positional signature this size means the run is not quotable either way. Re-run on a quiet box.")
else:
    print(f"    OK: the two static positions agree to {abs(cold)*100:.1f}% -- no positional signature in either direction.")
print()
print("  ⚠ n=2 per arm. This bounds the drift; it does not estimate variance. A tie here means")
print("  'not distinguishable at n=2', not 'proven equal'.")
PY2

# ★ OUTPUT SANITY -- grade each paged position's short-answer sample against its position-matched
# static reference. This is the corruption sampler the parity gate never had: needle=PASS validates
# the SPEED number, it was never a corruption detector (a fluent-but-wrong answer still contains the
# passcode). Exit meanings from output_sanity.py: 0 OK, 1 DEGENERATE (real defect), 3 DIVERGENT
# (reported, not failed), 2 VOID.
# ⚠ The exit status is read via PIPESTATUS[0], not $? -- piping through tee replaces the exit
# status, and reading tee's 0 as the grader's verdict is the exact shape that let a REFUSED privacy
# guard approve a commit on 2026-08-10.
for _sp in 1 2; do
    if [ -s "$D/paged$_sp.sanity.txt" ] || [ -s "$D/static$_sp.sanity.txt" ]; then
        echo "  output-sanity pos-matched (paged$_sp vs static$_sp):" | tee -a "$OUT"
        python3 "$(dirname "$0")/output_sanity.py" --text "$D/static$_sp.sanity.txt" "$D/paged$_sp.sanity.txt" 2>&1 | tee -a "$OUT"
        _src=${PIPESTATUS[0]}
        if [ "$_src" = "1" ]; then
            echo "  ⚠⚠ DEGENERATE PAGED OUTPUT at position $_sp. The ratios above are speed numbers from a" | tee -a "$OUT"
            echo "     run whose paged text failed sanity -- quote them ONLY with this caveat attached." | tee -a "$OUT"
        fi
    else
        echo "  output-sanity: no sample for position $_sp (arm died before the sanity request, or an older gate)." | tee -a "$OUT"
    fi
done
fi

if [ "$ORDER" != abba ]; then
python3 - "$D" <<'PY' | tee -a "$OUT"
import json, os, sys
D = sys.argv[1]
try:
    s = json.load(open(f"{D}/static.res")); p = json.load(open(f"{D}/paged.res"))
except Exception:
    print("  VOID: one or both arms produced no result -- nothing to compare."); raise SystemExit(2)
if not (s["ok"] and p["ok"]):
    print(f"  VOID: needle failed (static={s['ok']} paged={p['ok']}). A speed number from a wrong")
    print("  answer is not a speed number, so this pair is not readable as parity."); raise SystemExit(2)
r = p["wall"] / s["wall"]
print()
print(f"  WALL RATIO paged/static = {r:.4f}   ->  paged is {'FASTER' if r < 1 else 'SLOWER'} by {abs(1-r)*100:.1f}%")
print(f"  PREFILL  static {s['pp']:.1f} vs paged {p['pp']:.1f} tok/s   ({p['pp']/max(s['pp'],1e-9):.3f}x)")
print(f"  DECODE   static {s['tg']:.2f} vs paged {p['tg']:.2f} tok/s   ({p['tg']/max(s['tg'],1e-9):.3f}x)")
print()
print("  ⚠ ONE ORDERING. Arm order is a measured confound on this box (29% positional drift once gave")
print("  1.41x in both directions when the truth was 1.317x). Re-run with PP_ORDER=paged-first;")
print("  AGREEMENT ACROSS BOTH ORDERS is the claim. A single order is a data point.")
PY
fi
echo "log: $OUT" | tee -a "$OUT"
