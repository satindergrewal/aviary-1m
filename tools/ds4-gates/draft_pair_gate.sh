#!/usr/bin/env bash
# DRAFT-PAIR GATE -- does a speculative-decode DRAFT head run correctly over a PAGED KV cache?
#
# WHY A SECOND GATE. arch_serve_gate.sh compares two arms of ONE model. Three of the owner's nineteen
# are not models at all, they are draft heads, and they refuse to load standalone:
#
#   llama-context.cpp:182   Gemma4Assistant   "requires ctx_other to be set"
#   llama-context.cpp:191   EAGLE3, DFLASH    same, when the GGUF has no tok_embd/output
#
# One harness unblocks three rows: gemma4-assistant, eagle3, dflash.
#
# ★ THREE ARMS, AND THE MIDDLE ONE IS THE ARBITER CHECK:
#
#   a  target alone, static        the reference
#   b  target + draft, static      MUST equal (a). Greedy speculative decoding is LOSSLESS -- the
#                                  draft only proposes, the target verifies. If (b) != (a) then the
#                                  spec path or this harness is broken and PAGING IS NOT YET IN
#                                  QUESTION. Stop there. Same discipline as refusing to score against
#                                  a degenerate reference.
#   c  target + draft, PAGED       must equal (a)
#
# Without arm (b), a failure in (c) has two possible causes and no way to separate them.
#
# ⚠⚠ THE VACUITY MODE THIS GATE EXISTS TO AVOID: SILENTLY-INERT DRAFTING. If -md is ignored, or the
# draft proposes nothing, then (b) == (a) TRIVIALLY and (c) tests the target's paging while proving
# nothing whatsoever about the draft. Text agreement cannot detect it -- lossless spec-decode means
# agreement is the expected result either way. So draft ACTIVITY is required from the response JSON
# (timings.draft_n > 0), and for read-only architectures the DS4P-READONLY marker is required too:
# without it a PASS proves the TARGET paged, which is not the row under test.
#
# usage: draft_pair_gate.sh <draft-arch> <target.gguf> <draft.gguf>
set -uo pipefail
. "$(dirname "$0")/_no_abs_paths.sh" 2>/dev/null || true

ARCH="${1:-}"; TARGET="${2:-}"; DRAFT="${3:-}"
if [ -z "$ARCH" ] || [ -z "$TARGET" ] || [ -z "$DRAFT" ]; then
    echo "usage: $0 <draft-arch> <target.gguf> <draft.gguf>" >&2
    echo "  refusing without an EXPECTED draft arch -- see arch_serve_gate.sh for why a gate that" >&2
    echo "  accepts whatever it finds cannot catch a wrong-vehicle download." >&2
    exit 2
fi
[ -f "$TARGET" ] || { echo "missing target: $(basename "$TARGET")" >&2; exit 2; }
[ -f "$DRAFT"  ] || { echo "missing draft: $(basename "$DRAFT")" >&2; exit 2; }

WT=${WT:-$HOME/Documents/GitHub/llama.cpp-ds4ports}
SRV=${SRV:-$WT/build-metal/bin/llama-server}
CTX=${DP_CTX:-4096}
NPRED=${DP_NPRED:-24}          # long enough that the draft gets several chances to propose
PROMPT=${DP_PROMPT:-"List three European capital cities and one short fact about each."}
LOGDIR=${CLAUDE_JOB_DIR:-/tmp}/draftpair
OUT=${OUT:-$HOME/Documents/GitHub/ornith-1m/tools/ds4-gates/results/draft-$ARCH-$(date +%Y%m%d-%H%M).txt}
LOCK=$LOGDIR/.lock
mkdir -p "$LOGDIR" "$(dirname "$OUT")"

if ! mkdir "$LOCK" 2>/dev/null; then
    echo "ANOTHER RUN HOLDS $LOCK -- refusing rather than racing it." >&2; exit 2
fi
SRVPID=""
cleanup() { [ -n "$SRVPID" ] && kill "$SRVPID" 2>/dev/null; rmdir "$LOCK" 2>/dev/null; scrub_abs_paths "${OUT:-}" 2>/dev/null; }
trap cleanup EXIT

# ⚠ Stamp the binary, not the tip. A run that straddles a rebuild is not a measurement, and the tip
# printed at launch is not necessarily what any arm exec'd. Learned the hard way on the watch gate.
BINSHA=$(shasum -a 1 "$SRV" 2>/dev/null | cut -c1-12)
echo "draft-pair gate: draft arch=$ARCH" | tee "$OUT"
echo "  target=$(basename "$TARGET")  draft=$(basename "$DRAFT")" | tee -a "$OUT"
echo "  binary sha1=$BINSHA  -c $CTX  n_predict=$NPRED" | tee -a "$OUT"

pick_port() {
    local p
    for p in $(seq "${DP_PORT:-9401}" $(( ${DP_PORT:-9401} + 60 ))); do
        lsof -nP -iTCP:"$p" -sTCP:LISTEN >/dev/null 2>&1 || { echo "$p"; return 0; }
    done
    return 1
}

# ★★ -md ALONE IS INERT IN THIS FORK, AND IT FAILS SILENTLY.
#
# Passing only -md loads the draft model, logs "loading draft model ...", prints its arch, and then
# drafts NOTHING: common/speculative.cpp:2488 logs "no implementations specified for speculative
# decoding" and returns nullptr. The server serves normally. The output is correct. `timings` simply
# has no draft_n key at all.
#
# An implementation type is required: --spec-type <name>, from common/speculative.cpp:2186 --
#   draft-simple  draft-eagle3  draft-mtp  draft-dflash  draft-dspark
#   ngram-simple  ngram-map-k   ngram-map-k4v  ngram-mod  ngram-cache
#
# Measured on Qwen3.5-4B + its DFlash draft:
#   -md alone                      timings has NO draft keys, statistics line absent
#   -md --spec-type draft-dflash   draft_n=39  draft_n_accepted=25   (64% accepted)
#
# ⚠ This is why the gate demands draft_n > 0 rather than trusting that -md did something. Its very
# first real run VOIDed on exactly this, before any conclusion about dflash could be drawn.
spec_type_for() { # $1 draft arch -> --spec-type name
    case "$1" in
        dflash)           echo draft-dflash ;;
        eagle3)           echo draft-eagle3 ;;
        gemma4-assistant) echo draft-mtp    ;;   # NextN head; override with DP_SPEC_TYPE if wrong
        *)                echo ""           ;;
    esac
}
SPEC_TYPE=${DP_SPEC_TYPE:-$(spec_type_for "$ARCH")}
if [ -z "$SPEC_TYPE" ]; then
    echo "refusing: no --spec-type known for draft arch '$ARCH'. Set DP_SPEC_TYPE explicitly." >&2
    echo "  Running without one loads the draft and drafts nothing, which this gate would VOID" >&2
    echo "  anyway -- but silently guessing a type is worse than refusing." >&2
    exit 2
fi
echo "  spec type: $SPEC_TYPE" | tee -a "$OUT"

start() { # $1 label  $2 "draft"|""  $3 "paged"|""
    PORT=$(pick_port) || { echo "  no free port" | tee -a "$OUT"; return 1; }
    local flags=()
    [ -n "$2" ] && flags+=(-md "$DRAFT" -ngld 99 --spec-type "$SPEC_TYPE")
    [ -n "$3" ] && flags+=(--kv-paged)
    local stamp; stamp=$(shasum -a 1 "$SRV" 2>/dev/null | cut -c1-12)
    if [ "$stamp" != "$BINSHA" ]; then
        echo "VOID: binary changed mid-run ($BINSHA -> $stamp). Arms must differ in ONE thing." | tee -a "$OUT"
        return 1
    fi
    # shellcheck disable=SC2086
    env ${DP_ENV:-} "$SRV" -m "$TARGET" -ngl 99 -c "$CTX" -np 1 -b 512 -ub 512 \
        --port "$PORT" --no-warmup -lv 5 "${flags[@]}" > "$LOGDIR/$1.log" 2>&1 &
    SRVPID=$!
    local i
    for i in $(seq 1 400); do
        [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ] && return 0
        kill -0 "$SRVPID" 2>/dev/null || break
        sleep 1
    done
    # ⚠ DESIGNED REFUSALS MATCH ON MESSAGE TEXT, NOT ON AN ARCH LIST. That is what caught the
    # Gemma4Assistant ctx_other guard, which lives nine lines from the eagle3/dflash one and was
    # missed by a grep for the arch names. Spec-decode plus --kv-paged may well be guarded somewhere
    # too (the paged scheduler's n_batch == n_ubatch assert is a candidate).
    if grep -qaE "not yet supported|needs DS4P_|requires DS4P_|not supported with speculative|incompatible with" "$LOGDIR/$1.log" 2>/dev/null; then
        echo "  $1: REFUSED BY DESIGN -- guarded configuration:" | tee -a "$OUT"
        grep -aE "not yet supported|needs DS4P_|requires DS4P_|not supported with speculative|incompatible with" \
            "$LOGDIR/$1.log" | head -1 | cut -c1-170 | sed 's/^/  | /' | tee -a "$OUT"
        return 3
    fi
    echo "  $1: DID NOT SERVE" | tee -a "$OUT"
    local why
    why=$(grep -aE "GGML_ASSERT|GGML_ABORT|error:|failed to " "$LOGDIR/$1.log" | grep -av "^ *[0-9]* " | tail -3)
    printf '%s\n' "${why:-<no assert/error line>}" | cut -c1-190 | sed 's/^/  ! /' | tee -a "$OUT"
    return 1
}

ask() { # -> "text<TAB>draft_n<TAB>draft_accepted"
    python3 -c "
import json
print(json.dumps({'prompt': '''$PROMPT''', 'n_predict': $NPRED, 'temperature': 0,
                  'seed': 1, 'cache_prompt': False, 'timings_per_token': True}))" > "$LOGDIR/req.json"
    curl -s --max-time 900 -X POST "http://127.0.0.1:$PORT/completion" \
        -H 'Content-Type: application/json' -d @"$LOGDIR/req.json" \
      | python3 -c "
import json,sys
try:
    d = json.load(sys.stdin)
except Exception:
    print('MALFORMED\t0\t0'); raise SystemExit
t = d.get('timings', {}) or {}
print('%s\t%s\t%s' % (d.get('content','').replace(chr(10),' ').replace(chr(9),' '),
                      t.get('draft_n', 0), t.get('draft_n_accepted', 0)))"
}

run_arm() { # $1 label  $2 draft?  $3 paged?  -> sets TXT DN DA, returns start()'s code
    start "$1" "$2" "$3"; local rc=$?
    [ "$rc" -ne 0 ] && return "$rc"
    local line; line=$(ask)
    TXT=$(printf '%s' "$line" | cut -f1)
    DN=$(printf  '%s' "$line" | cut -f2)
    DA=$(printf  '%s' "$line" | cut -f3)
    kill "$SRVPID" 2>/dev/null; wait "$SRVPID" 2>/dev/null; SRVPID=""; sleep 2
    printf '  %-22s draft_n=%-6s accepted=%-6s out=[%s]\n' "$1" "${DN:-0}" "${DA:-0}" "$TXT" | tee -a "$OUT"
    return 0
}

# ---------------- arm (a): target alone, static ----------------
run_arm "a target-static" "" ""; rc=$?
[ "$rc" -eq 3 ] && { echo "log: $OUT"; exit 3; }
[ "$rc" -ne 0 ] && { echo "log: $OUT"; exit 1; }
A_TXT="$TXT"

# ⚠ IS THE REFERENCE SANE? Everything below is measured against it. A degenerate arbiter cannot judge
# anything, and two broken arms agreeing would read as a PASS.
read -r LETTERS DISTINCT <<EOF
$(printf '%s' "$A_TXT" | python3 -c "
import sys
s = sys.stdin.read()
print(sum(c.isalpha() for c in s), len(set(s.strip())))")
EOF
if [ "${LETTERS:-0}" -lt 5 ] || [ "${DISTINCT:-0}" -lt 6 ]; then
    echo "VOID: the target's own static answer is degenerate -- [$A_TXT] ($LETTERS letters, $DISTINCT distinct)." | tee -a "$OUT"
    echo "  Fix the TARGET vehicle first. This says nothing about the draft or about paging." | tee -a "$OUT"
    echo "log: $OUT"; exit 2
fi

# ---------------- arm (b): target + draft, static ----------------
run_arm "b target+draft-static" draft ""; rc=$?
[ "$rc" -eq 3 ] && { echo "log: $OUT"; exit 3; }
[ "$rc" -ne 0 ] && { echo "log: $OUT"; exit 1; }
B_TXT="$TXT"; B_DN="$DN"

D_ARCH=$(grep -a "print_info: arch" "$LOGDIR/b target+draft-static.log" 2>/dev/null | sed 's/.*= *//' | tr -d ' \r' | tail -1)
echo "  draft loader arch (last print_info): ${D_ARCH:-<none>}" | tee -a "$OUT"
if [ -n "$D_ARCH" ] && [ "$D_ARCH" != "$ARCH" ]; then
    echo "STRIKE: the second model loaded as '$D_ARCH', not the expected draft arch '$ARCH'." | tee -a "$OUT"
    echo "  Wrong draft vehicle. Nothing below would be about '$ARCH'." | tee -a "$OUT"
    echo "log: $OUT"; exit 2
fi

# ⚠ PRESENCE, NOT ABSENCE. Lossless spec-decode means (b) == (a) whether the draft worked or was
# ignored entirely. Only a positive activity count separates those.
if [ "${B_DN:-0}" -eq 0 ]; then
    echo "VOID: draft_n = 0 -- the draft proposed NOTHING, so -md was inert." | tee -a "$OUT"
    echo "  (b) matching (a) would then be trivially true and (c) would test the target's paging" | tee -a "$OUT"
    echo "  while proving nothing about the draft. This gate cannot answer for '$ARCH' in that state." | tee -a "$OUT"
    echo "log: $OUT"; exit 2
fi
if [ "$A_TXT" != "$B_TXT" ]; then
    echo "*** STOP: (b) != (a) WITH PAGING NOT YET INVOLVED. ***" | tee -a "$OUT"
    echo "    target alone  [$A_TXT]" | tee -a "$OUT"
    echo "    target+draft  [$B_TXT]" | tee -a "$OUT"
    echo "  Greedy speculative decoding is lossless, so these must match. The spec path or this" | tee -a "$OUT"
    echo "  harness is broken. Fix that before reading anything about paged." | tee -a "$OUT"
    echo "log: $OUT"; exit 1
fi

# ---------------- arm (c): target + draft, paged ----------------
run_arm "c target+draft-paged" draft paged; rc=$?
[ "$rc" -eq 3 ] && { echo "log: $OUT"; exit 3; }
[ "$rc" -ne 0 ] && { echo "log: $OUT"; exit 1; }
C_TXT="$TXT"; C_DN="$DN"
CLOG="$LOGDIR/c target+draft-paged.log"
C_CONS=$(grep -ac "DS4P-CONSUME" "$CLOG")
C_RO=$(grep -ac "DS4P-READONLY" "$CLOG")
printf '  paged markers: DS4P-CONSUME=%s  DS4P-READONLY=%s\n' "$C_CONS" "$C_RO" | tee -a "$OUT"
echo "-----" | tee -a "$OUT"

rc=0
if [ "${C_CONS:-0}" -eq 0 ]; then
    echo "VOID: no DS4P-CONSUME in the paged arm -- no graph consumed a paged context." | tee -a "$OUT"
    echo "log: $OUT"; exit 2
fi
if [ "${C_DN:-0}" -eq 0 ]; then
    echo "VOID: draft_n = 0 under paging. The draft went inert exactly where it is under test." | tee -a "$OUT"
    echo "log: $OUT"; exit 2
fi
# ⚠ THE ROW UNDER TEST IS THE DRAFT, NOT THE TARGET. gemma4-assistant's NextN head is the only thing
# in this pair that takes the read-only path, so without DS4P-READONLY a PASS here proves the TARGET
# paged correctly and says nothing at all about the arch named on the row.
if [ "$ARCH" = gemma4-assistant ] && [ "${C_RO:-0}" -eq 0 ]; then
    echo "VOID: gemma4-assistant but no DS4P-READONLY marker under paging." | tee -a "$OUT"
    echo "  Its NextN head is the only read-only consumer in this pair. Without that marker a PASS" | tee -a "$OUT"
    echo "  would be about the TARGET's paging, not about this row." | tee -a "$OUT"
    echo "log: $OUT"; exit 2
fi
if [ "$A_TXT" != "$C_TXT" ]; then
    echo "*** FAIL: paged draft output diverges from the static reference. ***" | tee -a "$OUT"
    echo "    (a) target-static      [$A_TXT]" | tee -a "$OUT"
    echo "    (c) target+draft-paged [$C_TXT]" | tee -a "$OUT"
    echo "  (b) already matched (a), so the spec path is sound and PAGING is the difference." | tee -a "$OUT"
    rc=1
fi
if [ "$rc" -eq 0 ]; then
    echo "PASS: draft arch=$ARCH over a paged cache." | tee -a "$OUT"
    echo "  (a)==(b)==(c), draft active in both draft arms (b:$B_DN c:$C_DN proposals), CONSUME=$C_CONS" | tee -a "$OUT"
    echo "  ⚠ SCOPE: short prompt at -c $CTX. Nothing about long context, nothing about speed, and" | tee -a "$OUT"
    echo "  the ~50k intermittent paged defect is open underneath this row like every other." | tee -a "$OUT"
fi
echo "log: $OUT"
exit $rc
