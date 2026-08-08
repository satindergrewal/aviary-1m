#!/usr/bin/env bash
# LINT: every paged consumer must announce itself, or arch_serve_gate will call a working arch broken.
#
# ⚠ WHY THIS EXISTS. `arch_serve_gate` decides whether ANY graph consumed the paged context by counting
# `DS4P-CONSUME` in the server log. That marker is emitted from the two helper funnels in
# llama-graph.cpp. On 2026-08-09 a sweep found a THIRD route: inkling.cpp calls
# build_attn_inp_kv_paged() then ggml_paged_attn_banded DIRECTLY, bypassing both helpers, and emitted
# no marker. The gate would have printed
#
#     DS4P-CONSUME banded=0  auto=0
#     VOID: no graph consumed the paged context
#
# for an architecture that pages correctly -- and that is the IDENTICAL output that meant a REAL defect
# on qwen35moe the same day (pool allocated, every layer static). **Same reading, opposite meanings.**
#
# ⇒ The gate's correctness therefore rested on an ENUMERATION IN A COMMENT that nothing checked. This
#   checks it. A fourth route is caught here, at the source, instead of six weeks later as a false VOID
#   on somebody's model.
#
# ⚠ WHAT THIS CANNOT DO: it cannot prove a marked consumer is CORRECT, only that a consumer announces
#   itself. It is a lint, not a gate. It answers "would arch_serve_gate be able to see this route?"
#
# usage: lint_paged_consumers.sh [fork-path]      exit 0 = every route announces itself
set -uo pipefail
WT="${1:-$HOME/Documents/GitHub/llama.cpp-ds4ports}"
[ -d "$WT/src/models" ] || { echo "not a fork tree: $WT" >&2; exit 2; }

# The ops that ARE paged attention. Reaching one of these is what makes a file a consumer.
OPS='ggml_paged_attn_banded|ggml_paged_attn\b'
# The helpers that already announce. A file routing through these inherits the marker.
HELPERS='build_attn_paged_or_null|build_attn_inp_kv_auto'
# The announcement itself.
MARK='ds4p_note_paged_consumer'

# ⚠⚠ MATCH CODE, NOT COMMENTS -- AND THIS LINT SHIPPED WITHOUT THAT AND ITS NEGATIVE CONTROL CAUGHT IT.
# First version grepped the raw file. inkling.cpp's own comment says "...rather than through
# build_attn_paged_or_null (banded funnel) or build_attn_inp_kv_auto" -- explaining that it does NOT use
# them -- and the lint read that as "routes via helper, marker inherited" and PASSED with the marker
# deliberately deleted. That is the exact trap named in AUDIT-bar-feeding-gates.md hours earlier: a file
# can contain the keyword inside a comment explaining why it does NOT do that thing. I wrote the warning
# and then built the instrument with the defect.
code_only() { sed 's|//.*||' "$1"; }

bad=0 checked=0
echo "lint: paged consumers must announce themselves (DS4P-CONSUME / $MARK)"
echo "tree: $(cd "$WT" && git rev-parse --short HEAD)"
for f in "$WT"/src/models/*.cpp; do
    code_only "$f" | grep -qE "$OPS" || continue          # not a direct-op consumer (CODE, not comments)
    checked=$((checked+1))
    b=$(basename "$f")
    if code_only "$f" | grep -q "$MARK"; then
        printf '  %-24s direct op + announces        OK\n' "$b"
    elif code_only "$f" | grep -qE "$HELPERS"; then
        printf '  %-24s direct op, routes via helper  OK (marker inherited)\n' "$b"
    else
        printf '  %-24s *** DIRECT OP, NO MARKER ***  arch_serve_gate would VOID this arch\n' "$b"
        bad=$((bad+1))
    fi
done

# ⚠ THE LINT MUST BE ABLE TO FIND ANYTHING AT ALL. If the op names are renamed upstream this loop
# silently checks zero files and exits 0 -- a clean pass from an instrument that looked at nothing,
# which is the failure this whole directory exists to prevent.
if [ "$checked" -eq 0 ]; then
    echo "VOID: matched ZERO files. The op names ($OPS) no longer appear in src/models/, so this lint"
    echo "  examined nothing and its exit code means nothing. Fix the pattern before trusting a pass."
    exit 2
fi

echo "-----"
if [ "$bad" -ne 0 ]; then
    echo "FAIL: $bad of $checked direct-op consumers do not announce themselves."
    echo "  Add the pair used by both helper funnels, next to the op:"
    echo "      LLAMA_LOG_DEBUG(\"%s: DS4P-CONSUME banded layer %d\\n\", __func__, il);"
    echo "      ds4p_note_paged_consumer();"
    exit 1
fi
echo "PASS: all $checked direct-op paged consumers announce themselves."
echo "  ⚠ This says arch_serve_gate can SEE these routes. It says nothing about whether they are"
echo "  correct -- that is a gate run per architecture, and it is a different claim."
