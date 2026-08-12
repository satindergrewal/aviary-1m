#!/usr/bin/env bash
# Serve Kimi-K3 REAP80 IQ2_KT for an interactive session.
#
# Refuses to start unless the artifact and the hardware are actually ready, because
# both failure modes are silent-ish and waste a GPU window:
#   1. a still-being-written GGUF loads as a truncated file
#   2. too little free VRAM silently spills to CPU and the session crawls
#
# Address note: no host addresses are hardcoded. Bind is 127.0.0.1 by default.
# Set K3_HOST=0.0.0.0 to expose on the LAN. Nothing here should ever be public.
set -uo pipefail

# ★ MUST be llama.cpp-k3kt. It is the ONLY tree with BOTH halves of the requirement:
#     llama.cpp-k3    KIMI_K3=yes  KT types=NO  (GGML_TYPE_COUNT=43 -> cannot read type 153)
#     llama.cpp-kt    KIMI_K3=no   KT types=yes
#     llama.cpp-k3kt  KIMI_K3=yes  KT types=yes  <- branch k3-kt, "Merge kt-quants into k3-kt"
#   I originally pointed this at llama.cpp-k3 and my own preflight PASSED it, because the gate
#   checked KIMI_K3 and never checked the quant types. Half a requirement is not a gate.
TREE=${BOX_MNT:?set BOX_MNT to the box storage root}/llama.cpp-k3kt
BIN=$TREE/build-cuda/bin/llama-server
# The WORKING artifact: grid family + 52-chunk imatrix, verdict COHERENT on the
# pre-registered probes (2026-07-31). The IQ2_KT trellis build at 2.15 bpw is kept
# on disk but is BROKEN (degenerate output) - do not serve it.
MODEL=${BOX_MNT:?set BOX_MNT to the box storage root}/bigmodels/k3-reap80-ours-IQ2_XXS-imat.gguf
TMPL=${BOX_MNT:?set BOX_MNT to the box storage root}/bigmodels/k3_chat_fixed.jinja
LOG=/tmp/k3_serve.log

CTX=${K3_CTX:-32768}
PORT=${K3_PORT:-8090}
HOST=${K3_HOST:-127.0.0.1}
NEED_GIB=${K3_NEED_GIB:-170}     # 148 model + KV + compute buffers, with margin

echo "== preflight =========================================================="

# 1. binary
[ -x "$BIN" ] || { echo "FAIL: no llama-server at $BIN"; exit 1; }
echo "ok   server binary"

# 2a. arch support: K3 (the kt tree has CUDA + KT types but KIMI_K3=0)
grep -q KIMI_K3 "$TREE/src/llama-arch.h" 2>/dev/null \
  && echo "ok   tree has KIMI_K3" || { echo "FAIL: tree lacks KIMI_K3"; exit 1; }

# 2b. ★ QUANT-TYPE support. This gate exists because 2a alone PASSED on llama.cpp-k3 and the
#     server then died with `invalid ggml type 153. should be in [0, 43)`. IQ2_KT is type 153,
#     so the tree's GGML_TYPE_COUNT must exceed it or the GGUF reader rejects the file before
#     arch handling is ever reached. Checking one half of a requirement is not a check.
TC=$(grep -oE 'GGML_TYPE_COUNT *= *[0-9]+' "$TREE/ggml/include/ggml.h" 2>/dev/null | grep -oE '[0-9]+')
if [ -z "$TC" ] || [ "$TC" -le 153 ]; then
  echo "FAIL: tree GGML_TYPE_COUNT=${TC:-unknown} cannot represent IQ2_KT (type 153)."
  echo "      Use ${BOX_MNT:?set BOX_MNT to the box storage root}/llama.cpp-k3kt - it is the only tree with BOTH kimi-k3 and KT types."
  exit 1
fi
echo "ok   tree can read KT types (GGML_TYPE_COUNT=$TC > 153)"

# 3. the quantize must be DONE. `pgrep -x`, never `-f`: -f matches the full cmdline
#    of every process INCLUDING this script, and this script mentions the name.
if pgrep -x llama-quantize >/dev/null 2>&1; then
  echo "FAIL: llama-quantize is still running -> $MODEL is INCOMPLETE."
  echo "      Loading a truncated GGUF is not a test, it is a crash. Wait for it."
  exit 1
fi
echo "ok   no quantize running"

# 4. model present and plausibly complete
[ -s "$MODEL" ] || { echo "FAIL: model missing"; exit 1; }
MG=$(( $(stat -c%s "$MODEL") / 1024/1024/1024 ))
echo "ok   model present: ${MG} GiB"
[ "$MG" -ge 140 ] || { echo "FAIL: ${MG} GiB is short of the expected ~150 GiB - truncated?"; exit 1; }

# 5. chat template. K3 ships NONE, so without this there is no chat at all.
[ -s "$TMPL" ] || { echo "FAIL: chat template missing at $TMPL"; exit 1; }
grep -q 'namespace(entries=' "$TMPL" \
  && echo "ok   chat template present AND carries the tool-result fix" \
  || { echo "FAIL: template lacks the namespace(entries=) fix - it will abort on turn 2"; exit 1; }

# 6. VRAM. Fail loudly rather than spilling to CPU and crawling.
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk '{s+=$1} END{print int(s/1024)}')
echo "     free VRAM: ${FREE} GiB across all cards (need >= ${NEED_GIB})"
if [ "$FREE" -lt "$NEED_GIB" ]; then
  echo "FAIL: not enough free VRAM. Something else holds the cards:"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits \
    | awk -F, '{printf "        pid %s: %.1f GiB\n", $1, $2/1024}'
  echo "      Refusing to start: it would silently offload to CPU and be unusable."
  echo "      Override with K3_NEED_GIB=<n> only if you know what you are doing."
  exit 1
fi
echo "ok   VRAM sufficient"

echo "== launching =========================================================="
# ★★ --no-repack IS LOAD-BEARING. DO NOT REMOVE IT.
#    Measured 2026-07-30 on this exact artifact, one variable, everything else equal:
#      repack ON (llama.cpp DEFAULT) -> "### ( ( ( ( ( ( ( ("      <- degenerate garbage
#      repack OFF (--no-repack)      -> "The capital of France is"  <- coherent
#    CPU weight repacking silently corrupts this model. Prime suspect: the 93 `iq4_nl`
#    fallback tensors (ssm_f_b = KDA recurrent state, attn_k_b = MLA key-B), which are
#    repack-eligible and sit on the two most sensitive paths in the network.
#    It fails SILENTLY - no crash, no warning - so the only symptom is bad output that
#    looks like a bad quant. Anyone who drops this flag will re-derive that wrong
#    conclusion about a model that is actually fine.
#
# -ub 512 is a CEILING. graph_max_nodes gives KIMI_K3 max(n_tokens*40, 32*n_tensors);
#         raising ubatch past 512 risks overflowing the graph. Patch 40->160 first.
# --jinja is THE enabler for the chat template + tool calling.
# temp/top-p per Moonshot's published sampling for K3.
nohup "$BIN" \
  -m "$MODEL" \
  --chat-template-file "$TMPL" \
  --jinja \
  --no-repack \
  -ngl 99 -sm layer \
  -c "$CTX" -b 512 -ub 512 \
  --temp 1.0 --top-p 0.95 \
  --host "$HOST" --port "$PORT" \
  > "$LOG" 2>&1 < /dev/null &
PID=$!
echo "pid=$PID  log=$LOG  http://${HOST}:${PORT}"
echo "$PID" > /tmp/k3_serve.pid

echo "== waiting for readiness (kill -0 on the PID, not a pattern) =========="
for i in $(seq 1 180); do
  kill -0 "$PID" 2>/dev/null || { echo "DIED during load. Last log:"; tail -25 "$LOG"; exit 1; }
  grep -q "server is listening" "$LOG" 2>/dev/null && { echo "READY after ~${i}0s"; break; }
  sleep 10
done

echo "== what we actually got (read, do not assume) ========================="
grep -E "load_tensors: .* model buffer size|llama_context: .*(KV|compute)|n_ctx *=|resolve_fused_ops|not supported, set to disabled|server is listening" "$LOG" \
  | tail -20
echo
echo "stop with: kill \$(cat /tmp/k3_serve.pid)"
