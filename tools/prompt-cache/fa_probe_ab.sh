#!/usr/bin/env bash
# fa_probe_ab.sh -- runtime witness for "-fa on skips the support probe that -fa auto runs".
#
# The claim was source-only (llama-context.cpp:654 `if (cparams.auto_fa) { resolve(...) }`).
# Its CORE is testable without any unsupported KV combo and without CUDA: under `-fa auto` the
# probe runs and announces a resolution; under `-fa on` it must not run at all.
#
# Witness: presence/absence of the resolve_fused_ops output ("Flash Attention enabled" or
# "... not supported, set to disabled"). Absence under -fa on == guard skipped.
#
# Vehicle: Qwen3-4B Q8_0, Mac METAL. This is NOT CUDA and NOT GLM -- the probe code is
# backend-agnostic (src/llama-context.cpp) but the *set* of supported KV types is not.
set -u

BIN=~/Documents/GitHub/llama.cpp-dspark-metal/build/bin/llama-server
MODEL=~/AI/ktdev/qwen3-4b-dev-Q8_0.gguf
PORT=8796
OUT="$(dirname "$0")/fa_results"
mkdir -p "$OUT"

probe_arm () {  # probe_arm <label> <fa-mode> [extra flags...]
  local label="$1" fa="$2"; shift 2
  local log="$OUT/$label.log"
  "$BIN" -m "$MODEL" --port "$PORT" -c 2048 -np 1 -ngl 99 -fa "$fa" "$@" \
         -lv 4 --no-warmup > "$log" 2>&1 &
  local pid=$!
  local up=0
  for i in $(seq 1 90); do
    curl -s -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"' && { up=1; break; }
    kill -0 $pid 2>/dev/null || break
    sleep 1
  done
  kill $pid 2>/dev/null; wait $pid 2>/dev/null

  local resolved fa_state
  resolved=$(grep -cE "resolve_fused_ops|Flash Attention (enabled|not supported)" "$log")
  fa_state=$(grep -oE "flash_attn *= *[01]|Flash Attention (enabled|not supported[^\"]*)" "$log" | head -2 | tr '\n' ' ')
  printf "%-22s fa=%-5s up=%s  probe_lines=%-3s  %s\n" "$label" "$fa" "$up" "$resolved" "$fa_state"
}

echo "=== does the probe run? (witness = resolve_fused_ops output) ==="
probe_arm "auto_f16"      auto
probe_arm "on_f16"        on
probe_arm "auto_q80kv"    auto -ctk q8_0 -ctv q8_0
probe_arm "on_q80kv"      on   -ctk q8_0 -ctv q8_0
probe_arm "auto_mixed_kv" auto -ctk q8_0 -ctv q4_0
probe_arm "on_mixed_kv"   on   -ctk q8_0 -ctv q4_0

echo
echo "################ RAW EVIDENCE ################"
for f in auto_f16 on_f16 auto_mixed_kv on_mixed_kv; do
  echo "--- $f: every flash/probe line ---"
  grep -iE "flash|resolve_fused|auto_fa|not supported" "$OUT/$f.log" | head -6
  echo
done
