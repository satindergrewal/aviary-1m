#!/usr/bin/env bash
# KLD battery for the K3 REAP80 quant ladder, path-matched against the byte-exact
# MXFP4 cell zero.
#
# WHY THIS SCRIPT EXISTS RATHER THAN A ONE-LINER
# ----------------------------------------------
# resolve_fused_ops() breaks on the FIRST layer whose fused-op device disagrees
# with its weight device, and then disables that fused op for ALL layers. So:
#
#   cell zero  374 GiB  -> must partially offload -> mismatch -> fused GDN OFF
#                                                 -> build_delta_net_chunking
#   IQ2_KT    ~150 GiB  -> fits 191 GiB VRAM      -> no mismatch -> fused GDN ON
#                                                 -> build_delta_net_fused
#
# Two different graph paths. Their float results differ. If the reference and the
# quant run on different paths, part of the reported KLD is fused-vs-chunked
# rounding, NOT quantization loss - on the one number the exercise exists to
# produce. There is no CLI flag for fused_gdn_*: auto_fgdn is always true and
# probe-resolved, so DEVICE LAYOUT IS THE ONLY LEVER.
#
# Therefore every arm is launched with an IDENTICAL layout, and step 3 refuses to
# report numbers unless the probe lines in both logs agree. The guard is the point.
#
# Usage:
#   ./kld_battery.sh pilot          # 5 chunks, measures per-chunk cost, sizes N
#   ./kld_battery.sh base   <N>     # write reference logits from cell zero
#   ./kld_battery.sh quant  <N> <gguf> <tag>
#   ./kld_battery.sh verify <tagA> <tagB> ...   # probe-match gate + report
set -uo pipefail

TREE=<BOX>/llama.cpp-k3
BIN=$TREE/build-cuda/bin
ZERO=<BOX>/bigmodels/k3-reap80-ours-mxfp4-bf16.gguf
CORPUS=<BOX>/llama.cpp-kt/wikitext-2-raw/wiki.test.raw
OUT=<BOX>/k3-kld
LOGITS=$OUT/base-cellzero.logits

# ---- PATH-MATCHED LAYOUT -------------------------------------------------------
# One layout, applied UNCHANGED to the reference and to every quant arm. Choose the
# profile with K3_PROFILE=cpu|split (default cpu).
#
#   cpu   PREFERRED CONTROL, not a fallback. Everything on one device, so
#         device_fused == device_layer everywhere, so NO fused op is ever disabled
#         and both arms are trivially guaranteed the same graph path. Removes the
#         confound by construction instead of by assertion. Costs speed: cell zero
#         is 374 GiB against 125 GiB RAM, so it streams from NVMe. Price it with
#         `pilot` - do NOT estimate it from disk bandwidth.
#
#   split Faster, but only usable when the cards are actually free. Cell zero at
#         374 GiB cannot fit 191 GiB VRAM, so this arm is necessarily mixed CPU/GPU,
#         which trips resolve_fused_ops and disables fused GDN. That is FINE as long
#         as every arm is mixed the same way - which is what the verify gate checks.
#         Tune --n-cpu-moe to whatever the free VRAM actually allows; the value below
#         assumes both cards empty.
#
#   -ub 512 is a CEILING in both profiles. graph_max_nodes gives KIMI_K3
#           max(n_tokens*40, 32*2573=82336); at Unsloth's ~160 nodes/token, 512
#           clears by 0.5% and 1024 blows up. Do not raise it without patching
#           40 -> 160 first.
#   -fa     left at default deliberately. This is a numerics measurement; forcing FA
#           changes the kernel mix, and on a split model `-fa auto` can disable it
#           outright on a device mismatch anyway.
case "${K3_PROFILE:=cpu}" in
  cpu)   LAYOUT=(-ngl 0                    -c 512 -b 512 -ub 512 --no-warmup -t 16) ;;
  split) LAYOUT=(-ngl 99 --n-cpu-moe 46 -sm layer -c 512 -b 512 -ub 512 --no-warmup) ;;
  *) echo "K3_PROFILE must be cpu or split"; exit 1 ;;
esac
echo "profile=$K3_PROFILE  layout: ${LAYOUT[*]}"

mkdir -p "$OUT"

# ---- TWO WITNESSES, because one of them is silent in the cpu profile ----------
# probe_lines catches an EXPLICIT fused-op decision. But: mismatch -> WARN (prints at
# default), success -> INFO (needs -lv 4). Measured by Fable-DSpark 2026-07-30:
#     lv=3  WARN=2  INFO=0        lv=4  WARN=2  INFO=9
# So in the cpu profile NOTHING is ever printed - no mismatch can occur, and the
# success line is INFO. probe_lines returns empty for every arm, empty == empty, and
# a gate built on it alone PASSES VACUOUSLY while verifying nothing.
#
# layout_fp is therefore the primary witness: which devices actually hold weights.
# `load_tensors: <DEV> model buffer size` prints at DEFAULT verbosity in every
# profile, and device placement is precisely what drives the mismatch. It cannot be
# silent, so it cannot pass vacuously.
probe_lines() {
  grep -hoE '(Flash Attention|Lightning Indexer|fused Gated Delta Net \([a-z]+\)) (enabled|not supported, set to disabled)' "$1" 2>/dev/null | sort -u
}
layout_fp() {
  grep -hoE 'load_tensors: +[A-Za-z0-9_]+ model buffer size' "$1" 2>/dev/null \
    | sed -E 's/load_tensors: +([A-Za-z0-9_]+) model buffer size/\1/' | sort -u | paste -sd, -
}

case "${1:-}" in

pilot)
  # Measure, do not extrapolate. 5 chunks is enough to price a chunk and NOT
  # enough to quote a rate for anything else.
  L=$OUT/pilot.log
  echo "== pilot: 5 chunks on cell zero, timing only" | tee "$L"
  /usr/bin/time -v "$BIN/llama-perplexity" -m "$ZERO" -f "$CORPUS" "${LAYOUT[@]}" \
      --chunks 5 --kl-divergence-base "$OUT/pilot.logits" >>"$L" 2>&1
  echo "--- probe outcome (this is the path the reference will take):"
  probe_lines "$L"
  echo "--- per-chunk seconds:"
  grep -oE '\[[0-9]+\][0-9.]+' "$L" | tail -5
  echo
  echo "SIZE N FROM THE MEASURED per-chunk COST, then run: base <N>"
  echo "logits bytes/token = 2*((163840+1)/2)+4 = 163844 uint16 = 327688 B"
  echo "  100 chunks = 16.8 GB   200 = 33.6 GB   283 = 47.5 GB (their +/-0.107%)"
  rm -f "$OUT/pilot.logits"
  ;;

base)
  N=${2:?need chunk count}
  L=$OUT/base.log
  echo "== reference: cell zero, $N chunks -> $LOGITS" | tee "$L"
  "$BIN/llama-perplexity" -m "$ZERO" -f "$CORPUS" "${LAYOUT[@]}" \
      --chunks "$N" --kl-divergence-base "$LOGITS" >>"$L" 2>&1
  rc=$?
  echo "rc=$rc  logits=$(du -h "$LOGITS" 2>/dev/null | cut -f1)"
  probe_lines "$L" | sed 's/^/  probe: /'
  exit $rc
  ;;

quant)
  N=${2:?need chunk count}; G=${3:?need gguf}; TAG=${4:?need tag}
  L=$OUT/$TAG.log
  [ -s "$LOGITS" ] || { echo "FATAL: no reference logits. run 'base <N>' first."; exit 1; }
  echo "== $TAG: $G, $N chunks, SAME layout as reference" | tee "$L"
  "$BIN/llama-perplexity" -m "$G" -f "$CORPUS" "${LAYOUT[@]}" \
      --chunks "$N" --kl-divergence-base "$LOGITS" --kl-divergence >>"$L" 2>&1
  rc=$?
  echo "rc=$rc"; probe_lines "$L" | sed 's/^/  probe: /'
  exit $rc
  ;;

verify)
  shift
  [ -s "$OUT/base.log" ] || { echo "GATE ERROR: $OUT/base.log missing/empty. Nothing to compare against."; exit 1; }
  BASEP=$(probe_lines "$OUT/base.log")
  BASEF=$(layout_fp  "$OUT/base.log")
  echo "=== PATH-MATCH GATE ==============================================="
  # A gate that cannot fail is not a gate. layout_fp must be non-empty or we are
  # about to "verify" by comparing two empty strings.
  [ -n "$BASEF" ] || { echo "GATE ERROR: reference layout fingerprint is EMPTY."; \
      echo "  No 'load_tensors: <DEV> model buffer size' lines in base.log."; \
      echo "  Refusing to run a gate whose witness is silent - it would pass vacuously."; exit 1; }
  echo "reference devices : $BASEF"
  echo "reference fused   : ${BASEP:-(none printed - expected in cpu profile, INFO needs -lv 4)}"
  fail=0
  for t in "$@"; do
    [ -s "$OUT/$t.log" ] || { echo "  [MISSING]  $t  (no log)"; fail=1; continue; }
    F=$(layout_fp "$OUT/$t.log"); P=$(probe_lines "$OUT/$t.log")
    if [ "$F" = "$BASEF" ] && [ "$P" = "$BASEP" ]; then
      echo "  [MATCH]    $t  devices=$F"
    else
      echo "  [MISMATCH] $t  <-- KLD IS CONFOUNDED, DO NOT REPORT IT"
      [ "$F" = "$BASEF" ] || echo "      devices: reference=$BASEF  arm=$F"
      [ "$P" = "$BASEP" ] || { echo "      fused-op decision differs:"; \
          diff <(echo "$BASEP") <(echo "$P") | sed 's/^/        /'; }
      fail=1
    fi
  done
  [ $fail -eq 0 ] || { echo; echo "GATE FAILED. Re-run mismatched arms with the reference layout."; exit 1; }

  echo
  echo "=== KLD (lead with this; PPL is not cross-comparable to anyone) ====="
  for t in "$@"; do
    echo "--- $t"
    grep -E '^(Mean KLD|Maximum KLD|99\.9%|99\.0%|95\.0%|90\.0%|Median|Same top p|Mean PPL\(Q\)|Mean PPL\(base\)|Mean ln\(PPL)' \
        "$OUT/$t.log" | sed 's/^/    /'
  done
  cat <<'NOTE'

REPORTING RULES
  - Lead with Mean KLD + the percentile ladder. A good mean with a fat 99.9%
    tail is a model that is fine until it catastrophically is not.
  - "top-1" in Unsloth's tables IS llama.cpp's `Same top p` (format string
    matched at perplexity.cpp:2005). Ours is directly placeable next to theirs.
  - NEVER print our PPL beside their 1.4581. Different corpus and/or n_ctx;
    1.4581 is nowhere near wiki.test.raw. KLD and Same top p transfer because
    each is measured against its OWN full-precision reference. PPL does not.
  - This measures the QUANT half only. REAP80 dropped 80% of experts and there
    is no local full-K3 reference, so the PRUNING half needs task benchmarks
    against Moonshot's official numbers. Do not imply KLD covered it.
NOTE
  ;;

*) sed -n '1,30p' "$0"; exit 1 ;;
esac
