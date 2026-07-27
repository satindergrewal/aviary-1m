#!/usr/bin/env bash
# N3: IQ1_KT depth CONTROL arm.
# The depth matrix ran the CANDIDATE (IQ2_KT, "never loops at any depth") without
# a control. That result is only meaningful if the harness can still DETECT a loop
# at the same depths. IQ1_KT loops 4/8 at shallow depth, so it is the positive
# control: if it stops showing loops at 32K, the harness broke, not the model.
#
# Now also reports NON-TERMINATION, which the original matrix could not see.
set -uo pipefail
cd <BOX>/ktdev
OUT=<BOX>/ktdev/n3_results
mkdir -p "$OUT"
PORT=8091
PREFILL=<BOX>/llama.cpp-kt/wikitext-2-raw/wiki.train.raw

for depth in 0 8000 32000; do
  label="Qwen27B-IQ1_KT|1.75|d${depth}"
  echo "=== depth ${depth} $(date -Is) ==="
  if [ "$depth" = "0" ]; then
    python3 loop_rate.py --host 127.0.0.1 --port "$PORT" \
      --label "$label" --tsv "$OUT/n3.tsv" --json "$OUT/n3_d${depth}.json" \
      --save-samples "$OUT/samples"
  else
    python3 loop_rate.py --host 127.0.0.1 --port "$PORT" \
      --label "$label" --prefill-tokens "$depth" --prefill-file "$PREFILL" \
      --tsv "$OUT/n3.tsv" --json "$OUT/n3_d${depth}.json" \
      --save-samples "$OUT/samples"
  fi
  echo
done
echo "=== N3 COMPLETE $(date -Is) ==="
