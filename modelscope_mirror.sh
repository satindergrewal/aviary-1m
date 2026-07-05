#!/bin/bash
export MODELSCOPE_ENDPOINT=https://www.modelscope.ai
MS=~/.local/bin/modelscope
SP=/private/tmp/claude-501/-Users-satinder-Documents-GitHub-ornith-1m/45a478b3-69b0-4f18-b949-04b99bb04a0f/scratchpad
M=~/Documents/GitHub/ornith-models
G=~/Documents/GitHub/ornith-1m

up() { ${MS} upload "$1" "$2" "$3" --repo-type model 2>&1 | tail -1; }

echo "=== 9B repo ==="
up satgeze/Ornith-1.0-9B-1M-GGUF ${SP}/hf-card-9b.md README.md
up satgeze/Ornith-1.0-9B-1M-GGUF ${G}/assets/ornith-1m-banner.jpeg banner.jpeg
up satgeze/Ornith-1.0-9B-1M-GGUF ${G}/niah_heatmap_9b.png niah_heatmap_9b.png
up satgeze/Ornith-1.0-9B-1M-GGUF ${G}/prefill_time_9b.png prefill_time_9b.png
for f in ${M}/ornith-1.0-9b-1M-*.gguf ${M}/mmproj-ornith-9b-f16.gguf; do
  up satgeze/Ornith-1.0-9B-1M-GGUF "$f" "$(basename $f)"
done
echo "=== 35B repo ==="
up satgeze/Ornith-1.0-35B-1M-GGUF ${SP}/hf-card-35b.md README.md
up satgeze/Ornith-1.0-35B-1M-GGUF ${G}/assets/ornith-1m-banner.jpeg banner.jpeg
up satgeze/Ornith-1.0-35B-1M-GGUF ${G}/niah_heatmap.png niah_heatmap.png
up satgeze/Ornith-1.0-35B-1M-GGUF ${G}/prefill_speed.png prefill_speed.png
for f in ${M}/ornith-1.0-35b-1M-*.gguf ${M}/mmproj-ornith-35b-f16.gguf; do
  up satgeze/Ornith-1.0-35B-1M-GGUF "$f" "$(basename $f)"
done
echo "=== 397B repo ==="
up satgeze/Ornith-1.0-397B-1M-GGUF ${SP}/hf-card-397b.md README.md
up satgeze/Ornith-1.0-397B-1M-GGUF ${G}/assets/ornith-1m-banner.jpeg banner.jpeg
for f in ${M}/397b/ornith-1.0-397b-1M-*.gguf; do
  up satgeze/Ornith-1.0-397B-1M-GGUF "$f" "$(basename $f)"
done
echo "MODELSCOPE MIRROR COMPLETE"
