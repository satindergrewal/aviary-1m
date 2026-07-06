#!/bin/bash
export PATH=$HOME/AI/ruler-env/bin:$PATH
RE=~/AI/ruler-env/bin/python
RD=~/AI/RULER/scripts
LS=~/AI/ROCmFPX/build/bin/llama-server
GGUF=/mnt/nvme2/ornith/models/gemma4-12b-uncensored-1M-Q4.gguf
LEN=${1:-131072}; NS=${2:-25}; THINK=${3:-think}; PORT=8044
OUT=~/AI/ruler_out/run_${LEN}_${THINK}
mkdir -p $OUT/pred
nohup $LS -m $GGUF -c $((LEN+6144)) -np 1 -ngl 99 --jinja --port $PORT > $OUT/server.log 2>&1 &
SP=$!
for w in $(seq 1 600); do curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break; kill -0 $SP 2>/dev/null || break; sleep 3; done
curl -sf http://127.0.0.1:$PORT/health >/dev/null 2>&1 || { echo 'CLEAN SERVE FAIL'; kill $SP 2>/dev/null; exit 1; }
echo "server up, RULER clean @ $LEN ($NS samples/task, $THINK)"
FLAG=''
[ "$THINK" = 'nothink' ] && FLAG='--no-thinking'
cd $RD
for TASK in niah_multivalue niah_multiquery vt cwe fwe; do
  [ -f ~/AI/ruler_out/data_$LEN/$TASK/validation.jsonl ] || $RE data/prepare.py --save_dir ~/AI/ruler_out/data_$LEN --benchmark synthetic --task $TASK --tokenizer_path unsloth/gemma-2-9b --tokenizer_type hf --max_seq_length $LEN --num_samples $NS --model_template_type base --random_seed 42 > $OUT/prep_$TASK.log 2>&1 || { echo "$TASK PREP FAIL"; continue; }
  $RE ~/AI/ruler_run_clean.py ~/AI/ruler_out/data_$LEN/$TASK/validation.jsonl $OUT/pred/$TASK.jsonl http://127.0.0.1:$PORT $FLAG >> $OUT/run_$TASK.log 2>&1
  echo "TASK-DONE $TASK"
done
$RE eval/evaluate.py --data_dir $OUT/pred --benchmark synthetic > $OUT/scores.txt 2>&1
echo '=== SCORES ==='; cat $OUT/pred/summary*.csv 2>/dev/null; grep -A20 -i score $OUT/scores.txt | head -20
kill $SP 2>/dev/null
echo "CLEAN RULER $LEN $THINK DONE"
