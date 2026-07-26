#!/usr/bin/env bash
# F2 verification: does the MTP pass actually collect data for the nextn block?
# Control (--no-mtp) = the old behaviour; Treatment (default) = the patch.
# Same binary, same model, same corpus. Only the MTP pass differs.
set -u
LC=~/Documents/GitHub/llama.cpp
M=~/AI/mtptest/Qwen3.6-27B-Q4_K_M.gguf
S=<SCRATCH>/-Users-<user>-Documents-GitHub-ornith-1m/eb691b13-b1a4-4013-9767-83af97b8d410/scratchpad
CAL=$S/calib.txt

echo "=== CONTROL: --no-mtp (old behaviour) ==="
"$LC/build/bin/llama-imatrix" -m "$M" -f "$CAL" -o "$S/imat_nomtp.gguf" \
  -c 512 --chunks 2 --no-ppl --no-mtp -ngl 99 2>&1 | grep -iE "mtp|nextn|error|chunks" | head -5

echo
echo "=== TREATMENT: MTP pass on (the patch) ==="
"$LC/build/bin/llama-imatrix" -m "$M" -f "$CAL" -o "$S/imat_mtp.gguf" \
  -c 512 --chunks 2 --no-ppl -ngl 99 2>&1 | grep -iE "mtp|nextn|error|chunks" | head -5

echo
echo "=== GATE: blk.64 (the MTP block) entries in each imatrix ==="
for F in imat_nomtp imat_mtp; do
  N=$(python3 "$S/gguf_names.py" "$S/$F.gguf" 2>/dev/null | grep -oE "total tensors: [0-9]+" | grep -oE "[0-9]+")
  echo "--- $F: $N collected entries"
  python3 - "$S/$F.gguf" << 'PY'
import struct, sys
f=open(sys.argv[1],"rb"); f.read(4); struct.unpack("<I",f.read(4))
nt=struct.unpack("<Q",f.read(8))[0]; nkv=struct.unpack("<Q",f.read(8))[0]
def rstr():
    n=struct.unpack("<Q",f.read(8))[0]; return f.read(n).decode("utf-8","replace")
FIXED={0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
def rval(t):
    if t in FIXED: return f.read(FIXED[t])
    if t==8: return rstr()
    if t==9:
        et=struct.unpack("<I",f.read(4))[0]; n=struct.unpack("<Q",f.read(8))[0]
        return [rval(et) for _ in range(n)]
for _ in range(nkv):
    rstr(); t=struct.unpack("<I",f.read(4))[0]; rval(t)
names=[]
for _ in range(nt):
    nm=rstr(); nd=struct.unpack("<I",f.read(4))[0]
    for _ in range(nd): f.read(8)
    f.read(4); f.read(8); names.append(nm)
b64=sorted({n.split(".weight")[0] for n in names if n.startswith("blk.64.")})
print("   blk.64 matmuls with imatrix data:", len(b64))
for n in b64: print("     ", n)
PY
done
