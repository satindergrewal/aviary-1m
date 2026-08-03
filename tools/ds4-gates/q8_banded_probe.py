#!/usr/bin/env python3
"""q8_banded_probe.py - B4 arc-2 CPU probe: is Q8_0 KV correct through the banded path?

Design: banded and fallback are the SAME math via different kernels, so their FP spread is
nonzero but small. Arm 1 (f16 KV) CALIBRATES that spread; arm 2 (q8 KV) must land in the
same ballpark (<= 3x). Each boot is WITNESSED by its reserve compute-buffer size (banded is
several times smaller than fallback at the same config) - engagement is proven per boot,
never assumed.

Usage: q8_banded_probe.py <server-bin> <model.gguf>
Boots four servers itself (f16/q8 x fallback/banded). CPU-only, port 8392.
"""
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

BIN, MODEL = sys.argv[1], sys.argv[2]
PORT = 8392
URL = f"http://127.0.0.1:{PORT}"

SEQ = [(3 + i*7) % 127 + 1 for i in range(256)]


def boot(ctk, banded, log):
    env = dict(os.environ)
    env.pop("DS4P_BANDED_QKV", None)
    if banded:
        env["DS4P_BANDED_QKV"] = "1"
    # banded arm: -fa on (banding engages when the type gate admits the KV type).
    # fallback/reference arm: -fa off = the dense rel path - there is no
    # disable-banding-only env in the tree, and -fa off is the honest mathematical
    # reference both KV types share.
    args = [BIN, "-m", MODEL, "--port", str(PORT), "-np", "1", "-c", "16384",
            "-ub", "512", "-b", "512", "-ngl", "0", "-fa", "on" if banded else "off",
            "--embeddings", "--pooling", "none", "-lv", "5"]
    if ctk:
        # K only: quantized V without flash attention refuses to boot ("V cache quantization
        # requires flash_attn"), and K's vec_dot path is where banded-vs-CUDA divergence
        # lives. V stays f16 in all arms.
        args += ["-ctk", ctk]
    p = subprocess.Popen(args, stdout=open(log, "w"), stderr=subprocess.STDOUT, env=env)
    for _ in range(60):
        try:
            urllib.request.urlopen(URL + "/health", timeout=2)
            return p
        except Exception:
            time.sleep(1)
    p.kill()
    raise RuntimeError(f"server failed to boot ({log})")


def stop(p):
    p.terminate()
    try:
        p.wait(timeout=10)
    except Exception:
        p.kill()


def embed(tokens):
    req = urllib.request.Request(URL + "/embedding",
        data=json.dumps({"input": tokens}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.load(r)
    if isinstance(d, list):
        d = d[0]
    e = d["embedding"]
    if e and isinstance(e[0], (int, float)):
        e = [e]
    return e


def buf_mib(log):
    txt = open(log).read()
    m = re.search(r"compute buffer size =\s+([0-9.]+) MiB", txt)
    return float(m.group(1)) if m else -1.0


def run_arm(ctk, banded, log):
    p = boot(ctk, banded, log)
    try:
        e = embed(SEQ)
    finally:
        stop(p)
    return e, buf_mib(log)


def spread(a, b):
    m = 0.0
    for va, vb in zip(a, b):
        for xa, xb in zip(va, vb):
            m = max(m, abs(xa - xb))
    return m


def main():
    results = {}
    for name, ctk, banded in [
            ("f16-fallback", None, False),
            ("f16-banded",   None, True),
            ("q8-fallback",  "q8_0", False),
            ("q8-banded",    "q8_0", True)]:
        log = f"/tmp/q8probe_{name}.log"
        e, buf = run_arm(ctk, banded, log)
        results[name] = (e, buf)
        print(f"  {name:14s} compute_buf = {buf:8.2f} MiB")

    # witness: within each KV type, the banded arm's buffer must be well below fallback's.
    ok_witness = True
    for t in ("f16", "q8"):
        bb, fb = results[f"{t}-banded"][1], results[f"{t}-fallback"][1]
        if not (0 < bb < 0.7*fb):
            print(f"  WITNESS FAIL ({t}): banded buf {bb} vs fallback {fb} - paths not distinct")
            ok_witness = False
    if not ok_witness:
        print("== RESULT: INVALID (engagement not witnessed) ==")
        return 2

    # second witness: q8 must actually be ACTIVE. The AUTHORITATIVE evidence is the boot
    # log's cache line ("K (q8_0): ..."), not the output delta - on a random-weight
    # synthetic the softmax can peak identically under quantization and outputs then agree
    # to fp32 noise (measured 2.6e-08 with a PROVEN q8_0 cache), so an output-delta
    # threshold false-alarms. The delta stays printed as information only.
    for name in ("q8-fallback", "q8-banded"):
        if "K (q8_0)" not in open(f"/tmp/q8probe_{name}.log").read():
            print(f"== RESULT: INVALID ({name}: boot log does not show a q8_0 K cache) ==")
            return 2
    quant_noise = spread(results["f16-fallback"][0], results["q8-fallback"][0])
    print(f"  q8 K-cache WITNESSED in boot logs; f16-vs-q8 output delta (info) = {quant_noise:.3e}")

    cal = spread(results["f16-fallback"][0], results["f16-banded"][0])
    q8  = spread(results["q8-fallback"][0],  results["q8-banded"][0])
    print(f"  path FP spread: f16(calibration) = {cal:.3e}   q8 = {q8:.3e}")

    limit = max(3.0*cal, 1e-6)
    if q8 <= limit:
        print(f"== RESULT: PASS (q8 spread within 3x of f16 calibration, limit {limit:.3e}) ==")
        return 0
    print(f"== RESULT: FAIL (q8 spread {q8:.3e} > limit {limit:.3e}) ==")
    return 1


if __name__ == "__main__":
    sys.exit(main())
