#!/usr/bin/env python3
"""banded_equiv_gate.py - arc-1 correctness net (numerical equivalence).

Proves the per-stream banded slicing is arithmetically right: the SAME two token
sequences produce the SAME per-token hidden states whether decoded singly (idle
second slot -> single-stream ubatches -> the unchanged n_stream==1 path) or
concurrently (joint multi-stream ubatches -> the new per-stream banded calls).

Uses --embeddings --pooling none so the full forward pass is observable on a
no-vocab synthetic model (no tokenizer/sampler involvement). CPU execution is
deterministic, so the tolerance is tight.

Usage: banded_equiv_gate.py <server-url> <server-log-path>
The server must already run: --embeddings --pooling none -np 2 -fa on, CPU, -lv 5.
The log path makes the gate SELF-WITNESSING: it fails as INVALID unless a joint batch
(n_tokens = 2 x seq len) actually appears in the runtime log - a serialized phase B
passes the numeric compare vacuously.
"""
import json
import re
import sys
import threading
import urllib.request

URL = sys.argv[1]
SRVLOG = sys.argv[2]

# n_vocab = 128 (cycle within it). Embedding tasks cannot split across ubatches (the server
# rejects inputs > n_ubatch) and n_batch is forced to n_ubatch in embeddings mode - so joint
# batching of two tasks requires 2 x seq_len <= n_ubatch. 256 + 256 fills one 512 ubatch.
# Co-queueing is forced by FLOODING (16 concurrent pairs saturating both slots); the
# n_seqs=2 witness below decides validity either way. (8 earlier runs with 2 solo requests
# were VACUOUS - ms-scale prefills serialized every time.)
SEQ_X = [(3 + i*7)  % 127 + 1 for i in range(256)]
SEQ_Y = [(11 + i*13) % 127 + 1 for i in range(256)]
N_FLOOD = 16


def embed(tokens):
    req = urllib.request.Request(
        URL + "/embedding",
        data=json.dumps({"input": tokens}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.load(r)
    # native endpoint: list of {index, embedding: [[per-token vectors]]} or {embedding: ...}
    if isinstance(d, list):
        d = d[0]
    e = d["embedding"]
    # pooling none -> list of per-token vectors
    if e and isinstance(e[0], (int, float)):
        e = [e]
    return e


def max_abs_diff(a, b):
    assert len(a) == len(b), f"token count mismatch {len(a)} vs {len(b)}"
    m = 0.0
    for va, vb in zip(a, b):
        assert len(va) == len(vb)
        for xa, xb in zip(va, vb):
            m = max(m, abs(xa - xb))
    return m


def main():
    print("# phase A: singly decoded (single-stream path)")
    ax = embed(SEQ_X)
    ay = embed(SEQ_Y)
    print(f"  X: {len(ax)} tokens x {len(ax[0])} dims; Y: {len(ay)} tokens")

    print(f"# phase B: flood - {N_FLOOD} concurrent X/Y pairs saturating both slots")
    res_x = [None]*N_FLOOD
    res_y = [None]*N_FLOOD
    def worker(store, idx, seq):
        store[idx] = embed(seq)
    threads = []
    for i in range(N_FLOOD):
        threads.append(threading.Thread(target=worker, args=(res_x, i, SEQ_X)))
        threads.append(threading.Thread(target=worker, args=(res_y, i, SEQ_Y)))
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    dx = max(max_abs_diff(ax, r) for r in res_x)
    dy = max(max_abs_diff(ay, r) for r in res_y)
    print(f"  worst over {N_FLOOD} pairs: max|dX| = {dx:.3e}   max|dY| = {dy:.3e}")

    # self-witness: a joint MULTI-SEQUENCE ubatch must actually have happened during
    # phase B. Batch SIZE cannot discriminate (joint and single prefill chunks both cap at
    # n_ubatch); the ubatch's n_seqs can.
    with open(SRVLOG) as f:
        log = f.read()
    counts = {}
    for m in re.finditer(r"n_seqs\s*=\s*(\d+)", log):
        counts[int(m.group(1))] = counts.get(int(m.group(1)), 0) + 1
    print(f"  n_seqs histogram: {dict(sorted(counts.items(), reverse=True))}")
    if counts.get(2, 0) == 0:
        print("== RESULT: INVALID - no multi-sequence ubatch occurred (phase B serialized) ==")
        return 2

    tol = 1e-4  # CPU-deterministic; slack only for f16 cast boundary effects
    if dx <= tol and dy <= tol:
        print(f"== RESULT: PASS (tol {tol:.0e}, joint batching witnessed) ==")
        return 0
    print(f"== RESULT: FAIL (tol {tol:.0e}) ==")
    return 1


if __name__ == "__main__":
    sys.exit(main())
