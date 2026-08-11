#!/usr/bin/env python3
"""Honest progress for a running llama-quantize job.

Tensor count lies on MoE models. Kimi-K3 has 2573 tensors, but 276 of them are
[3584, 3072, 179] expert stacks carrying ~77% of the mass, so "286/2573 done"
reads about 3x faster than reality. This cost me a publicly-wrong 7.3 h ETA on a
job that was actually ~23 h.

Progress by SOURCE BYTES is the metric that does not lie: sum the `size = N MiB`
fields llama-quantize prints per tensor and divide by the model's total.

    quant_progress.py <logfile> [--total-gib 374] [--elapsed-min N]

--elapsed-min turns it into a projection. Take at least two readings a few
minutes apart before believing any ETA: the rate changes sharply when the run
crosses from small dense tensors into the expert stacks.
"""
import argparse, re, sys

SIZE_RE = re.compile(r"size =\s+([0-9.]+) MiB")
TENSOR_RE = re.compile(r"^\[\s*(\d+)/(\d+)\]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--total-gib", type=float, default=374.0,
                    help="source model size in GiB (K3 REAP80 = 374)")
    ap.add_argument("--elapsed-min", type=float, default=0.0)
    a = ap.parse_args()

    src = 0.0
    n = total = 0
    experts = 0
    last = ""
    for line in open(a.log, errors="ignore"):
        m = SIZE_RE.search(line)
        if m:
            src += float(m.group(1))
        t = TENSOR_RE.match(line)
        if t:
            n, total = int(t.group(1)), int(t.group(2))
            if "_exps.weight" in line:
                experts += 1
            last = line.strip()

    gib = src / 1024
    pct = gib / a.total_gib * 100

    print(f"tensors   : {n}/{total}   (expert stacks seen: {experts})")
    print(f"bytes     : {gib:.1f} GiB of {a.total_gib:.0f}  = {pct:.2f}%")
    if a.elapsed_min and pct > 0:
        total_h = a.elapsed_min / (pct / 100) / 60
        print(f"elapsed   : {a.elapsed_min:.0f} min")
        print(f"projected : {total_h:.1f} h total, {total_h - a.elapsed_min/60:.1f} h remaining")
        if experts == 0:
            print("  WARNING: no expert stacks processed yet - this projection is a "
                  "FLOOR, not an estimate. The rate drops sharply once they start.")
    if last:
        print(f"current   : {last[:110]}")


if __name__ == "__main__":
    main()
