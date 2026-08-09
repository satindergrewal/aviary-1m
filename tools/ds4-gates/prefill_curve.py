#!/usr/bin/env python3
"""Extract a WITHIN-RUN prefill rate curve from a llama-server log.

WHY THIS EXISTS. Every prefill number this lane has produced is an ENDPOINT AVERAGE: one
`pp` figure per arm, covering the whole 400k prompt. From those averages I concluded
"both paths decay with context, static decays harder" -- a claim about the SHAPE of a
curve, derived from its two endpoints across DIFFERENT rungs (8k, 32k, ... 512k), each a
separate server launch with its own cold state.

`slot print_timing` already emits a cumulative pair every `-b` tokens:

    ... prompt processing, n_tokens = 203264, progress = 0.51, t = 881.90 s / 230.49 tokens per second

so the shape was in the logs the whole time. Differencing consecutive lines turns the
CUMULATIVE rate (which is a running average, and therefore lags) into an INTERVAL rate:

    rate_i = (n_i - n_{i-1}) / (t_i - t_{i-1})

⚠ THE CUMULATIVE COLUMN IS NOT THE CURVE. `230.49 tokens per second` at the 51% mark is the
average over everything up to that point, so it is dominated by the fast early region and
UNDERSTATES the decay. Reading the printed column as the curve would flatten exactly the
effect being measured. The differenced rate is what varies with depth.

WHAT IT IS FOR, concretely: two arms of the same ABBA can be compared AT MATCHED DEPTH
rather than at their endpoints, which removes the confound that endpoint averages carry --
a slow arm and an arm that decays faster produce the same average.

⚠ SCOPE, stated because the vocabulary does not cover it: this is PREFILL ONLY. The decode
phase emits no per-interval line at all; its timing appears once, in the final
`print_timing` block. So an early-vs-late split INSIDE decode is not available from this
log at any verbosity, and no amount of parsing changes that. Named rather than implied.

usage:  prefill_curve.py <log> [more logs...]   [--bucket N]
"""
import re, sys

# ⚠⚠ THE TWO ARMS PRINT PREFILL PROGRESS IN DIFFERENT FORMATS, AND THE FIRST VERSION OF THIS
# REGEX SAW ONLY ONE OF THEM. Measured, both from the same 512k ABBA:
#
#   static  slot print_timing:      ... prompt processing, n_tokens = 203264, progress = 0.51, ...
#   paged   populate_batch_from:    ... prompt processing, n_tokens =  19968 / 399181, progress = 0.05, ...
#
# The paged line carries a `/ total` the static line does not, so `n_tokens =\s*(\d+), progress`
# matched 409 static lines and **ZERO paged lines**. I wrote the pattern against the arm I happened
# to open first -- the instrument-both-arms class, in the one file whose entire purpose is to
# COMPARE two arms.
#
# ⇒ What saved it was the `len(pts) < 3 -> VOID` guard: the paged arm printed "VOID -- 0 usable
#   progress lines", loudly, instead of a plausible one-arm curve. **A parser that returns a
#   partial answer where it should return none is how a one-armed measurement gets reported as a
#   comparison.** The guard was worth more than the pattern.
#
# They also differ in RATE: static emits every `-b` tokens (409 lines), paged is time-limited
# (142 lines), so the paged curve is coarser at the same depth. Bucketing absorbs that; a
# point-for-point zip would not have.
LINE = re.compile(r'prompt processing, n_tokens =\s*(\d+)(?:\s*/\s*\d+)?, progress = ([\d.]+), t =\s*([\d.]+) s')


# A bucket must be at least this fraction covered by an arm's measured tokens before that arm's
# rate for it may be compared against the other's. 0.8 is a judgement, printed with every row so a
# reader can see what was excluded rather than having to trust the threshold.
MIN_COVER = 0.8


def curve(path):
    pts = []
    try:
        with open(path, 'r', errors='replace') as fh:
            for line in fh:
                m = LINE.search(line)
                if m:
                    pts.append((int(m.group(1)), float(m.group(3))))
    except OSError as e:
        print(f"  ! cannot read {path}: {e}", file=sys.stderr)
        return []
    # ⚠ A log may hold MORE THAN ONE request (warm-up prelude, retries). n_tokens resets to a
    # small value when a new one starts. Keep the LONGEST monotonic run, and say how many were
    # discarded -- silently concatenating two requests would splice a fast early region into the
    # middle of a slow late one and manufacture a "recovery" that never happened.
    runs, cur = [], []
    for p in pts:
        if cur and p[0] <= cur[-1][0]:
            runs.append(cur); cur = []
        cur.append(p)
    if cur:
        runs.append(cur)
    if not runs:
        return []
    best = max(runs, key=lambda r: r[-1][0] if r else 0)
    if len(runs) > 1:
        print(f"  note: {len(runs)} requests in this log; kept the longest "
              f"({best[-1][0]} tok), discarded {len(runs)-1}")
    return best


def report(path, bucket):
    pts = curve(path)
    if len(pts) < 3:
        print(f"{path}: VOID -- {len(pts)} usable progress lines (need >=3 to difference)")
        return None
    print(f"\n{path}")
    print(f"  {len(pts)} progress lines, {pts[0][0]} -> {pts[-1][0]} tok over {pts[-1][1]:.1f}s"
          f"   (cumulative avg {pts[-1][0]/pts[-1][1]:.1f} tok/s)")
    # bucket the differenced rates by depth so the shape is readable
    out = []
    b = {}
    for i in range(1, len(pts)):
        dn = pts[i][0] - pts[i-1][0]
        dt = pts[i][1] - pts[i-1][1]
        if dn <= 0 or dt <= 0:
            continue
        depth = pts[i-1][0] // bucket
        b.setdefault(depth, [0, 0.0])
        b[depth][0] += dn
        b[depth][1] += dt
    print(f"  depth range        interval tok/s   cover   (bucket {bucket//1000}k)")
    for d in sorted(b):
        dn, dt = b[d]
        r = dn / dt
        # ⚠⚠ A BUCKET IS NOT A BUCKET UNLESS BOTH ARMS ACTUALLY SPAN IT. The first version of this
        # printed a matched-depth ratio of 1.263 for the 0-50k bucket -- and that number was an
        # ARTIFACT of unequal coverage, not a measurement:
        #
        #     static  first progress line at   4096   -> its 0-50k bucket covers 4k..50k
        #     paged   first progress line at  19968   -> its 0-50k bucket covers 20k..50k
        #
        # Prefill rate falls STEEPLY in exactly that region (650 -> 300 tok/s across the first
        # 100k), so static's bucket included the fastest 16k of the prompt and paged's did not.
        # The comparison was crediting static with a head start the paged arm was never measured
        # over. Same shape for the LAST bucket of a still-running arm: static's 200-250k held only
        # 200k..213k, the fastest 13k of that band.
        #
        # ⇒ Carry each bucket's COVERED FRACTION and refuse to compare buckets that are not both
        #   substantially complete. This is the arms-must-differ-in-one-thing rule applied to the
        #   x-axis: if the depth ranges differ, the rates are not measuring the same thing.
        cov = dn / bucket
        out.append((d * bucket, r, cov))
        bar = '#' * max(1, int(r / 8))
        print(f"    {d*bucket:>7d}-{(d+1)*bucket:<7d} {r:8.1f}   {cov*100:3.0f}%   {bar}")
    full = [o for o in out if o[2] >= MIN_COVER]
    if len(full) >= 2:
        print(f"  ⇒ decay across FULL buckets only: {full[0][1]:.1f} -> {full[-1][1]:.1f} tok/s "
              f"= {full[0][1]/full[-1][1]:.2f}x slower at depth")
    return out


def main():
    args = [a for a in sys.argv[1:]]
    bucket = 50000
    if '--bucket' in args:
        i = args.index('--bucket')
        bucket = int(args[i+1]); del args[i:i+2]
    if not args:
        print(__doc__)
        return 2
    curves = {}
    for p in args:
        c = report(p, bucket)
        if c:
            curves[p] = c
    # ★ MATCHED-DEPTH COMPARISON -- the entire point. Endpoint averages cannot distinguish
    # "uniformly slower" from "decays faster", and those have different mechanisms.
    if len(curves) == 2:
        (pa, ca), (pb, cb) = list(curves.items())
        da = {d: (r, c) for d, r, c in ca}
        db = {d: (r, c) for d, r, c in cb}
        common = sorted(set(da) & set(db))
        print(f"\n  MATCHED-DEPTH RATIO   {pb.split('/')[-1]} / {pa.split('/')[-1]}")
        usable = []
        for d in common:
            ra, ca_ = da[d]
            rb, cb_ = db[d]
            ok = ca_ >= MIN_COVER and cb_ >= MIN_COVER
            tag = '' if ok else '   ⚠ EXCLUDED: partial bucket, the arms do not span the same depths'
            print(f"    depth {d:>7d}   {ra:7.1f} ({ca_*100:3.0f}%)  vs {rb:7.1f} ({cb_*100:3.0f}%)"
                  f"   ratio {rb/ra:5.3f}{tag}")
            if ok:
                usable.append((d, rb / ra))
        if len(usable) >= 2:
            print(f"  ⇒ over COMPARABLE buckets the ratio moves {usable[0][1]:.3f} -> {usable[-1][1]:.3f}.")
            print("    FLAT  => one arm is uniformly slower (a level effect: cold start, clocks, load).")
            print("    TREND => the two decay at different rates, which is the only thing that")
            print("             supports a 'decays harder' claim.")
            print("  ★ A TREND SURVIVES THE COLD-ARM CONFOUND and a level difference does not:")
            print("    a cold penalty scales an arm uniformly, so it cannot manufacture a")
            print("    depth-DEPENDENT ratio. The LEVEL here is confounded; the SLOPE is not.")
        elif not common:
            print("    no overlapping depth buckets -- one arm may still be running")
        else:
            print("    fewer than 2 comparable buckets -- no trend claim is available yet")
    return 0


if __name__ == '__main__':
    sys.exit(main())
