#!/usr/bin/env python3
"""Merge two GGUF imatrix files, taking each tensor entry from whichever source
you name for it. Built for one specific job:

    Unsloth's GLM-5.2 imatrix surveyed 811,008 tokens (88 chunks x 9216) across
    blocks 0-77 and contains NOTHING for the MTP layer, because upstream
    llama.cpp discards NextN tensors at load time so they never activate.

    Our own run collects blk.78 (the MTP layer) but cannot match that trunk
    coverage: the 1.4 TB BF16 master streams past 95 GB of RAM on every pass, so
    reaching 811k tokens would take ~95 hours of CPU.

    So take their blocks 0-77 and our block 78. Better than either alone.

An imatrix GGUF holds two entries per tensor, `<tensor>.in_sum2` (the summed
squared activations) and `<tensor>.counts` (how many times it was accumulated).
Both must travel together or the consumer divides by the wrong count.

Pure stdlib. Tensor payloads are copied as raw bytes, so nothing is recomputed.

    imatrix_merge.py --base unsloth.gguf --overlay ours.imatrix --out merged.imatrix
    imatrix_merge.py --base a --overlay b --out c --take 'blk\\.78\\.'
"""
import argparse, re, struct, sys

U8, I8, U16, I16, U32, I32, F32, BOOL, STR, ARR, U64, I64, F64 = range(13)
FIXED = {U8: 1, I8: 1, U16: 2, I16: 2, U32: 4, I32: 4, F32: 4, BOOL: 1,
         U64: 8, I64: 8, F64: 8}
GGML = {0: (4, 1), 1: (2, 1)}          # imatrix payloads are F32/F16 only


def read(path):
    f = open(path, "rb")
    if f.read(4) != b"GGUF":
        sys.exit(f"{path}: not a GGUF file")
    ver = struct.unpack("<I", f.read(4))[0]
    n_tensors = struct.unpack("<Q", f.read(8))[0]
    n_kv = struct.unpack("<Q", f.read(8))[0]

    def rstr():
        n = struct.unpack("<Q", f.read(8))[0]
        return f.read(n)

    def rval(t):
        if t in FIXED:
            return f.read(FIXED[t])
        if t == STR:
            return rstr()
        if t == ARR:
            et = struct.unpack("<I", f.read(4))[0]
            n = struct.unpack("<Q", f.read(8))[0]
            return (et, n, [rval(et) for _ in range(n)])
        sys.exit(f"unhandled kv type {t}")

    kvs, align = [], 32
    for _ in range(n_kv):
        k = rstr()
        t = struct.unpack("<I", f.read(4))[0]
        v = rval(t)
        if k == b"general.alignment":
            align = struct.unpack("<I", v)[0]
        kvs.append((k, t, v))

    tensors = []
    for _ in range(n_tensors):
        nm = rstr()
        nd = struct.unpack("<I", f.read(4))[0]
        dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)]
        tt = struct.unpack("<I", f.read(4))[0]
        off = struct.unpack("<Q", f.read(8))[0]
        tensors.append({"name": nm, "dims": dims, "type": tt, "off": off})

    data_start = (f.tell() + align - 1) // align * align
    return f, ver, kvs, tensors, align, data_start


def nbytes(t):
    n = 1
    for d in t["dims"]:
        n *= d
    if t["type"] not in GGML:
        sys.exit(f"unhandled ggml type {t['type']} in {t['name']}")
    bs, bn = GGML[t["type"]]
    return n // bn * bs


def wstr(out, b):
    out.write(struct.pack("<Q", len(b)))
    out.write(b)


def wval(out, t, v):
    if t == STR:
        wstr(out, v)
    elif t in FIXED:
        out.write(v)
    elif t == ARR:
        et, n, items = v
        out.write(struct.pack("<I", et))
        out.write(struct.pack("<Q", n))
        for it in items:
            wval(out, et, it)
    else:
        sys.exit(f"unhandled kv type {t}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="entries come from here by default")
    ap.add_argument("--overlay", required=True, help="entries matching --take come from here")
    ap.add_argument("--out", required=True)
    ap.add_argument("--take", default=r"blk\.(\d+)\.",
                    help="regex; overlay entries whose name matches are preferred. "
                         "Default takes any block the base does not have.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    bf, bver, bkvs, bts, balign, bdata = read(args.base)
    of, over, okvs, ots, oalign, odata = read(args.overlay)

    bnames = {t["name"] for t in bts}
    pat = re.compile(args.take.encode())

    # Anything in the overlay that the base lacks, plus anything explicitly matched.
    picked = [t for t in ots if t["name"] not in bnames or pat.search(t["name"])]
    picked_names = {t["name"] for t in picked}

    kept = [t for t in bts if t["name"] not in picked_names]
    merged = kept + picked

    def blocks(ts):
        out = set()
        for t in ts:
            m = re.search(rb"blk\.(\d+)\.", t["name"])
            if m:
                out.add(int(m.group(1)))
        return out

    print(f"base    : {args.base}")
    print(f"          {len(bts)} entries, blocks {min(blocks(bts))}-{max(blocks(bts))}")
    print(f"overlay : {args.overlay}")
    print(f"          {len(ots)} entries, blocks {min(blocks(ots))}-{max(blocks(ots))}")
    print(f"taken from overlay : {len(picked)} entries"
          f"{' (blocks ' + str(sorted(blocks(picked))) + ')' if picked else ''}")
    print(f"kept from base     : {len(kept)} entries")
    print(f"merged             : {len(merged)} entries, "
          f"blocks {min(blocks(merged))}-{max(blocks(merged))}")

    # A tensor and its .counts partner must come from the SAME source, or the
    # consumer divides summed activations by a count from a different run.
    stems = {}
    for t in merged:
        nm = t["name"].decode()
        for suf in (".in_sum2", ".counts"):
            if nm.endswith(suf):
                stems.setdefault(nm[: -len(suf)], set()).add(
                    "overlay" if t["name"] in picked_names else "base")
    split = [s for s, srcs in stems.items() if len(srcs) > 1]
    if split:
        sys.exit(f"ABORT: {len(split)} tensors would take in_sum2 and counts from "
                 f"different sources, e.g. {split[:3]}")
    print("partner check      : OK, every in_sum2 travels with its own counts")

    if args.dry_run:
        print("\n--dry-run, nothing written")
        return

    # KV comes from the base, but record that this file is a merge so the
    # provenance is not silently lost.
    kvs = list(bkvs)
    kvs.append((b"imatrix.merged_from", STR, args.overlay.encode()))
    kvs.append((b"imatrix.merged_entries", U32, struct.pack("<I", len(picked))))

    with open(args.out, "wb") as out:
        out.write(b"GGUF")
        out.write(struct.pack("<I", bver))
        out.write(struct.pack("<Q", len(merged)))
        out.write(struct.pack("<Q", len(kvs)))
        for k, t, v in kvs:
            wstr(out, k)
            out.write(struct.pack("<I", t))
            wval(out, t, v)

        off, infos = 0, []
        for t in merged:
            infos.append((t, off))
            off += (nbytes(t) + balign - 1) // balign * balign
        for t, o in infos:
            wstr(out, t["name"])
            out.write(struct.pack("<I", len(t["dims"])))
            for d in t["dims"]:
                out.write(struct.pack("<Q", d))
            out.write(struct.pack("<I", t["type"]))
            out.write(struct.pack("<Q", o))

        pad = (balign - out.tell() % balign) % balign
        out.write(b"\0" * pad)
        base_off = out.tell()

        for t, o in infos:
            src, sdata = (of, odata) if t["name"] in picked_names else (bf, bdata)
            sz = nbytes(t)
            src.seek(sdata + t["off"])
            out.seek(base_off + o)
            left = sz
            while left:
                chunk = src.read(min(1 << 24, left))
                if not chunk:
                    sys.exit(f"short read on {t['name']}")
                out.write(chunk)
                left -= len(chunk)
        out.seek(0, 2)
        while out.tell() % balign:
            out.write(b"\0")

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
