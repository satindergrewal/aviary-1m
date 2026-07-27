#!/usr/bin/env python3
"""Extract the MTP/NextN block of a GGUF into a standalone file, mmproj-style.

Why not llama-quantize --prune-layers: it RENUMBERS the surviving blocks
(remap_layer in llama-quant.cpp), so pruning 0..N-1 to keep the MTP block turns
blk.78 into blk.0 and the loader, which identifies MTP layers by `il >= n_layer()`,
no longer recognises it. This keeps the original block index intact.

The complementary trunk-only file DOES come free from the existing tool:
    llama-quantize --prune-layers <MTP_INDEX> in.gguf trunk.gguf COPY
because blocks below the pruned index keep their numbers, and the loader's
trunk_only probe already tolerates "metadata declares MTP, tensors absent".

Pure stdlib: tensor payloads are copied as raw bytes, so no numpy and no
dequant/requant round-trip. Quantized data passes through untouched.
"""
import struct, sys, os, shutil

# ggml type -> (block_size_bytes, elements_per_block)
GGML = {0:(4,1),1:(2,1),2:(18,32),3:(20,32),6:(22,32),7:(24,32),8:(34,32),9:(36,32),
        10:(84,256),11:(110,256),12:(144,256),13:(176,256),14:(210,256),15:(292,256),
        16:(66,256),17:(74,256),18:(98,256),19:(50,256),20:(18,32),21:(110,256),
        22:(82,256),23:(136,256),24:(1,1),25:(2,1),26:(4,1),27:(4,1),28:(8,1),
        29:(52,256),30:(2,1)}

# gguf value type ids
U8,I8,U16,I16,U32,I32,F32,BOOL,STR,ARR,U64,I64,F64 = range(13)
FIXED = {U8:1,I8:1,U16:2,I16:2,U32:4,I32:4,F32:4,BOOL:1,U64:8,I64:8,F64:8}


def read_gguf(path):
    f = open(path, "rb")
    assert f.read(4) == b"GGUF", "not a GGUF file"
    ver = struct.unpack("<I", f.read(4))[0]
    n_tensors = struct.unpack("<Q", f.read(8))[0]
    n_kv = struct.unpack("<Q", f.read(8))[0]

    def rstr():
        n = struct.unpack("<Q", f.read(8))[0]
        return f.read(n)                      # keep raw bytes, re-emit verbatim

    def rval(t):
        if t in FIXED:  return f.read(FIXED[t])
        if t == STR:    return rstr()
        if t == ARR:
            et = struct.unpack("<I", f.read(4))[0]
            n  = struct.unpack("<Q", f.read(8))[0]
            return (et, n, [rval(et) for _ in range(n)])
        raise ValueError(f"unknown kv type {t}")

    kvs, align = [], 32
    for _ in range(n_kv):
        k = rstr(); t = struct.unpack("<I", f.read(4))[0]; v = rval(t)
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


def nbytes(dims, tt):
    n = 1
    for d in dims: n *= d
    if tt not in GGML:
        raise ValueError(f"unhandled ggml type {tt}")
    bs, bn = GGML[tt]
    if n % bn:
        raise ValueError(f"element count {n} not a multiple of block {bn}")
    return n // bn * bs


def wstr(out, b):
    out.write(struct.pack("<Q", len(b))); out.write(b)


def wval(out, t, v):
    if t in FIXED or t == STR:
        if t == STR: wstr(out, v)
        else:        out.write(v)
        return
    if t == ARR:
        et, n, items = v
        out.write(struct.pack("<I", et)); out.write(struct.pack("<Q", n))
        for it in items: wval(out, et, it)
        return
    raise ValueError(t)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        print("usage: gguf_extract_mtp.py <in.gguf> <out-mtp.gguf> [block_index]")
        sys.exit(1)
    src, dst = sys.argv[1], sys.argv[2]
    f, ver, kvs, tensors, align, data_start = read_gguf(src)

    # locate the MTP block: explicit arg, else the block owning a `nextn` tensor
    if len(sys.argv) > 3:
        blk = int(sys.argv[3])
    else:
        cand = {int(t["name"].decode().split(".")[1])
                for t in tensors
                if t["name"].startswith(b"blk.") and b"nextn" in t["name"]}
        if not cand:
            print("ERROR: no nextn tensors found; this model has no MTP block")
            sys.exit(2)
        blk = sorted(cand)[-1]

    prefix = f"blk.{blk}.".encode()
    keep = [t for t in tensors if t["name"].startswith(prefix)]
    if not keep:
        print(f"ERROR: no tensors for blk.{blk}")
        sys.exit(2)

    total = sum(nbytes(t["dims"], t["type"]) for t in keep)
    print(f"source     : {src}")
    print(f"MTP block  : blk.{blk}  ({len(keep)} tensors, {total/1073741824:.2f} GiB)")

    # metadata is copied wholesale so the loader still sees block_count,
    # nextn_predict_layers, arch and the tokenizer; only the tensor set shrinks.
    with open(dst, "wb") as out:
        out.write(b"GGUF")
        out.write(struct.pack("<I", ver))
        out.write(struct.pack("<Q", len(keep)))
        out.write(struct.pack("<Q", len(kvs)))
        for k, t, v in kvs:
            wstr(out, k); out.write(struct.pack("<I", t)); wval(out, t, v)

        # tensor infos need fresh, densely-packed offsets
        off = 0
        infos = []
        for t in keep:
            infos.append((t, off))
            off += (nbytes(t["dims"], t["type"]) + align - 1) // align * align
        for t, o in infos:
            wstr(out, t["name"])
            out.write(struct.pack("<I", len(t["dims"])))
            for d in t["dims"]: out.write(struct.pack("<Q", d))
            out.write(struct.pack("<I", t["type"]))
            out.write(struct.pack("<Q", o))

        pad = (align - out.tell() % align) % align
        out.write(b"\0" * pad)
        base = out.tell()
        for t, o in infos:
            sz = nbytes(t["dims"], t["type"])
            f.seek(data_start + t["off"])
            out.seek(base + o)
            remaining = sz
            while remaining:
                chunk = f.read(min(1 << 24, remaining))
                if not chunk: raise IOError("short read on tensor data")
                out.write(chunk); remaining -= len(chunk)
        out.seek(0, os.SEEK_END)
        while out.tell() % align: out.write(b"\0")

    print(f"wrote      : {dst}  ({os.path.getsize(dst)/1073741824:.2f} GiB)")
    print()
    print("complementary trunk-only file (existing tool, no new code needed):")
    print(f"  llama-quantize --prune-layers {blk} {src} trunk.gguf COPY")


if __name__ == "__main__":
    main()
