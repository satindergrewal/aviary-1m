#!/usr/bin/env python3
"""Read general.architecture out of a remote GGUF's header WITHOUT downloading the model.

WHY. Choosing a test vehicle by name has failed twice in this lane:

    intended `starcoder` -> downloaded StarCoder2-3B  -> loads as `starcoder2`, a different file
    intended `nemotron`  -> Llama-3.1-Nemotron-Nano-8B -> loads as `llama`

The first cost a download and a retracted verification. The second was caught before downloading, by
asking which arch the repo loads as. This makes that question cheap and mechanical: a ranged GET of
the first megabyte, parsed locally. Hundreds of KB instead of gigabytes, and it reads the ARTIFACT
rather than reasoning forward from a source repo's config.json through a converter I hope was the one
actually used.

⚠ THE PARSER MUST DIE RATHER THAN GUESS. The first version returned early from a string-array without
consuming its elements, so every later offset was garbage and it fell over on a bogus type tag. That
crash was the good outcome. Had it silently resynchronised it would have printed a confident wrong
arch, which is worse than no answer -- the whole point of this tool is to be trusted about exactly one
string. Unknown type tags, short reads and nested arrays all raise.

usage:
    gguf_arch_probe.py <repo> <file.gguf> [<repo> <file.gguf> ...]
    gguf_arch_probe.py --expect <arch> <repo> <file.gguf>     # exit 1 unless it matches
"""
import json
import struct
import subprocess
import sys

HDR_BYTES = 1 << 20  # 1 MiB: the tokenizer vocab array can be ~500 KB and general.* precede it

SCALAR = {0: '<B', 1: '<b', 2: '<H', 3: '<h', 4: '<I', 5: '<i',
          6: '<f', 7: '<?', 10: '<Q', 11: '<q', 12: '<d'}


class Truncated(Exception):
    """Ran off the end of the fetched window -- distinct from a malformed file."""


def fetch_head(repo: str, fname: str) -> bytes:
    url = f"https://huggingface.co/{repo}/resolve/main/{fname}"
    out = subprocess.run(
        ["curl", "-sL", "-r", f"0-{HDR_BYTES - 1}", "--max-time", "120", url],
        capture_output=True)
    if out.returncode != 0:
        raise RuntimeError(f"curl failed rc={out.returncode}")
    return out.stdout


class Reader:
    def __init__(self, b: bytes):
        self.b, self.o = b, 0

    def raw(self, fmt: str):
        n = struct.calcsize(fmt)
        if self.o + n > len(self.b):
            raise Truncated()
        v = struct.unpack_from(fmt, self.b, self.o)
        self.o += n
        return v

    def string(self) -> str:
        (n,) = self.raw('<Q')
        if n > (1 << 24):                       # a sane key/value; a huge one means desynchronised
            raise ValueError(f"absurd string length {n} -- parser lost sync")
        if self.o + n > len(self.b):
            raise Truncated()
        s = self.b[self.o:self.o + n].decode('utf-8', 'replace')
        self.o += n
        return s

    def value(self, t: int):
        if t == 8:
            return self.string()
        if t == 9:
            (et,) = self.raw('<I')
            (n,) = self.raw('<Q')
            if et == 9:
                raise ValueError("nested array -- unsupported, refusing to guess")
            if et == 8:
                for _ in range(n):              # ⚠ MUST consume; skipping desyncs everything after
                    self.string()
                return f"<array[str] n={n}>"
            if et not in SCALAR:
                raise ValueError(f"unknown array element type {et}")
            self.o += struct.calcsize(SCALAR[et]) * n
            if self.o > len(self.b):
                raise Truncated()
            return f"<array n={n}>"
        if t not in SCALAR:
            raise ValueError(f"unknown value type {t} -- parser lost sync")
        return self.raw(SCALAR[t])[0]


def parse(b: bytes) -> dict:
    if b[:4] != b'GGUF':
        # ⚠ DISTINGUISH "wrong file" FROM "wrong arch". HF answers a bad path with an HTML/text error
        # page, and curl -sL hands it over with rc=0. Reported as a wrong arch it would look like a
        # meaningful STRIKE -- which is how a control ends up vacuous: it refused, but it never
        # tested the thing it was written to test.
        head = b[:40].decode('utf-8', 'replace').replace('\n', ' ')
        if b[:1] in (b'<', b'{') or b'not found' in b[:400].lower():
            raise ValueError(f"server returned an error page, not a file -- check repo/filename: {head!r}")
        raise ValueError(f"not a GGUF file (magic {b[:4]!r}, head {head!r})")
    r = Reader(b)
    r.o = 4
    (ver,) = r.raw('<I')
    (ntens,) = r.raw('<Q')
    (nkv,) = r.raw('<Q')
    kv = {'_version': ver, '_tensors': ntens, '_kv_count': nkv}
    for _ in range(nkv):
        try:
            k = r.string()
            (t,) = r.raw('<I')
            kv[k] = r.value(t)
        except Truncated:
            kv['_truncated'] = True             # fine IF the key we want was already read
            break
    return kv


INTEREST = ('general.architecture', 'general.name', 'general.size_label')


def report(repo: str, fname: str) -> str | None:
    try:
        kv = parse(fetch_head(repo, fname))
    except Exception as e:                      # noqa: BLE001 -- any failure means "do not trust"
        print(f"  {repo}/{fname}\n    PROBE FAILED: {type(e).__name__}: {e}")
        return None
    arch = kv.get('general.architecture')
    print(f"  {repo}/{fname}")
    for k in INTEREST:
        if k in kv:
            print(f"    {k:<28s} {kv[k]}")
    for k in sorted(kv):
        if arch and k.startswith(f"{arch}.") and ('head_count' in k or 'block_count' in k
                                                  or 'context_length' in k or 'expert' in k):
            print(f"    {k:<28s} {kv[k]}")
    if kv.get('_truncated') and arch is None:
        print("    (header exceeds the 1 MiB window AND the arch was not in it -- inconclusive)")
    return arch


def main(argv: list[str]) -> int:
    expect = None
    if argv[:1] == ['--expect']:
        expect, argv = argv[1], argv[2:]
    if not argv or len(argv) % 2:
        print(__doc__)
        print("refusing: need <repo> <file.gguf> pairs", file=sys.stderr)
        return 2
    rc = 0
    for i in range(0, len(argv), 2):
        arch = report(argv[i], argv[i + 1])
        if expect is not None:
            if arch == expect:
                print(f"    -> MATCH: loads as '{expect}'")
            else:
                print(f"    -> STRIKE: expected '{expect}', header says {arch!r}. Do not download.")
                rc = 1
    return rc


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
