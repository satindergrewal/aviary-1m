#!/usr/bin/env python3
"""Assemble the GLM 5.2 calibration corpus per docs/GLM-CALIBRATION-CORPUS-DESIGN.md.

Stage (a) builds the static shares: code (own-stack + broad), English technical,
long documents. The Chinese and chat-template/reasoning shares are model-self-generated
(stage b, separate tool) and their budget is reserved in the manifest.

Every output share is a single UTF-8 text file. The manifest records source, byte
count, and sha256 per share so the calibration claim is auditable alongside the quant.

Privacy: a scrub pass drops any line matching IP-like, key-like, or long-hex/base64
patterns. Sources are restricted to public repositories and permissive corpora.
"""
import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

CODE_EXT = {".go", ".rs", ".swift", ".py", ".c", ".cc", ".cpp", ".h", ".hpp",
            ".cu", ".cuh", ".m", ".mm", ".ts", ".js", ".sh", ".metal"}
DOC_EXT = {".md", ".txt", ".rst"}
SKIP_DIRS = {".git", "node_modules", "vendor", "build", "dist", ".venv", "venv",
             "__pycache__", "assets", "img", "images", "fonts", "testdata"}
SKIP_NAME_PAT = re.compile(r"(secret|token|credential|\.env|\.pem|\.key$)", re.I)

SCRUB_PATTERNS = [
    re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),          # IPv4-like
    re.compile(r"\b[0-9a-fA-F:]{2,4}(::?[0-9a-fA-F]{1,4}){3,}\b"),  # IPv6-like
    re.compile(r"-----BEGIN [A-Z ]*KEY-----"),
    re.compile(r"\b[A-Za-z0-9+/]{64,}={0,2}\b"),                     # long base64
    re.compile(r"\b[0-9a-fA-F]{48,}\b"),                             # long hex
    re.compile(r"\b(Qm|12D3Koo)[1-9A-HJ-NP-Za-km-z]{20,}\b"),        # libp2p peer ids
]

def scrub(text: str) -> tuple[str, int]:
    kept, dropped = [], 0
    for line in text.splitlines():
        if any(p.search(line) for p in SCRUB_PATTERNS):
            dropped += 1
            continue
        kept.append(line)
    return "\n".join(kept), dropped

def iter_files(root: Path, exts):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in sorted(filenames):
            if SKIP_NAME_PAT.search(name):
                continue
            p = Path(dirpath) / name
            if p.suffix.lower() in exts:
                yield p

def read_text(p: Path, max_bytes: int) -> str | None:
    try:
        data = p.read_bytes()
    except OSError:
        return None
    if len(data) > max_bytes or b"\x00" in data[:4096]:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None

def collect(sources, exts, budget, max_file=262144, min_file=256):
    """Round-robin across sources until budget bytes collected."""
    out, total, dropped_lines, provenance = [], 0, 0, {}
    per_source = [list(iter_files(Path(s), exts)) for s in sources]
    idx = [0] * len(sources)
    while total < budget and any(i < len(fs) for i, fs in zip(idx, per_source)):
        for si, files in enumerate(per_source):
            if total >= budget or idx[si] >= len(files):
                continue
            p = files[idx[si]]
            idx[si] += 1
            text = read_text(p, max_file)
            if text is None or len(text) < min_file:
                continue
            text, d = scrub(text)
            dropped_lines += d
            if len(text) < min_file:
                continue
            header = f"\n\n// ==== {p.name} ====\n\n"
            out.append(header + text)
            total += len(text) + len(header)
            provenance[str(Path(sources[si]).name)] = provenance.get(str(Path(sources[si]).name), 0) + len(text)
    return "".join(out), dropped_lines, provenance

def collect_long_docs(sources, budget, min_file=131072):
    """Long-document share: whole files above min_file bytes, largest first."""
    candidates = []
    for s in sources:
        for p in iter_files(Path(s), CODE_EXT | DOC_EXT):
            try:
                sz = p.stat().st_size
            except OSError:
                continue
            if sz >= min_file:
                candidates.append((sz, p))
    candidates.sort(reverse=True)
    out, total, dropped, prov = [], 0, 0, {}
    for sz, p in candidates:
        if total >= budget:
            break
        text = read_text(p, 8 * 1024 * 1024)
        if text is None:
            continue
        text, d = scrub(text)
        dropped += d
        if len(text) < min_file // 2:
            continue
        out.append(f"\n\n// ==== LONGDOC {p.name} ====\n\n" + text)
        total += len(text)
        prov[p.name] = len(text)
    return "".join(out), dropped, prov

def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--total-mb", type=int, default=75)
    ap.add_argument("--wikitext-train", help="path to wiki.train.raw (English share)")
    ap.add_argument("--own-code", nargs="+", required=True, help="own-stack repo roots (public repos only)")
    ap.add_argument("--broad-code", nargs="+", required=True, help="permissive broad-code roots")
    ap.add_argument("--docs", nargs="+", default=[], help="extra English technical doc roots")
    args = ap.parse_args()

    total = args.total_mb * 1024 * 1024
    # spec mix: 40% code (70/30 own/broad), 15% zh (reserved), 20% en, 15% chat+think (reserved), 10% long
    budgets = {
        "code_own":   int(total * 0.40 * 0.70),
        "code_broad": int(total * 0.40 * 0.30),
        "english":    int(total * 0.20),
        "longdocs":   int(total * 0.10),
        "chinese_reserved":  int(total * 0.15),
        "chat_think_reserved": int(total * 0.15),
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"design": "docs/GLM-CALIBRATION-CORPUS-DESIGN.md", "total_target_bytes": total,
                "shares": {}, "reserved": {}}

    jobs = [
        ("code_own", lambda: collect(args.own_code, CODE_EXT, budgets["code_own"])),
        ("code_broad", lambda: collect(args.broad_code, CODE_EXT, budgets["code_broad"])),
        ("longdocs", lambda: collect_long_docs(args.own_code + args.broad_code + args.docs, budgets["longdocs"])),
    ]
    for name, job in jobs:
        text, dropped, prov = job()
        path = out / f"{name}.txt"
        path.write_text(text, encoding="utf-8")
        manifest["shares"][name] = {"bytes": len(text), "sha256": sha256(text),
                                    "scrubbed_lines": dropped, "provenance_bytes": prov}
        print(f"{name}: {len(text)/1e6:.1f} MB ({dropped} lines scrubbed)")

    if args.wikitext_train:
        wt = Path(args.wikitext_train).read_text(encoding="utf-8", errors="replace")
        en_docs = ""
        if args.docs:
            en_docs, dropped, prov = collect(args.docs, DOC_EXT, max(0, budgets["english"] - len(wt)))
        text = (wt[: budgets["english"]] + en_docs)
        path = out / "english.txt"
        path.write_text(text, encoding="utf-8")
        manifest["shares"]["english"] = {"bytes": len(text), "sha256": sha256(text),
                                         "note": "wikitext-2-raw TRAIN split (never test) + open docs"}
        print(f"english: {len(text)/1e6:.1f} MB")

    manifest["reserved"]["chinese"] = {"bytes": budgets["chinese_reserved"],
                                       "source": "GLM-5.2 self-generated under production serve (stage b)"}
    manifest["reserved"]["chat_think"] = {"bytes": budgets["chat_think_reserved"],
                                          "source": "GLM-5.2 self-generated, --jinja, reasoning enabled (stage b)"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"manifest -> {out}/manifest.json")

if __name__ == "__main__":
    sys.exit(main())
