#!/usr/bin/env python3
"""Copy a Qwen3.5-family GGUF with YaRN 1M-context metadata baked in.

Reads general.architecture from the input (qwen35moe, qwen35, ...) and writes
the rope-scaling keys under that arch prefix.

Usage: bake_yarn.py input.gguf output.gguf
"""
import os
import sys
LLAMA_CPP = os.environ.get("LLAMA_CPP", os.path.expanduser("~/Documents/GitHub/llama.cpp"))
sys.path.insert(0, os.path.join(LLAMA_CPP, "gguf-py/gguf/scripts"))
sys.path.insert(0, os.path.join(LLAMA_CPP, "gguf-py"))
import gguf
from gguf_new_metadata import MetadataDetails, copy_with_new_metadata

inp, outp = sys.argv[1], sys.argv[2]

reader = gguf.GGUFReader(inp, "r")
arch = reader.get_field("general.architecture").contents()
native = reader.get_field(f"{arch}.context_length").contents()
assert native == 262144, f"unexpected native context: {native}"

new_metadata = {
    f"{arch}.context_length": MetadataDetails(gguf.GGUFValueType.UINT32, 1048576),
    f"{arch}.rope.scaling.type": MetadataDetails(gguf.GGUFValueType.STRING, "yarn"),
    f"{arch}.rope.scaling.factor": MetadataDetails(gguf.GGUFValueType.FLOAT32, 4.0),
    f"{arch}.rope.scaling.original_context_length": MetadataDetails(gguf.GGUFValueType.UINT32, 262144),
}

writer = gguf.GGUFWriter(outp, arch=arch, endianess=reader.endianess)
copy_with_new_metadata(reader, writer, new_metadata, [])
print(f"baked yarn keys (arch={arch}) into {outp}")
