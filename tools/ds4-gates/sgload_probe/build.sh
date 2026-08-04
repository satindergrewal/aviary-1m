#!/usr/bin/env bash
# Build the standalone simdgroup_load pitch probe. Seconds, no llama.cpp rebuild.
set -euo pipefail
cd "$(dirname "$0")"
# No offline Metal toolchain needed: probe.metal is compiled at RUNTIME by the host
# (newLibraryWithSource:), the same path ggml-metal uses. Run from this directory.
clang++ -std=c++17 -fobjc-arc -framework Foundation -framework Metal probe.mm -o probe
echo "built: $(pwd)/probe   (run: ./probe <SRC_W> <LOAD_PITCH> <STORE_PITCH>)"
