#!/usr/bin/env python3
"""Render 9B NIAH heatmap (with failures) + prefill time chart from 5090 results."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

SURFACE = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"
GRID = "#e1e0d9"; GOOD = "#0ca30c"; CRIT = "#d03b3b"; BLUE = "#2a78d6"

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "text.color": INK, "axes.edgecolor": "#c3c2b7",
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK2,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})

runs = [json.loads(l) for l in open("results_9b_5090.jsonl")]
main = {r["target_tokens"]: r for r in runs[:5]}
CTX = [32768, 131072, 262144, 524288, 1048576]
LBL = ["32K", "131K", "262K", "524K", "1M"]
DEPTHS = [round((i + 0.5) / 10, 2) for i in range(10)]

fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=200)
for j, c in enumerate(CTX):
    fails = {d["depth"] for d in main[c]["depths"] if not d["hit"]}
    for i, d in enumerate(DEPTHS):
        ok = d not in fails
        cell = FancyBboxPatch((j + 0.06, i + 0.06), 0.88, 0.88,
                              boxstyle="round,pad=0,rounding_size=0.10",
                              linewidth=0, facecolor=GOOD if ok else CRIT, alpha=0.92)
        ax.add_patch(cell)
        ax.text(j + 0.5, i + 0.5, "✓" if ok else "✗", ha="center",
                va="center", fontsize=11, color="#ffffff", fontweight="bold")
ax.set_xlim(0, 5); ax.set_ylim(10, 0)
ax.set_xticks([j + 0.5 for j in range(5)]); ax.set_xticklabels(LBL, fontsize=10)
ax.set_yticks([i + 0.5 for i in range(10)])
ax.set_yticklabels([f"{int(d*100)}%" for d in DEPTHS], fontsize=9)
ax.set_xlabel("Context length (tokens)", fontsize=10)
ax.set_ylabel("Needle depth", fontsize=10)
for s in ax.spines.values(): s.set_visible(False)
ax.tick_params(length=0)
ax.axvline(3, color=INK2, linewidth=1.2, linestyle=(0, (4, 3)))
ax.text(2.94, -0.25, "native 262K", ha="right", fontsize=8.5, color=INK2)
ax.text(3.06, -0.25, "YaRN ×4", ha="left", fontsize=8.5, color=INK2)
ax.set_title("Ornith-9B needle retrieval: perfect to 524K, deep-middle loss at 1M",
             fontsize=11.5, color=INK, pad=44, loc="left")
ax.text(0, -1.15, "Q4_K_M weights · q8_0 KV cache · RTX 5090 · temperature 0 · main-run of each rung",
        fontsize=8.5, color=MUTED)
fig.tight_layout(); fig.savefig("niah_heatmap_9b.png", bbox_inches="tight")
print("wrote niah_heatmap_9b.png")

fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=200)
times = [main[c]["wall_seconds"] for c in CTX]
ax.plot(range(5), times, color=BLUE, linewidth=2, marker="o", markersize=7)
for i, t in enumerate(times):
    lab = f"{t:.0f}s" if t < 300 else f"{t/60:.1f}min"
    ax.annotate(lab, (i, t), textcoords="offset points", xytext=(0, 10),
                ha="center", fontsize=9, color=INK2)
ax.set_yscale("log")
ax.set_xticks(range(5)); ax.set_xticklabels(LBL, fontsize=10)
ax.set_ylabel("Prefill + answer wall time (s, log)", fontsize=10)
ax.set_xlabel("Context length", fontsize=10)
ax.grid(axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
for side in ("top", "right"): ax.spines[side].set_visible(False)
ax.tick_params(length=0)
ax.set_title("Ornith-9B wall time per NIAH run — RTX 5090 (32GB)",
             fontsize=11.5, color=INK, pad=14, loc="left")
fig.tight_layout(); fig.savefig("prefill_time_9b.png", bbox_inches="tight")
print("wrote prefill_time_9b.png")
