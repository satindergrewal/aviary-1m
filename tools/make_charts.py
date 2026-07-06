#!/usr/bin/env python3
"""Render NIAH heatmap + prefill throughput curve for the model card."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
GOOD = "#0ca30c"
BLUE = "#2a78d6"

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "text.color": INK, "axes.edgecolor": BASELINE,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.labelcolor": INK2, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})

CTX = [32768, 131072, 262144, 524288, 1048576]
CTX_LABELS = ["32K", "131K", "262K", "524K", "1M"]
DEPTHS = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]

# ---- Chart 1: NIAH heatmap (all cells pass) -------------------------------
fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=200)
for j, _ in enumerate(CTX):
    for i, d in enumerate(DEPTHS):
        cell = FancyBboxPatch((j + 0.06, i + 0.06), 0.88, 0.88,
                              boxstyle="round,pad=0,rounding_size=0.10",
                              linewidth=0, facecolor=GOOD, alpha=0.92)
        ax.add_patch(cell)
        ax.text(j + 0.5, i + 0.5, "✓", ha="center", va="center",
                fontsize=11, color="#ffffff", fontweight="bold")
ax.set_xlim(0, 5); ax.set_ylim(10, 0)
ax.set_xticks([j + 0.5 for j in range(5)])
ax.set_xticklabels(CTX_LABELS, fontsize=10)
ax.set_yticks([i + 0.5 for i in range(10)])
ax.set_yticklabels([f"{d}%" for d in DEPTHS], fontsize=9)
ax.set_xlabel("Context length (tokens)", fontsize=10)
ax.set_ylabel("Needle depth", fontsize=10)
for s in ax.spines.values():
    s.set_visible(False)
ax.tick_params(length=0)
# native/YaRN boundary between 262K and 524K columns
ax.axvline(3, color=INK2, linewidth=1.2, linestyle=(0, (4, 3)))
ax.text(2.94, -0.25, "native 262K", ha="right", fontsize=8.5, color=INK2)
ax.text(3.06, -0.25, "YaRN ×4", ha="left", fontsize=8.5, color=INK2)
ax.set_title("Needle-in-a-haystack retrieval — 50/50 needles recovered",
             fontsize=12, color=INK, pad=44, loc="left")
ax.text(0, -1.15, "Ornith-1.0-35B Q8_0 · YaRN factor 4 · temperature 0 · 10 needles per run",
        fontsize=8.5, color=MUTED, transform=ax.transData)
fig.tight_layout()
fig.savefig("assets/niah_heatmap.png", bbox_inches="tight")
print("wrote niah_heatmap.png")

# ---- Chart 2: prefill throughput vs position (stitched 1M run) ------------
rows = [json.loads(l) for l in open("prefill_timing.jsonl")]
def series(task, tok_off=0.0, t_off=0.0):
    pts = [(r["n_tokens"] + tok_off, r["t"] + t_off) for r in rows if r["task"] == task]
    return sorted(pts)

pts = series(1549) + series(1941, tok_off=798720, t_off=14360.46)
# instantaneous throughput between consecutive checkpoints, light smoothing
xs, ys = [], []
window = []
for (n0, t0), (n1, t1) in zip(pts, pts[1:]):
    if t1 <= t0:
        continue
    window.append((n1, (n1 - n0) / (t1 - t0)))
    if len(window) == 8:
        xs.append(sum(w[0] for w in window) / 8)
        ys.append(sum(w[1] for w in window) / 8)
        window = []

fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=200)
ax.plot(xs, ys, color=BLUE, linewidth=2)
ax.set_xlim(0, 1_060_000)
ax.set_ylim(0, max(ys) * 1.12)
ax.set_xticks(CTX)
ax.set_xticklabels(CTX_LABELS, fontsize=9.5)
ax.set_xlabel("Tokens into prompt", fontsize=10)
ax.set_ylabel("Prefill throughput (tokens/s)", fontsize=10)
ax.grid(axis="y", color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.tick_params(length=0)
ax.axvline(262144, color=INK2, linewidth=1.2, linestyle=(0, (4, 3)))
ax.text(262144 * 1.04, max(ys) * 1.02, "native 262K → YaRN", fontsize=8.5, color=INK2)
# direct labels at ends
ax.annotate(f"{ys[0]:.0f} tok/s", (xs[0], ys[0]), textcoords="offset points",
            xytext=(8, 4), fontsize=9, color=INK2)
ax.annotate(f"{ys[-1]:.0f} tok/s", (xs[-1], ys[-1]), textcoords="offset points",
            xytext=(-6, 8), fontsize=9, color=INK2, ha="right")
ax.set_title("Prefill throughput across a 1M-token prompt — M3 Max 128GB",
             fontsize=12, color=INK, pad=14, loc="left")
fig.tight_layout()
fig.savefig("assets/prefill_speed.png", bbox_inches="tight")
print("wrote prefill_speed.png")
