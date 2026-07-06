import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
S="#fcfcfb"; INK="#0b0b0b"; I2="#52514e"; MU="#898781"; G="#0ca30c"; R="#d03b3b"
plt.rcParams.update({"font.family":["Helvetica Neue","Arial","DejaVu Sans"],"figure.facecolor":S,"axes.facecolor":S,"savefig.facecolor":S,"text.color":INK,"xtick.color":MU,"ytick.color":MU,"axes.labelcolor":I2})
runs=[json.loads(l) for l in open("results_9b_5090.jsonl")]
main={r["target_tokens"]:r for r in runs[:5]}
CTX=[32768,131072,262144,524288,1048576]
LBL=["32K","131K","262K","524K","1M"]
cols=[(l,{d["depth"] for d in main[c]["depths"] if not d["hit"]}) for l,c in zip(LBL,CTX)]
cols.append(("1M\nQ8+f16KV", set()))  # Mac pure-config: 10/10
depths=[round((i+.5)/10,2) for i in range(10)]
fig, ax = plt.subplots(figsize=(8.2,5.4), dpi=200)
for j,(lab,fails) in enumerate(cols):
    for i,d in enumerate(depths):
        ok=d not in fails
        ax.add_patch(FancyBboxPatch((j+.06,i+.06),.88,.88,boxstyle="round,pad=0,rounding_size=.1",linewidth=0,facecolor=G if ok else R,alpha=.92))
        ax.text(j+.5,i+.5,"✓" if ok else "✗",ha="center",va="center",fontsize=10.5,color="#fff",fontweight="bold")
ax.set_xlim(0,6); ax.set_ylim(10,0)
ax.set_xticks([j+.5 for j in range(6)]); ax.set_xticklabels([c[0] for c in cols],fontsize=9)
ax.set_yticks([i+.5 for i in range(10)]); ax.set_yticklabels([f"{int(d*100)}%" for d in depths],fontsize=9)
ax.set_xlabel("Context length (budget config: Q4_K_M + q8_0 KV on RTX 5090; last column: max-quality on M3 Max)",fontsize=8.5)
ax.set_ylabel("Needle depth",fontsize=10)
for s in ax.spines.values(): s.set_visible(False)
ax.tick_params(length=0)
ax.axvline(3,color=I2,linewidth=1.2,linestyle=(0,(4,3)))
ax.axvline(5,color=I2,linewidth=1.6)
ax.text(2.94,-.25,"native 262K",ha="right",fontsize=8.5,color=I2)
ax.text(3.06,-.25,"YaRN ×4",ha="left",fontsize=8.5,color=I2)
ax.text(5.5,-.25,"isolation",ha="center",fontsize=8.5,color=I2)
ax.set_title("Ornith-9B needle retrieval: budget config ladder + max-quality 1M isolation",fontsize=11.5,color=INK,pad=40,loc="left")
ax.text(0,-1.1,"Quantization tax isolated: identical model scores 7/10 (Q4+q8KV) vs 10/10 (Q8+f16KV) at 1,048,576 tokens",fontsize=8.5,color=MU)
fig.tight_layout(); fig.savefig("assets/niah_heatmap_9b.png",bbox_inches="tight"); print("combined chart written")
