import json
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullFormatter

import style

palette = sys.argv[1] if len(sys.argv) > 1 else "oxide"
out = sys.argv[2] if len(sys.argv) > 2 else str(style.OUT / "interaction.pdf")
colours = style.apply(palette)

seeds = json.loads((style.TABLES / "seed_variance.json").read_text())["cells"]
dual = json.loads((style.TABLES / "fid_dual_2048.json").read_text())["cells"]
dual |= json.loads((style.TABLES / "fid_dual_d3pm.json").read_text())["cells"]

gens = [("transformer", "AR"), ("maskgit", "MaskGIT"), ("d3pm", "D3PM")]
quants = ["VQ", "LFQ", "FSQ"]
x = range(len(gens))
offset = {"VQ": -0.06, "LFQ": 0.0, "FSQ": 0.06}

fig, axes = plt.subplots(1, 2, figsize=(style.TEXT_WIDTH, 2.25), gridspec_kw={"wspace": 0.42})

ax = axes[0]
for q in quants:
    cells = [seeds[f"chestmnist_{q}1024_{g}"] for g, _ in gens]
    xs = [i + offset[q] for i in x]
    ax.plot(xs, [c["mean"] for c in cells], color=colours[q], lw=1.4, zorder=2)
    ax.errorbar(xs, [c["mean"] for c in cells], yerr=[c["std"] for c in cells], fmt=style.MARKERS[q],
                color=colours[q], ms=5, mec="white", mew=0.8, elinewidth=1.0, capsize=0, zorder=3)
ax.set_yscale("log")
ax.set_ylim(0.3, 2.8)
ax.yaxis.set_major_locator(FixedLocator([0.3, 0.5, 1, 2]))
ax.yaxis.set_major_formatter(lambda v, _: f"{v:g}")
ax.yaxis.set_minor_formatter(NullFormatter())
ax.set_ylabel("FID-192 (mean $\\pm$ SD, 3 seeds)")
ax.set_title("(a) FID-192, three training seeds", loc="left")

ax = axes[1]
for q in quants:
    ys = [dual[f"{q}1024_{g}"]["fid2048"] for g, _ in gens]
    ax.plot(x, ys, color=colours[q], lw=1.4, marker=style.MARKERS[q], ms=5, mec="white", mew=0.8)
ax.set_yscale("log")
ax.set_ylim(30, 330)
ax.yaxis.set_major_locator(FixedLocator([30, 50, 100, 200, 300]))
ax.yaxis.set_major_formatter(lambda v, _: f"{v:g}")
ax.yaxis.set_minor_formatter(NullFormatter())
ax.set_ylabel("FID-2048 (single pass)")
ax.set_title("(b) FID-2048, same generated samples", loc="left")

for ax in axes:
    ax.set_xticks(list(x), [name for _, name in gens])
    ax.set_xlim(-0.35, len(gens) - 0.65)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", length=0, pad=4, colors=style.INK)

handles = [Line2D([], [], color=colours[q], marker=style.MARKERS[q], lw=1.4, ms=5, mec="white", mew=0.8,
                      label=f"{q}-1024") for q in quants]
fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.07), handlelength=1.8,
           columnspacing=1.6)
fig.savefig(out)
fig.savefig(str(out).replace(".pdf", ".png"))
print(out)
