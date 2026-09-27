import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator
from scipy.stats import pearsonr

import style

colours = style.apply("oxide")
CKPT = style.RESULTS / "chestmnist_factorial"
mem = json.loads((CKPT / "memorization.json").read_text())
fid = json.loads((CKPT / "eval_results.json").read_text())

DISCRETE_GENS = ["transformer", "maskgit", "flow", "d3pm", "sedd", "bayesian_flow"]
CONTINUOUS = ["VAEc1", "VAEc2", "VAEc4", "VAEc8", "VAEc16", "VAE1e5", "VAE1e7", "AEdisc"]
cells = [(f"{q}{v}", g, q) for q in ("VQ", "LFQ", "FSQ") for v in (1024, 2048, 4096) for g in DISCRETE_GENS]
cells += [(t, g, "Continuous") for t in CONTINUOUS for g in ("diffusion", "continuous_flow")]
rows = [(fam, round(fid[f"chestmnist_{t}_{g}"]["fid"], 2), mem[f"chestmnist_{t}_{g}"]["memorization_ratio"],
         mem[f"chestmnist_{t}_{g}"]["auth_pct"]) for t, g, fam in cells]
assert len(rows) == 70

fig, axes = plt.subplots(1, 2, figsize=(style.TEXT_WIDTH, 2.3), gridspec_kw={"width_ratios": [1.7, 1], "wspace": 0.3})

ax = axes[0]
lf = np.array([(f, r) for fam, f, r, _ in rows if fam in ("LFQ", "FSQ")])
slope, intercept = np.polyfit(lf[:, 0], lf[:, 1], 1)
r = pearsonr(lf[:, 0], lf[:, 1])[0]
xs = np.geomspace(0.3, 9, 200)
ax.plot(xs, intercept + slope * xs, color=style.MUTED, lw=0.8, ls=(0, (1.5, 1.5)), zorder=1)
ax.text(0.03, 0.95, f"dotted: linear fit to the 36 LFQ and FSQ cells, $r = {r:.3f}$", transform=ax.transAxes,
        ha="left", va="top", fontsize=6.5, color=style.MUTED)
ax.axhline(1, color=style.INK, lw=0.7, ls=(0, (4, 2)), zorder=1)
ax.annotate("$R = 1$: training-set self-distance", (10.5, 1), xytext=(0, -3), textcoords="offset points",
            fontsize=6.5, color=style.INK, ha="right", va="top")
for fam in ("Continuous", "LFQ", "FSQ", "VQ"):
    pts = [(f, r_) for fam_, f, r_, _ in rows if fam_ == fam]
    ax.plot(*zip(*pts), ls="none", marker=style.MARKERS[fam], ms=4.2, color=colours[fam], mec="white",
            mew=0.5, alpha=0.95, zorder=3)
ax.set_xscale("log")
ax.set_xlim(0.05, 11)
ax.set_ylim(0.55, 7.3)
ax.xaxis.set_major_locator(FixedLocator([0.1, 0.3, 1, 3, 10]))
ax.xaxis.set_major_formatter(lambda v, _: f"{v:g}")
ax.set_xlabel("FID-192 (default sampling, 10K samples)")
ax.set_ylabel("Nearest-neighbour ratio $R$")
ax.set_title("(a) Distance to training set against FID, 70 cells", loc="left")
ax.grid(which="minor", visible=False)

ax = axes[1]
rng = np.random.default_rng(0)
order = ["Continuous", "LFQ", "FSQ", "VQ"]
for i, fam in enumerate(order):
    vals = np.array([a for fam_, _, _, a in rows if fam_ == fam])
    ax.plot(i + rng.uniform(-0.16, 0.16, len(vals)), vals, ls="none", marker=style.MARKERS[fam], ms=3.6,
            color=colours[fam], mec="white", mew=0.4, alpha=0.9)
    ax.plot([i - 0.26, i + 0.26], [np.median(vals)] * 2, color=style.INK, lw=1.1)
ax.set_xticks(range(len(order)), ["VAE/AE", "LFQ", "FSQ", "VQ"])
ax.tick_params(axis="x", length=0, colors=style.INK)
ax.set_xlim(-0.6, len(order) - 0.4)
ax.set_ylim(-0.005, 0.12)
ax.set_ylabel("AuthPct")
ax.set_title("(b) AuthPct by tokenizer family", loc="left")
ax.grid(axis="x", visible=False)

handles = [Line2D([], [], ls="none", marker=style.MARKERS[f], ms=4.5, color=colours[f], mec="white",
                  label="VAE / AE" if f == "Continuous" else f) for f in order]
handles.append(Line2D([], [], color=style.INK, lw=1.1, label="median"))
fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.08), handletextpad=0.2,
           columnspacing=1.2)
out = style.OUT / "memorization_scatter.pdf"
fig.savefig(out)
fig.savefig(out.with_suffix(".png"))
print(out, f"r={r:.3f}", f"minR={min(x[2] for x in rows):.3f}", f"maxA={max(x[3] for x in rows):.3f}")
