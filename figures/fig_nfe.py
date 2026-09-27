import json

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullFormatter

import style

colours = style.apply("oxide")
CKPT = style.RESULTS / "chestmnist_factorial"
validation = json.loads((CKPT / "sweep_validation.json").read_text())
selected = json.loads((style.TABLES / "hp_revalidation.json").read_text())["cells"]

NFE_KEY = {"d3pm": "num_timesteps", "sedd": "num_timesteps", "bayesian_flow": "num_steps",
           "maskgit": "num_steps", "flow": "num_euler_steps"}
DEFAULT_NFE = {"transformer": 64, "maskgit": 12, "flow": 100, "d3pm": 1000, "sedd": 1000, "bayesian_flow": 1000}
LABEL = {"transformer": "AR", "maskgit": "MaskGIT", "flow": "DFM", "d3pm": "D3PM", "sedd": "SEDD",
         "bayesian_flow": "BFN"}
FOCUS = {"d3pm", "sedd"}


def colour(model):
    return colours["LFQ"] if model in FOCUS else style.MUTED


fig, ax = plt.subplots(figsize=(style.TEXT_WIDTH * 0.72, 2.45))

for model, ls in (("sedd", (0, (4, 1.5))), ("d3pm", "-")):
    pts = sorted((v["hp"]["num_timesteps"], v["fid"]) for v in validation.values()
                 if v["run"] == f"chestmnist_LFQ1024_{model}")
    ax.plot(*zip(*pts), color=colour(model), lw=1.3, ls=ls, zorder=2)
    ax.plot(*zip(*pts), ls="none", marker="o", ms=3, color=colour(model), mec="white", mew=0.5, zorder=3)

for model, nfe in DEFAULT_NFE.items():
    cell = selected[f"LFQ1024_{model}"]
    x0, y0 = nfe, cell["default_fid"]
    x1 = cell["selected_hp"].get(NFE_KEY.get(model), nfe)
    y1 = cell["test_fid_10k"]
    c = colour(model)
    if model not in FOCUS and (x0, round(y0, 2)) != (x1, round(y1, 2)):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=c, lw=0.7,
                    shrinkA=4.5, shrinkB=5.5, mutation_scale=6), zorder=2)
    ax.plot([x0], [y0], ls="none", marker="o", ms=6.5, mfc="white", mec=c, mew=1.0, zorder=4)
    ax.plot([x1], [y1], ls="none", marker="*", ms=9, color=c, mec="white", mew=0.6, zorder=5)

labels = {"transformer": (64, 0.33, (0, 7), "center"), "maskgit": (12, 1.91, (0, 7), "center"),
          "flow": (100, 1.77, (-8, -1), "right"), "bayesian_flow": (1000, 2.27, (0, 7), "center"),
          "d3pm": (100, 0.0898, (-6, -1), "right"), "sedd": (500, 0.098, (6, -4), "left")}
for model, (x, y, off, ha) in labels.items():
    ax.annotate(LABEL[model], (x, y), xytext=off, textcoords="offset points", ha=ha, va="center",
                fontsize=7, color=colour(model) if model in FOCUS else style.INK)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(2.5, 2000)
ax.set_ylim(0.07, 3.4)
ax.yaxis.set_major_locator(FixedLocator([0.1, 0.2, 0.5, 1, 2]))
ax.yaxis.set_major_formatter(lambda v, _: f"{v:g}")
ax.yaxis.set_minor_formatter(NullFormatter())
ax.xaxis.set_major_formatter(lambda v, _: f"{v:g}")
ax.set_xlabel("Function evaluations per sample")
ax.set_ylabel("FID-192 on LFQ-1024 (10K test)")
ax.grid(which="minor", visible=False)

key = [Line2D([], [], ls="none", marker="o", ms=6.5, mfc="white", mec=style.INK, mew=1.0, label="default"),
       Line2D([], [], ls="none", marker="*", ms=9, color=style.INK, mec="white", label="validation-selected"),
       Line2D([], [], color=colours["LFQ"], lw=1.3, marker="o", ms=3, mec="white", label="step sweep, T = 0.7")]
ax.legend(handles=key, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.14), handletextpad=0.3,
          columnspacing=1.0)
out = style.OUT / "nfe_vs_fid.pdf"
fig.savefig(out)
fig.savefig(out.with_suffix(".png"))
print(out)
