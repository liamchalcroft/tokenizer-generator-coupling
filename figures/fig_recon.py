import json

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

import style

colours = style.apply("oxide")
EVAL = json.loads((style.RESULTS / "chestmnist_factorial/eval_results.json").read_text())

PSNR = {"VQ1024": 28.0, "VQ2048": 29.3, "VQ4096": 29.4, "LFQ1024": 28.0, "LFQ2048": 28.1, "LFQ4096": 28.6,
        "FSQ1024": 29.4, "FSQ2048": 29.6, "FSQ4096": 29.7, "VAEc1": 24.6, "VAEc2": 29.3, "VAEc4": 32.4,
        "VAEc8": 34.4, "VAEc16": 32.3, "AEdisc": 32.3, "VAE1e5": 28.2, "VAE1e7": 32.0}
LABEL = {"VQ1024": "VQ-1024", "VQ4096": "VQ-4096", "LFQ1024": "LFQ-1024", "FSQ4096": "FSQ-4096",
         "VAEc1": "c = 1", "VAEc2": "c = 2", "VAEc4": "c = 4", "VAEc8": "c = 8", "VAEc16": "c = 16",
         "AEdisc": "AE", "VAE1e5": r"$10^{-5}$", "VAEc4_kl": r"$10^{-6}$", "VAE1e7": r"$10^{-7}$"}
TUNED = {"LFQ1024": 0.0898, "FSQ1024": 0.13}
GENS = ["transformer", "maskgit", "flow", "d3pm", "sedd", "bayesian_flow", "diffusion", "continuous_flow"]

DISCRETE = [f"{q}{v}" for q in ("VQ", "LFQ", "FSQ") for v in (1024, 2048, 4096)]
CHANNEL = ["VAEc1", "VAEc2", "VAEc4", "VAEc8", "VAEc16"]
KL = ["VAE1e5", "VAEc4", "VAE1e7", "AEdisc"]


def best(tok):
    return min(EVAL[f"chestmnist_{tok}_{g}"]["fid"] for g in GENS if f"chestmnist_{tok}_{g}" in EVAL)


def fam(tok):
    for q in ("VQ", "LFQ", "FSQ"):
        if tok.startswith(q):
            return q
    return "Continuous"


fig, axes = plt.subplots(1, 3, figsize=(style.TEXT_WIDTH, 1.95), gridspec_kw={"width_ratios": [1.25, 1, 1],
                                                                                  "wspace": 0.42})
panels = [(axes[0], DISCRETE, "(a) Discrete"), (axes[1], CHANNEL, "(b) VAE channels"), (axes[2], KL, "(c) KL weight, c = 4")]
offsets = {"VQ1024": (5, 0), "VQ4096": (0, -8), "LFQ1024": (5, -3), "FSQ4096": (5, -1),
           "VAEc1": (5, 0), "VAEc2": (0, 7), "VAEc4": (5, 2), "VAEc8": (-5, 0), "VAEc16": (5, -3),
           "VAE1e5": (4, 0), "VAEc4_kl": (-3, 5), "VAE1e7": (-4, 0), "AEdisc": (5, 0)}
ha = {"VAEc8": "right", "VAE1e7": "right", "VAEc4_kl": "right", "VAEc2": "center", "VQ4096": "center"}

for ax, toks, title in panels:
    xs, ys = [PSNR[t] for t in toks], [best(t) for t in toks]
    rho = spearmanr(xs, ys)[0]
    for t, x, y in zip(toks, xs, ys):
        f = fam(t)
        marker = "D" if t == "AEdisc" else style.MARKERS[f] if f != "Continuous" else "o"
        ax.plot([x], [y], ls="none", marker=marker, ms=5.2, color=colours[f], mec="white", mew=0.7, zorder=3)
        key = "VAEc4_kl" if (t == "VAEc4" and toks is KL) else t
        if key in LABEL:
            dx, dy = offsets[key]
            ax.annotate(LABEL[key], (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=6.5,
                        ha=ha.get(key, "left"), va="center", color=style.INK)
    if toks is DISCRETE:
        for t, v in TUNED.items():
            ax.plot([PSNR[t]], [v], ls="none", marker=style.MARKERS[fam(t)], ms=5.2, mfc="white",
                    mec=colours[fam(t)], mew=0.9, zorder=3)
    ax.set_title(rf"{title}, $\rho = {rho:+.2f}$", loc="left")
    ax.set_xlabel("Reconstruction PSNR (dB)")
    ax.margins(x=0.12, y=0.14)
axes[0].set_ylabel("Best observed FID-192")
axes[0].set_ylim(bottom=0)
axes[0].set_xlim(27.8, 30.25)

handles = [Line2D([], [], ls="none", marker=style.MARKERS[q], ms=5, color=colours[q], mec="white", label=q)
           for q in ("VQ", "LFQ", "FSQ")]
handles += [Line2D([], [], ls="none", marker="o", ms=5, color=colours["Continuous"], mec="white", label="VAE"),
            Line2D([], [], ls="none", marker="D", ms=4.5, color=colours["Continuous"], mec="white", label="AE"),
            Line2D([], [], ls="none", marker="o", ms=5, mfc="white", mec=style.INK, mew=0.9,
                   label="tuned D3PM (validation-selected)")]
fig.legend(handles=handles, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.1), handletextpad=0.2,
           columnspacing=1.0)
out = style.OUT / "recon_vs_gen.pdf"
fig.savefig(out)
fig.savefig(out.with_suffix(".png"))
print(out)
