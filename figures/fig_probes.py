import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import style
import tiles

colours = style.apply("oxide")


def row_label(ax, text, sub=None):
    ax.set_axis_off()
    ax.text(0.95, 0.56 if sub else 0.5, text, ha="right", va="center", fontsize=7.5, transform=ax.transAxes)
    if sub:
        ax.text(0.95, 0.36, sub, ha="right", va="center", fontsize=6.5, color=style.MUTED, transform=ax.transAxes)


def show(ax, img, **kw):
    ax.imshow(img, interpolation="nearest", **({"cmap": "gray", "vmin": 0, "vmax": 1} | kw))
    ax.set_axis_off()


def inpaint():
    rgb, rows, cols = tiles.grid(style.INPUTS / "inpaint_demo_original.png", x_min=250)
    labels = [("Test image", None), ("Model input", "centre 4 × 4 tokens masked"), ("Inpainted", "LFQ-1024 + MaskGIT")]
    fig = plt.figure(figsize=(style.TEXT_WIDTH, 2.2))
    gs = fig.add_gridspec(3, 7, width_ratios=[1.55] + [1] * 6, wspace=0.04, hspace=0.06, left=0, right=1, top=1,
                          bottom=0)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            show(fig.add_subplot(gs[i, j + 1]), tiles.native(rgb, r, c))
        row_label(fig.add_subplot(gs[i, 0]), *labels[i])
    out = style.OUT / "inpaint_demo.pdf"
    fig.savefig(out, dpi=300)
    fig.savefig(out.with_suffix(".png"), dpi=300)


def anomaly():
    rgb = np.asarray(plt.imread(style.INPUTS / "anomaly_demo_original.png"))[:, :, :3].astype(float)
    _, _, cols = tiles.grid(style.INPUTS / "anomaly_demo_original.png", x_min=280)
    rows = [(29 + 204 * k, 206 + 204 * k) for k in range(6)]
    real, inserted, counterfactual, _, pred, truth = [[tiles.native(rgb, r, c, grey=False) for c in cols]
                                                         for r in rows]
    fig = plt.figure(figsize=(style.TEXT_WIDTH, 3.05))
    gs = fig.add_gridspec(4, 7, width_ratios=[1.55] + [1] * 6, wspace=0.04, hspace=0.06, left=0, right=1, top=0.93,
                          bottom=0)
    labels = [("Test image", None), ("Input", "bright ellipsoid added"), ("Counterfactual", "affected tokens inpainted"),
              ("Residual mask", "prediction vs. truth")]
    for j in range(len(cols)):
        show(fig.add_subplot(gs[0, j + 1]), real[j].mean(-1))
        show(fig.add_subplot(gs[1, j + 1]), inserted[j].mean(-1))
        show(fig.add_subplot(gs[2, j + 1]), counterfactual[j].mean(-1))
        ax = fig.add_subplot(gs[3, j + 1])
        show(ax, real[j].mean(-1), alpha=0.6)
        pm = (pred[j][..., 0] - pred[j][..., 1]) > 0.3
        tm = (truth[j][..., 1] - truth[j][..., 0]) > 0.15
        ax.contour(tm, levels=[0.5], colors=[colours["LFQ"]], linewidths=0.9)
        ax.contour(pm, levels=[0.5], colors=[colours["VQ"]], linewidths=0.9, linestyles=[(0, (2, 1))])
    for i, lab in enumerate(labels):
        row_label(fig.add_subplot(gs[i, 0]), *lab)
    handles = [Line2D([], [], color=colours["VQ"], lw=0.9, ls=(0, (2, 1)), label="predicted from residual"),
               Line2D([], [], color=colours["LFQ"], lw=0.9, label="inserted ellipsoid")]
    fig.legend(handles=handles, loc="upper right", ncol=2, bbox_to_anchor=(1, 1.0), handlelength=2)
    out = style.OUT / "anomaly_demo.pdf"
    fig.savefig(out, dpi=300)
    fig.savefig(out.with_suffix(".png"), dpi=300)


inpaint()
anomaly()
print("ok")
