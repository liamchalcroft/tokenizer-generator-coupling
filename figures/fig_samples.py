import matplotlib.pyplot as plt

import style
import tiles

style.apply("oxide")
rgb, rows, cols = tiles.grid(style.INPUTS / "sample_grid_original.png", x_min=250)
assert len(rows) == 6 and len(cols) == 8

labels = [("Real test images", None), ("LFQ-1024 + AR", "0.33"), ("LFQ-1024 + D3PM", "0.44"),
          ("VAE-c2 + LDM", "0.14"), ("VAE-c16 + RF", "0.11"), ("AE + RF", "0.07")]

fig = plt.figure(figsize=(style.TEXT_WIDTH, 3.62))
gs = fig.add_gridspec(7, 9, width_ratios=[1.75] + [1] * 8, height_ratios=[1, 0.14] + [1] * 5, wspace=0.04, hspace=0.06,
                      left=0, right=1, top=1, bottom=0)
for i, r in enumerate(rows):
    g = i if i == 0 else i + 1
    for j, c in enumerate(cols):
        ax = fig.add_subplot(gs[g, j + 1])
        ax.imshow(tiles.native(rgb, r, c), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_axis_off()
    ax = fig.add_subplot(gs[g, 0])
    ax.set_axis_off()
    name, fid = labels[i]
    ax.text(0.94, 0.56 if fid else 0.5, name, ha="right", va="center", fontsize=7.5, transform=ax.transAxes)
    if fid:
        ax.text(0.94, 0.36, f"FID-192 {fid}", ha="right", va="center", fontsize=6.5, color=style.MUTED,
                transform=ax.transAxes)

rule = fig.add_subplot(gs[1, 1:])
rule.set_axis_off()
rule.axhline(0.5, color=style.INK, lw=0.6)
rule.set_ylim(0, 1)
out = style.OUT / "sample_grid.pdf"
fig.savefig(out, dpi=300)
fig.savefig(out.with_suffix(".png"), dpi=300)
print(out)
