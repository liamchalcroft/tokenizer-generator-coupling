import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import style

style.apply("oxide")
data = json.loads((style.TABLES / "derived_per_position_entropy_1024.json").read_text())["entropy_bits"]
cmap = LinearSegmentedColormap.from_list("ink", ["#F4F1EA", "#9DB3CC", "#2B5E93", "#12263F"])

fig, axes = plt.subplots(1, 3, figsize=(style.TEXT_WIDTH * 0.82, 1.62), gridspec_kw={"wspace": 0.08})
for ax, q in zip(axes, ("VQ", "LFQ", "FSQ")):
    grid = np.array(data[q])
    im = ax.imshow(grid, cmap=cmap, vmin=8.0, vmax=10.0, interpolation="nearest")
    ax.set_title(f"{q}-1024", fontsize=8)
    ax.set_xlabel(f"mean {grid.mean():.2f} bits", fontsize=7, color=style.MUTED, labelpad=3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.015, shrink=0.93, ticks=[8, 8.5, 9, 9.5, 10])
cbar.set_label("Entropy per position (bits)", fontsize=7)
cbar.outline.set_visible(False)
cbar.ax.tick_params(labelsize=6.5, length=2)
out = style.OUT / "token_entropy_1024.pdf"
fig.savefig(out)
fig.savefig(out.with_suffix(".png"))
print(out)
