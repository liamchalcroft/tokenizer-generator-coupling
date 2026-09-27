import matplotlib.pyplot as plt
import numpy as np

import style

style.apply("oxide")
pairs = np.load(style.RESULTS / "samples" / "memorization_gallery_pairs.npz")
labels = {"chestmnist_AEdisc_continuous_flow": ("AE + RF", "0.07"), "chestmnist_LFQ1024_transformer": ("LFQ-1024 + AR", "0.33"),
          "chestmnist_LFQ1024_d3pm": ("LFQ-1024 + D3PM", "0.44"), "chestmnist_VQ1024_maskgit": ("VQ-1024 + MaskGIT", "1.86"),
          "chestmnist_FSQ4096_bayesian_flow": ("FSQ-4096 + BFN", "8.46")}
cells = pairs["cells"].tolist()
assert cells == list(labels)

fig = plt.figure(figsize=(style.TEXT_WIDTH, 3.3))
ratios = [1.75, 1, 1, 0.16, 1, 1, 0.16, 1, 1]
gs = fig.add_gridspec(6, len(ratios), width_ratios=ratios, height_ratios=[0.22] + [1] * 5, wspace=0.03,
                      hspace=0.07, left=0, right=1, top=1, bottom=0)
slots = [1, 2, 4, 5, 7, 8]
for k, slot in enumerate(slots):
    ax = fig.add_subplot(gs[0, slot])
    ax.set_axis_off()
    ax.text(0.5, 0.2, "generated" if k % 2 == 0 else "nearest training", ha="center", va="bottom", fontsize=6.5,
            color=style.MUTED, transform=ax.transAxes)
for i, cell in enumerate(cells):
    for k in range(3):
        for slot, img in ((slots[2 * k], pairs["generated"][i, k]), (slots[2 * k + 1], pairs["train"][i, k])):
            ax = fig.add_subplot(gs[i + 1, slot])
            ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            ax.set_axis_off()
    ax = fig.add_subplot(gs[i + 1, 0])
    ax.set_axis_off()
    name, fid = labels[cell]
    ax.text(0.95, 0.56, name, ha="right", va="center", fontsize=7.5, transform=ax.transAxes)
    ax.text(0.95, 0.36, f"FID-192 {fid}", ha="right", va="center", fontsize=6.5, color=style.MUTED,
            transform=ax.transAxes)
out = style.OUT / "memorization_gallery.pdf"
fig.savefig(out, dpi=300)
fig.savefig(out.with_suffix(".png"), dpi=300)
print(out)
