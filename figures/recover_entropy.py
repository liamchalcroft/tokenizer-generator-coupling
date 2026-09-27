import json

import matplotlib
import numpy as np
from PIL import Image

import style

lut = matplotlib.colormaps["viridis"](np.linspace(0, 1, 1024))[:, :3]
out = {}
for q in ("VQ", "LFQ", "FSQ"):
    img = np.asarray(Image.open(style.INPUTS / "token_entropy" / f"per_pos_entropy_{q}1024.png").convert("RGB")) / 255
    h, w, _ = img.shape
    d = np.min(np.linalg.norm(img[:, :, None, :] - lut[None, None, ::16, :], axis=-1), axis=-1)
    mask = d < 0.02
    cols = np.where(mask[:, : int(w * 0.78)].sum(0) > h * 0.3)[0]
    rows = np.where(mask[:, cols.min():cols.max()].sum(1) > (cols.max() - cols.min()) * 0.5)[0]
    x0, x1, y0, y1 = cols.min(), cols.max(), rows.min(), rows.max()
    grid = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            cy = int(y0 + (i + 0.5) * (y1 - y0) / 8)
            cx = int(x0 + (j + 0.5) * (x1 - x0) / 8)
            patch = img[cy - 3:cy + 4, cx - 3:cx + 4].reshape(-1, 3).mean(0)
            k = np.argmin(np.linalg.norm(lut - patch, axis=1))
            grid[i, j] = 10 * k / 1023
    out[q] = grid.round(3).tolist()
    print(q, (x0, x1, y0, y1), f"min {grid.min():.2f} max {grid.max():.2f} mean {grid.mean():.3f}")
(style.TABLES / "derived_per_position_entropy_1024.json").write_text(json.dumps(
    {"_description": "Per-position token entropy (bits) at vocabulary 1024, recovered by inverting the viridis "
     "colour map (range 0-10 bits) of figures/inputs/token_entropy/per_pos_entropy_*1024.png, because the source "
     "token arrays are not available locally. Resolution is limited by 8-bit colour, about 0.01 bits.",
     "entropy_bits": out}, indent=1))
