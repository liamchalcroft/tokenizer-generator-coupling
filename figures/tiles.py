import numpy as np
from PIL import Image


def bands(profile, thresh, min_len):
    on = profile > thresh
    out, start = [], None
    for i, v in enumerate(np.append(on, False)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_len:
                out.append((start, i))
            start = None
    return out


def grid(path, x_min=0, min_len=60, thresh=0.3):
    rgb = np.asarray(Image.open(path).convert("RGB")).astype(float) / 255
    dark = rgb.mean(-1)[:, x_min:] < 0.97
    cols = [(a + x_min, b + x_min) for a, b in bands(dark.mean(0), thresh, min_len)]
    span = rgb.mean(-1)[:, cols[0][0]:cols[-1][1]] < 0.995
    rows = bands(span.mean(1), thresh, min_len)
    return rgb, rows, cols


def native(rgb, r, c, size=64, inset=4, grey=True):
    tile = rgb[r[0] + inset:r[1] - inset, c[0] + inset:c[1] - inset]
    img = Image.fromarray((tile * 255).astype(np.uint8))
    img = img.resize((size, size), Image.BOX)
    arr = np.asarray(img).astype(float) / 255
    return arr.mean(-1) if grey else arr
