#!/usr/bin/env python
"""Exact per-position 8 x 8 token entropy maps for the vocabulary-1024 tokenizers.

analyze_token_entropy.py renders these maps as PNGs but never stored the arrays, so
results/tables/derived_per_position_entropy_1024.json had to be recovered from the
colour-mapped renders. This recomputes them from the tokenized test split with the
same estimator (analyze_token_entropy.shannon_entropy) and reports how far the
recovered values sit from the exact ones.

Output: results/tables/token_entropy_per_position_1024.json
"""

import json

import numpy as np

from analyze_token_entropy import TOKENIZED, shannon_entropy
from paths import TABLES

TOKENIZERS = {"VQ": "VQ1024", "LFQ": "LFQ1024", "FSQ": "FSQ1024"}
VOCAB = 1024


def per_position(name: str) -> np.ndarray:
    codes = np.load(TOKENIZED / f"chestmnist_{name}" / "test.npz")["codes"].astype(np.int64)
    _, h, w = codes.shape
    out = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            counts = np.bincount(codes[:, i, j], minlength=VOCAB)
            out[i, j] = shannon_entropy(counts / counts.sum())
    return out


def main():
    derived = json.loads((TABLES / "derived_per_position_entropy_1024.json").read_text())
    maps, check = {}, {}
    for key, name in TOKENIZERS.items():
        m = per_position(name)
        maps[key] = np.round(m, 4).tolist()
        diff = np.abs(m - np.asarray(derived["entropy_bits"][key]))
        check[key] = {
            "max_abs_diff_bits": float(diff.max()),
            "mean_abs_diff_bits": float(diff.mean()),
        }
        print(
            f"{name}: mean {m.mean():.3f} min {m.min():.3f} max {m.max():.3f} bits; "
            f"vs recovered max|diff| {diff.max():.3f}, mean {diff.mean():.3f}"
        )
    out = {
        "_description": (
            "Exact per-position Shannon entropy (bits, log2) of the 8 x 8 token grid on the "
            "ChestMNIST test split at vocabulary 1024, computed from the tokenized arrays "
            "by scripts/token_entropy_maps.py. Supersedes the colour-map recovery in "
            "derived_per_position_entropy_1024.json; the comparison block gives the "
            "recovery error."
        ),
        "vocab": VOCAB,
        "max_entropy_bits": float(np.log2(VOCAB)),
        "entropy_bits": maps,
        "vs_derived_colourmap_recovery": check,
    }
    path = TABLES / "token_entropy_per_position_1024.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
