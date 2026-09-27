#!/usr/bin/env python
"""Create a train-subsampled copy of a tokenized dataset directory.

For the training-set-size control: does VQ overtake LFQ at small training-set
size on ChestMNIST itself (matching the PneumoniaMNIST inversion)? We subsample the
train split to a target size, keeping val/test intact, so the only thing that
changes is how much training data the generator sees.

Slices the train `codes` array to N rows with a fixed RNG, writes a new directory
with the subsampled train.npz and val/test copied through, plus metadata.json.

Usage:
    python scripts/subsample_tokens.py --src .../tokenized/chestmnist_LFQ1024 \
        --n 4700 --dst .../tokenized/chestmnist_LFQ1024_n4700
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np


def load_codes(npz_path: Path):
    z = np.load(npz_path)
    for k in ("codes", "tokens", "train_tokens", "arr_0"):
        if k in z.files:
            return z[k], k
    if len(z.files) == 1:
        return z[z.files[0]], z.files[0]
    raise KeyError(f"{npz_path}: keys {z.files}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="source tokenized dir with train/val/test.npz")
    ap.add_argument("--n", type=int, required=True, help="target train size")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    train, key = load_codes(src / "train.npz")
    n = min(args.n, train.shape[0])
    idx = np.random.default_rng(args.seed).choice(train.shape[0], size=n, replace=False)
    idx.sort()
    np.savez_compressed(dst / "train.npz", **{key: train[idx]})
    print(f"train {train.shape[0]} -> {n} (key={key})")

    for split in ("val", "test"):
        if (src / f"{split}.npz").exists():
            shutil.copy2(src / f"{split}.npz", dst / f"{split}.npz")
    if (src / "metadata.json").exists():
        shutil.copy2(src / "metadata.json", dst / "metadata.json")
    print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
