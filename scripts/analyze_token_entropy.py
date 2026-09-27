"""Per-position entropy and codebook-usage analysis on tokenized ChestMNIST.

Reads chestmnist_{tok}/test.npz from medtokenizers and produces:
- Per-position Shannon entropy heatmaps (8×8) per discrete tokenizer
- Per-tokenizer codebook utilization histograms
- Aggregate stats (overall entropy, max possible entropy, normalised entropy)
- Latent magnitude/std for each continuous tokenizer

Outputs PNGs to figures/inputs/token_entropy/ and a JSON of summary stats to
results/tables/token_entropy_summary.json.
"""

import json
import os
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paths import FIGURE_INPUTS, MEDTOKENIZERS_ROOT, TABLES

TOKENIZED = Path(
    os.environ.get("TOKENIZED_ROOT", MEDTOKENIZERS_ROOT / "checkpoints" / "medmnist" / "tokenized")
)
OUT_FIG = FIGURE_INPUTS / "token_entropy"
OUT_JSON = TABLES / "token_entropy_summary.json"
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

DISCRETE = [
    ("VQ1024", 1024),
    ("VQ2048", 2048),
    ("VQ4096", 4096),
    ("LFQ1024", 1024),
    ("LFQ2048", 2048),
    ("LFQ", 4096),  # legacy: chestmnist_LFQ ↔ vocab 4096
    ("FSQ1024", 1024),
    ("FSQ2048", 2048),
    ("FSQ4096", 4096),
]
CONTINUOUS = ["VAEc1", "VAEc2", "VAEc4", "VAEc8", "VAEc16", "VAE", "VAE1e5", "VAE1e7", "AEdisc"]


def shannon_entropy(probs: np.ndarray, eps: float = 1e-12) -> float:
    p = probs[probs > eps]
    return float(-(p * np.log2(p)).sum())


def analyze_discrete(name: str, vocab: int) -> dict:
    test = np.load(TOKENIZED / f"chestmnist_{name}" / "test.npz")
    codes = test["codes"].astype(np.int64)  # (N, H, W)
    n, h, w = codes.shape

    # Per-position entropy
    per_pos = np.zeros((h, w), dtype=np.float64)
    for i in range(h):
        for j in range(w):
            counts = np.bincount(codes[:, i, j], minlength=vocab)
            probs = counts / counts.sum()
            per_pos[i, j] = shannon_entropy(probs)

    # Aggregate codebook usage
    flat = codes.reshape(-1)
    counts_all = np.bincount(flat, minlength=vocab)
    probs_all = counts_all / counts_all.sum()
    used = int((counts_all > 0).sum())
    overall_entropy = shannon_entropy(probs_all)
    max_entropy = float(np.log2(vocab))

    # Heatmap figure. constrained_layout reserves space for the title and
    # colorbar so the (short) title cannot overrun the colorbar; the panel
    # tokenizer name and metric are also given by the LaTeX caption/labels.
    fig, ax = plt.subplots(1, 1, figsize=(3.4, 3.2), constrained_layout=True)
    im = ax.imshow(per_pos, cmap="viridis", vmin=0, vmax=max_entropy)
    ax.set_title(f"{name} entropy", fontsize=10)
    ax.set_xlabel("col")
    ax.set_ylabel("row")
    fig.colorbar(im, ax=ax, label="bits", shrink=0.85)
    fig.savefig(OUT_FIG / f"per_pos_entropy_{name}.png", dpi=150)
    plt.close(fig)

    # Codebook usage histogram (sorted)
    fig, ax = plt.subplots(1, 1, figsize=(4.0, 2.5))
    sorted_counts = np.sort(counts_all)[::-1]
    ax.semilogy(sorted_counts, lw=1)
    ax.set_xlabel("Codebook index (sorted by frequency)")
    ax.set_ylabel("Count")
    ax.set_title(f"{name} codebook usage")
    fig.tight_layout()
    fig.savefig(OUT_FIG / f"codebook_usage_{name}.png", dpi=150)
    plt.close(fig)

    return {
        "tokenizer": name,
        "vocab": vocab,
        "n_samples": int(n),
        "spatial": [int(h), int(w)],
        "codebook_used": used,
        "codebook_utilization": used / vocab,
        "overall_entropy_bits": overall_entropy,
        "max_entropy_bits": max_entropy,
        "normalised_entropy": overall_entropy / max_entropy,
        "per_pos_entropy_min": float(per_pos.min()),
        "per_pos_entropy_max": float(per_pos.max()),
        "per_pos_entropy_mean": float(per_pos.mean()),
        "per_pos_entropy_std": float(per_pos.std()),
    }


def analyze_continuous(name: str) -> dict:
    path = TOKENIZED / f"chestmnist_{name}" / "test.npz"
    if not path.exists():
        return {"tokenizer": name, "error": "metadata not found"}
    test = np.load(path)
    key = "latents" if "latents" in test else "codes"
    z = test[key].astype(np.float32)  # (N, C, H, W)
    n = z.shape[0]
    C = z.shape[1] if z.ndim == 4 else 1
    return {
        "tokenizer": name,
        "n_samples": int(n),
        "shape": list(z.shape[1:]),
        "channels": int(C),
        "mean": float(z.mean()),
        "std": float(z.std()),
        "min": float(z.min()),
        "max": float(z.max()),
        "abs_mean": float(np.abs(z).mean()),
    }


def main():
    summary = {"discrete": [], "continuous": []}

    for name, vocab in DISCRETE:
        try:
            r = analyze_discrete(name, vocab)
            summary["discrete"].append(r)
            print(
                f"  {name:10s}  vocab={vocab:>5}  util={r['codebook_utilization'] * 100:5.1f}%  "
                f"H={r['overall_entropy_bits']:5.2f}/{r['max_entropy_bits']:.0f} "
                f"({r['normalised_entropy']:.3f} norm)"
            )
        except Exception as e:
            print(f"  {name}: FAIL {e}")
            summary["discrete"].append({"tokenizer": name, "error": str(e)})

    for name in CONTINUOUS:
        try:
            r = analyze_continuous(name)
            summary["continuous"].append(r)
            if "error" not in r:
                print(
                    f"  {name:8s}  shape={r['shape']}  mean={r['mean']:+.3f}  std={r['std']:.3f}  "
                    f"|z|_mean={r['abs_mean']:.3f}"
                )
            else:
                print(f"  {name}: SKIP ({r['error']})")
        except Exception as e:
            print(f"  {name}: FAIL {e}")

    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {OUT_JSON}")
    print(f"Wrote heatmap PNGs to {OUT_FIG}/")


if __name__ == "__main__":
    main()
