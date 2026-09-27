"""Generate a sample grid figure for the paper.

For each dataset, picks 8 samples from the trained generator cells in
GEN_CKPT_ROOT (which is dataset-aware via $MEDGEN_DATASET) and composes a
real-vs-generated grid.

  MEDGEN_DATASET=chestmnist     uv run python scripts/generate_paper_samples.py
  MEDGEN_DATASET=pneumoniamnist uv run python scripts/generate_paper_samples.py
  MEDGEN_DATASET=organamnist    uv run python scripts/generate_paper_samples.py

Output: figures/inputs/sample_grid_original.png for ChestMNIST, which
figures/fig_samples.py restyles, otherwise figures/inputs/sample_grid_{dataset}.png.
Use --plot_only after the first run to redraw labels/layout from the cached
image rows without resampling.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from evaluate_chestmnist_generation import (
    DATASET,
    GEN_CKPT_ROOT,
    RAW_DATA_PATH,
    decode_continuous_samples,
    decode_discrete_samples,
    generate_continuous_samples,
    generate_discrete_samples,
    load_gen_model,
    load_tokenizer_decoder,
)
from paths import FIGURE_INPUTS

OUT_NAME = "sample_grid_original.png" if DATASET == "chestmnist" else f"sample_grid_{DATASET}.png"
OUT = FIGURE_INPUTS / OUT_NAME
CACHE = FIGURE_INPUTS / f"sample_grid_{DATASET}_cache.pt"
OUT.parent.mkdir(parents=True, exist_ok=True)
N_PER_ROW = 8

# Cells to display, per-dataset. The names match what was trained: the
# ChestMNIST run used vocab-suffixed tokenizer names (LFQ1024 etc.) while
# the Pneumonia/Organa runs use the default save names (LFQ etc.).
CELLS_BY_DATASET = {
    "chestmnist": [
        ("LFQ-1024 + AR", "chestmnist_LFQ1024_transformer", "LFQ1024"),
        ("LFQ-1024 + D3PM", "chestmnist_LFQ1024_d3pm", "LFQ1024"),
        ("VAE-c2 + LDM", "chestmnist_VAEc2_diffusion", "VAEc2"),
        ("VAE-c16 + RF", "chestmnist_VAEc16_continuous_flow", "VAEc16"),
        ("AE + RF", "chestmnist_AEdisc_continuous_flow", "AEdisc"),
    ],
    "pneumoniamnist": [
        ("LFQ-1024 + AR", f"{DATASET}_LFQ_transformer", "LFQ"),
        ("LFQ-1024 + D3PM", f"{DATASET}_LFQ_d3pm", "LFQ"),
        ("FSQ-1024 + AR", f"{DATASET}_FSQ_transformer", "FSQ"),
        ("VQ-1024 + AR", f"{DATASET}_VQ_transformer", "VQ"),
        ("VAE-c2 + LDM", f"{DATASET}_VAE_diffusion", "VAE"),
        ("AE + RF", f"{DATASET}_AE_continuous_flow", "AE"),
    ],
    "organamnist": [
        ("LFQ-1024 + AR", f"{DATASET}_LFQ_transformer", "LFQ"),
        ("LFQ-1024 + D3PM", f"{DATASET}_LFQ_d3pm", "LFQ"),
        ("FSQ-1024 + AR", f"{DATASET}_FSQ_transformer", "FSQ"),
        ("VQ-1024 + AR", f"{DATASET}_VQ_transformer", "VQ"),
        ("VAE-c2 + LDM", f"{DATASET}_VAE_diffusion", "VAE"),
        ("AE + RF", f"{DATASET}_AE_continuous_flow", "AE"),
    ],
}
CELLS = CELLS_BY_DATASET.get(DATASET, CELLS_BY_DATASET["chestmnist"])


def to_uint8(x: torch.Tensor) -> np.ndarray:
    """(N, 1, H, W) float in [0, 1] -> (N, H, W) uint8."""
    x = x.detach().float().cpu().clamp(0, 1)
    if x.ndim == 4 and x.shape[1] == 1:
        x = x.squeeze(1)
    return (x * 255).numpy().astype(np.uint8)


def build_rows(n_per_row: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(123)

    # Real reference images
    raw = np.load(str(RAW_DATA_PATH))
    real = raw["test_images"][:n_per_row]  # (N, H, W) uint8 already

    rows = [("Real (reference)", real)]
    tokenizer_cache = {}

    for label, run_name, tok_type in CELLS:
        ckpt = GEN_CKPT_ROOT / run_name / "checkpoint_best.pt"
        if not ckpt.exists():
            print(f"SKIP {label}: no checkpoint at {ckpt}")
            continue
        print(f"Loading {label}...")
        gen_model, gen_args = load_gen_model(str(ckpt), device)

        if tok_type not in tokenizer_cache:
            tokenizer_cache[tok_type] = load_tokenizer_decoder(tok_type, device)
        tokenizer = tokenizer_cache[tok_type]

        is_discrete = gen_args["model"] in {
            "transformer",
            "maskgit",
            "flow",
            "d3pm",
            "sedd",
            "bayesian_flow",
        }
        if is_discrete:
            tokens = generate_discrete_samples(gen_model, gen_args, n_per_row, n_per_row, device)
            spatial_shape = gen_args.get("spatial_shape", None)
            imgs = decode_discrete_samples(tokenizer, tokens, spatial_shape, n_per_row, device)
        else:
            latents = generate_continuous_samples(gen_model, gen_args, n_per_row, n_per_row, device)
            imgs = decode_continuous_samples(tokenizer, latents, n_per_row, device)

        rows.append((label, to_uint8(imgs)))
        del gen_model
        torch.cuda.empty_cache()

    return rows


def plot_rows(rows, n_per_row: int, output: Path):
    # Compose figure
    n_rows = len(rows)
    fig, axes = plt.subplots(
        n_rows,
        n_per_row + 1,
        figsize=(1.0 * (n_per_row + 1), 1.0 * n_rows),
        gridspec_kw={"width_ratios": [1.4] + [1.0] * n_per_row},
    )
    for r, (label, imgs) in enumerate(rows):
        # Row label cell
        ax = axes[r, 0]
        ax.text(0.5, 0.5, label, ha="center", va="center", fontsize=8, transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        # Images
        for c in range(n_per_row):
            axc = axes[r, c + 1]
            axc.imshow(imgs[c], cmap="gray", vmin=0, vmax=255)
            axc.set_xticks([])
            axc.set_yticks([])
    fig.subplots_adjust(left=0.02, right=0.99, top=0.99, bottom=0.01, wspace=0.05, hspace=0.05)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output}")


def main():
    parser = argparse.ArgumentParser(description="Generate or plot the paper sample grid")
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--n_per_row", type=int, default=N_PER_ROW)
    parser.add_argument("--plot_only", action="store_true")
    parser.add_argument("--refresh_cache", action="store_true")
    args = parser.parse_args()

    if args.plot_only and not args.cache.exists():
        raise FileNotFoundError(f"{args.cache} does not exist; rerun without --plot_only first")
    if args.plot_only or (args.cache.exists() and not args.refresh_cache):
        rows = torch.load(args.cache, map_location="cpu", weights_only=False)
    else:
        rows = build_rows(args.n_per_row)
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(rows, args.cache)
        print(f"Wrote {args.cache}")
    plot_rows(rows, args.n_per_row, args.output)


if __name__ == "__main__":
    main()
