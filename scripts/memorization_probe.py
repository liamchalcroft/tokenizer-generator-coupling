"""Memorization probe across all (dataset, cell) combinations.

Uses the medlatents.evaluation.memorization library module so the same
analysis is reproducible by anyone using the library on their own
generators.

For each cell:
  1. Generate ``--num_samples`` samples from the trained checkpoint.
  2. Lift Inception pool3 features (192-d, matches FID-192).
  3. Cache training-set features (one per dataset).
  4. Compute memorization metrics (memorization ratio R, AuthPct, NN
     distance medians) using ``medlatents.evaluation.memorization``.
  5. Persist per-cell results to ``results/{dataset}_factorial/
     memorization.json``.
  6. Optionally save the top-k suspect (generated, training-NN) pairs
     for the qualitative gallery.

Usage::

    MEDGEN_DATASET=chestmnist uv run python scripts/memorization_probe.py \
        --num_samples 1000 \
        --filter LFQ1024_  # optional, restrict to a subset of cells

The script is dataset-aware via $MEDGEN_DATASET, mirroring the
evaluate_chestmnist_generation.py and generate_paper_samples.py pattern.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from medlatents.evaluation.memorization import (
    InceptionPool3FeatureExtractor,
    memorization_metrics,
    memorization_pairs,
)

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
    parse_run_name,
    prepare_for_metrics,
)
from paths import CHECKPOINTS, RESULTS

CACHE_DIR = CHECKPOINTS / f"{DATASET}_factorial" / "memorization_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = RESULTS / f"{DATASET}_factorial" / "memorization.json"


def get_train_features(extractor: InceptionPool3FeatureExtractor) -> torch.Tensor:
    cache = CACHE_DIR / "train_features.npy"
    if cache.exists():
        print(f"  loading cached training features from {cache}")
        return torch.from_numpy(np.load(cache))
    print(f"  computing training features for {DATASET}")
    raw = np.load(str(RAW_DATA_PATH))
    train = raw["train_images"]  # (N, H, W) uint8
    images = torch.from_numpy(train).unsqueeze(1).to(torch.uint8)  # (N, 1, H, W)
    feats = extractor.extract(images)
    np.save(cache, feats.numpy())
    return feats


@torch.inference_mode()
def generate_samples(model, gen_args: dict, num_samples: int, batch_size: int, device: str):
    is_discrete = gen_args["model"] in {
        "transformer",
        "maskgit",
        "flow",
        "d3pm",
        "sedd",
        "bayesian_flow",
    }
    if is_discrete:
        return generate_discrete_samples(model, gen_args, num_samples, batch_size, device)
    return generate_continuous_samples(model, gen_args, num_samples, batch_size, device)


@torch.inference_mode()
def decode(tokenizer, samples, gen_args, batch_size, device):
    is_discrete = gen_args["model"] in {
        "transformer",
        "maskgit",
        "flow",
        "d3pm",
        "sedd",
        "bayesian_flow",
    }
    if is_discrete:
        spatial_shape = gen_args.get("spatial_shape", None)
        return decode_discrete_samples(tokenizer, samples, spatial_shape, batch_size, device)
    return decode_continuous_samples(tokenizer, samples, batch_size, device)


def main():
    parser = argparse.ArgumentParser(description="Memorization probe for trained generators")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--filter", type=str, default=None, help="Substring filter for cell names (e.g. 'LFQ1024_')"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--top_k_pairs",
        type=int,
        default=8,
        help="Save top-k (gen, train) NN pairs per cell for the gallery",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Skip cells already in the output JSON"
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    print(f"Memorization probe -- DATASET={DATASET}")
    print(f"  GEN_CKPT_ROOT = {GEN_CKPT_ROOT}")
    print(f"  out  = {OUT_PATH}")

    extractor = InceptionPool3FeatureExtractor(device=args.device, batch_size=64)
    train_feats = get_train_features(extractor)
    print(f"  training features: {tuple(train_feats.shape)}")

    # Discover cells
    cells: list[tuple[str, str, str, str]] = []
    for entry in sorted(GEN_CKPT_ROOT.iterdir()):
        if not entry.is_dir():
            continue
        ckpt = entry / "checkpoint_best.pt"
        if not ckpt.exists():
            continue
        parsed = parse_run_name(entry.name)
        if parsed is None:
            continue
        tok_type, model_type = parsed
        if args.filter and args.filter not in entry.name:
            continue
        cells.append((entry.name, tok_type, model_type, str(ckpt)))
    print(f"  found {len(cells)} cells to probe")

    results: dict[str, dict] = {}
    if args.resume and OUT_PATH.exists():
        results = json.loads(OUT_PATH.read_text())
        print(f"  resumed {len(results)} existing entries")

    tokenizer_cache: dict[str, torch.nn.Module] = {}

    for cell_name, tok_type, model_type, ckpt_path in cells:
        if args.resume and cell_name in results and "memorization_ratio" in results[cell_name]:
            print(f"SKIP {cell_name} (already done)")
            continue
        print(f"\n=== {cell_name} ===")
        try:
            t0 = time.time()
            gen_model, gen_args = load_gen_model(ckpt_path, args.device)
            if tok_type not in tokenizer_cache:
                tokenizer_cache[tok_type] = load_tokenizer_decoder(tok_type, args.device)
            tokenizer = tokenizer_cache[tok_type]

            samples = generate_samples(
                gen_model, gen_args, args.num_samples, args.batch_size, args.device
            )
            images = decode(tokenizer, samples, gen_args, args.batch_size, args.device)
            images_uint8 = prepare_for_metrics(images)  # (N, 3, H, W) uint8

            gen_feats = extractor.extract(images_uint8)
            metrics = memorization_metrics(gen_feats, train_feats, device=args.device)
            metrics["tokenizer"] = tok_type
            metrics["model"] = model_type
            metrics["time_s"] = time.time() - t0
            metrics["num_samples"] = args.num_samples

            # Save top-k suspect pairs (just indices + distances; the gallery
            # script regenerates the actual images on demand).
            pairs = memorization_pairs(
                gen_feats, train_feats, top_k=args.top_k_pairs, device=args.device
            )
            metrics["top_k_pairs"] = [
                {"gen_idx": g, "train_idx": t, "distance": d} for g, t, d in pairs
            ]

            results[cell_name] = metrics
            print(
                f"  R={metrics['memorization_ratio']:.3f}  AuthPct={metrics['auth_pct']:.3f}  "
                f"med d_g→t={metrics['median_d_gen_to_train']:.3f}  "
                f"med d_t→t={metrics['median_d_train_to_train']:.3f}  "
                f"({metrics['time_s']:.0f}s)"
            )

            del gen_model, samples, images, images_uint8, gen_feats
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            results[cell_name] = {"error": str(e)}
            torch.cuda.empty_cache()

        # Atomic save after each cell (resumable).
        tmp = OUT_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(results, indent=2))
        tmp.replace(OUT_PATH)

    print(f"\nWrote {len(results)} results to {OUT_PATH}")


if __name__ == "__main__":
    main()
