"""Bootstrap CIs for FID-192 on ChestMNIST.

Two analyses:

  (1) Real-vs-real noise floor. Split the 22K-image test set into two
      random halves of size N each, compute FID-192(half_A, half_B),
      repeat B=200 times. This characterises the *estimator* variance
      of FID-192 on this domain at a given sample size --- no generation
      needed.

  (2) Real-vs-generated CI. For 3 representative cells (best, mid, worst
      by reported FID), generate 10K samples, extract pool3 features,
      bootstrap-resample (real_idx, gen_idx) of size N, recompute FID
      B=200 times, report 95% CI.

Together these justify the per-cell uncertainty band claimed in Table 3.

Outputs:
  results/tables/fid_noise_floor.json
  results/tables/fid_per_cell_ci.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from medlatents.evaluation import (
    InceptionPool3FeatureExtractor,
    bootstrap_fid_noise_floor,
    bootstrap_fid_real_vs_gen,
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
from paths import CHECKPOINTS, TABLES

OUT_DIR = TABLES
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE = CHECKPOINTS / f"{DATASET}_factorial" / "fid_bootstrap_cache"
CACHE.mkdir(parents=True, exist_ok=True)


# `fid_from_features`, `bootstrap_fid_noise_floor`, `bootstrap_fid_real_vs_gen`
# now live in `medlatents.evaluation.fid` --- this script just wires them up
# to the chestmnist eval pipeline.


def get_test_features(extractor: InceptionPool3FeatureExtractor) -> np.ndarray:
    cache = CACHE / "test_features.npy"
    if cache.exists():
        print(f"  loading cached test features from {cache}")
        return np.load(cache)
    print("  extracting test features")
    raw = np.load(str(RAW_DATA_PATH))
    test = raw["test_images"]  # (N, H, W) uint8
    images = torch.from_numpy(test).unsqueeze(1).to(torch.uint8)
    feats = extractor.extract(images).numpy()
    np.save(cache, feats)
    print(f"  saved {feats.shape} -> {cache}")
    return feats


# Real-vs-real noise floor is now a one-line library call:
# bootstrap_fid_noise_floor(features, n_per_half=[...], B=200, verbose=True)


@torch.inference_mode()
def get_gen_features(
    cell_name: str,
    ckpt_path: str,
    num_samples: int,
    batch_size: int,
    device: str,
    extractor: InceptionPool3FeatureExtractor,
    tokenizer_cache: dict,
) -> np.ndarray:
    cache = CACHE / f"gen_features_{cell_name}.npy"
    if cache.exists():
        print(f"  loading cached gen features from {cache.name}")
        return np.load(cache)
    print(f"  generating {num_samples} samples for {cell_name}")
    parsed = parse_run_name(cell_name)
    if parsed is None:
        raise ValueError(f"cannot parse {cell_name}")
    tok_type, model_type = parsed
    gen_model, gen_args = load_gen_model(ckpt_path, device)
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
        samples = generate_discrete_samples(gen_model, gen_args, num_samples, batch_size, device)
        spatial = gen_args.get("spatial_shape", None)
        images = decode_discrete_samples(tokenizer, samples, spatial, batch_size, device)
    else:
        samples = generate_continuous_samples(gen_model, gen_args, num_samples, batch_size, device)
        images = decode_continuous_samples(tokenizer, samples, batch_size, device)
    images_uint8 = prepare_for_metrics(images)
    feats = extractor.extract(images_uint8).numpy()
    np.save(cache, feats)
    print(f"  saved {feats.shape} -> {cache.name}")
    del gen_model, samples, images, images_uint8
    torch.cuda.empty_cache()
    return feats


# Real-vs-gen CI is also a library call:
# bootstrap_fid_real_vs_gen(real_feats, gen_feats, n=10000, B=200, verbose=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cells",
        nargs="*",
        default=[
            "chestmnist_AEdisc_diffusion",  # AE + LDM (FID 0.09)
            "chestmnist_LFQ1024_transformer",  # best discrete  (FID 0.33)
            "chestmnist_LFQ1024_maskgit",  # mid discrete    (FID 1.91)
            "chestmnist_FSQ4096_bayesian_flow",  # worst overall   (FID 8.46)
        ],
    )
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--bootstrap_n", type=int, default=10000)
    parser.add_argument("--B", type=int, default=200, help="bootstrap reps")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_real_vs_real", action="store_true")
    parser.add_argument("--skip_real_vs_gen", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    extractor = InceptionPool3FeatureExtractor(device=args.device, batch_size=64)

    # ---- Real vs real noise floor ----
    if not args.skip_real_vs_real:
        print("\n=== Real-vs-real noise floor ===")
        test_feats = get_test_features(extractor)
        print(f"  test features: {test_feats.shape}")
        noise = bootstrap_fid_noise_floor(
            test_feats,
            n_per_half=[1000, 2500, 5000, 10000],
            B=args.B,
            seed=args.seed,
            verbose=True,
        )
        out_path = OUT_DIR / "fid_noise_floor.json"
        out_path.write_text(json.dumps(noise, indent=2))
        print(f"  wrote {out_path}")

    # ---- Real vs gen CI for representative cells ----
    if not args.skip_real_vs_gen:
        print("\n=== Real-vs-generated CI ===")
        test_feats = get_test_features(extractor)
        tokenizer_cache: dict = {}
        per_cell = {}
        for cell in args.cells:
            ckpt = GEN_CKPT_ROOT / cell / "checkpoint_best.pt"
            if not ckpt.exists():
                print(f"  SKIP {cell}: no checkpoint")
                continue
            print(f"\n--- {cell} ---")
            try:
                gen_feats = get_gen_features(
                    cell,
                    str(ckpt),
                    args.num_samples,
                    args.batch_size,
                    args.device,
                    extractor,
                    tokenizer_cache,
                )
                # Use min(real_n, gen_n, bootstrap_n) for bootstrap size
                n = min(args.bootstrap_n, test_feats.shape[0], gen_feats.shape[0])
                ci = bootstrap_fid_real_vs_gen(
                    test_feats, gen_feats, n=n, B=args.B, seed=args.seed, verbose=True
                )
                per_cell[cell] = ci
                print(
                    f"  FID(real,gen) mean={ci['mean']:.3f} std={ci['std']:.3f} "
                    f"95%CI=[{ci['p2.5']:.3f},{ci['p97.5']:.3f}] (width {ci['ci_width_95']:.3f}, {ci['time_s']:.0f}s)"
                )
            except Exception as e:
                print(f"  ERROR: {type(e).__name__}: {e}")
                per_cell[cell] = {"error": str(e)}

        out_path = OUT_DIR / "fid_per_cell_ci.json"
        out_path.write_text(json.dumps(per_cell, indent=2))
        print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
