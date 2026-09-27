#!/usr/bin/env python
"""Validation-selected, test-evaluated sampling-hyperparameter selection.

For each vocabulary-1024 (tokenizer, generator) cell:

  1. Screen the full sampling-HP grid (SAMPLING_GRIDS in sweep_sampling_hps.py)
     at 2K samples, scoring FID-192 against the validation split (val_images).
  2. Select the single best config by validation FID-192.
  3. Evaluate that one config once at 10K samples against the test split.

It also reports the validation real-vs-real FID-192 noise floor (selection
uncertainty). Resume-safe: the results JSON is rewritten (atomic) after every
config, so the job can be killed and restarted without losing work. Set
MEDGEN_SAVE_TOKENS_DIR to keep the generated tokens of every screened and final
configuration (results/generated_tokens/chestmnist/sweep/ was written this way).

Usage:
    uv run python scripts/revalidate_on_val.py
    uv run python scripts/revalidate_on_val.py --only LFQ1024_d3pm,LFQ1024_sedd
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from evaluate_chestmnist_generation import (
    GEN_CKPT_ROOT,
    RAW_DATA_PATH,
    compute_fid_is,
    decode_discrete_samples,
    load_gen_model,
    load_tokenizer_decoder,
    parse_run_name,
    prepare_for_metrics,
)
from paths import RESULTS
from sweep_sampling_hps import (
    SAMPLING_GRIDS,
    expand_grid,
    generate_with_hps,
)

EVAL_RESULTS = RESULTS / "chestmnist_factorial" / "eval_results.json"

TARGET_CELLS = [
    "chestmnist_LFQ1024_d3pm",
    "chestmnist_LFQ1024_sedd",
    "chestmnist_FSQ1024_d3pm",
    "chestmnist_FSQ1024_sedd",
    "chestmnist_LFQ1024_transformer",
    "chestmnist_LFQ1024_maskgit",
    "chestmnist_LFQ1024_flow",
    "chestmnist_LFQ1024_bayesian_flow",
    "chestmnist_FSQ1024_transformer",
    "chestmnist_FSQ1024_maskgit",
    "chestmnist_FSQ1024_flow",
    "chestmnist_FSQ1024_bayesian_flow",
    "chestmnist_VQ1024_transformer",
    "chestmnist_VQ1024_maskgit",
    "chestmnist_VQ1024_flow",
    "chestmnist_VQ1024_bayesian_flow",
    "chestmnist_VQ1024_d3pm",
    "chestmnist_VQ1024_sedd",
]


def atomic_save(obj, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def load_split_uint8(key: str) -> torch.Tensor:
    raw = np.load(str(RAW_DATA_PATH))
    imgs = torch.from_numpy(raw[key]).float() / 255.0
    imgs = imgs.unsqueeze(1)
    return prepare_for_metrics(imgs)


@torch.inference_mode()
def fid_for_config(gen_model, gen_args, hp, num_samples, batch_size, device, real_uint8):
    tokens = generate_with_hps(gen_model, gen_args, hp, num_samples, batch_size, device)
    save_dir = os.environ.get("MEDGEN_SAVE_TOKENS_DIR")
    if save_dir:
        slug = "_".join(f"{k}-{v}" for k, v in sorted(hp.items()))
        path = Path(save_dir) / gen_args.get("_cell", "cell") / f"{slug}_n{num_samples}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, tokens=tokens.cpu().numpy().astype(np.int16), hp=json.dumps(hp))
    spatial_shape = gen_args.get("spatial_shape", None)
    tokenizer = gen_args["_tokenizer"]
    gen_images = decode_discrete_samples(tokenizer, tokens, spatial_shape, batch_size, device)
    gen_uint8 = prepare_for_metrics(gen_images)
    m = compute_fid_is(real_uint8, gen_uint8, device, batch_size=batch_size)
    del tokens, gen_images, gen_uint8
    torch.cuda.empty_cache()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen_samples", type=int, default=2000)
    ap.add_argument("--final_samples", type=int, default=10000)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--output",
        default=str(RESULTS / "chestmnist_factorial" / "revalidation_results.json"),
    )
    ap.add_argument("--only", default=None, help="comma list of TOK_MODEL keys to restrict to")
    ap.add_argument("--skip_noise_floor", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device

    default_fid = {k: round(v["fid"], 2) for k, v in json.loads(EVAL_RESULTS.read_text()).items()}

    cells = TARGET_CELLS
    if args.only:
        keep = set(args.only.split(","))
        cells = [c for c in cells if c.replace("chestmnist_", "") in keep]

    print(f"Loading val/test splits from {RAW_DATA_PATH}")
    val_uint8 = load_split_uint8("val_images")
    test_uint8 = load_split_uint8("test_images")
    print(f"  val:  {tuple(val_uint8.shape)}   test: {tuple(test_uint8.shape)}")

    results = {}
    if os.path.exists(args.output):
        with open(args.output) as f:
            results = json.load(f)
    results.setdefault("cells", {})
    results.setdefault("noise_floor_val", {})
    results.setdefault(
        "meta",
        {
            "screen_split": "val",
            "final_split": "test",
            "screen_samples": args.screen_samples,
            "final_samples": args.final_samples,
            "n_val": int(val_uint8.shape[0]),
            "n_test": int(test_uint8.shape[0]),
        },
    )

    # ---- Validation real-vs-real noise floor (selection uncertainty) --------
    if not args.skip_noise_floor and not results["noise_floor_val"]:
        try:
            from medlatents.evaluation import (
                InceptionPool3FeatureExtractor,
                bootstrap_fid_noise_floor,
            )

            print("\n=== Validation real-vs-real FID-192 noise floor ===")
            extractor = InceptionPool3FeatureExtractor(device=device, batch_size=64)
            val_feats = extractor.extract(val_uint8).numpy()
            # n_per_half=5000 uses 10k of 11,219 val images; 10k-per-half is
            # infeasible on val (only 11,219 images) -> see test floor at 10k.
            nf = bootstrap_fid_noise_floor(
                val_feats, n_per_half=[1000, 2000, 5000], B=200, seed=args.seed, verbose=True
            )
            results["noise_floor_val"] = nf
            results["noise_floor_val"]["_note"] = (
                "val has 11,219 images; n_per_half=10000 infeasible. For the 10K "
                "final-eval reference use the test floor in fid_noise_floor.json "
                "(n_per_half=10000 mean 0.0018)."
            )
            atomic_save(results, args.output)
            del extractor, val_feats
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            print(f"  noise floor skipped: {type(e).__name__}: {e}")

    # ---- Per-cell: screen on val, select, eval selected on test -------------
    for cell in cells:
        key = cell.replace("chestmnist_", "")
        ckpt = GEN_CKPT_ROOT / cell / "checkpoint_best.pt"
        if not ckpt.exists():
            print(f"SKIP {cell}: no checkpoint at {ckpt}")
            continue
        parsed = parse_run_name(cell)
        if parsed is None:
            print(f"SKIP {cell}: unparseable")
            continue
        tok_type, model_type = parsed

        cell_rec = results["cells"].setdefault(key, {"screen": {}})
        grid = SAMPLING_GRIDS.get(model_type, {"temperature": [0.9]})
        hp_configs = expand_grid(grid)

        print(
            f"\n{'=' * 70}\n{cell}  ({len(hp_configs)} configs, screen on val @ {args.screen_samples})\n{'=' * 70}"
        )

        gen_model, gen_args = load_gen_model(str(ckpt), device)
        gen_args["_tokenizer"] = load_tokenizer_decoder(tok_type, device)
        gen_args["_cell"] = key

        # --- screening on val ---
        for i, hp in enumerate(hp_configs):
            hp_key = json.dumps(hp, sort_keys=True)
            if hp_key in cell_rec["screen"]:
                continue
            t0 = time.time()
            try:
                m = fid_for_config(
                    gen_model, gen_args, hp, args.screen_samples, args.batch_size, device, val_uint8
                )
                cell_rec["screen"][hp_key] = {
                    "hp": hp,
                    "val_fid": m["fid"],
                    "val_is": m["is_mean"],
                    "time_s": time.time() - t0,
                }
                print(
                    f"  [{i + 1}/{len(hp_configs)}] {hp}  val_fid={m['fid']:.3f}  ({time.time() - t0:.0f}s)"
                )
            except Exception as e:  # noqa: BLE001
                cell_rec["screen"][hp_key] = {"hp": hp, "error": str(e)}
                print(f"  [{i + 1}/{len(hp_configs)}] {hp}  ERROR: {e}")
            atomic_save(results, args.output)

        # --- select best by val fid ---
        valid = [v for v in cell_rec["screen"].values() if "val_fid" in v]
        if not valid:
            print(f"  no valid configs for {cell}")
            del gen_model
            torch.cuda.empty_cache()
            continue
        best = min(valid, key=lambda v: v["val_fid"])
        cell_rec["selected_hp"] = best["hp"]
        cell_rec["val_fid_2k"] = best["val_fid"]
        print(f"  >> selected (val): {best['hp']}  val_fid_2k={best['val_fid']:.3f}")

        # --- final eval of selected config on test @ 10k ---
        if "test_fid_10k" not in cell_rec:
            t0 = time.time()
            try:
                m = fid_for_config(
                    gen_model,
                    gen_args,
                    best["hp"],
                    args.final_samples,
                    args.batch_size,
                    device,
                    test_uint8,
                )
                cell_rec["test_fid_10k"] = m["fid"]
                cell_rec["test_is_10k"] = m["is_mean"]
                cell_rec["final_time_s"] = time.time() - t0
                cell_rec["default_fid"] = default_fid.get(cell)
                print(f"  >> FINAL test_fid_10k={m['fid']:.3f}  (default {default_fid.get(cell)})")
            except Exception as e:  # noqa: BLE001
                cell_rec["test_fid_10k_error"] = str(e)
                print(f"  FINAL ERROR: {e}")
            atomic_save(results, args.output)

        del gen_model
        torch.cuda.empty_cache()

    # ---- summary table ------------------------------------------------------
    print(f"\n{'=' * 90}\nSUMMARY (val-selected, test-evaluated)\n{'=' * 90}")
    print(f"{'cell':28s} {'selected HP':34s} {'valFID2K':>9s} {'testFID10K':>11s} {'default':>8s}")
    for key, r in results["cells"].items():
        if "test_fid_10k" not in r:
            continue
        print(
            f"{key:28s} {str(r.get('selected_hp')):34s} {r.get('val_fid_2k', float('nan')):9.3f} "
            f"{r['test_fid_10k']:11.3f} {str(r.get('default_fid')):>8s}"
        )
    print(f"\nResults -> {args.output}")


if __name__ == "__main__":
    main()
