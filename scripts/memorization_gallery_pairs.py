#!/usr/bin/env python
"""Closest generated-to-training pairs for the memorization gallery.

Generates each cell's samples once (seed 42, reset per cell), computes the nearest
training neighbours of exactly those samples with the feature extractor and pair
function of memorization_probe.py, and saves the images together with the pairs, so
each generated image is shown next to its own nearest training image. The probe seeds
once and samples every cell in turn, so its stored gen_idx values index a different
batch for every cell but the first; the gallery therefore does not reuse them.

Same five cells as the paper's gallery. Generator outputs are saved too.

Output: results/samples/memorization_gallery_pairs.npz
  cells str [5], generated uint8 [5, K, 64, 64], train uint8 [5, K, 64, 64],
  distance float [5, K], gen_idx int [5, K], train_idx int [5, K],
  and raw_<i> (tokens or latents of all generated samples for cell i)
plus memorization_gallery_pairs.json with the same pairs and per-cell metrics.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from medlatents.evaluation import (
    InceptionPool3FeatureExtractor,
    memorization_metrics,
    memorization_pairs,
)

from evaluate_chestmnist_generation import (
    GEN_CKPT_ROOT,
    RAW_DATA_PATH,
    load_gen_model,
    load_tokenizer_decoder,
    parse_run_name,
    prepare_for_metrics,
)
from memorization_probe import decode, generate_samples
from paths import RESULTS

CELLS = [
    "chestmnist_AEdisc_continuous_flow",
    "chestmnist_LFQ1024_transformer",
    "chestmnist_LFQ1024_d3pm",
    "chestmnist_VQ1024_maskgit",
    "chestmnist_FSQ4096_bayesian_flow",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--num_samples", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--pairs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--train_features", default=None, help="cached train features (.npy)")
    args = ap.parse_args()

    raw = np.load(str(RAW_DATA_PATH))
    train_images = raw["train_images"]
    extractor = InceptionPool3FeatureExtractor(device=args.device, batch_size=64)
    if args.train_features:
        train_feats = torch.from_numpy(np.load(args.train_features))
    else:
        train_u8 = prepare_for_metrics(torch.from_numpy(train_images).float().div(255).unsqueeze(1))
        train_feats = extractor.extract(train_u8)

    out = {"gen": [], "train": [], "dist": [], "gidx": [], "tidx": [], "raw": []}
    meta = {}
    for cell in CELLS:
        tok, model_type = parse_run_name(cell)
        model, gen_args = load_gen_model(
            str(GEN_CKPT_ROOT / cell / "checkpoint_best.pt"), args.device
        )
        tokenizer = load_tokenizer_decoder(tok, args.device)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        samples = generate_samples(model, gen_args, args.num_samples, args.batch_size, args.device)
        images = prepare_for_metrics(
            decode(tokenizer, samples, gen_args, args.batch_size, args.device)
        )
        feats = extractor.extract(images)
        pairs = memorization_pairs(feats, train_feats, top_k=args.pairs, device=args.device)
        metrics = memorization_metrics(feats, train_feats, device=args.device)
        out["gen"].append(np.stack([images[g, 0].numpy() for g, _, _ in pairs]))
        out["train"].append(np.stack([train_images[t] for _, t, _ in pairs]))
        out["dist"].append([d for _, _, d in pairs])
        out["gidx"].append([g for g, _, _ in pairs])
        out["tidx"].append([t for _, t, _ in pairs])
        out["raw"].append(samples.cpu().numpy())
        meta[cell] = {
            "pairs": [{"gen_idx": g, "train_idx": t, "distance": d} for g, t, d in pairs],
            "memorization_ratio": metrics["memorization_ratio"],
            "auth_pct": metrics["auth_pct"],
        }
        print(cell, meta[cell]["pairs"], flush=True)
        del model, tokenizer
        torch.cuda.empty_cache()

    path = RESULTS / "samples" / "memorization_gallery_pairs.npz"
    np.savez_compressed(
        path,
        cells=np.array(CELLS),
        generated=np.stack(out["gen"]).astype(np.uint8),
        train=np.stack(out["train"]).astype(np.uint8),
        distance=np.array(out["dist"], dtype=np.float64),
        gen_idx=np.array(out["gidx"]),
        train_idx=np.array(out["tidx"]),
        **{f"raw_{i}": r for i, r in enumerate(out["raw"])},
    )
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "_description": (
                    "Closest generated-to-training pairs (InceptionV3-192 cosine distance, "
                    "memorization_pairs) computed on the same samples that are saved, seed "
                    f"{args.seed} reset per cell, {args.num_samples} samples per cell. "
                    "raw_<i> in the npz holds cell i's generator output (tokens or latents)."
                ),
                "cells": meta,
            },
            indent=2,
        )
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
