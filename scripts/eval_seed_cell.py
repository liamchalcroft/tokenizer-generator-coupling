#!/usr/bin/env python
"""Evaluate one trained generator checkpoint at its DEFAULT sampling config and
record the 10K test FID-192 into results/tables/seed_variance.json under a seed
label. Used by the multi-seed variance sweep (seeds 42/43/44).
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from evaluate_chestmnist_generation import (
    RAW_DATA_PATH,
    compute_fid_is,
    decode_continuous_samples,
    decode_discrete_samples,
    generate_continuous_samples,
    generate_discrete_samples,
    load_gen_model,
    load_tokenizer_decoder,
    prepare_for_metrics,
)
from paths import TABLES

DISCRETE = {"transformer", "maskgit", "flow", "d3pm", "sedd", "bayesian_flow"}
# Default-config FID-192 (10K) for seed 42, from the main table, to pre-populate.
SEED42 = {"chestmnist_LFQ1024_d3pm": 0.44}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", required=True, help="tokenizer name, e.g. LFQ1024")
    ap.add_argument(
        "--cell", required=True, help="canonical cell key, e.g. chestmnist_LFQ1024_d3pm"
    )
    ap.add_argument("--seed_label", required=True)
    ap.add_argument("--num_samples", type=int, default=10000)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--out", default=str(TABLES / "seed_variance.json"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = args.device
    raw = np.load(str(RAW_DATA_PATH))
    test_uint8 = prepare_for_metrics(
        torch.from_numpy(raw["test_images"]).float().div(255).unsqueeze(1)
    )

    model, gen_args = load_gen_model(args.checkpoint, device)
    tok = load_tokenizer_decoder(args.tokenizer, device)
    t0 = time.time()
    if gen_args["model"] in DISCRETE:
        s = generate_discrete_samples(model, gen_args, args.num_samples, args.batch_size, device)
        imgs = decode_discrete_samples(
            tok, s, gen_args.get("spatial_shape"), args.batch_size, device
        )
    else:
        s = generate_continuous_samples(model, gen_args, args.num_samples, args.batch_size, device)
        imgs = decode_continuous_samples(tok, s, args.batch_size, device)
    m = compute_fid_is(test_uint8, prepare_for_metrics(imgs), device, batch_size=args.batch_size)
    eval_s = time.time() - t0

    out = {}
    if os.path.exists(args.out):
        with open(args.out) as f:
            out = json.load(f)
    out.setdefault(
        "_description",
        "Training-seed variance of default-config FID-192 (10K samples, test split). "
        "seed 42 = original main-table run; seeds 43/44 retrained from scratch (frozen tokenizer).",
    )
    cells = out.setdefault("cells", {})
    rec = cells.setdefault(args.cell, {"default_config": True, "seeds": {}})
    if args.cell in SEED42:
        rec["seeds"].setdefault("42", SEED42[args.cell])
    rec["seeds"][args.seed_label] = round(m["fid"], 4)
    rec.setdefault("eval_time_s", {})[args.seed_label] = round(eval_s, 0)
    # mean/std across available seeds
    vals = list(rec["seeds"].values())
    if len(vals) >= 2:
        rec["mean"] = round(sum(vals) / len(vals), 4)
        rec["std"] = round((sum((v - rec["mean"]) ** 2 for v in vals) / len(vals)) ** 0.5, 4)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    print(
        f"{args.cell} seed {args.seed_label}: FID={m['fid']:.4f} (eval {eval_s:.0f}s). seeds now: {rec['seeds']}"
    )


if __name__ == "__main__":
    main()
