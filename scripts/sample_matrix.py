#!/usr/bin/env python
"""Fixed-seed, uncurated sample matrix over every ChestMNIST factorial cell.

For each of the 54 discrete and 16 continuous cells, plus the two tuned LFQ-1024
settings from results/tables/hp_revalidation.json, draws one batch of 16 samples
at torch.manual_seed(0) and decodes them to 64 x 64 greyscale. Sampling goes
through the same functions as the paper's evaluation: default cells use
generate_discrete_samples / generate_continuous_samples from
evaluate_chestmnist_generation.py, and tuned cells use generate_with_hps from
sweep_sampling_hps.py. Pixel conversion uses prepare_for_metrics, so the stored
uint8 values are exactly what FID was computed on.

Outputs, under results/samples/:
  sample_matrix.npz   images uint8 [n_cells, 16, 64, 64], cells str [n_cells]
  sample_matrix.json  per-cell sampler settings, checkpoint sha256, FID-192, timing
  real_test.npz       first 16 ChestMNIST test images, same preprocessing
  contact_sheet.png   rows = cells, columns = the 16 samples

Usage:
    CHECKPOINT_ROOT=... MEDTOKENIZERS_ROOT=... uv run python scripts/sample_matrix.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from importlib.metadata import version

import numpy as np
import torch
from PIL import Image, ImageDraw

from evaluate_chestmnist_generation import (
    CONTINUOUS_MODELS,
    GEN_CKPT_ROOT,
    RAW_DATA_PATH,
    _find_tokenizer_checkpoint,
    decode_continuous_samples,
    decode_discrete_samples,
    generate_continuous_samples,
    generate_discrete_samples,
    load_gen_model,
    load_tokenizer_decoder,
    prepare_for_metrics,
)
from paths import RESULTS, TABLES
from sweep_sampling_hps import generate_with_hps

OUT = RESULTS / "samples"
N = 16
SEED = 0

DISCRETE_TOKS = [f"{q}{v}" for q in ("VQ", "LFQ", "FSQ") for v in (1024, 2048, 4096)]
DISCRETE_GENS = ["transformer", "maskgit", "flow", "d3pm", "sedd", "bayesian_flow"]
CONTINUOUS_TOKS = ["VAEc1", "VAEc2", "VAEc4", "VAEc8", "VAEc16", "VAE1e5", "VAE1e7", "AEdisc"]
CONTINUOUS_GENS = ["diffusion", "continuous_flow"]
TUNED = ["LFQ1024_d3pm", "LFQ1024_sedd"]


def default_settings(model_type: str, a: dict) -> dict:
    """The sampler settings hardcoded in the evaluation's generate_* functions."""
    if model_type == "transformer":
        return {"temperature": 0.9}
    if model_type == "maskgit":
        return {"num_steps": 12, "temperature": 0.9}
    if model_type == "flow":
        return {"num_euler_steps": 100, "source_dist": a.get("source_dist", "uniform")}
    if model_type in ("d3pm", "sedd"):
        return {"num_timesteps": a.get("diffusion_steps", 1000), "temperature": 0.9}
    if model_type == "bayesian_flow":
        return {"num_steps": a.get("bfn_num_steps", 1000), "temperature": 0.9}
    if model_type == "diffusion":
        return {"num_timesteps": a.get("diffusion_steps", 1000), "temperature": 0.9}
    if model_type == "continuous_flow":
        return {"num_steps": a.get("flow_num_steps", 100), "base_std": a.get("base_std", 1.0)}
    raise ValueError(model_type)


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def to_uint8(images: torch.Tensor) -> np.ndarray:
    """(N, 1, 64, 64) float -> (N, 64, 64) uint8, via the evaluation's conversion."""
    return prepare_for_metrics(images)[:, 0].numpy()


def contact_sheet(images: np.ndarray, labels: list[str], path) -> None:
    label_w, pad = 190, 2
    n, k, h, w = images.shape
    sheet = Image.new("L", (label_w + k * (w + pad), n * (h + pad)), 255)
    draw = ImageDraw.Draw(sheet)
    for i in range(n):
        y = i * (h + pad)
        draw.text((4, y + h // 2 - 6), labels[i], fill=0)
        for j in range(k):
            sheet.paste(Image.fromarray(images[i, j]), (label_w + j * (w + pad), y))
    sheet.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--filter", default=None, help="only run cells containing this substring")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    eval_results = json.loads((RESULTS / "chestmnist_factorial" / "eval_results.json").read_text())
    hp_cells = json.loads((TABLES / "hp_revalidation.json").read_text())["cells"]

    cells = [(t, g, False) for t in DISCRETE_TOKS for g in DISCRETE_GENS]
    cells += [(t, g, False) for t in CONTINUOUS_TOKS for g in CONTINUOUS_GENS]
    cells += [(*c.split("_", 1), True) for c in TUNED]
    if args.filter:
        cells = [c for c in cells if args.filter in f"{c[0]}_{c[1]}"]

    raw = np.load(str(RAW_DATA_PATH))
    real = torch.from_numpy(raw["test_images"][:N]).float().div(255.0).unsqueeze(1)
    np.savez_compressed(OUT / "real_test.npz", images=to_uint8(real))

    tokenizers: dict[str, torch.nn.Module] = {}
    sha_cache: dict[str, str] = {}
    images, names, records = [], [], []
    t_all = time.time()

    for tok, gen, tuned in cells:
        name = f"{tok}_{gen}" + ("_tuned" if tuned else "")
        run = f"chestmnist_{tok}_{gen}"
        ckpt = GEN_CKPT_ROOT / run / "checkpoint_best.pt"
        rec = {"cell": name, "tokenizer": tok, "generator": gen, "tuned": tuned, "seed": SEED}
        print(f"{name} ...", flush=True)
        try:
            model, gen_args = load_gen_model(str(ckpt), args.device)
            gen_args["model"] = gen
            continuous = gen in CONTINUOUS_MODELS
            if tuned:
                hp = dict(hp_cells[f"{tok}_{gen}"]["selected_hp"])
                settings = dict(hp)
            else:
                settings = default_settings(gen, gen_args)

            torch.cuda.synchronize()
            t0 = time.time()
            torch.manual_seed(SEED)
            if tuned:
                out = generate_with_hps(model, gen_args, hp, N, N, args.device)
            elif continuous:
                out = generate_continuous_samples(model, gen_args, N, N, args.device)
            else:
                out = generate_discrete_samples(model, gen_args, N, N, args.device)
            torch.cuda.synchronize()
            t_gen = time.time() - t0
            del model
            torch.cuda.empty_cache()

            if tok not in tokenizers:
                tokenizers[tok] = load_tokenizer_decoder(tok, args.device)
            t0 = time.time()
            if continuous:
                dec = decode_continuous_samples(tokenizers[tok], out, N, args.device)
            else:
                dec = decode_discrete_samples(
                    tokenizers[tok], out, gen_args.get("spatial_shape"), N, args.device
                )
            t_dec = time.time() - t0
            if not torch.isfinite(dec).all():
                raise ValueError("non-finite values in decoded images")
            img = to_uint8(dec)

            if str(ckpt) not in sha_cache:
                sha_cache[str(ckpt)] = sha256(ckpt)
            fid_default = eval_results.get(run, {}).get("fid")
            rec.update(
                sampler=settings,
                checkpoint=str(ckpt),
                checkpoint_sha256=sha_cache[str(ckpt)],
                tokenizer_checkpoint=str(_find_tokenizer_checkpoint(tok)),
                fid192=hp_cells[f"{tok}_{gen}"]["test_fid_10k"] if tuned else fid_default,
                fid192_source=(
                    "results/tables/hp_revalidation.json test_fid_10k"
                    if tuned
                    else "results/chestmnist_factorial/eval_results.json"
                ),
                generation_time_s=round(t_gen, 3),
                decode_time_s=round(t_dec, 3),
                pixel_mean=float(img.mean()),
                pixel_std=float(img.std()),
            )
            images.append(img)
            names.append(name)
            print(f"  {t_gen:.1f}s  mean={img.mean():.1f}  fid={rec['fid192']}", flush=True)
        except Exception as e:  # record and continue, so one bad cell cannot sink the matrix
            rec["error"] = f"{type(e).__name__}: {e}"
            print(f"  FAILED {rec['error']}", flush=True)
            torch.cuda.empty_cache()
        records.append(rec)

    stack = np.stack(images).astype(np.uint8)
    np.savez_compressed(OUT / "sample_matrix.npz", images=stack, cells=np.array(names))
    meta = {
        "_description": (
            "Uncurated fixed-seed samples: one batch of 16 per cell, torch.manual_seed(0) "
            "immediately before sampling, kept in generation order. Default cells use the "
            "evaluation's sampler settings; *_tuned cells use the validation-selected "
            "settings from results/tables/hp_revalidation.json."
        ),
        "n_cells": len(names),
        "n_failed": sum("error" in r for r in records),
        "samples_per_cell": N,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "medlatents": version("medlatents"),
        "medtokenizers": version("medtokenizers"),
        "total_generation_time_s": round(sum(r.get("generation_time_s", 0) for r in records), 1),
        "total_wall_time_s": round(time.time() - t_all, 1),
        "cells": records,
    }
    (OUT / "sample_matrix.json").write_text(json.dumps(meta, indent=2))
    contact_sheet(stack, names, OUT / "contact_sheet.png")
    print(f"\n{len(names)} cells, {meta['n_failed']} failed, wrote {OUT}")


if __name__ == "__main__":
    main()
