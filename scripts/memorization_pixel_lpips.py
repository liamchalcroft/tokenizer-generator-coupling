#!/usr/bin/env python
"""Pixel-space + LPIPS nearest-neighbour memorization probe.

Complements the InceptionV3-192 feature probe (memorization.py) with two more
distance spaces. For each headline cell we generate N samples and measure the
generated->train nearest-neighbour distance in (a) raw pixel L2 and (b) LPIPS,
versus a real-vs-real baseline (held-out test->train NN in the same spaces).
A memorization signature would be a cell whose median gen->train NN distance is
well below the test->train baseline (ratio R < 1) or a near-zero min distance.

LPIPS over the full 78k train set per sample is infeasible (a network forward
per pair), so we use the standard two-stage scheme: take the top-K pixel-L2
nearest train candidates per sample, then rerank those by LPIPS and take the
min. Memorized samples are close in both spaces, so this is a faithful screen.

Output: results/tables/memorization_pixel_lpips.json  (resume-safe per cell)
"""

from __future__ import annotations

import json

import numpy as np
import torch

from evaluate_chestmnist_generation import (
    GEN_CKPT_ROOT,
    RAW_DATA_PATH,
    decode_continuous_samples,
    decode_discrete_samples,
    generate_continuous_samples,
    generate_discrete_samples,
    load_gen_model,
    load_tokenizer_decoder,
    parse_run_name,
)
from paths import TABLES

OUT = TABLES / "memorization_pixel_lpips.json"
DISCRETE = {"transformer", "maskgit", "flow", "d3pm", "sedd", "bayesian_flow"}
# Headline cells spanning regimes; continuous + best-discrete are the highest
# memorization risk (closest to train under the inception probe).
HEADLINE = [
    "chestmnist_AEdisc_continuous_flow",  # AE + RF (best continuous)
    "chestmnist_AEdisc_diffusion",  # AE + LDM
    "chestmnist_VAEc8_continuous_flow",  # highest-recon continuous
    "chestmnist_LFQ1024_transformer",  # best default discrete (AR)
    "chestmnist_LFQ1024_d3pm",  # tuned diffusion
    "chestmnist_VQ1024_maskgit",  # VQ best
    "chestmnist_FSQ4096_bayesian_flow",  # worst FID (contrast)
]
N = 1000
TOPK = 50
SEED = 42


@torch.no_grad()
def gen_images(cell, device, batch_size=128):
    tok, model_type = parse_run_name(cell)
    ckpt = GEN_CKPT_ROOT / cell / "checkpoint_best.pt"
    model, gen_args = load_gen_model(str(ckpt), device)
    tokenizer = load_tokenizer_decoder(tok, device)
    if gen_args["model"] in DISCRETE:
        s = generate_discrete_samples(model, gen_args, N, batch_size, device)
        imgs = decode_discrete_samples(
            tokenizer, s, gen_args.get("spatial_shape"), batch_size, device
        )
    else:
        s = generate_continuous_samples(model, gen_args, N, batch_size, device)
        imgs = decode_continuous_samples(tokenizer, s, batch_size, device)
    del model, tokenizer
    torch.cuda.empty_cache()
    return imgs.clamp(0, 1).float()  # (N,1,64,64)


def pixel_nn(query, train_flat, device, topk):
    """Return (min L2 dist per query, topk train indices per query)."""
    q = query.reshape(query.shape[0], -1).to(device)
    mins = torch.full((q.shape[0],), float("inf"), device=device)
    # accumulate global top-k across train chunks via running candidate pool
    best_d = torch.full((q.shape[0], topk), float("inf"), device=device)
    best_i = torch.zeros((q.shape[0], topk), dtype=torch.long, device=device)
    for s in range(0, train_flat.shape[0], 8192):
        tchunk = train_flat[s : s + 8192].to(device)
        d = torch.cdist(q, tchunk)  # (Q, chunk)
        cat_d = torch.cat([best_d, d], dim=1)
        cat_i = torch.cat(
            [best_i, torch.arange(s, s + tchunk.shape[0], device=device).expand(q.shape[0], -1)],
            dim=1,
        )
        best_d, sel = cat_d.topk(topk, dim=1, largest=False)
        best_i = torch.gather(cat_i, 1, sel)
        del tchunk, d
    mins = best_d[:, 0]
    return mins.cpu(), best_i.cpu()


def lpips_nn(query, train_imgs, cand_idx, lpips_model, device, batch=256):
    """LPIPS min over the pixel-topk candidates. query/train (N,1,64,64) in [0,1]."""

    def to_lpips(x):  # (N,1,64,64)[0,1] -> (N,3,64,64)[-1,1]
        return x.repeat(1, 3, 1, 1) * 2 - 1

    out = torch.full((query.shape[0],), float("inf"))
    for i in range(query.shape[0]):
        cands = train_imgs[cand_idx[i]]  # (K,1,64,64)
        q = query[i : i + 1].repeat(cands.shape[0], 1, 1, 1)
        ds = []
        for s in range(0, cands.shape[0], batch):
            a = to_lpips(q[s : s + batch]).to(device)
            b = to_lpips(cands[s : s + batch]).to(device)
            ds.append(lpips_model(a, b).flatten().cpu())
        out[i] = torch.cat(ds).min()
    return out


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    import lpips

    lp = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    raw = np.load(str(RAW_DATA_PATH))
    train = (torch.from_numpy(raw["train_images"]).float() / 255.0).unsqueeze(1)  # (Nt,1,64,64)
    test = (torch.from_numpy(raw["test_images"]).float() / 255.0).unsqueeze(1)
    train_flat = train.reshape(train.shape[0], -1)
    print(f"train {tuple(train.shape)} test {tuple(test.shape)}")

    results = {}
    if OUT.exists():
        with open(OUT) as f:
            results = json.load(f)
    results.setdefault(
        "_description",
        f"Pixel-L2 and LPIPS(alex) gen->train NN vs test->train baseline. N={N} per cell, "
        f"LPIPS via top-{TOPK} pixel-candidate rerank. R=median(gen->train)/median(test->train); "
        "R<1 or near-zero min would indicate memorization.",
    )
    results.setdefault("cells", {})

    # real-vs-real baseline (test sample -> train NN)
    if "baseline_real_test" not in results:
        idx = torch.randperm(test.shape[0])[:N]
        ts = test[idx]
        p_min, cand = pixel_nn(ts, train_flat, device, TOPK)
        l_min = lpips_nn(ts, train, cand, lp, device)
        results["baseline_real_test"] = {
            "n": int(N),
            "pixel_med": float(p_min.median()),
            "pixel_min": float(p_min.min()),
            "lpips_med": float(l_min.median()),
            "lpips_min": float(l_min.min()),
        }
        with open(OUT, "w") as f:
            json.dump(results, f, indent=2)
        print(
            f"baseline test->train: pixel_med={p_min.median():.3f} lpips_med={l_min.median():.4f}"
        )

    base = results["baseline_real_test"]
    for cell in HEADLINE:
        if cell in results["cells"]:
            print(f"skip cached {cell}")
            continue
        ck = GEN_CKPT_ROOT / cell / "checkpoint_best.pt"
        if not ck.exists():
            print(f"SKIP {cell}: no ckpt")
            continue
        print(f"--- {cell} ---", flush=True)
        try:
            g = gen_images(cell, device)
            p_min, cand = pixel_nn(g, train_flat, device, TOPK)
            l_min = lpips_nn(g, train, cand, lp, device)
            results["cells"][cell] = {
                "n_gen": int(g.shape[0]),
                "pixel_med_gen_train": float(p_min.median()),
                "pixel_min_gen_train": float(p_min.min()),
                "lpips_med_gen_train": float(l_min.median()),
                "lpips_min_gen_train": float(l_min.min()),
                "R_pixel": float(p_min.median() / base["pixel_med"]),
                "R_lpips": float(l_min.median() / base["lpips_med"]),
            }
            r = results["cells"][cell]
            print(
                f"  pixel R={r['R_pixel']:.3f} (min {r['pixel_min_gen_train']:.3f})  "
                f"lpips R={r['R_lpips']:.3f} (min {r['lpips_min_gen_train']:.4f})"
            )
        except Exception as e:  # noqa: BLE001
            results["cells"][cell] = {"error": str(e)}
            print(f"  ERROR: {e}")
        with open(OUT, "w") as f:
            json.dump(results, f, indent=2)
        torch.cuda.empty_cache()
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
