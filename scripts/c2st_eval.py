#!/usr/bin/env python
"""Classifier two-sample test: how separable are generated samples from real ones?

A natural question is whether tokenizer preferences translate into downstream task
utility. The generators are unconditional, so a synth-then-classify study is
ill-posed: there are no labels to train a downstream classifier on. This script
provides a label-free alternative.

A C2ST trains a discriminator to tell real test images from generated ones and
reports held-out AUC. 0.5 means the two sets are indistinguishable to the
discriminator; 1.0 means trivially separable. Unlike FID it makes no Gaussian
assumption about the feature distribution and needs no Inception network, so
agreement between the C2ST ranking and the FID-192 ranking is evidence
that the FID-192 ordering is not an artifact of either choice.

Reads the sample cache written by scripts/eval_fid_dual.py --save_samples, so it
costs no extra generation. Roughly a minute per cell on a GPU.

Usage:
    python scripts/c2st_eval.py --smoke
    MEDTOKENIZERS_ROOT=../medtokenizers \
    python scripts/c2st_eval.py --samples_dir /path/to/cache
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from paths import TABLES


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUC via the rank-sum identity. Ties get average ranks."""
    order = np.argsort(scores, kind="mergesort")
    s = scores[order]
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i : j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    y = labels[order]
    n_pos = float(y.sum())
    n_neg = float(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def spearman(x, y) -> float:
    x, y = list(x), list(y)

    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(x), ranks(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return float(num / (dx * dy)) if dx and dy else float("nan")


def build_discriminator(torch):
    """Small CNN. Deliberately modest: the question is separability, not how good a
    discriminator we can build, and an over-powerful one saturates every cell at 1.0."""
    nn = torch.nn
    return nn.Sequential(
        nn.Conv2d(1, 32, 3, stride=2, padding=1),
        nn.GroupNorm(8, 32),
        nn.SiLU(),
        nn.Conv2d(32, 64, 3, stride=2, padding=1),
        nn.GroupNorm(8, 64),
        nn.SiLU(),
        nn.Conv2d(64, 128, 3, stride=2, padding=1),
        nn.GroupNorm(8, 128),
        nn.SiLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(128, 1),
    )


def c2st(
    real_u8: np.ndarray,
    gen_u8: np.ndarray,
    device: str,
    epochs: int = 4,
    batch_size: int = 128,
    lr: float = 3e-4,
    seed: int = 0,
) -> dict:
    """Train a discriminator on a balanced split, return held-out AUC and accuracy."""
    import torch

    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)

    n = min(len(real_u8), len(gen_u8))
    # Subsample at random rather than truncating: the real split is 22,433 against
    # 10,000 generated, and MedMNIST ordering is not guaranteed to be shuffled, so
    # taking a prefix could hand the discriminator real ordering structure.
    rng = np.random.default_rng(seed)
    ridx = rng.choice(len(real_u8), size=n, replace=False) if len(real_u8) > n else slice(None)
    gidx = rng.choice(len(gen_u8), size=n, replace=False) if len(gen_u8) > n else slice(None)
    real = torch.from_numpy(np.ascontiguousarray(real_u8[ridx])).float().div(255)
    gen = torch.from_numpy(np.ascontiguousarray(gen_u8[gidx])).float().div(255)
    if real.ndim == 3:
        real = real.unsqueeze(1)
    if gen.ndim == 3:
        gen = gen.unsqueeze(1)

    x = torch.cat([real, gen], 0)
    y = torch.cat([torch.zeros(n), torch.ones(n)], 0)
    perm = torch.randperm(len(x), generator=g)
    x, y = x[perm], y[perm]

    n_train = int(0.7 * len(x))
    xtr, ytr = x[:n_train], y[:n_train]
    xte, yte = x[n_train:], y[n_train:]

    model = build_discriminator(torch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    lossf = torch.nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(epochs):
        idx = torch.randperm(len(xtr), generator=g)
        for s in range(0, len(xtr), batch_size):
            b = idx[s : s + batch_size]
            opt.zero_grad()
            out = model(xtr[b].to(device)).squeeze(1)
            loss = lossf(out, ytr[b].to(device))
            loss.backward()
            opt.step()

    model.eval()
    scores = []
    with torch.no_grad():
        for s in range(0, len(xte), batch_size):
            scores.append(model(xte[s : s + batch_size].to(device)).squeeze(1).cpu())
    scores = torch.cat(scores).numpy()
    labels = yte.numpy()

    auc = roc_auc(scores, labels)
    acc = float(((scores > 0).astype(np.float64) == labels).mean())
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return {"auc": auc, "accuracy": acc, "n_per_class": int(n), "n_held_out": int(len(xte))}


def smoke() -> int:
    """Validate AUC maths and the end-to-end path on separable and identical sets."""
    print("smoke: roc_auc")
    s = np.array([0.1, 0.4, 0.35, 0.8])
    y = np.array([0, 0, 1, 1])
    assert abs(roc_auc(s, y) - 0.75) < 1e-9, roc_auc(s, y)
    assert abs(roc_auc(np.array([1.0, 2.0]), np.array([0, 1])) - 1.0) < 1e-9
    assert abs(roc_auc(np.array([1.0, 1.0]), np.array([0, 1])) - 0.5) < 1e-9
    print("  ok")

    print("smoke: spearman")
    assert abs(spearman([1, 2, 3], [1, 2, 3]) - 1.0) < 1e-9
    print("  ok")

    try:
        import torch  # noqa: F401
    except ImportError as e:
        print(f"smoke: torch unavailable here, skipping c2st path: {e}")
        print("smoke: ok")
        return 0

    rng = np.random.default_rng(0)
    print("smoke: c2st on obviously different distributions (expect AUC near 1)")
    a = rng.integers(0, 80, (400, 1, 64, 64)).astype(np.uint8)
    b = rng.integers(180, 255, (400, 1, 64, 64)).astype(np.uint8)
    r1 = c2st(a, b, "cpu", epochs=2, batch_size=64)
    print(f"  AUC {r1['auc']:.4f} acc {r1['accuracy']:.4f}")
    assert r1["auc"] > 0.9, "clearly separable sets should give high AUC"

    print("smoke: c2st on identical distributions (expect AUC near 0.5)")
    c = rng.integers(0, 255, (400, 1, 64, 64)).astype(np.uint8)
    d = rng.integers(0, 255, (400, 1, 64, 64)).astype(np.uint8)
    r2 = c2st(c, d, "cpu", epochs=2, batch_size=64)
    print(f"  AUC {r2['auc']:.4f} acc {r2['accuracy']:.4f}")
    assert 0.3 < r2["auc"] < 0.7, "indistinguishable sets should sit near chance"
    print("smoke: ok")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument(
        "--samples_dir",
        default=None,
        help="cache written by eval_fid_dual.py --save_samples",
    )
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--device", default=None)
    ap.add_argument("--fid_json", default=str(TABLES / "fid_dual_2048.json"))
    ap.add_argument("--out", default=str(TABLES / "c2st.json"))
    args = ap.parse_args()

    if args.smoke:
        return smoke()

    import torch

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if not args.samples_dir:
        raise SystemExit(
            "pass --samples_dir; run eval_fid_dual.py --save_samples DIR first so this "
            "reuses those samples instead of regenerating them"
        )

    from evaluate_chestmnist_generation import RAW_DATA_PATH

    raw = np.load(str(RAW_DATA_PATH))
    real = raw["test_images"]
    if real.ndim == 3:
        real = real[:, None]
    print(f"real test images: {real.shape}")

    fid_lookup = {}
    fid_path = Path(args.fid_json)
    if fid_path.exists():
        fid_lookup = json.loads(fid_path.read_text()).get("cells", {})

    cache = sorted(Path(args.samples_dir).glob("*.npz"))
    if not cache:
        raise SystemExit(f"no cached samples in {args.samples_dir}")

    rows = {}
    for path in cache:
        name = path.stem
        gen = np.load(path)["samples"]
        print(f"\n{name}: {gen.shape}")
        aucs, accs = [], []
        for seed in args.seeds:
            r = c2st(real, gen, args.device, args.epochs, args.batch_size, seed=seed)
            aucs.append(r["auc"])
            accs.append(r["accuracy"])
            print(f"  seed {seed}: AUC {r['auc']:.4f} acc {r['accuracy']:.4f}")
        a = np.asarray(aucs)
        rows[name] = {
            "auc_mean": float(a.mean()),
            "auc_std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "auc_per_seed": aucs,
            "accuracy_mean": float(np.mean(accs)),
            "n_seeds": len(args.seeds),
        }
        if name in fid_lookup and "fid192" in fid_lookup[name]:
            rows[name]["fid192"] = fid_lookup[name]["fid192"]
        print(f"  mean AUC {a.mean():.4f} +/- {rows[name]['auc_std']:.4f}")

    results = {
        "_description": (
            "Classifier two-sample test. AUC 0.5 means a discriminator cannot separate "
            "generated from real; 1.0 means trivially separable. Offered because "
            "unconditional generators make a synth-then-classify utility study "
            "ill-posed. Makes no Gaussian assumption and uses no Inception features, "
            "so agreement with the FID-192 ordering is independent evidence."
        ),
        "meta": {
            "epochs": args.epochs,
            "seeds": args.seeds,
            "device": args.device,
            "samples_dir": args.samples_dir,
        },
        "cells": rows,
    }

    paired = [(v["fid192"], v["auc_mean"]) for v in rows.values() if "fid192" in v]
    if len(paired) >= 3:
        r = spearman([p[0] for p in paired], [p[1] for p in paired])
        results["spearman_fid192_vs_c2st_auc"] = r
        results["spearman_n"] = len(paired)
        print(f"\nSpearman FID-192 vs C2ST AUC over {len(paired)} cells: {r:.4f}")
        print("Positive means the two metrics agree on ordering.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(results, indent=2))
    tmp.replace(out)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
