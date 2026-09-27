#!/usr/bin/env python
"""Compute FID-192 and FID-2048 from a single generation pass per cell.

FID-192 values are not comparable to results outside this paper, so this script
provides an anchor against standard FID-2048. Generating once and computing both
metrics from the same image tensor makes the second metric nearly free and removes
sampling variation as an explanation for any disagreement between them.

Also measures the real-vs-real floor at both dimensionalities, which shows that
standard FID-2048 is well conditioned at 64x64 with 10K samples.

Sampling, decoding and pixel conversion are imported from
scripts/evaluate_chestmnist_generation.py, so default-config cells go through the
same code path as the main table.

Usage:
    # validate the logic without data or a GPU
    python scripts/eval_fid_dual.py --smoke

    MEDTOKENIZERS_ROOT=../medtokenizers \
    uv run python scripts/eval_fid_dual.py --cells default12
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from paths import TABLES

# Twelve cells spanning the full observed FID-192 range (0.07 to 8.44), chosen so
# a rank correlation is computed over the whole quality span rather than a cluster.
# "tuned" entries carry the validation-selected config from hp_revalidation.json.
DEFAULT_CELLS = [
    {"tok": "AEdisc", "model": "continuous_flow", "label": "AE + RF"},
    {
        "tok": "LFQ1024",
        "model": "d3pm",
        "label": "LFQ-1024 + D3PM (tuned)",
        "temperature": 0.7,
        "steps": 100,
    },
    {"tok": "VAEc2", "model": "diffusion", "label": "VAE-c2 + LDM"},
    {"tok": "LFQ1024", "model": "transformer", "label": "LFQ-1024 + AR"},
    {"tok": "FSQ1024", "model": "transformer", "label": "FSQ-1024 + AR"},
    {"tok": "LFQ1024", "model": "d3pm", "label": "LFQ-1024 + D3PM"},
    {"tok": "FSQ1024", "model": "maskgit", "label": "FSQ-1024 + MaskGIT"},
    {"tok": "VQ1024", "model": "maskgit", "label": "VQ-1024 + MaskGIT"},
    {"tok": "LFQ1024", "model": "flow", "label": "LFQ-1024 + DFM"},
    {"tok": "LFQ1024", "model": "maskgit", "label": "LFQ-1024 + MaskGIT"},
    {"tok": "VQ1024", "model": "transformer", "label": "VQ-1024 + AR"},
    {"tok": "FSQ4096", "model": "bayesian_flow", "label": "FSQ-4096 + BFN"},
]

# The two D3PM interaction cells missing from the default set (LFQ-1024 + D3PM is
# already there). With those three, the D3PM quantizer interaction can be checked
# under FID-2048 as well as FID-192.
D3PM_INTERACTION_CELLS = [
    {"tok": "FSQ1024", "model": "d3pm", "label": "FSQ-1024 + D3PM"},
    {"tok": "VQ1024", "model": "d3pm", "label": "VQ-1024 + D3PM"},
]

DISCRETE = {"transformer", "maskgit", "flow", "d3pm", "sedd", "bayesian_flow"}


def compute_fid_dims(
    real,
    gen,
    dims: list[int],
    device: str,
    batch_size: int = 64,
) -> dict[int, float]:
    """FID at several Inception feature dimensionalities from the same tensors.

    Mirrors compute_fid_is: normalize=True on float [0,1] inputs. One metric object
    at a time, since the 2048-dim accumulator is the memory-hungry one.
    """
    import torch
    from torchmetrics.image.fid import FrechetInceptionDistance

    if real.dtype == torch.uint8:
        real = real.float() / 255.0
    if gen.dtype == torch.uint8:
        gen = gen.float() / 255.0

    out = {}
    for d in dims:
        fid = FrechetInceptionDistance(feature=d, normalize=True).to(device)
        for s in range(0, len(real), batch_size):
            fid.update(real[s : s + batch_size].to(device), real=True)
        for s in range(0, len(gen), batch_size):
            fid.update(gen[s : s + batch_size].to(device), real=False)
        out[d] = float(fid.compute().item())
        del fid
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return out


def real_vs_real_floor(
    real_all,
    dims: list[int],
    device: str,
    n_per_half: int = 10000,
    n_boot: int = 20,
    batch_size: int = 64,
    seed: int = 0,
) -> dict:
    """Floor from disjoint halves of the real split, at each dimensionality.

    A negative or wildly unstable value at 2048 is itself the finding: it would
    substantiate the paper's stability argument with a measurement instead of an
    assertion. A well-behaved value means the argument needs restating.
    """
    import numpy as np
    import torch

    g = torch.Generator().manual_seed(seed)
    n = len(real_all)
    if n < 2 * n_per_half:
        n_per_half = n // 2

    per_dim: dict[int, list[float]] = {d: [] for d in dims}
    for _ in range(n_boot):
        perm = torch.randperm(n, generator=g)
        a = real_all[perm[:n_per_half]]
        c = real_all[perm[n_per_half : 2 * n_per_half]]
        scores = compute_fid_dims(a, c, dims, device, batch_size)
        for d in dims:
            per_dim[d].append(scores[d])

    summary = {}
    for d in dims:
        v = np.asarray(per_dim[d], dtype=np.float64)
        summary[str(d)] = {
            "mean": float(v.mean()),
            "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "min": float(v.min()),
            "max": float(v.max()),
            "n_negative": int((v < 0).sum()),
            "n_per_half": int(n_per_half),
            "n_boot": int(n_boot),
        }
    return summary


def generate_discrete_with_config(
    model,
    gen_args: dict,
    num_samples: int,
    batch_size: int,
    device: str,
    temperature: float | None,
    steps: int | None,
):
    """Discrete sampling with an explicit temperature and step count.

    generate_discrete_samples hardcodes temperature=0.9 and reads step counts from
    gen_args, so a validation-selected config cannot be expressed through it. Only
    d3pm and sedd are parameterised here, which covers every tuned cell we evaluate;
    default-config cells go through generate_discrete_samples unchanged.
    """
    import torch

    from evaluate_chestmnist_generation import generate_discrete_samples

    model_type = gen_args["model"]
    if temperature is None and steps is None:
        return generate_discrete_samples(model, gen_args, num_samples, batch_size, device)

    if model_type not in {"d3pm", "sedd"}:
        raise NotImplementedError(
            f"config override for {model_type} not implemented; "
            "use scripts/sweep_sampling_hps.py for that generator"
        )

    from medlatents.diffusion.d3pm import D3PM

    temp = 0.9 if temperature is None else temperature
    num_timesteps = steps if steps is not None else gen_args.get("diffusion_steps", 1000)
    seq_len = gen_args["seq_len"]
    vocab_size = gen_args["vocab_size"]

    d3pm = D3PM(
        num_classes=vocab_size,
        num_timesteps=num_timesteps,
        schedule_type=gen_args.get("diffusion_schedule", "cosine"),
        transition_type=gen_args.get("d3pm_transition", "absorbing"),
        device=device,
    )

    target = model
    if model_type == "sedd":
        # SEDD trains on continuous t in [0,1) but D3PM.sample passes integers.
        class SEDDModelWrapper(torch.nn.Module):
            def __init__(self, m, T):
                super().__init__()
                self.model = m
                self.num_timesteps = T

            def forward(self, x, t, **kw):
                return self.model(x=x, t=t.float() / self.num_timesteps, **kw)

        target = SEDDModelWrapper(model, num_timesteps)

    chunks = []
    for start in range(0, num_samples, batch_size):
        bs = min(batch_size, num_samples - start)
        s = d3pm.sample(target, (bs, seq_len), temperature=temp)
        chunks.append(s.clamp(0, vocab_size - 1).cpu())
    return torch.cat(chunks, dim=0)


def spearman(x: list[float], y: list[float]) -> float:
    """Rank correlation without a scipy dependency. Ties get average ranks."""

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


def smoke() -> int:
    """Validate metric plumbing and rank logic on synthetic data. No GPU, no data."""
    print("smoke: spearman")
    assert abs(spearman([1, 2, 3, 4], [1, 2, 3, 4]) - 1.0) < 1e-9
    assert abs(spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1.0) < 1e-9
    assert abs(spearman([1, 2, 3], [1, 3, 2]) - 0.5) < 1e-9
    print("  ok")

    print("smoke: compute_fid_dims on random images (cpu, small)")
    try:
        import torch

        torch.manual_seed(0)
        real = torch.rand(24, 3, 64, 64)
        gen = torch.rand(24, 3, 64, 64)
        out = compute_fid_dims(real, gen, [192], "cpu", batch_size=8)
        print(f"  FID-192 on noise: {out[192]:.4f}")
        assert 192 in out
    except ImportError as e:
        print(f"  skipped, torch/torchmetrics unavailable here: {e}")

    print("smoke: cell table")
    assert len(DEFAULT_CELLS) == 12
    tuned = [c for c in DEFAULT_CELLS if "temperature" in c]
    assert len(tuned) == 1 and tuned[0]["steps"] == 100
    print(f"  {len(DEFAULT_CELLS)} cells, {len(tuned)} tuned")
    print("smoke: ok")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="self-test, no data or GPU needed")
    ap.add_argument("--cells", default="default12")
    ap.add_argument("--num_samples", type=int, default=10000)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--dims", type=int, nargs="+", default=[192, 2048])
    ap.add_argument("--floor_boot", type=int, default=20)
    ap.add_argument("--skip_floor", action="store_true")
    ap.add_argument(
        "--save_samples",
        default=None,
        help="directory to cache decoded uint8 samples per cell, so scripts/c2st_eval.py "
        "can reuse them instead of regenerating (roughly 41 MB per cell at 10K)",
    )
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=str(TABLES / "fid_dual_2048.json"))
    args = ap.parse_args()

    if args.smoke:
        return smoke()

    import numpy as np
    import torch

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    from evaluate_chestmnist_generation import (
        GEN_CKPT_ROOT,
        RAW_DATA_PATH,
        decode_continuous_samples,
        decode_discrete_samples,
        generate_continuous_samples,
        load_gen_model,
        load_tokenizer_decoder,
        prepare_for_metrics,
    )

    device = args.device
    cell_sets = {"default12": DEFAULT_CELLS, "d3pm_interaction": D3PM_INTERACTION_CELLS}
    if args.cells not in cell_sets:
        raise SystemExit(f"--cells must be one of {list(cell_sets)}")
    cells_to_run = cell_sets[args.cells]

    raw = np.load(str(RAW_DATA_PATH))
    real = prepare_for_metrics(torch.from_numpy(raw["test_images"]).float().div(255).unsqueeze(1))
    print(f"real test images: {tuple(real.shape)}")

    results: dict = {
        "_description": (
            "FID at 192 and 2048 Inception dims computed from one generation pass per "
            "cell, so the two metrics see identical samples."
        ),
        "meta": {
            "num_samples": args.num_samples,
            "dims": args.dims,
            "device": device,
            "n_real": int(len(real)),
        },
        "cells": {},
    }

    if not args.skip_floor:
        print(f"real-vs-real floor, {args.floor_boot} resamples per dim")
        t0 = time.time()
        results["real_vs_real_floor"] = real_vs_real_floor(
            real, args.dims, device, n_boot=args.floor_boot, batch_size=args.batch_size
        )
        print(f"  {time.time() - t0:.0f}s")
        for d, v in results["real_vs_real_floor"].items():
            neg = v["n_negative"]
            print(
                f"  dim {d}: mean {v['mean']:.5f} sd {v['std']:.5f} "
                f"range [{v['min']:.5f}, {v['max']:.5f}] negatives {neg}/{v['n_boot']}"
            )

    for cell in cells_to_run:
        key = f"{cell['tok']}_{cell['model']}"
        tuned = "temperature" in cell or "steps" in cell
        name = key + ("_tuned" if tuned else "")
        ckpt = GEN_CKPT_ROOT / f"chestmnist_{key}" / "checkpoint_best.pt"
        if not ckpt.exists():
            print(f"SKIP {name}: no checkpoint at {ckpt}")
            results["cells"][name] = {"error": "missing checkpoint", "path": str(ckpt)}
            continue

        print(f"\n{name}")
        t0 = time.time()
        model, gen_args = load_gen_model(str(ckpt), device)
        tok = load_tokenizer_decoder(cell["tok"], device)

        if gen_args["model"] in DISCRETE:
            s = generate_discrete_with_config(
                model,
                gen_args,
                args.num_samples,
                args.batch_size,
                device,
                cell.get("temperature"),
                cell.get("steps"),
            )
            imgs = decode_discrete_samples(
                tok, s, gen_args.get("spatial_shape"), args.batch_size, device
            )
        else:
            s = generate_continuous_samples(
                model, gen_args, args.num_samples, args.batch_size, device
            )
            imgs = decode_continuous_samples(tok, s, args.batch_size, device)

        gen_time = time.time() - t0
        gen_u8 = prepare_for_metrics(imgs)

        if args.save_samples:
            cache_dir = Path(args.save_samples)
            cache_dir.mkdir(parents=True, exist_ok=True)
            # Single channel is enough: prepare_for_metrics only replicates greyscale
            # to 3 channels, so storing one third the bytes loses nothing.
            np.savez_compressed(cache_dir / f"{name}.npz", samples=gen_u8[:, :1].numpy())
            # Raw generator output too, so later metrics or decodes need no re-sampling.
            (cache_dir / "raw").mkdir(exist_ok=True)
            raw_key = "tokens" if gen_args["model"] in DISCRETE else "latents"
            np.savez_compressed(cache_dir / "raw" / f"{name}.npz", **{raw_key: s.numpy()})

        scores = compute_fid_dims(real, gen_u8, args.dims, device, args.batch_size)
        entry = {
            "label": cell["label"],
            "tokenizer": cell["tok"],
            "generator": cell["model"],
            "tuned": tuned,
            "gen_time_s": round(gen_time, 1),
        }
        for d in args.dims:
            entry[f"fid{d}"] = scores[d]
        results["cells"][name] = entry
        print("  " + "  ".join(f"FID-{d} {scores[d]:.4f}" for d in args.dims))

        del model, tok, imgs, gen_u8
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    ok = [c for c in results["cells"].values() if "fid192" in c and "fid2048" in c]
    if len(ok) >= 3:
        r = spearman([c["fid192"] for c in ok], [c["fid2048"] for c in ok])
        results["spearman_fid192_vs_fid2048"] = r
        results["spearman_n"] = len(ok)
        print(f"\nSpearman FID-192 vs FID-2048 over {len(ok)} cells: {r:.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(results, indent=2))
    tmp.replace(out)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
