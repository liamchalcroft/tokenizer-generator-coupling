#!/usr/bin/env python
"""Compute-accounting for the study -> results/tables/compute_budget.json.

Two confidence tiers:
  MEASURED (high confidence): generation/eval and HP-sweep GPU-seconds are the
    recorded wall-clocks of those jobs (generation_time_s / time_s fields).
  ESTIMATED (rough): training GPU-hours are inferred from the wall-clock gap
    between consecutive checkpoint-file writes for sequentially-trained runs.
    This includes per-run eval/checkpoint/idle overhead, so it is an upper
    bound on pure-train compute; deltas > a per-type cap are treated as
    batch-boundary idle and dropped (the first run of each batch is unmeasured).
All on a single NVIDIA A6000 (48 GB).
"""

from __future__ import annotations

import glob
import json
import os
import re
import statistics

from paths import CHECKPOINTS, MEDTOKENIZERS_ROOT, RESULTS, TABLES

OUT = TABLES / "compute_budget.json"

ARCH_SUFFIXES = [
    "continuous_flow",
    "bayesian_flow",
    "transformer",
    "maskgit",
    "d3pm",
    "sedd",
    "diffusion",
    "flow",
]  # specific-first; 'flow' last


def gen_hours(p):
    if not os.path.exists(p):
        return 0.0, 0
    with open(p) as f:
        d = json.load(f)
    secs = [
        v["generation_time_s"]
        for v in d.values()
        if isinstance(v, dict) and "generation_time_s" in v
    ]
    return sum(secs) / 3600.0, len(secs)


def sweep_hours(p, screen_field="time_s"):
    if not os.path.exists(p):
        return 0.0
    with open(p) as f:
        d = json.load(f)
    return (
        sum(v[screen_field] for v in d.values() if isinstance(v, dict) and screen_field in v)
        / 3600.0
    )


def reval_hours():
    with open(RESULTS / "chestmnist_factorial/revalidation_results.json") as f:
        d = json.load(f)
    screen = (
        sum(
            v["time_s"]
            for r in d["cells"].values()
            for v in r.get("screen", {}).values()
            if "time_s" in v
        )
        / 3600.0
    )
    final = sum(r["final_time_s"] for r in d["cells"].values() if "final_time_s" in r) / 3600.0
    return screen, final


def tok_family(name):
    s = re.sub(r"(chestmnist_|_final)", "", name)
    for fam in ("RESFSQ", "VQ", "LFQ", "FSQ"):
        if s.startswith(fam):
            return fam
    if s.startswith("AE"):
        return "AE"
    return "VAE"  # VAE, VAE1e5/1e7, VAEc1-16


def gen_arch(name):
    for a in ARCH_SUFFIXES:
        if name.endswith("_" + a):
            return a
    return "other"


def train_deltas(dirs, cap_h):
    dirs = [d for d in dirs if os.path.exists(d)]  # skip broken symlinks
    runs = sorted(((os.path.getmtime(d), os.path.basename(d)) for d in dirs), key=lambda x: x[0])
    out = [(runs[i][1], (runs[i][0] - runs[i - 1][0]) / 3600.0) for i in range(1, len(runs))]
    return [(n, dt) for n, dt in out if 0.05 < dt < cap_h]


def grouped(deltas, keyfn):
    g = {}
    for n, dt in deltas:
        g.setdefault(keyfn(n), []).append(dt)
    return {
        k: {"n_runs": len(v), "median_h": round(statistics.median(v), 2), "sum_h": round(sum(v), 1)}
        for k, v in sorted(g.items())
    }


def main():
    # ---- MEASURED ----
    gen = {}
    for ds, p in [
        ("chestmnist", "chestmnist_factorial/eval_results.json"),
        ("pneumoniamnist", "pneumoniamnist_factorial/eval_results.json"),
        ("organamnist", "organamnist_factorial/eval_results.json"),
    ]:
        h, n = gen_hours(RESULTS / p)
        gen[ds] = {"gpu_hours": round(h, 1), "n_cells": n}
    gen_total = round(sum(v["gpu_hours"] for v in gen.values()), 1)

    sweep2k = round(sweep_hours(RESULTS / "chestmnist_factorial/sweep_results.json"), 1)
    sweep10k = round(
        sweep_hours(RESULTS / "chestmnist_factorial/sweep_validation.json"), 1
    )
    rscreen, rfinal = reval_hours()
    hp_total = round(sweep2k + sweep10k + rscreen + rfinal, 1)

    # ---- ESTIMATED (training) ----
    tok = train_deltas(glob.glob(str(MEDTOKENIZERS_ROOT / "checkpoints/medmnist/chestmnist_*_final")), cap_h=24)
    genr = train_deltas(
        [
            d
            for d in glob.glob(str(CHECKPOINTS / "chestmnist_factorial/chestmnist_*"))
            if os.path.isdir(d)
        ],
        cap_h=12,
    )
    tok_by_fam = grouped(tok, tok_family)
    gen_by_arch = grouped(genr, gen_arch)
    tok_train_h = round(sum(dt for _, dt in tok), 1)
    gen_train_h = round(sum(dt for _, dt in genr), 1)

    measured = round(gen_total + hp_total, 1)
    estimated = round(tok_train_h + gen_train_h, 1)

    out = {
        "_description": (
            "Single NVIDIA A6000 (48 GB). MEASURED = recorded job wall-clocks "
            "(generation_time_s / sweep time_s). ESTIMATED training = wall-clock "
            "between consecutive checkpoint writes (upper bound; includes per-run "
            "overhead, first run of each batch unmeasured). ChestMNIST unless noted."
        ),
        "measured_gpu_hours": {
            "generation_eval": {**gen, "total": gen_total},
            "hp_sweep": {
                "chestmnist_2k_screening": sweep2k,
                "lfq1024_10k_validation": sweep10k,
                "val_reselection_2k_screen": round(rscreen, 1),
                "val_reselection_10k_finals": round(rfinal, 1),
                "total": hp_total,
            },
            "measured_total": measured,
        },
        "estimated_training_gpu_hours": {
            "method": "wall-clock between consecutive checkpoint writes; per-run, batch-idle dropped (cap tok 24h / gen 12h)",
            "tokenizers_by_family": tok_by_fam,
            "tokenizers_total": tok_train_h,
            "generators_by_architecture": gen_by_arch,
            "generators_total": gen_train_h,
            "estimated_total": estimated,
        },
        "grand_total_a6000_hours": {
            "measured": measured,
            "estimated_training": estimated,
            "approx_total": round(measured + estimated, 1),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
