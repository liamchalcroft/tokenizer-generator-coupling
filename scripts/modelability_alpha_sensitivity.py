#!/usr/bin/env python
"""Smoothing sensitivity of the generator-free predictors.

latent_modelability.py stores, per tokenizer and axis, the predictive gain at every
smoothing constant in its grid (sweep_left / sweep_up) and the validation-selected
constant. This reports VQ-vs-rest rank-AUC and Spearman against best FID for every
statistic at every grid value, next to the validation-selected result, so the
sensitivity to alpha can be stated rather than inferred from one point.

Conditional entropy at each alpha is recovered exactly as marginal minus gain. The
alpha-independent statistics (PSNR, marginal entropy, per-position dispersion) are
listed once. Nothing is refitted; the input is results/tables/latent_modelability.json.

Output: results/tables/latent_modelability_alpha_sensitivity.json
"""

import argparse
import json
import math
from pathlib import Path

from latent_modelability import group_auc, spearman
from paths import TABLES


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=TABLES / "latent_modelability.json")
    ap.add_argument(
        "--out", type=Path, default=TABLES / "latent_modelability_alpha_sensitivity.json"
    )
    args = ap.parse_args()
    src = json.loads(args.src.read_text())
    rows = src["tokenizers"]
    names = list(rows)
    vq = {n for n in names if n.startswith("VQ")}
    fid = [rows[n]["best_fid"] for n in names]
    grid = [f"{a:g}" for a in src["meta"]["alpha_grid"]]

    def summarise(values: dict, sign: float = 1.0) -> dict:
        tmp = {n: {"v": values[n]} for n in names}
        return {
            "vq_vs_rest_auc": group_auc(tmp, "v", vq, sign),
            "spearman_vs_best_fid": spearman([values[n] for n in names], fid),
        }

    out = {
        "_description": (
            "VQ-vs-rest rank-AUC (1.0 = the three VQ tokenizers all score on the same "
            "side of the six LFQ/FSQ ones) and Spearman against best "
            "observed FID-192, for each generator-free statistic at every smoothing "
            "constant in the grid, plus the validation-selected result. Derived from "
            "latent_modelability.json by scripts/modelability_alpha_sensitivity.py; "
            "alpha is never chosen against FID."
        ),
        "alpha_grid": grid,
        "alpha_independent": {
            "psnr": summarise({n: rows[n]["psnr"] for n in names}, sign=-1.0),
            "marginal_entropy_norm": summarise(
                {n: rows[n]["marginal_entropy_norm"] for n in names}
            ),
            "per_pos_dispersion_norm": summarise(
                {n: rows[n]["per_pos_dispersion_norm"] for n in names}
            ),
        },
        "alpha_dependent": {},
        "validation_selected_alpha": {
            n: {ax: rows[n][f"alpha_selected_{ax}"] for ax in ("left", "up")} for n in names
        },
    }

    for ax in ("left", "up"):
        gain = {a: {n: rows[n][f"sweep_{ax}"][a] for n in names} for a in grid}
        cond = {
            a: {
                n: (rows[n]["marginal_entropy_bits"] - gain[a][n]) / math.log2(rows[n]["vocab"])
                for n in names
            }
            for a in grid
        }
        for stat, table, valsel_key in (
            (f"predictive_gain_{ax}", gain, f"predictive_gain_{ax}_valsel"),
            (f"cond_entropy_{ax}_norm", cond, f"cond_entropy_{ax}_valsel_norm"),
        ):
            per_alpha = {a: summarise(table[a]) for a in grid}
            aucs = [v["vq_vs_rest_auc"] for v in per_alpha.values()]
            out["alpha_dependent"][stat] = {
                "per_alpha": per_alpha,
                "validation_selected": summarise({n: rows[n][valsel_key] for n in names}),
                "auc_min": min(aucs),
                "auc_max": max(aucs),
                "alphas_with_auc_1": [a for a in grid if per_alpha[a]["vq_vs_rest_auc"] == 1.0],
            }

    path = args.out
    path.write_text(json.dumps(out, indent=2))

    print(f"{'statistic':<28}" + "".join(f"{a:>7}" for a in grid) + "  valsel")
    for stat, v in out["alpha_dependent"].items():
        cells = "".join(f"{v['per_alpha'][a]['vq_vs_rest_auc']:>7.3f}" for a in grid)
        print(f"{stat:<28}{cells}  {v['validation_selected']['vq_vs_rest_auc']:.3f}")
    for stat, v in out["alpha_independent"].items():
        print(f"{stat:<28}{'(alpha-independent)':>30}  {v['vq_vs_rest_auc']:.3f}")
    print("\nvalidation-selected alpha (left, up):")
    for n, a in out["validation_selected_alpha"].items():
        print(f"  {n:<8} {a['left']:g}, {a['up']:g}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
