#!/usr/bin/env python
"""Analyze factor contributions in the 54-cell discrete FID grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from medlatents.evaluation import (
    decompose_balanced_three_factor,
    discrete_factorial_cells_from_results,
    mean_ranks_by_factor,
)

from paths import RESULTS, TABLES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=RESULTS / "chestmnist_factorial" / "eval_results.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=TABLES / "factorial_effects.json",
    )
    parser.add_argument("--dataset", default="chestmnist")
    args = parser.parse_args()

    results = json.loads(args.results.read_text())
    cells = discrete_factorial_cells_from_results(results, dataset=args.dataset)
    decomposition = decompose_balanced_three_factor(cells, log_transform=True)
    ranks = mean_ranks_by_factor(cells)
    payload = {
        "dataset": args.dataset,
        "num_cells": len(cells),
        "design": "balanced 3x3x6 VQ/LFQ/FSQ x 1024/2048/4096 x six discrete generators",
        "response": "log(FID-192)",
        "method": "descriptive orthogonal fixed-effect corrected sum-of-squares decomposition",
        "inferential": False,
        "decomposition": decomposition,
        "mean_ranks": ranks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
