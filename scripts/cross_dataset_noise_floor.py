#!/usr/bin/env python
"""Real-vs-real FID-192 noise floor for the cross-dataset replication sets.

The cross-dataset table reports FID-192 for 6 cells each on PneumoniaMNIST
(test n=624) and OrganAMNIST (test n=17,778), using 10K generated samples vs
the full test split. This script characterises the *estimator* variance of
FID-192 on each domain at the relevant sample sizes by splitting the test set
into two disjoint random halves and bootstrapping FID-192(half_A, half_B).
No generation needed.

Output: results/tables/cross_dataset_noise_floor.json
"""

from __future__ import annotations

import json

import numpy as np
import torch
from medlatents.evaluation import (
    InceptionPool3FeatureExtractor,
    bootstrap_fid_noise_floor,
)

from paths import MEDTOKENIZERS_ROOT, TABLES

OUT = TABLES / "cross_dataset_noise_floor.json"

# Cross-dataset table: 10K generated samples vs the full test split.
DATASETS = {
    "pneumoniamnist": {"n_per_half": [100, 200, 312]},  # test 624 -> max half 312
    "organamnist": {"n_per_half": [1000, 2500, 5000, 8889]},  # test 17,778 -> max half 8889
}
GENERATED_N = 10000
B = 200
SEED = 42


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = InceptionPool3FeatureExtractor(device=device, batch_size=64)
    results = {
        "_description": (
            "Real-vs-real FID-192 noise floor (estimator variance) for the "
            "cross-dataset replication sets. FID-192 in the cross-dataset table "
            f"uses {GENERATED_N} generated samples vs the full test split; this "
            "floor splits the test set into two disjoint halves of n_per_half "
            "each, B=200 bootstrap reps. Pneumonia's small test set (624) implies "
            "a relatively high floor / wide CI, which bounds how finely its "
            "cross-dataset FIDs can be ranked."
        ),
        "generated_samples_per_cell": GENERATED_N,
        "B": B,
        "datasets": {},
    }
    for ds, cfg in DATASETS.items():
        npz = MEDTOKENIZERS_ROOT / "data" / ds / f"{ds}_64.npz"
        test = np.load(str(npz))["test_images"]
        n_test = int(test.shape[0])
        print(f"\n=== {ds}: test n={n_test} ===")
        imgs = torch.from_numpy(test).unsqueeze(1).to(torch.uint8)  # (N,1,64,64)
        feats = extractor.extract(imgs).numpy()
        nf = bootstrap_fid_noise_floor(
            feats, n_per_half=cfg["n_per_half"], B=B, seed=SEED, verbose=True
        )
        results["datasets"][ds] = {"n_test": n_test, "noise_floor": nf}
        del feats
        if device != "cpu":
            torch.cuda.empty_cache()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {OUT}")
    # Compact summary
    for ds, r in results["datasets"].items():
        floors = {k: round(v["mean"], 4) for k, v in r["noise_floor"].items() if k.isdigit()}
        print(f"  {ds} (n_test={r['n_test']}): floor means {floors}")


if __name__ == "__main__":
    main()
