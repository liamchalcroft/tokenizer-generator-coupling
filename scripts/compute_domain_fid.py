#!/usr/bin/env python
"""Compute ChestMNIST domain-FID with a trained classifier backbone."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from medlatents.evaluation import (
    build_grayscale_resnet18,
    extract_classifier_features,
    fid_from_features,
)

from evaluate_chestmnist_generation import (
    DISCRETE_MODELS,
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
from paths import CHECKPOINTS, TABLES

DEFAULT_CELLS = [
    "chestmnist_LFQ1024_transformer",
    "chestmnist_FSQ1024_transformer",
    "chestmnist_VQ1024_transformer",
    "chestmnist_LFQ1024_d3pm",
    "chestmnist_VAEc2_diffusion",
    "chestmnist_AEdisc_continuous_flow",
]


def load_classifier(path: Path, device: str) -> tuple[torch.nn.Module, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    num_classes = len(ckpt.get("pathology_names", [])) or 14
    model = build_grayscale_resnet18(num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def load_fid192_scores(path: Path) -> dict[str, float]:
    raw = json.loads(path.read_text())
    return {name: float(result["fid"]) for name, result in raw.items() if "fid" in result}


def rankdata(values: list[float]) -> np.ndarray:
    values_arr = np.asarray(values, dtype=float)
    order = np.argsort(values_arr, kind="mergesort")
    ranks = np.empty(len(values_arr), dtype=float)
    sorted_values = values_arr[order]
    start = 0
    while start < len(values_arr):
        end = start + 1
        while end < len(values_arr) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman_rho(a: list[float], b: list[float]) -> float:
    ar = rankdata(a)
    br = rankdata(b)
    if np.std(ar) == 0 or np.std(br) == 0:
        return float("nan")
    return float(np.corrcoef(ar, br)[0, 1])


def spearman_rho_or_none(rows: list[dict]) -> float | None:
    if len(rows) < 2:
        return None
    return spearman_rho([r["fid192"] for r in rows], [r["domain_fid"] for r in rows])


def real_classifier_features(
    model: torch.nn.Module,
    device: str,
    cache_dir: Path,
    batch_size: int,
) -> np.ndarray:
    path = cache_dir / "real_test_classifier_features.npy"
    if path.exists():
        print(f"loading cached real features: {path}")
        return np.load(path)
    raw = np.load(str(RAW_DATA_PATH))
    images = torch.from_numpy(raw["test_images"]).unsqueeze(1).to(torch.uint8)
    print(f"extracting real classifier features: {tuple(images.shape)}")
    feats = extract_classifier_features(model, images, device=device, batch_size=batch_size).numpy()
    np.save(path, feats)
    return feats


@torch.inference_mode()
def generated_classifier_features(
    cell_name: str,
    model: torch.nn.Module,
    device: str,
    cache_dir: Path,
    num_samples: int,
    gen_batch_size: int,
    feature_batch_size: int,
    tokenizer_cache: dict[str, torch.nn.Module],
) -> np.ndarray:
    path = cache_dir / f"{cell_name}_classifier_features_n{num_samples}.npy"
    if path.exists():
        print(f"loading cached generated features: {path.name}")
        return np.load(path)

    ckpt_path = GEN_CKPT_ROOT / cell_name / "checkpoint_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    parsed = parse_run_name(cell_name)
    if parsed is None:
        raise ValueError(f"cannot parse run name: {cell_name}")
    tok_type, model_type = parsed

    print(f"generating {num_samples} samples for {cell_name}")
    gen_model, gen_args = load_gen_model(str(ckpt_path), device)
    gen_args["model"] = model_type
    is_discrete = model_type in DISCRETE_MODELS
    if is_discrete:
        samples = generate_discrete_samples(
            gen_model, gen_args, num_samples, gen_batch_size, device
        )
    else:
        samples = generate_continuous_samples(
            gen_model, gen_args, num_samples, gen_batch_size, device
        )
    del gen_model
    torch.cuda.empty_cache()

    if tok_type not in tokenizer_cache:
        print(f"loading {tok_type} tokenizer decoder")
        tokenizer_cache[tok_type] = load_tokenizer_decoder(tok_type, device)
    tokenizer = tokenizer_cache[tok_type]
    if is_discrete:
        images = decode_discrete_samples(
            tokenizer,
            samples,
            gen_args.get("spatial_shape", None),
            gen_batch_size,
            device,
        )
    else:
        images = decode_continuous_samples(tokenizer, samples, gen_batch_size, device)
    images = images.clamp(0, 1)
    feats = extract_classifier_features(
        model,
        images,
        device=device,
        batch_size=feature_batch_size,
    ).numpy()
    np.save(path, feats)
    del samples, images
    torch.cuda.empty_cache()
    return feats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", nargs="*", default=DEFAULT_CELLS)
    parser.add_argument(
        "--classifier", type=Path, default=CHECKPOINTS / "chestmnist_classifier_real.pt"
    )
    parser.add_argument("--fid192-results", type=Path, default=GEN_CKPT_ROOT / "eval_results.json")
    parser.add_argument(
        "--output", type=Path, default=TABLES / "domain_fid_validation.json"
    )
    parser.add_argument("--cache-dir", type=Path, default=GEN_CKPT_ROOT / "domain_fid_cache")
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--gen-batch-size", type=int, default=128)
    parser.add_argument("--feature-batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    classifier, classifier_ckpt = load_classifier(args.classifier, args.device)
    fid192_scores = load_fid192_scores(args.fid192_results)
    real_feats = real_classifier_features(
        classifier,
        device=args.device,
        cache_dir=args.cache_dir,
        batch_size=args.feature_batch_size,
    )

    tokenizer_cache: dict[str, torch.nn.Module] = {}
    rows = []
    t0 = time.time()
    for cell in args.cells:
        gen_feats = generated_classifier_features(
            cell,
            classifier,
            device=args.device,
            cache_dir=args.cache_dir,
            num_samples=args.num_samples,
            gen_batch_size=args.gen_batch_size,
            feature_batch_size=args.feature_batch_size,
            tokenizer_cache=tokenizer_cache,
        )
        domain_fid = fid_from_features(real_feats, gen_feats)
        rows.append(
            {
                "cell": cell,
                "fid192": fid192_scores[cell],
                "domain_fid": float(domain_fid),
                "num_real": int(len(real_feats)),
                "num_gen": int(len(gen_feats)),
            }
        )
        print(f"{cell}: FID-192={fid192_scores[cell]:.4f} domain-FID={domain_fid:.4f}")

        tmp = {
            "classifier": str(args.classifier),
            "classifier_test_mean_auc": classifier_ckpt.get("test_mean_auc"),
            "feature_dim": int(real_feats.shape[1]),
            "num_samples": args.num_samples,
            "rows": rows,
            "spearman_rho": spearman_rho_or_none(rows),
            "time_s": time.time() - t0,
        }
        args.output.write_text(json.dumps(tmp, indent=2))

    rho = spearman_rho_or_none(rows)
    out = {
        "classifier": str(args.classifier),
        "classifier_test_mean_auc": classifier_ckpt.get("test_mean_auc"),
        "classifier_best_val_mean_auc": classifier_ckpt.get("best_val_mean_auc"),
        "feature_dim": int(real_feats.shape[1]),
        "num_samples": args.num_samples,
        "rows": rows,
        "spearman_rho": rho,
        "time_s": time.time() - t0,
    }
    args.output.write_text(json.dumps(out, indent=2))

    print("\nSummary")
    print(f"Spearman rho: {rho:.4f}" if rho is not None else "Spearman rho: n/a")
    for row in sorted(rows, key=lambda r: r["domain_fid"]):
        print(
            f"  {row['cell']:<38s} domain-FID={row['domain_fid']:.4f} FID-192={row['fid192']:.4f}"
        )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
