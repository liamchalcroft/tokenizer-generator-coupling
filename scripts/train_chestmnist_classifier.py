"""Train a small ResNet-18 multi-label classifier on ChestMNIST.

Thin wrapper around ``medlatents.evaluation.classifier_utility`` --- this
script is just dataset glue. The training loop, AUC eval, model factory,
and feature extraction all live in the library so external users can
reproduce the analysis on their own datasets.

Used for two paper analyses:

  (1) **Domain-FID sanity check (Day 1b)**: penultimate (512-d) features
      provide a medical-domain alternative to Inception pool3 (192-d,
      ImageNet-trained). Re-evaluating the headline FID rankings with
      this backbone confirms whether they survive the choice of feature
      extractor.

  (2) **TSTR evaluation (Day 5)**: same classifier architecture trained
      on synthetic samples, tested on real test set, mean AUC across 14
      pathologies vs the real-data baseline.

Usage::

    uv run python scripts/train_chestmnist_classifier.py \
        --train_source real \
        --output checkpoints/chestmnist_classifier_real.pt

For TSTR, --train_source can point to a directory containing
``train_images.npy`` (uint8, (N, 64, 64) or (N, 1, 64, 64)) and
``train_labels.npy`` ((N, 14) float).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from medlatents.evaluation import (
    build_grayscale_resnet18,
    evaluate_auc,
    train_classifier,
)
from torch.utils.data import DataLoader, TensorDataset

from evaluate_chestmnist_generation import RAW_DATA_PATH
from paths import CHECKPOINTS

NUM_CLASSES = 14
PATHOLOGY_NAMES = [
    "atelectasis",
    "cardiomegaly",
    "effusion",
    "infiltration",
    "mass",
    "nodule",
    "pneumonia",
    "pneumothorax",
    "consolidation",
    "edema",
    "emphysema",
    "fibrosis",
    "pleural_thickening",
    "hernia",
]


def load_real_dataset() -> dict[str, torch.Tensor]:
    raw = np.load(str(RAW_DATA_PATH))
    out = {}
    for split in ("train", "val", "test"):
        imgs = torch.from_numpy(raw[f"{split}_images"]).unsqueeze(1)  # (N, 1, H, W)
        labels = torch.from_numpy(raw[f"{split}_labels"]).float()
        out[f"{split}_images"] = imgs
        out[f"{split}_labels"] = labels
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_source",
        default="real",
        help="'real' or path to a directory containing train_images.npy + train_labels.npy",
    )
    parser.add_argument("--output", default=str(CHECKPOINTS / "chestmnist_classifier_real.pt"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Training source: {args.train_source}")
    real = load_real_dataset()

    if args.train_source == "real":
        train_imgs = real["train_images"]
        train_labels = real["train_labels"]
    else:
        src = Path(args.train_source)
        train_imgs = torch.from_numpy(np.load(src / "train_images.npy"))
        if train_imgs.ndim == 3:
            train_imgs = train_imgs.unsqueeze(1)
        train_labels = torch.from_numpy(np.load(src / "train_labels.npy")).float()

    print(f"  train: {tuple(train_imgs.shape)}  labels: {tuple(train_labels.shape)}")
    print(f"  val:   {tuple(real['val_images'].shape)}")
    print(f"  test:  {tuple(real['test_images'].shape)}")

    train_loader = DataLoader(
        TensorDataset(train_imgs, train_labels),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        TensorDataset(real["val_images"], real["val_labels"]),
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(real["test_images"], real["test_labels"]),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = build_grayscale_resnet18(num_classes=NUM_CLASSES).to(args.device)
    history = train_classifier(
        model, train_loader, val_loader, epochs=args.epochs, device=args.device, lr=args.lr
    )

    model.load_state_dict(history["best_state_dict"])
    test = evaluate_auc(model, test_loader, device=args.device, class_names=PATHOLOGY_NAMES)
    print(f"\nbest val mean AUC: {history['best_val_mean_auc']:.4f}")
    print(f"test mean AUC:     {test['mean_auc']:.4f}")
    print("test per-class AUCs:")
    for k, v in test["per_class_auc"].items():
        print(f"  {k:<22s} {v:.4f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": history["best_state_dict"],
            "best_val_mean_auc": history["best_val_mean_auc"],
            "test_mean_auc": test["mean_auc"],
            "test_per_class_auc": test["per_class_auc"],
            "args": vars(args),
            "pathology_names": PATHOLOGY_NAMES,
        },
        out_path,
    )
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
