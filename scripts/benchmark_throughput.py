#!/usr/bin/env python
"""Generation throughput for every paper generator at its exact paper configuration.

Sampling cost depends only on architecture, sequence length, vocabulary and step
count, so models are randomly initialised. Each model is built by the training
script's own constructor from the run_factorial.sh flags, serialised in memory, and
reloaded through evaluate_chestmnist_generation.load_gen_model; samples are drawn
through the same functions the paper's evaluation used (default rows:
generate_discrete_samples / generate_continuous_samples; tuned rows:
sweep_sampling_hps.generate_with_hps, as in revalidate_on_val.py). Evaluation ran
in float32 without autocast, so this does too.

Usage:
    uv run python scripts/benchmark_throughput.py
    uv run python scripts/benchmark_throughput.py --smoke --steps_override 10
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from unittest import mock

import torch

import train_medmnist2d_tokens as train_script
from evaluate_chestmnist_generation import (
    decode_continuous_samples,
    decode_discrete_samples,
    generate_continuous_samples,
    generate_discrete_samples,
    load_gen_model,
)
from paths import REPO, TABLES
from sweep_sampling_hps import generate_with_hps

VOCAB_SIZE = 1024
SEQ_LEN = 64
SPATIAL_SHAPE = [8, 8]
LATENT_SHAPE = [4, 8, 8]

COMMON_FLAGS = [
    "--tokens_path", "unused",
    "--hidden_size", "512", "--depth", "12", "--num_heads", "8",
    "--batch_size", "128", "--mixed_precision", "bf16",
    "--use_ema", "--no-use_wandb",
]  # fmt: skip

GENERATORS = {
    "ar": ("transformer", [], None),
    "maskgit": ("maskgit", ["--use_curriculum", "--mask_schedule", "cosine"], None),
    "dfm": (
        "flow",
        ["--source_dist", "mask", "--scheduler_type", "polynomial", "--scheduler_power", "3.0",
         "--flow_loss_type", "cross_entropy"],
        None,
    ),
    "d3pm": (
        "d3pm",
        ["--d3pm_transition", "absorbing", "--d3pm_loss_type", "cross_entropy",
         "--diffusion_steps", "1000", "--diffusion_schedule", "cosine"],
        None,
    ),
    "d3pm_tuned": (
        "d3pm",
        ["--d3pm_transition", "absorbing", "--d3pm_loss_type", "cross_entropy",
         "--diffusion_steps", "1000", "--diffusion_schedule", "cosine"],
        {"temperature": 0.7, "num_timesteps": 100},
    ),
    "sedd": (
        "sedd",
        ["--d3pm_transition", "absorbing", "--diffusion_steps", "1000",
         "--diffusion_schedule", "cosine"],
        None,
    ),
    "sedd_tuned": (
        "sedd",
        ["--d3pm_transition", "absorbing", "--diffusion_steps", "1000",
         "--diffusion_schedule", "cosine"],
        {"temperature": 0.7, "num_timesteps": 500},
    ),
    "bfn": ("bayesian_flow", [], None),
    "ldm": ("diffusion", ["--diffusion_steps", "1000", "--diffusion_schedule", "cosine"], None),
    "rf": ("continuous_flow", ["--flow_num_steps", "100", "--base_std", "1.0"], None),
}  # fmt: skip

CONTINUOUS = {"diffusion", "continuous_flow"}

TOKENIZER_COMMON = {
    "dim": 2,
    "in_channels": 1,
    "out_channels": 1,
    "z_channels": 64,
    "channels": 64,
    "channels_mult": (1, 2, 4),
    "num_res_blocks": 2,
    "attn_resolutions": (16,),
    "dropout": 0.0,
    "resolution": 64,
    "spatial_compression": 8,
}
DISCRETE_TOKENIZERS = {
    "VQ1024": {"quantizer": "VQ", "num_embeddings": 1024, "use_ema": True, "use_norm": True},
    "LFQ1024": {"quantizer": "LFQ", "codebook_size": 1024, "codebook_dim": 10, "num_codebooks": 1},
    "FSQ1024": {"quantizer": "FSQ", "levels": [4, 4, 4, 4, 4]},
}
CONTINUOUS_TOKENIZERS = {
    "VAEc4": {"latent_channels": 4, "formulation": "VAE"},
    "AE": {"latent_channels": 4, "formulation": "AE"},
}


def training_args(model_type: str, flags: list[str]) -> argparse.Namespace:
    argv = ["train_medmnist2d_tokens.py", *COMMON_FLAGS, "--model", model_type, *flags]
    with mock.patch.object(sys, "argv", argv):
        args = train_script.parse_args()
    # Values the training script infers from the tokenized ChestMNIST data.
    args.seq_len = SEQ_LEN
    args.vocab_size = VOCAB_SIZE
    args.spatial_shape = SPATIAL_SHAPE
    args.latent_shape = LATENT_SHAPE
    args.latent_channels = LATENT_SHAPE[0]
    args.latent_mean = 0.0
    args.latent_std = 1.0
    return args


def build_model(model_type: str, flags: list[str], device: str):
    args = training_args(model_type, flags)
    if model_type in CONTINUOUS:
        model = train_script.create_continuous_model(args)
    else:
        model = train_script.create_discrete_model(args)
    buf = io.BytesIO()
    torch.save({"args": vars(args), "net": model.state_dict()}, buf)
    buf.seek(0)
    del model
    gen_model, gen_args = load_gen_model(buf, device)
    gen_args["model"] = model_type
    return gen_model, gen_args


def build_tokenizer(name: str, device: str) -> torch.nn.Module:
    from medtokenizers.networks import ContinuousTokenizer, DiscreteTokenizer

    if name in DISCRETE_TOKENIZERS:
        model = DiscreteTokenizer(
            **TOKENIZER_COMMON, embedding_dim=16, name=name, **DISCRETE_TOKENIZERS[name]
        )
    else:
        model = ContinuousTokenizer(**TOKENIZER_COMMON, name=name, **CONTINUOUS_TOKENIZERS[name])
    return model.to(device).eval()


def sampling_steps(key: str, gen_args: dict, hp: dict | None) -> int:
    model_type = gen_args["model"]
    if hp is not None:
        return hp["num_timesteps"]
    if model_type == "transformer":
        return gen_args["seq_len"]
    if model_type == "maskgit":
        return 12
    if model_type == "flow":
        return 100
    if model_type in {"d3pm", "sedd", "diffusion"}:
        return gen_args["diffusion_steps"]
    if model_type == "bayesian_flow":
        return gen_args["bfn_num_steps"]
    return gen_args["flow_num_steps"]


def apply_steps_override(gen_args: dict, hp: dict | None, steps: int) -> None:
    if hp is not None:
        hp["num_timesteps"] = steps
    gen_args["diffusion_steps"] = steps
    gen_args["bfn_num_steps"] = steps
    gen_args["flow_num_steps"] = steps


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def generate(model, gen_args, hp, num_samples, batch_size, device):
    if hp is not None:
        return generate_with_hps(model, gen_args, hp, num_samples, batch_size, device)
    if gen_args["model"] in CONTINUOUS:
        return generate_continuous_samples(model, gen_args, num_samples, batch_size, device)
    return generate_discrete_samples(model, gen_args, num_samples, batch_size, device)


def decode(tokenizer, samples, gen_args, batch_size, device):
    if gen_args["model"] in CONTINUOUS:
        return decode_continuous_samples(tokenizer, samples, batch_size, device)
    return decode_discrete_samples(tokenizer, samples, SPATIAL_SHAPE, batch_size, device)


def timed(fn, device: str):
    synchronize(device)
    t0 = time.perf_counter()
    out = fn()
    synchronize(device)
    return out, time.perf_counter() - t0


def git_sha() -> str:
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPO,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return f"{sha}-dirty" if dirty else sha


def device_name(device: str) -> str:
    if device.startswith("cuda"):
        return torch.cuda.get_device_name(torch.device(device))
    if device == "mps":
        return f"Apple MPS ({platform.machine()})"
    return platform.processor() or platform.machine()


def meta_block(args) -> dict:
    import medtokenizers

    return {
        "device": args.device,
        "device_name": device_name(args.device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "medlatents": importlib.metadata.version("medlatents"),
        "medtokenizers": importlib.metadata.version("medtokenizers"),
        "medtokenizers_path": str(Path(medtokenizers.__file__).parent),
        "git_sha": git_sha(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "host": platform.node(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "num_samples": args.num_samples,
        "repeats": args.repeats,
        "seed": args.seed,
        "steps_override": args.steps_override,
        "smoke": args.smoke,
        "weights": "random init (sampling cost is weight-independent)",
    }


def benchmark(key: str, args, tokenizers: dict) -> dict:
    model_type, flags, tuned_hp = GENERATORS[key]
    hp = dict(tuned_hp) if tuned_hp is not None else None
    torch.manual_seed(args.seed)
    model, gen_args = build_model(model_type, flags, args.device)
    if args.steps_override is not None:
        apply_steps_override(gen_args, hp, args.steps_override)

    # sweep_sampling_hps caps D3PM/SEDD batches at 64 for the tuned (generate_with_hps) path.
    batch_size = min(args.batch_size, 64) if hp is not None else args.batch_size
    tok_name = args.continuous_tokenizer if model_type in CONTINUOUS else args.discrete_tokenizer
    if tok_name not in tokenizers:
        tokenizers[tok_name] = build_tokenizer(tok_name, args.device)
    tokenizer = tokenizers[tok_name]

    warm = generate(model, gen_args, hp, min(batch_size, args.num_samples), batch_size, args.device)
    decode(tokenizer, warm, gen_args, args.batch_size, args.device)

    gen_seconds, decode_seconds = [], []
    for _ in range(args.repeats):
        samples, dt = timed(
            partial(generate, model, gen_args, hp, args.num_samples, batch_size, args.device),
            args.device,
        )
        gen_seconds.append(dt)
        _, dt = timed(
            partial(decode, tokenizer, samples, gen_args, args.batch_size, args.device),
            args.device,
        )
        decode_seconds.append(dt)

    median_gen = statistics.median(gen_seconds)
    median_dec = statistics.median(decode_seconds)
    entry = {
        "model_type": model_type,
        "steps": sampling_steps(key, gen_args, hp),
        "temperature": hp["temperature"] if hp is not None else 0.9,
        "batch_size": batch_size,
        "dtype": "float32",
        "autocast": None,
        "sampling_path": "sweep_sampling_hps.generate_with_hps"
        if hp is not None
        else "evaluate_chestmnist_generation.generate_"
        + ("continuous" if model_type in CONTINUOUS else "discrete")
        + "_samples",
        "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
        "vocab_size": gen_args["vocab_size"],
        "seq_len": gen_args["seq_len"],
        "latent_shape": gen_args["latent_shape"] if model_type in CONTINUOUS else None,
        "num_samples": args.num_samples,
        "gen_seconds": gen_seconds,
        "gen_seconds_median": median_gen,
        "gen_seconds_min": min(gen_seconds),
        "samples_per_s_median": args.num_samples / median_gen,
        "samples_per_s_max": args.num_samples / min(gen_seconds),
        "decode_tokenizer": tok_name,
        "decode_batch_size": args.batch_size,
        "decode_seconds": decode_seconds,
        "decode_seconds_median": median_dec,
        "decode_seconds_min": min(decode_seconds),
        "end_to_end_samples_per_s_median": args.num_samples / (median_gen + median_dec),
    }
    del model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return entry


def main():
    parser = argparse.ArgumentParser(description="Benchmark generation throughput")
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--generators", nargs="+", default=list(GENERATORS), choices=list(GENERATORS)
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--discrete_tokenizer", default="LFQ1024", choices=list(DISCRETE_TOKENIZERS)
    )
    parser.add_argument(
        "--continuous_tokenizer", default="VAEc4", choices=list(CONTINUOUS_TOKENIZERS)
    )
    parser.add_argument(
        "--steps_override",
        type=int,
        default=None,
        help="Replace the step count of D3PM/SEDD/BFN/LDM/RF (smoke testing only). "
        "AR, MaskGIT and DFM step counts are fixed by the evaluation code.",
    )
    parser.add_argument("--smoke", action="store_true", help="num_samples=64, repeats=1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out", type=str, default=str(TABLES / "throughput_benchmark.json")
    )
    args = parser.parse_args()
    if args.smoke:
        args.num_samples = 64
        args.repeats = 1

    results = {"meta": meta_block(args), "generators": {}}
    tokenizers: dict[str, torch.nn.Module] = {}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        for key in args.generators:
            print(f"[{key}] ...", flush=True)
            entry = benchmark(key, args, tokenizers)
            results["generators"][key] = entry
            print(
                f"[{key}] steps={entry['steps']} bs={entry['batch_size']} "
                f"gen={entry['gen_seconds_median']:.2f}s "
                f"({entry['samples_per_s_median']:.2f} samples/s, "
                f"{entry['gen_seconds_median'] / entry['steps'] / math.ceil(args.num_samples / entry['batch_size']) * 1e3:.1f} ms/step/batch) "
                f"decode={entry['decode_seconds_median']:.2f}s",
                flush=True,
            )
            tmp = out.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(results, indent=2))
            os.replace(tmp, out)

    print(f"\n{'Generator':<12} {'Steps':>6} {'BS':>4} {'Params':>7} {'Gen s (med)':>12} "
          f"{'Gen s (min)':>12} {'Samples/s':>10} {'Decode s':>9}")  # fmt: skip
    for key, e in results["generators"].items():
        print(
            f"{key:<12} {e['steps']:>6} {e['batch_size']:>4} {e['params_M']:>6.1f}M "
            f"{e['gen_seconds_median']:>12.2f} {e['gen_seconds_min']:>12.2f} "
            f"{e['samples_per_s_median']:>10.2f} {e['decode_seconds_median']:>9.2f}"
        )
    print(f"\n{results['meta']['device_name']} | torch {results['meta']['torch']} | "
          f"CUDA {results['meta']['cuda']} | medlatents {results['meta']['medlatents']} | "
          f"{results['meta']['git_sha']}")  # fmt: skip
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
