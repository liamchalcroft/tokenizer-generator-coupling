#!/usr/bin/env python
"""Sweep sampling hyperparameters for trained ChestMNIST generative models.

Loads a trained checkpoint once, then generates samples under many different
sampling configurations and evaluates FID/IS for each. No retraining required.

Usage:
    # Sweep all models with 2K screening samples
    uv run python scripts/sweep_sampling_hps.py --num_samples 2000

    # Sweep a specific model
    uv run python scripts/sweep_sampling_hps.py --filter LFQ_transformer --num_samples 2000

    # Full evaluation of top configs
    uv run python scripts/sweep_sampling_hps.py --filter LFQ_transformer --num_samples 10000 \
        --temperature 0.8 --top_k 256
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from medlatents.utils.numerics import EPS_PROB

from evaluate_chestmnist_generation import (
    GEN_CKPT_ROOT,
    RAW_DATA_PATH,
    compute_fid_is,
    decode_continuous_samples,
    decode_discrete_samples,
    load_gen_model,
    load_tokenizer_decoder,
    parse_run_name,
    prepare_for_metrics,
)
from paths import RESULTS

DISCRETE_MODELS = {"transformer", "maskgit", "flow", "d3pm", "sedd", "bayesian_flow"}

# ---------------------------------------------------------------------------
# Sampling HP grids per model type
# ---------------------------------------------------------------------------

SAMPLING_GRIDS = {
    "transformer": {
        "temperature": [0.5, 0.7, 0.8, 0.9, 1.0, 1.2],
        "top_k": [0, 64, 256],
    },
    "maskgit": {
        "temperature": [0.5, 0.7, 0.9, 1.0, 1.2],
        "num_steps": [4, 8, 12, 16, 24, 32],
        "mask_schedule": ["cosine", "linear"],
    },
    "flow": {
        "temperature": [0.5, 0.7, 0.9, 1.0, 1.2],
        "num_euler_steps": [10, 25, 50, 100, 200],
    },
    "d3pm": {
        "temperature": [0.5, 0.7, 0.9, 1.0, 1.2],
        "num_timesteps": [100, 250, 500, 1000],
    },
    "sedd": {
        "temperature": [0.5, 0.7, 0.9, 1.0, 1.2],
        "num_timesteps": [100, 250, 500, 1000],
    },
    "bayesian_flow": {
        "temperature": [0.3, 0.5, 0.7, 0.9, 1.0, 1.2],
        "num_steps": [10, 25, 50, 100, 200],
    },
    "diffusion": {
        "temperature": [0.5, 0.7, 0.8, 0.9, 1.0],
        "num_inference_steps": [25, 50, 100, 250, 1000],
    },
}


# Every per-generator name for the sampler step count, so a single --num_steps
# override can target whichever one a given grid uses.
_STEP_KEYS = {"num_steps", "num_timesteps", "num_euler_steps", "num_inference_steps"}


def expand_grid(grid: dict) -> list[dict]:
    """Expand a dict of lists into a list of dicts (cartesian product)."""
    keys = list(grid.keys())
    values = list(grid.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


# ---------------------------------------------------------------------------
# Model-specific generation with configurable sampling HPs
# ---------------------------------------------------------------------------


@torch.inference_mode()
def generate_with_hps(
    model: torch.nn.Module,
    args: dict,
    hp: dict,
    num_samples: int,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    """Generate samples with specific sampling hyperparameters."""
    model_type = args["model"]

    if model_type == "transformer":
        return _gen_transformer(model, args, hp, num_samples, batch_size, device)
    elif model_type == "maskgit":
        return _gen_maskgit(model, args, hp, num_samples, batch_size, device)
    elif model_type == "flow":
        return _gen_flow(model, args, hp, num_samples, batch_size, device)
    elif model_type == "d3pm":
        return _gen_d3pm(model, args, hp, num_samples, batch_size, device)
    elif model_type == "sedd":
        return _gen_sedd(model, args, hp, num_samples, batch_size, device)
    elif model_type == "bayesian_flow":
        return _gen_bfn(model, args, hp, num_samples, batch_size, device)
    elif model_type == "diffusion":
        return _gen_diffusion(model, args, hp, num_samples, batch_size, device)
    else:
        raise ValueError(f"Unknown model: {model_type}")


def _gen_transformer(model, args, hp, num_samples, batch_size, device):
    seq_len = args["seq_len"]
    temperature = hp.get("temperature", 0.9)
    top_k = hp.get("top_k", 0)

    all_samples = []
    for start in range(0, num_samples, batch_size):
        bs = min(batch_size, num_samples - start)
        bos = model.special_tokens.bos if getattr(model, "special_tokens", None) else 0
        prompt = torch.full((bs, 1), bos, dtype=torch.long, device=device)

        gen_kwargs = {"max_length": seq_len + 1, "temperature": temperature}
        if top_k > 0:
            gen_kwargs["top_k"] = top_k

        samples = model.generate(prompt, **gen_kwargs)
        samples = samples[:, 1:]  # strip BOS
        samples = samples.clamp(0, args["vocab_size"] - 1)
        all_samples.append(samples.cpu())

    return torch.cat(all_samples, dim=0)


def _gen_maskgit(model, args, hp, num_samples, batch_size, device):
    seq_len = args["seq_len"]
    temperature = hp.get("temperature", 0.9)
    num_steps = hp.get("num_steps", 12)
    mask_schedule = hp.get("mask_schedule", "cosine")

    all_samples = []
    for start in range(0, num_samples, batch_size):
        bs = min(batch_size, num_samples - start)
        x = torch.full((bs, seq_len), model.mask_token, dtype=torch.long, device=device)

        # Use external MaskGITScheduler for schedule control
        try:
            from medlatents.sampling.maskgit.sampling import MaskGITScheduler

            scheduler = MaskGITScheduler(
                num_steps=num_steps,
                seq_length=seq_len,
                mask_schedule=mask_schedule,
            )
            # Manual generation loop with scheduler
            for step in range(num_steps):
                logits = model(x)  # [B, L, V]
                probs = F.softmax(logits / temperature, dim=-1)

                # Sample tokens
                pred_tokens = torch.multinomial(
                    probs.reshape(-1, probs.size(-1)).clamp_min(EPS_PROB), 1
                ).reshape(bs, seq_len)

                # Get confidences for predicted tokens
                confidences = torch.gather(probs, -1, pred_tokens.unsqueeze(-1)).squeeze(-1)

                # Determine how many to unmask this step
                is_final = step == num_steps - 1
                if is_final:
                    x = pred_tokens
                else:
                    ratio = scheduler.get_mask_ratio(step, num_steps)
                    num_masked = max(1, int(seq_len * ratio))

                    # Keep highest confidence predictions, re-mask the rest
                    _, indices = confidences.sort(dim=-1, descending=True)
                    keep_mask = torch.zeros_like(x, dtype=torch.bool)
                    for b_idx in range(bs):
                        keep_mask[b_idx, indices[b_idx, : seq_len - num_masked]] = True

                    x = torch.where(keep_mask, pred_tokens, model.mask_token)

        except Exception:
            # Fallback to model's built-in generate (covers ImportError,
            # AttributeError, and TypeError from upstream MaskGITScheduler API
            # drift -- e.g. seq_length kwarg was removed).
            x = torch.full((bs, seq_len), model.mask_token, dtype=torch.long, device=device)
            x = model.generate(x, num_steps=num_steps, temperature=temperature)

        samples = x.clamp(0, args["vocab_size"] - 1)
        all_samples.append(samples.cpu())

    return torch.cat(all_samples, dim=0)


def _gen_flow(model, args, hp, num_samples, batch_size, device):
    from medlatents.flow_matching.core import MixtureDiscreteEulerSolver, ModelWrapper
    from medlatents.flow_matching.discrete import get_path, get_source_distribution

    seq_len = args["seq_len"]
    vocab_size = args["vocab_size"]
    temperature = hp.get("temperature", 0.9)
    num_euler_steps = hp.get("num_euler_steps", 100)

    path = get_path(args.get("scheduler_type", "polynomial"), args.get("scheduler_power", 2.0))
    source_dist = get_source_distribution(args.get("source_dist", "uniform"), vocab_size)

    class WrappedModel(ModelWrapper):
        @torch.no_grad()
        def forward(self, x, t, **extras):
            # MixtureDiscreteEulerSolver applies the softmax itself, so return logits.
            logits = self.model(x=x, t=t, **extras).float()
            return logits / temperature

    wrapped = WrappedModel(model=model)
    add_token = 1 if getattr(source_dist, "masked", False) else 0
    solver = MixtureDiscreteEulerSolver(
        model=wrapped,
        path=path,
        vocabulary_size=vocab_size + add_token,
    )

    all_samples = []
    for start in range(0, num_samples, batch_size):
        bs = min(batch_size, num_samples - start)
        x_init = source_dist.sample((bs, seq_len), device=device)
        # Same call as generate_discrete_samples, with the solver's default time grid.
        samples = solver.sample(
            x_init=x_init,
            step_size=1 / num_euler_steps,
            verbose=False,
        )
        samples = samples.clamp(0, vocab_size - 1)
        all_samples.append(samples.cpu())

    return torch.cat(all_samples, dim=0)


def _gen_d3pm(model, args, hp, num_samples, batch_size, device):
    from medlatents.diffusion.d3pm import D3PM

    vocab_size = args["vocab_size"]
    seq_len = args["seq_len"]
    temperature = hp.get("temperature", 0.9)
    num_timesteps = hp.get("num_timesteps", 1000)

    d3pm = D3PM(
        num_classes=vocab_size,
        num_timesteps=num_timesteps,
        schedule_type=args.get("diffusion_schedule", "cosine"),
        transition_type=args.get("d3pm_transition", "absorbing"),
        device=device,
    )

    all_samples = []
    # Use smaller batch for D3PM to avoid memory issues
    d3pm_batch = min(batch_size, 64)
    for start in range(0, num_samples, d3pm_batch):
        bs = min(d3pm_batch, num_samples - start)
        samples = d3pm.sample(model, (bs, seq_len), temperature=temperature)
        samples = samples.clamp(0, vocab_size - 1)
        all_samples.append(samples.cpu())

    return torch.cat(all_samples, dim=0)


def _gen_sedd(model, args, hp, num_samples, batch_size, device):
    """SEDD shares the D3PM sampling path but trains with continuous t in [0, 1).
    D3PM.sample() passes integer timesteps in [0, num_timesteps), so we wrap the
    model to normalize them before forwarding.
    """
    from medlatents.diffusion.d3pm import D3PM

    vocab_size = args["vocab_size"]
    seq_len = args["seq_len"]
    temperature = hp.get("temperature", 0.9)
    num_timesteps = hp.get("num_timesteps", 1000)

    d3pm = D3PM(
        num_classes=vocab_size,
        num_timesteps=num_timesteps,
        schedule_type=args.get("diffusion_schedule", "cosine"),
        transition_type=args.get("d3pm_transition", "absorbing"),
        device=device,
    )

    class SEDDModelWrapper(torch.nn.Module):
        def __init__(self, m, n):
            super().__init__()
            self.model = m
            self.num_timesteps = n

        def forward(self, x, t, **kwargs):
            t_cont = t.float() / self.num_timesteps
            return self.model(x=x, t=t_cont, **kwargs)

    wrapped = SEDDModelWrapper(model, num_timesteps)

    all_samples = []
    sedd_batch = min(batch_size, 64)
    for start in range(0, num_samples, sedd_batch):
        bs = min(sedd_batch, num_samples - start)
        samples = d3pm.sample(wrapped, (bs, seq_len), temperature=temperature)
        samples = samples.clamp(0, vocab_size - 1)
        all_samples.append(samples.cpu())

    return torch.cat(all_samples, dim=0)


def _gen_bfn(model, args, hp, num_samples, batch_size, device):
    seq_len = args["seq_len"]
    vocab_size = args["vocab_size"]
    temperature = hp.get("temperature", 0.9)
    num_steps = hp.get("num_steps", 100)

    all_samples = []
    for start in range(0, num_samples, batch_size):
        bs = min(batch_size, num_samples - start)
        samples = model.sample((bs, seq_len), num_steps=num_steps, temperature=temperature)
        samples = samples.clamp(0, vocab_size - 1)
        all_samples.append(samples.cpu())

    return torch.cat(all_samples, dim=0)


def _gen_diffusion(model, args, hp, num_samples, batch_size, device):
    from medlatents.diffusion.continuous import ContinuousGaussianDiffusion

    latent_shape = args.get("latent_shape", [4, 8, 8])
    C, H, W = latent_shape[0], latent_shape[1], latent_shape[2]
    seq_len = H * W
    temperature = hp.get("temperature", 0.9)
    num_inference_steps = hp.get("num_inference_steps", 1000)

    diffusion = ContinuousGaussianDiffusion(
        num_timesteps=args.get("diffusion_steps", 1000),
        schedule_type=args.get("diffusion_schedule", "cosine"),
        device=device,
    )

    all_samples = []
    for start in range(0, num_samples, batch_size):
        bs = min(batch_size, num_samples - start)
        samples = diffusion.sample(
            model,
            shape=(bs, seq_len, C),
            num_inference_steps=num_inference_steps,
            temperature=temperature,
        )
        samples = samples.view(bs, H, W, C).permute(0, 3, 1, 2).contiguous()
        all_samples.append(samples.cpu())

    result = torch.cat(all_samples, dim=0)

    # Un-normalize
    latent_mean = args.get("latent_mean", 0.0)
    latent_std = args.get("latent_std", 1.0)
    if latent_std != 1.0 or latent_mean != 0.0:
        result = result * latent_std + latent_mean

    return result


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Sweep sampling HPs for ChestMNIST models")
    parser.add_argument(
        "--num_samples", type=int, default=2000, help="Samples per config (2K for screening)"
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--filter", type=str, default=None, help="Only sweep runs matching this substring"
    )
    parser.add_argument(
        "--output", type=str, default=str(RESULTS / "chestmnist_factorial" / "sweep_results.json")
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--selection_split",
        type=str,
        default="val",
        choices=["val", "test"],
        help="Real-image split used as the FID reference for screening/selection. "
        "The default 'val' keeps selection off the test split used for the final eval.",
    )
    # Optional: override grid with specific values
    parser.add_argument("--temperature", type=float, nargs="+", default=None)
    parser.add_argument("--num_steps", type=int, nargs="+", default=None)
    parser.add_argument("--top_k", type=int, nargs="+", default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = args.device

    # Load real images for the FID reference. Selection MUST use a held-out
    # split ('val') to avoid HP-selection leakage: the final 10K eval uses
    # 'test', so selecting on 'test' tunes hyperparameters on the reporting set.
    split_key = f"{args.selection_split}_images"
    if args.selection_split == "test":
        print(
            "WARNING: selecting HPs on the TEST split -> HP-selection leakage "
            "(same split as the final eval). Use --selection_split val for clean results."
        )
    print(f"Loading real {args.selection_split} images...")
    raw = np.load(str(RAW_DATA_PATH))
    real_images = torch.from_numpy(raw[split_key]).float() / 255.0
    real_images = real_images.unsqueeze(1)
    real_uint8 = prepare_for_metrics(real_images)
    print(f"  Real {args.selection_split}: {real_uint8.shape}")

    # Load existing results
    results = {}
    if os.path.exists(args.output):
        with open(args.output) as f:
            results = json.load(f)

    # Discover checkpoints
    runs = []
    for entry in sorted(GEN_CKPT_ROOT.iterdir()):
        if not entry.is_dir():
            continue
        ckpt = entry / "checkpoint_best.pt"
        if not ckpt.exists():
            continue
        parsed = parse_run_name(entry.name)
        if parsed is None:
            continue
        tok_type, model_type = parsed
        if args.filter and args.filter not in entry.name:
            continue
        runs.append((entry.name, tok_type, model_type, str(ckpt)))

    print(f"Found {len(runs)} checkpoints to sweep")

    tokenizer_cache = {}

    for run_name, tok_type, model_type, ckpt_path in runs:
        # Build HP grid
        grid = dict(SAMPLING_GRIDS.get(model_type, {"temperature": [0.9]}))

        # Apply CLI overrides. The step-count key differs by generator
        # (num_steps / num_timesteps / num_euler_steps / num_inference_steps), so
        # --num_steps maps onto whichever one this grid actually uses.
        if args.temperature is not None:
            grid["temperature"] = args.temperature
        if args.num_steps is not None:
            step_keys = [k for k in grid if k in _STEP_KEYS]
            if len(step_keys) != 1:
                raise SystemExit(f"{run_name}: cannot map --num_steps onto grid keys {list(grid)}")
            grid[step_keys[0]] = args.num_steps
        if args.top_k is not None and "top_k" in grid:
            grid["top_k"] = args.top_k

        hp_configs = expand_grid(grid)
        print(f"\n{'=' * 60}")
        print(f"{run_name}: {len(hp_configs)} HP configs to sweep")
        print(f"{'=' * 60}")

        # Load model once
        try:
            gen_model, gen_args = load_gen_model(ckpt_path, device)
        except Exception as e:
            print(f"  SKIP: {e}")
            continue

        # Load tokenizer decoder
        if tok_type not in tokenizer_cache:
            tokenizer_cache[tok_type] = load_tokenizer_decoder(tok_type, device)
        tokenizer = tokenizer_cache[tok_type]

        for i, hp in enumerate(hp_configs):
            hp_key = f"{run_name}|{json.dumps(hp, sort_keys=True)}"

            if hp_key in results:
                print(f"  [{i + 1}/{len(hp_configs)}] SKIP (cached): {hp}")
                continue

            print(f"  [{i + 1}/{len(hp_configs)}] {hp} ...", end=" ", flush=True)
            t0 = time.time()

            try:
                is_discrete = model_type in DISCRETE_MODELS

                if is_discrete:
                    tokens = generate_with_hps(
                        gen_model,
                        gen_args,
                        hp,
                        args.num_samples,
                        args.batch_size,
                        device,
                    )
                    spatial_shape = gen_args.get("spatial_shape", None)
                    gen_images = decode_discrete_samples(
                        tokenizer,
                        tokens,
                        spatial_shape,
                        args.batch_size,
                        device,
                    )
                else:
                    latents = generate_with_hps(
                        gen_model,
                        gen_args,
                        hp,
                        args.num_samples,
                        args.batch_size,
                        device,
                    )
                    gen_images = decode_continuous_samples(
                        tokenizer,
                        latents,
                        args.batch_size,
                        device,
                    )

                gen_uint8 = prepare_for_metrics(gen_images)
                metrics = compute_fid_is(real_uint8, gen_uint8, device, batch_size=args.batch_size)
                metrics["hp"] = hp
                metrics["run"] = run_name
                metrics["time_s"] = time.time() - t0
                metrics["num_samples"] = args.num_samples
                results[hp_key] = metrics
                print(
                    f"FID={metrics['fid']:.1f}  IS={metrics['is_mean']:.2f}  ({metrics['time_s']:.0f}s)"
                )

            except Exception as e:
                results[hp_key] = {"error": str(e), "hp": hp, "run": run_name}
                print(f"ERROR: {e}")

            torch.cuda.empty_cache()

            # Save after each config (resume-safe) -- atomic rename so a crash
            # mid-write can't truncate the good file (cf. eval atomic-save fix).
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            tmp_path = args.output + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(results, f, indent=2)
            os.replace(tmp_path, args.output)

        del gen_model
        torch.cuda.empty_cache()

    # Summary: best config per run
    print(f"\n{'=' * 80}")
    print("BEST CONFIG PER MODEL")
    print(f"{'=' * 80}")

    best_per_run = {}
    for key, val in results.items():
        if "error" in val or "fid" not in val:
            continue
        run = val["run"]
        if run not in best_per_run or val["fid"] < best_per_run[run]["fid"]:
            best_per_run[run] = val

    print(f"{'Run':<40} {'FID':>8} {'IS':>8}  Best HPs")
    print("-" * 90)
    for run in sorted(best_per_run.keys()):
        r = best_per_run[run]
        print(f"{run:<40} {r['fid']:>8.1f} {r['is_mean']:>8.2f}  {r['hp']}")

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
