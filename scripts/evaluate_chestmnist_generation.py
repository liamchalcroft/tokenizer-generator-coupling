#!/usr/bin/env python
"""Generate samples from trained ChestMNIST models and evaluate with FID/IS.

End-to-end pipeline:
1. Load real test images (original pixel space)
2. For each (tokenizer, model) checkpoint:
   a. Load generative model, generate samples in latent space
   b. Load frozen tokenizer decoder, decode to pixel space
   c. Compute FID and IS against real test images
3. Output summary table and save results to JSON

Usage:
    uv run python scripts/evaluate_chestmnist_generation.py

    # Fewer samples for quick test
    uv run python scripts/evaluate_chestmnist_generation.py --num_samples 1000

    # Specific checkpoints only
    uv run python scripts/evaluate_chestmnist_generation.py --filter VQ_transformer
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from paths import CHECKPOINTS, MEDTOKENIZERS_ROOT, RESULTS

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TOKENIZED_ROOT = MEDTOKENIZERS_ROOT / "checkpoints" / "medmnist" / "tokenized"
TOKENIZER_CKPT_ROOT = MEDTOKENIZERS_ROOT / "checkpoints" / "medmnist"
# Dataset prefix is configurable via $MEDGEN_DATASET (default: chestmnist).
# Used to construct RAW_DATA_PATH, GEN_CKPT_ROOT, and tokenizer dir lookup
# prefixes -- enables Phase 4 multi-dataset replication on pneumoniamnist /
# organamnist without forking the script.
DATASET = os.environ.get("MEDGEN_DATASET", "chestmnist")
RAW_DATA_PATH = MEDTOKENIZERS_ROOT / "data" / DATASET / f"{DATASET}_64.npz"
GEN_CKPT_ROOT = CHECKPOINTS / f"{DATASET}_factorial"
RESULTS_DIR = RESULTS / f"{DATASET}_factorial"

# ---------------------------------------------------------------------------
# Tokenizer loading (adapted from medtokenizers/scripts/eval_all_medmnist.py)
# ---------------------------------------------------------------------------


def _build_tokenizer_config_from_metadata(tok_name: str) -> dict:
    """Build tokenizer constructor kwargs from metadata.json in the tokenized data dir."""
    common = {
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

    # Try versioned metadata first; fall back to base name (e.g. LFQ4096 → LFQ).
    # Some legacy tokenized dirs (chestmnist_LFQ) have empty config {}, so we also
    # parse the numeric vocab suffix from tok_name below and use it as a fallback.
    import re

    meta_path = TOKENIZED_ROOT / f"{DATASET}_{tok_name}" / "metadata.json"
    if not meta_path.exists():
        base = re.sub(r"\d+$", "", tok_name)
        if base != tok_name:
            meta_path = TOKENIZED_ROOT / f"{DATASET}_{base}" / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Tokenizer metadata not found: {meta_path}")

    with open(meta_path) as f:
        meta = json.load(f)

    config = meta.get("config", {})
    formulation = config.get("formulation", "")

    # Recover from empty metadata config (legacy chestmnist_LFQ): infer quantizer
    # from the name prefix and vocab size from the numeric suffix.
    name_vocab_match = re.search(r"(\d+)$", tok_name)
    name_vocab = int(name_vocab_match.group(1)) if name_vocab_match else None

    # Detect continuous tokenizers -- check config, metadata type, or name prefix
    is_continuous = (
        formulation in ("VAE", "AE")
        or meta.get("type") == "continuous"
        or tok_name.startswith(("VAE", "AE"))
    )
    if is_continuous:
        # Infer formulation from name if metadata is incomplete
        if not formulation:
            formulation = "AE" if tok_name.startswith("AE") else "VAE"
        latent_ch = config.get("latent_channels", 4)
        return dict(
            **common,
            latent_channels=latent_ch,
            formulation=formulation,
            name=f"{DATASET}_{tok_name}_tokenizer",
            cls="continuous",
        )

    quantizer = config.get("quantizer", "")
    if not quantizer:
        # Fall back to inferring from name prefix (e.g. "LFQ4096" -> "LFQ").
        qmatch = re.match(r"([A-Z]+)", tok_name)
        quantizer = qmatch.group(1) if qmatch else ""
    quantizer_kwargs: dict = {}

    if quantizer == "VQ":
        quantizer_kwargs["num_embeddings"] = config.get("num_embeddings") or name_vocab or 1024
        quantizer_kwargs["use_ema"] = config.get("use_ema", True)
        quantizer_kwargs["use_norm"] = config.get("use_norm", True)
    elif quantizer == "FSQ":
        quantizer_kwargs["levels"] = config.get("levels", [4, 4, 4, 4, 4])
    elif quantizer == "RESFSQ":
        quantizer_kwargs["levels"] = config.get("levels", [8, 8, 8])
        quantizer_kwargs["num_codebooks"] = config.get("num_codebooks", 2)
    elif quantizer == "LFQ":
        cbs = config.get("codebook_size") or name_vocab or 1024
        quantizer_kwargs["codebook_size"] = cbs
        quantizer_kwargs["codebook_dim"] = config.get("codebook_dim") or (cbs - 1).bit_length()
        quantizer_kwargs["num_codebooks"] = config.get("num_codebooks", 1)

    embedding_dim = config.get("embedding_dim", 16)

    return dict(
        **common,
        embedding_dim=embedding_dim,
        quantizer=quantizer,
        name=f"{DATASET}_{tok_name}_tokenizer",
        cls="discrete",
        **quantizer_kwargs,
    )


def _find_tokenizer_checkpoint(tok_name: str) -> Path:
    """Find the tokenizer checkpoint, trying versioned then base name."""
    # Try exact name first (e.g., chestmnist_VQ1024_final)
    for suffix in ("_final", ""):
        ckpt_dir = TOKENIZER_CKPT_ROOT / f"{DATASET}_{tok_name}{suffix}"
        safetensors_path = ckpt_dir / "model.safetensors"
        if safetensors_path.exists():
            return safetensors_path

    # Try base quantizer name (e.g., VQ1024 -> VQ)
    import re

    base = re.sub(r"\d+$", "", tok_name)
    if base != tok_name:
        for suffix in ("_final", ""):
            ckpt_dir = TOKENIZER_CKPT_ROOT / f"{DATASET}_{base}{suffix}"
            safetensors_path = ckpt_dir / "model.safetensors"
            if safetensors_path.exists():
                return safetensors_path

    raise FileNotFoundError(
        f"Tokenizer checkpoint not found for {tok_name}. "
        f"Searched: {TOKENIZER_CKPT_ROOT}/chestmnist_{tok_name}*"
    )


def load_tokenizer_decoder(tok_name: str, device: str) -> torch.nn.Module:
    """Load a frozen tokenizer decoder from checkpoint."""
    from medtokenizers.networks import ContinuousTokenizer, DiscreteTokenizer
    from safetensors.torch import load_file

    safetensors_path = _find_tokenizer_checkpoint(tok_name)
    cfg = _build_tokenizer_config_from_metadata(tok_name)
    cls_name = cfg.pop("cls")

    if cls_name == "continuous":
        model = ContinuousTokenizer(**cfg)
    else:
        model = DiscreteTokenizer(**cfg)

    state_dict = load_file(str(safetensors_path), device=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Generative model loading (from generate_medmnist.py)
# ---------------------------------------------------------------------------


def load_gen_model(checkpoint_path: str, device: str):
    """Load a trained generative model from checkpoint. Returns (model, args_dict)."""
    from medlatents.autoregressive import AutoregressiveTransformer
    from medlatents.bayesian_flow import BayesianFlowTransformer
    from medlatents.maskgit import MaskGIT
    from medlatents.networks import ContinuousDiT, DiscreteDiT
    from medlatents.utils import load_state_dict_compat

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    model_type = args.get("model", "transformer")

    seq_len = args.get("seq_len", 64)
    vocab_size = args.get("vocab_size", 512)
    hidden_size = args.get("hidden_size", 512)
    depth = args.get("depth", 12)
    num_heads = args.get("num_heads", 8)
    mlp_ratio = args.get("mlp_ratio", 4.0)

    if model_type == "transformer":
        model = AutoregressiveTransformer(
            seq_length=seq_len,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            gradient_checkpointing=False,
        )
    elif model_type == "maskgit":
        model = MaskGIT(
            seq_length=seq_len,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            gradient_checkpointing=False,
        )
    elif model_type == "flow":
        source_dist = args.get("source_dist", "uniform")
        model = DiscreteDiT(
            seq_length=seq_len,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            masked=source_dist == "mask",
            gradient_checkpointing=False,
            allow_dynamic_seq_length=True,
        )
    elif model_type == "d3pm":
        d3pm_transition = args.get("d3pm_transition", "absorbing")
        model = DiscreteDiT(
            seq_length=seq_len,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            masked=d3pm_transition == "absorbing",
            gradient_checkpointing=False,
            allow_dynamic_seq_length=True,
        )
    elif model_type == "bayesian_flow":
        model = BayesianFlowTransformer(
            seq_length=seq_len,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_steps=args.get("bfn_num_steps", 1000),
            beta=args.get("bfn_beta", 1.0),
            gradient_checkpointing=False,
        )
    elif model_type == "sedd":
        d3pm_transition = args.get("d3pm_transition", "absorbing")
        model = DiscreteDiT(
            seq_length=seq_len,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            masked=d3pm_transition == "absorbing",
            gradient_checkpointing=False,
            allow_dynamic_seq_length=True,
        )
    elif model_type == "continuous_flow" or model_type == "diffusion":
        import math

        latent_shape = args.get("latent_shape", [4, 8, 8])
        latent_channels = latent_shape[0]
        seq_length = math.prod(latent_shape[1:])
        model = ContinuousDiT(
            seq_length=seq_length,
            in_channels=latent_channels,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            gradient_checkpointing=False,
            allow_dynamic_seq_length=True,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    load_state_dict_compat(model, ckpt["net"])

    # Load EMA weights if available (better quality)
    if "ema" in ckpt and "shadow_params" in ckpt["ema"]:
        shadow_params = ckpt["ema"]["shadow_params"]
        for param, shadow in zip(model.parameters(), shadow_params):
            param.data.copy_(shadow.to(param.device))

    model.to(device)
    model.eval()
    return model, args


# ---------------------------------------------------------------------------
# Sample generation
# ---------------------------------------------------------------------------


@torch.inference_mode()
def generate_discrete_samples(
    model: torch.nn.Module,
    args: dict,
    num_samples: int,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    """Generate discrete token samples. Returns (N, seq_len) int64 tensor."""
    model_type = args["model"]
    seq_len = args["seq_len"]
    vocab_size = args["vocab_size"]

    all_samples = []
    for start in tqdm(
        range(0, num_samples, batch_size), desc=f"Generating ({model_type})", leave=False
    ):
        bs = min(batch_size, num_samples - start)

        if model_type == "transformer":
            bos = model.special_tokens.bos if getattr(model, "special_tokens", None) else 0
            prompt = torch.full((bs, 1), bos, dtype=torch.long, device=device)
            samples = model.generate(prompt, max_length=seq_len + 1, temperature=0.9)
            samples = samples[:, 1:]  # strip BOS token

        elif model_type == "maskgit":
            x = torch.full((bs, seq_len), model.mask_token, dtype=torch.long, device=device)
            samples = model.generate(x, num_steps=12, temperature=0.9)

        elif model_type == "flow":
            from medlatents.flow_matching.core import MixtureDiscreteEulerSolver, ModelWrapper
            from medlatents.flow_matching.discrete import get_path, get_source_distribution

            path = get_path(
                args.get("scheduler_type", "polynomial"), args.get("scheduler_power", 2.0)
            )
            source_dist = get_source_distribution(args.get("source_dist", "uniform"), vocab_size)

            class WrappedModel(ModelWrapper):
                @torch.no_grad()
                def forward(self, x, t, **extras):
                    # Return raw logits -- solver applies softmax internally
                    return self.model(x=x, t=t, **extras).float()

            solver = MixtureDiscreteEulerSolver(
                model=WrappedModel(model=model),
                path=path,
                vocabulary_size=vocab_size + (1 if getattr(source_dist, "masked", False) else 0),
            )
            x_init = source_dist.sample((bs, seq_len), device=device)
            num_flow_steps = 100
            samples = solver.sample(
                x_init=x_init,
                step_size=1 / num_flow_steps,
                verbose=False,
            )

        elif model_type == "d3pm":
            from medlatents.diffusion.d3pm import D3PM

            d3pm = D3PM(
                num_classes=vocab_size,
                num_timesteps=args.get("diffusion_steps", 1000),
                schedule_type=args.get("diffusion_schedule", "cosine"),
                transition_type=args.get("d3pm_transition", "absorbing"),
                hybrid_loss_coeff=args.get("d3pm_hybrid_coeff", 0.001),
                device=device,
            )
            samples = d3pm.sample(model, (bs, seq_len), temperature=0.9)

        elif model_type == "sedd":
            # SEDD trains with continuous t in [0,1) but D3PM.sample() passes
            # integer t. Wrap the model to normalize timesteps.
            from medlatents.diffusion.d3pm import D3PM

            num_timesteps = args.get("diffusion_steps", 1000)
            d3pm = D3PM(
                num_classes=vocab_size,
                num_timesteps=num_timesteps,
                schedule_type=args.get("diffusion_schedule", "cosine"),
                transition_type=args.get("d3pm_transition", "absorbing"),
                device=device,
            )

            class SEDDModelWrapper(torch.nn.Module):
                def __init__(self, model, num_timesteps):
                    super().__init__()
                    self.model = model
                    self.num_timesteps = num_timesteps

                def forward(self, x, t, **kwargs):
                    # Convert integer timesteps to continuous [0, 1)
                    t_cont = t.float() / self.num_timesteps
                    return self.model(x=x, t=t_cont, **kwargs)

            wrapped = SEDDModelWrapper(model, num_timesteps)
            samples = d3pm.sample(wrapped, (bs, seq_len), temperature=0.9)

        elif model_type == "bayesian_flow":
            bfn_num_steps = args.get("bfn_num_steps", 1000)
            samples = model.sample((bs, seq_len), num_steps=bfn_num_steps, temperature=0.9)

        else:
            raise ValueError(f"Unknown discrete model: {model_type}")

        # Clamp to valid codebook range
        samples = samples.clamp(0, vocab_size - 1)
        all_samples.append(samples.cpu())

    return torch.cat(all_samples, dim=0)


@torch.inference_mode()
def generate_continuous_samples(
    model: torch.nn.Module,
    args: dict,
    num_samples: int,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    """Generate continuous latent samples. Returns (N, C, H, W) float tensor."""

    model_type = args.get("model", "diffusion")
    latent_shape = args.get("latent_shape", [4, 8, 8])
    C, H, W = latent_shape[0], latent_shape[1], latent_shape[2]
    seq_len = H * W

    if model_type == "continuous_flow":
        from medlatents.flow_matching.continuous import RectifiedFlow

        flow = RectifiedFlow(
            base_std=args.get("base_std", 1.0),
            device=device,
        )
        flow.latent_shape = (seq_len, C)  # must be set before sampling
    else:
        from medlatents.diffusion.continuous import ContinuousGaussianDiffusion

        diffusion = ContinuousGaussianDiffusion(
            num_timesteps=args.get("diffusion_steps", 1000),
            schedule_type=args.get("diffusion_schedule", "cosine"),
            device=device,
        )

    all_samples = []
    desc = f"Generating ({model_type})"
    for start in tqdm(range(0, num_samples, batch_size), desc=desc, leave=False):
        bs = min(batch_size, num_samples - start)
        if model_type == "continuous_flow":
            samples = flow.sample(model, bs, num_steps=args.get("flow_num_steps", 100))
        else:
            samples = diffusion.sample(model, shape=(bs, seq_len, C), temperature=0.9)
        # Reshape sequence format (B, H*W, C) to (B, C, H, W)
        samples = samples.view(bs, H, W, C).permute(0, 3, 1, 2).contiguous()
        all_samples.append(samples.cpu())

    samples = torch.cat(all_samples, dim=0)

    # Un-normalize if model was trained with normalized latents
    latent_mean = args.get("latent_mean", 0.0)
    latent_std = args.get("latent_std", 1.0)
    if latent_std != 1.0 or latent_mean != 0.0:
        samples = samples * latent_std + latent_mean

    return samples


# ---------------------------------------------------------------------------
# Decoding latents -> pixel space
# ---------------------------------------------------------------------------


@torch.inference_mode()
def decode_discrete_samples(
    tokenizer: torch.nn.Module,
    tokens: torch.Tensor,
    spatial_shape: list | None,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    """Decode discrete tokens to pixel-space images. Returns (N, 1, 64, 64) float."""
    all_images = []
    for start in tqdm(range(0, len(tokens), batch_size), desc="Decoding", leave=False):
        batch = tokens[start : start + batch_size].to(device)

        # Reshape flat tokens back to spatial grid
        if spatial_shape is not None and len(spatial_shape) >= 2:
            batch = batch.view(batch.shape[0], *spatial_shape)

        images = tokenizer.detokenize(batch)
        all_images.append(images.cpu())

    return torch.cat(all_images, dim=0)


@torch.inference_mode()
def decode_continuous_samples(
    tokenizer: torch.nn.Module,
    latents: torch.Tensor,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    """Decode continuous latents to pixel-space images. Returns (N, 1, 64, 64) float."""
    all_images = []
    for start in tqdm(range(0, len(latents), batch_size), desc="Decoding", leave=False):
        batch = latents[start : start + batch_size].to(device)
        images = tokenizer.decode(batch)
        all_images.append(images.cpu())

    return torch.cat(all_images, dim=0)


# ---------------------------------------------------------------------------
# FID / IS computation
# ---------------------------------------------------------------------------


def prepare_for_metrics(images: torch.Tensor) -> torch.Tensor:
    """Convert (N, 1, 64, 64) float images to (N, 3, 64, 64) uint8 for FID/IS."""
    # Clamp to [0, 1]
    images = images.clamp(0, 1)
    # Grayscale -> 3-channel
    images = images.repeat(1, 3, 1, 1)
    # Float [0,1] -> uint8 [0,255]
    images = (images * 255).to(torch.uint8)
    return images


def prepare_for_metrics_float(images: torch.Tensor) -> torch.Tensor:
    """Convert (N, 1, 64, 64) float images to (N, 3, 64, 64) float [0,1] for FID/IS."""
    images = images.clamp(0, 1)
    return images.repeat(1, 3, 1, 1) if images.shape[1] == 1 else images


def compute_fid_is(
    real_images: torch.Tensor,
    gen_images: torch.Tensor,
    device: str,
    batch_size: int = 64,
) -> dict:
    """Compute FID and IS between real and generated images.

    Both inputs should be (N, 3, H, W) float [0,1] or uint8 [0,255] tensors.
    Uses normalize=True for numerical stability with small medical images.
    """
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.inception import InceptionScore

    # Convert uint8 to float if needed
    if real_images.dtype == torch.uint8:
        real_images = real_images.float() / 255.0
    if gen_images.dtype == torch.uint8:
        gen_images = gen_images.float() / 255.0

    # FID (normalize=True expects float [0,1])
    # Use feature=192 (pool3) instead of 2048 for numerical stability with
    # small 64×64 medical images. The 2048-dim features are degenerate at this
    # resolution, causing negative FID from ill-conditioned covariance matrices.
    fid = FrechetInceptionDistance(feature=192, normalize=True).to(device)
    for start in range(0, len(real_images), batch_size):
        fid.update(real_images[start : start + batch_size].to(device), real=True)
    for start in range(0, len(gen_images), batch_size):
        fid.update(gen_images[start : start + batch_size].to(device), real=False)
    fid_score = float(fid.compute().item())
    del fid
    torch.cuda.empty_cache()

    # IS (normalize=True expects float [0,1])
    inception = InceptionScore(normalize=True, feature="logits_unbiased").to(device)
    for start in range(0, len(gen_images), batch_size):
        inception.update(gen_images[start : start + batch_size].to(device))
    is_mean, is_std = inception.compute()
    is_mean, is_std = float(is_mean.item()), float(is_std.item())
    del inception
    torch.cuda.empty_cache()

    return {"fid": fid_score, "is_mean": is_mean, "is_std": is_std}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

DISCRETE_MODELS = {"transformer", "maskgit", "flow", "d3pm", "sedd", "bayesian_flow"}
CONTINUOUS_MODELS = {"diffusion", "continuous_flow"}


def parse_run_name(name: str) -> tuple[str, str] | None:
    """Parse 'chestmnist_{TOK}_{MODEL}' into (tokenizer_name, model_type).

    Handles both old naming (chestmnist_VQ_transformer) and versioned
    naming (chestmnist_VQ1024_transformer, chestmnist_VAEc4_diffusion).
    """
    prefix = f"{DATASET}_"
    if not name.startswith(prefix):
        return None
    rest = name[len(prefix) :]

    # Known model types (longest first to avoid partial matches)
    model_types = sorted(
        DISCRETE_MODELS | CONTINUOUS_MODELS,
        key=len,
        reverse=True,
    )

    for mt in model_types:
        suffix = f"_{mt}"
        if rest.endswith(suffix):
            tok_name = rest[: -len(suffix)]
            return tok_name, mt

    return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate ChestMNIST generation quality")
    parser.add_argument(
        "--num_samples", type=int, default=10000, help="Samples to generate per model"
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for generation/decoding"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Override checkpoint directory (default: checkpoints/{dataset}_factorial)",
    )
    parser.add_argument(
        "--filter", type=str, default=None, help="Only evaluate runs matching this substring"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: results/{dataset}_factorial/eval_results.json)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip", type=str, default=None, help="Comma-separated list of run name substrings to skip"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Skip runs already present in the output JSON"
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = args.device

    gen_ckpt_root = Path(args.checkpoint_dir) if args.checkpoint_dir else GEN_CKPT_ROOT
    output_path = args.output or str(RESULTS_DIR / "eval_results.json")

    # ------------------------------------------------------------------
    # 1. Load real test images
    # ------------------------------------------------------------------
    print("Loading real test images...")
    raw = np.load(str(RAW_DATA_PATH))
    real_images = torch.from_numpy(raw["test_images"]).float() / 255.0  # (N, 64, 64)
    real_images = real_images.unsqueeze(1)  # (N, 1, 64, 64)
    real_uint8 = prepare_for_metrics(real_images)
    print(f"  Real test images: {real_uint8.shape}")

    # ------------------------------------------------------------------
    # 2. Discover trained checkpoints
    # ------------------------------------------------------------------
    skip_list = args.skip.split(",") if args.skip else []

    runs = []
    for entry in sorted(gen_ckpt_root.iterdir()):
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
        if any(s in entry.name for s in skip_list):
            continue
        runs.append((entry.name, tok_type, model_type, str(ckpt)))

    print(f"Found {len(runs)} trained checkpoints")

    # ------------------------------------------------------------------
    # 3. Evaluate each run
    # ------------------------------------------------------------------
    # Resume from existing results if requested
    results = {}
    if args.resume and os.path.exists(output_path):
        with open(output_path) as f:
            results = json.load(f)
        print(f"Resumed {len(results)} existing results from {output_path}")

    tokenizer_cache: dict[str, torch.nn.Module] = {}

    for run_name, tok_type, model_type, ckpt_path in runs:
        if run_name in results and "fid" in results[run_name]:
            print(f"SKIP {run_name} (already evaluated, FID={results[run_name]['fid']:.2f})")
            continue

        print(f"\n{'=' * 60}")
        print(f"Evaluating: {run_name}")
        print(f"  Tokenizer: {tok_type}, Model: {model_type}")
        print(f"{'=' * 60}")

        t0 = time.time()

        # Load generative model
        try:
            gen_model, gen_args = load_gen_model(ckpt_path, device)
        except Exception as e:
            print(f"  SKIP: Failed to load model: {e}")
            results[run_name] = {"error": str(e)}
            continue

        # Generate samples
        try:
            is_discrete = model_type in DISCRETE_MODELS
            gen_args["model"] = model_type  # ensure model type is set
            if is_discrete:
                tokens = generate_discrete_samples(
                    gen_model,
                    gen_args,
                    args.num_samples,
                    args.batch_size,
                    device,
                )
            else:
                latents = generate_continuous_samples(
                    gen_model,
                    gen_args,
                    args.num_samples,
                    args.batch_size,
                    device,
                )
        except Exception as e:
            print(f"  SKIP: Generation failed: {e}")
            results[run_name] = {"error": f"generation: {e}"}
            del gen_model
            torch.cuda.empty_cache()
            continue

        del gen_model
        torch.cuda.empty_cache()

        # Load tokenizer decoder (cached per tokenizer type)
        if tok_type not in tokenizer_cache:
            print(f"  Loading {tok_type} tokenizer decoder...")
            tokenizer_cache[tok_type] = load_tokenizer_decoder(tok_type, device)
        tokenizer = tokenizer_cache[tok_type]

        # Decode to pixel space
        try:
            if is_discrete:
                spatial_shape = gen_args.get("spatial_shape", None)
                gen_images = decode_discrete_samples(
                    tokenizer,
                    tokens,
                    spatial_shape,
                    args.batch_size,
                    device,
                )
            else:
                gen_images = decode_continuous_samples(
                    tokenizer,
                    latents,
                    args.batch_size,
                    device,
                )
        except Exception as e:
            print(f"  SKIP: Decoding failed: {e}")
            results[run_name] = {"error": f"decoding: {e}"}
            torch.cuda.empty_cache()
            continue

        gen_uint8 = prepare_for_metrics(gen_images)

        # Compute metrics
        print(f"  Computing FID and IS ({len(gen_uint8)} generated vs {len(real_uint8)} real)...")
        metrics = compute_fid_is(real_uint8, gen_uint8, device, batch_size=args.batch_size)
        metrics["generation_time_s"] = time.time() - t0
        metrics["num_samples"] = len(gen_uint8)
        metrics["tokenizer"] = tok_type
        metrics["model"] = model_type

        results[run_name] = metrics
        print(f"  FID: {metrics['fid']:.2f}")
        print(f"  IS:  {metrics['is_mean']:.2f} +/- {metrics['is_std']:.2f}")
        print(f"  Time: {metrics['generation_time_s']:.0f}s")

        # Save incrementally so results survive crashes -- atomic rename so a
        # disk-full or kill mid-write can't truncate the good file.
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        tmp_path = output_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(results, f, indent=2)
        os.replace(tmp_path, output_path)

        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 4. Summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 80}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 80}")
    print(f"{'Run':<40} {'FID':>8} {'IS':>12}")
    print("-" * 64)

    for run_name in sorted(results.keys()):
        r = results[run_name]
        if "error" in r:
            print(f"{run_name:<40} {'ERROR':>8}   {r['error']}")
        else:
            print(f"{run_name:<40} {r['fid']:>8.2f}   {r['is_mean']:.2f} +/- {r['is_std']:.2f}")

    # Save results -- atomic rename (same rationale as incremental save above).
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(results, f, indent=2)
    os.replace(tmp_path, output_path)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
