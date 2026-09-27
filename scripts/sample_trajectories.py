#!/usr/bin/env python
"""Intermediate sampler states for every sample-matrix cell, as decoded uint8 frames.

Raw material for figures of generation in progress. Covers the same 72 cells, seed
and batch as sample_matrix.py (torch.manual_seed(0) immediately before sampling, 16
samples, no selection), and runs the same sampling calls. Intermediate states are
only observed, never altered, and nothing observed consumes random numbers; the
final frame of every cell is compared against results/samples/sample_matrix.npz and
the result is recorded in the manifest.

How the state at each step is observed:
  transformer (AR)      no hook needed: AR never revises a token, so the state after
                        k steps is the first k tokens of the final sequence
  maskgit, flow (DFM),  forward pre-hook on the generator network records the token
  d3pm, sedd            grid it is given at each call (masked positions included)
  bayesian_flow (BFN)   the network never sees tokens; the belief-update method is
                        wrapped and the most probable token under the belief after
                        each update is recorded
  diffusion (LDM),      forward pre-hook records the latent x_t given to the network
  continuous_flow (RF)  at each call, then un-normalised as the evaluation does

Frames: K evenly spaced recorded states plus the final sample. Discrete states are
decoded with the frozen tokenizer; masked or not-yet-generated positions are decoded
as token 0 and their 8 x 8 pixel patch is then set to 0, with the mask stored so
this can be undone. Continuous states are decoded directly.

Outputs, under results/trajectories/:
  <cell>.npz      images uint8 [F, 16, 64, 64], call_index int [F],
                  plus tokens int16 [F, 16, 64] (-1 = masked) and masked bool
                  [F, 16, 8, 8] for discrete cells, or latents float16
                  [F, 16, C, 8, 8] for continuous cells
  manifest.json   per cell: seed, sampler settings, state kind, network calls,
                  frame call indices, and the final-frame check
  contact_sheet.png   rows = cells, columns = the frames of sample 0

Usage:
    CHECKPOINT_ROOT=... MEDTOKENIZERS_ROOT=... python scripts/sample_trajectories.py
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from evaluate_chestmnist_generation import (
    CONTINUOUS_MODELS,
    GEN_CKPT_ROOT,
    decode_continuous_samples,
    decode_discrete_samples,
    generate_continuous_samples,
    generate_discrete_samples,
    load_gen_model,
    load_tokenizer_decoder,
    prepare_for_metrics,
)
from paths import RESULTS, TABLES
from sample_matrix import (
    CONTINUOUS_GENS,
    CONTINUOUS_TOKS,
    DISCRETE_GENS,
    DISCRETE_TOKS,
    SEED,
    TUNED,
    N,
    contact_sheet,
    default_settings,
)
from sweep_sampling_hps import generate_with_hps

OUT = RESULTS / "trajectories"
K = 16  # intermediate frames per cell, before the final sample


def to_uint8(images: torch.Tensor) -> np.ndarray:
    return prepare_for_metrics(images)[:, 0].numpy()


def pick(n: int, k: int = K) -> list[int]:
    return sorted(set(np.linspace(0, n - 1, min(k, n)).round().astype(int).tolist()))


def decode_tokens(tokenizer, states: torch.Tensor, vocab: int, spatial, device):
    """(F, B, L) token states, anything outside [0, vocab) masked -> frames, masks."""
    h, w = spatial
    masked = (states < 0) | (states >= vocab)
    filled = torch.where(masked, torch.zeros_like(states), states)
    f, b, _ = states.shape
    img = to_uint8(decode_discrete_samples(tokenizer, filled.view(f * b, -1), spatial, N, device))
    img = img.reshape(f, b, 64, 64)
    m = masked.view(f, b, h, w).numpy()
    ph, pw = 64 // h, 64 // w
    img[np.repeat(np.repeat(m, ph, axis=2), pw, axis=3)] = 0
    return img, m


def run_cell(tok, gen, tuned, hp_cells, tokenizers, ref, device):
    run = f"chestmnist_{tok}_{gen}"
    model, a = load_gen_model(str(GEN_CKPT_ROOT / run / "checkpoint_best.pt"), device)
    a["model"] = gen
    continuous = gen in CONTINUOUS_MODELS
    hp = dict(hp_cells[f"{tok}_{gen}"]["selected_hp"]) if tuned else None
    settings = dict(hp) if tuned else default_settings(gen, a)

    states: list[torch.Tensor] = []
    handle = None
    if gen == "bayesian_flow" and not tuned:
        orig = model.update_params_bayesian

        def update(*args, **kw):
            p = orig(*args, **kw)
            states.append(p.argmax(-1).cpu())
            return p

        model.update_params_bayesian = update
        kind = "bfn_belief_argmax_after_update"
    elif gen != "transformer":

        def pre_hook(module, args, kwargs):
            x = kwargs.get("x", args[0] if args else None)
            states.append(x.detach().cpu().clone())

        handle = model.register_forward_pre_hook(pre_hook, with_kwargs=True)
        kind = "network_input_latent" if continuous else "network_input_tokens"
    else:
        kind = "ar_prefix"

    torch.cuda.synchronize()
    t0 = time.time()
    torch.manual_seed(SEED)
    if tuned:
        out = generate_with_hps(model, a, hp, N, N, device)
    elif continuous:
        out = generate_continuous_samples(model, a, N, N, device)
    else:
        out = generate_discrete_samples(model, a, N, N, device)
    torch.cuda.synchronize()
    t_gen = time.time() - t0
    if handle is not None:
        handle.remove()
    del model
    torch.cuda.empty_cache()

    if tok not in tokenizers:
        # Keep one decoder resident: the GPU is shared, and cells come grouped by tokenizer.
        tokenizers.clear()
        torch.cuda.empty_cache()
        tokenizers[tok] = load_tokenizer_decoder(tok, device)
    tk = tokenizers[tok]
    rec = {"tokenizer": tok, "generator": gen, "tuned": tuned, "seed": SEED, "sampler": settings}
    arrays = {}

    if continuous:
        C, H, W = a.get("latent_shape", [4, 8, 8])
        mean, std = a.get("latent_mean", 0.0), a.get("latent_std", 1.0)
        idx = pick(len(states))
        lat = [states[i].view(N, H, W, C).permute(0, 3, 1, 2) * std + mean for i in idx]
        lat.append(out)
        lat = torch.stack(lat).float()
        frames = to_uint8(decode_continuous_samples(tk, lat.flatten(0, 1), N, device))
        frames = frames.reshape(len(lat), N, 64, 64)
        arrays["latents"] = lat.numpy().astype(np.float16)
    else:
        vocab, spatial = a["vocab_size"], a.get("spatial_shape", [8, 8])
        if kind == "ar_prefix":
            n_calls = out.shape[1]
            full = []
            for k in range(n_calls):
                s = out.clone()
                s[:, k:] = -1
                full.append(s)
            states = full
        idx = pick(len(states))
        st = torch.stack([states[i].long() for i in idx] + [out.long()])
        frames, masked = decode_tokens(tk, st, vocab, spatial, device)
        arrays["tokens"] = torch.where((st < 0) | (st >= vocab), -1, st).numpy().astype(np.int16)
        arrays["masked"] = masked

    n_calls = len(states)
    call_index = np.array(idx + [n_calls], dtype=np.int32)
    ref_img = ref.get(f"{tok}_{gen}" + ("_tuned" if tuned else ""))
    diff = None if ref_img is None else int(np.abs(frames[-1].astype(int) - ref_img).max())
    rec.update(
        state_kind=kind,
        n_recorded_states=n_calls,
        frame_call_index=call_index.tolist(),
        n_frames=int(len(frames)),
        # medtokenizers turns on cudnn.benchmark at import, so the decoder's conv
        # algorithm, and with it the last bit of some pixels, can differ between
        # processes even for identical tokens. One wrong token would move a whole
        # 8 x 8 patch by far more than one level, so <= 1 is the equality check.
        final_max_abs_diff_vs_sample_matrix=diff,
        final_matches_sample_matrix=diff is not None and diff <= 1,
        generation_time_s=round(t_gen, 3),
    )
    arrays["images"] = frames.astype(np.uint8)
    arrays["call_index"] = call_index
    return rec, arrays


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--filter", default=None)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    hp_cells = json.loads((TABLES / "hp_revalidation.json").read_text())["cells"]
    z = np.load(RESULTS / "samples" / "sample_matrix.npz")
    ref = dict(zip(z["cells"].tolist(), z["images"]))

    cells = [(t, g, False) for t in DISCRETE_TOKS for g in DISCRETE_GENS]
    cells += [(t, g, False) for t in CONTINUOUS_TOKS for g in CONTINUOUS_GENS]
    cells += [(*c.split("_", 1), True) for c in TUNED]
    if args.filter:
        cells = [c for c in cells if args.filter in f"{c[0]}_{c[1]}"]

    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"cells": {}}
    tokenizers: dict = {}
    for tok, gen, tuned in cells:
        name = f"{tok}_{gen}" + ("_tuned" if tuned else "")
        print(f"{name} ...", flush=True)
        try:
            rec, arrays = run_cell(tok, gen, tuned, hp_cells, tokenizers, ref, args.device)
            np.savez_compressed(OUT / f"{name}.npz", **arrays)
            print(
                f"  {rec['n_recorded_states']} states, {rec['n_frames']} frames, "
                f"final matches matrix: {rec['final_matches_sample_matrix']}, "
                f"{rec['generation_time_s']:.1f}s",
                flush=True,
            )
        except Exception as e:
            rec = {"error": f"{type(e).__name__}: {e}"}
            print(f"  FAILED {rec['error']}", flush=True)
            torch.cuda.empty_cache()
        manifest["cells"][name] = rec
        manifest_path.write_text(json.dumps(manifest, indent=2))

    manifest.update(
        _description=(
            "Intermediate sampler states for every sample-matrix cell, seed 0, 16 samples "
            "in generation order. See scripts/sample_trajectories.py for how each "
            "generator's state is observed and how masked positions are rendered. "
            "frame_call_index[i] is the number of sampler network calls (AR: tokens "
            "generated) before frame i; the last frame is the final sample."
        ),
        seed=SEED,
        samples_per_cell=N,
        intermediate_frames_per_cell=K,
        gpu=torch.cuda.get_device_name(0),
        torch=torch.__version__,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Contact sheet: sample 0 of every cell, one row per cell, frames left to right.
    rows, labels = [], []
    for n in manifest["cells"]:
        if "error" in manifest["cells"][n]:
            continue
        z = np.load(OUT / f"{n}.npz")
        rows.append(z["images"][:, 0])
        labels.append(n)
    width = max(r.shape[0] for r in rows)
    pad = [np.concatenate([r, np.full((width - len(r), 64, 64), 255, np.uint8)]) for r in rows]
    contact_sheet(np.stack(pad), labels, OUT / "contact_sheet.png")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
