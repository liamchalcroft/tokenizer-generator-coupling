#!/usr/bin/env python
"""Generator-free latent statistics, tested against observed generation quality.

Reconstruction quality does not predict which tokenizer makes a good generation
substrate (Spearman -0.03 on the discrete panel). This asks whether any intrinsic,
generator-free property of the token field does.

Statistics computed per discrete tokenizer, all on already-tokenized data:

  marginal entropy         pooled over positions, bits, and normalised by log2(V)
  per-position dispersion  sd of entropy across the 64 grid positions
  conditional entropy      H(X_i | left neighbour) and H(X_i | upper neighbour),
                           counts fitted on train and scored as held-out
                           cross-entropy on test, so an overfitted sparse model is
                           penalised rather than rewarded
  predictive gain          marginal minus conditional, i.e. how much a neighbour
                           actually tells you about a token

Deliberately not called mutual information: the marginal is empirical on test
while the conditional comes from a train-fitted model, so the difference can go
negative when neighbours are unhelpful, which is a meaningful outcome.

Entropies scale with log2(V), so cross-vocabulary comparison uses the normalised
columns and the within-vocabulary orderings, which are the fair comparison at
matched categorical rate.

No GPU, no model loading, no medtokenizers import. Runs on the tokenized NPZs.

Usage:
    python scripts/latent_modelability.py --smoke
    MEDTOKENIZERS_ROOT=../medtokenizers \
    python scripts/latent_modelability.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

from paths import RESULTS, TABLES

# Test-set reconstruction PSNR per tokenizer, the baseline predictor this analysis is
# trying to beat. best_fid and best_gen are filled in by load_best_fid().
TOKENIZERS = {
    "VQ1024": {"vocab": 1024, "psnr": 28.0},
    "VQ2048": {"vocab": 2048, "psnr": 29.3},
    "VQ4096": {"vocab": 4096, "psnr": 29.4},
    "LFQ1024": {"vocab": 1024, "psnr": 28.0},
    "LFQ2048": {"vocab": 2048, "psnr": 28.1},
    "LFQ4096": {"vocab": 4096, "psnr": 28.6},
    "FSQ1024": {"vocab": 1024, "psnr": 29.4},
    "FSQ2048": {"vocab": 2048, "psnr": 29.6},
    "FSQ4096": {"vocab": 4096, "psnr": 29.7},
}
GENERATOR_LABELS = {
    "transformer": "AR",
    "maskgit": "MaskGIT",
    "flow": "DFM",
    "d3pm": "D3PM",
    "sedd": "SEDD",
    "bayesian_flow": "BFN",
}


def load_best_fid(path: Path = RESULTS / "chestmnist_factorial" / "eval_results.json") -> None:
    """Best default-config FID-192 per tokenizer over the six discrete generators."""
    evals = json.loads(path.read_text())
    for name, meta in TOKENIZERS.items():
        fid, gen = min((evals[f"chestmnist_{name}_{g}"]["fid"], g) for g in GENERATOR_LABELS)
        meta["best_fid"] = round(fid, 4)
        meta["best_gen"] = GENERATOR_LABELS[gen]


# LFQ4096 was written under the pre-rename convention.
DIR_ALIASES = {"LFQ4096": ["chestmnist_LFQ4096", "chestmnist_LFQ"]}


def spearman(x, y) -> float:
    """Rank correlation, average ranks for ties, no scipy dependency."""
    x = list(x)
    y = list(y)

    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(x), ranks(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return float(num / (dx * dy)) if dx and dy else float("nan")


def load_tokens(tok_dir: Path, split: str) -> np.ndarray:
    """Load a token array, tolerating both on-disk layouts.

    Some dirs hold per-split files (train.npz), others a single combined archive
    keyed by {split}_tokens. Raises with the directory listing rather than
    guessing, because silently loading the wrong split would be invisible.
    """
    per_split = tok_dir / f"{split}.npz"
    if per_split.exists():
        z = np.load(per_split)
        for key in (f"{split}_tokens", "tokens", "codes", "arr_0"):
            if key in z.files:
                return z[key]
        if len(z.files) == 1:
            return z[z.files[0]]
        raise KeyError(f"{per_split}: ambiguous keys {z.files}")

    for cand in sorted(tok_dir.glob("*.npz")):
        z = np.load(cand)
        if f"{split}_tokens" in z.files:
            return z[f"{split}_tokens"]

    listing = sorted(p.name for p in tok_dir.glob("*"))
    raise FileNotFoundError(f"no {split} tokens under {tok_dir}; contents: {listing}")


def as_grid(tokens: np.ndarray) -> np.ndarray:
    """Normalise to (N, H, W) int64. Accepts (N, 64), (N, 8, 8), or (N, 1, 8, 8)."""
    t = np.asarray(tokens)
    if t.ndim == 4 and t.shape[1] == 1:
        t = t[:, 0]
    if t.ndim == 2:
        side = int(round(math.sqrt(t.shape[1])))
        if side * side != t.shape[1]:
            raise ValueError(f"token length {t.shape[1]} is not square")
        t = t.reshape(t.shape[0], side, side)
    if t.ndim != 3:
        raise ValueError(f"unexpected token shape {t.shape}")
    return t.astype(np.int64)


def entropy_bits(counts: np.ndarray) -> float:
    """Shannon entropy of a count vector, in bits. Zero counts contribute nothing."""
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0].astype(np.float64) / total
    return float(-(p * np.log2(p)).sum())


def marginal_stats(grid: np.ndarray, vocab: int) -> dict:
    """Pooled marginal entropy, per-position entropies, and codebook usage."""
    flat = grid.reshape(-1)
    pooled = np.bincount(flat, minlength=vocab)
    h_pooled = entropy_bits(pooled)

    _, hgt, wid = grid.shape
    per_pos = []
    for r in range(hgt):
        for c in range(wid):
            per_pos.append(entropy_bits(np.bincount(grid[:, r, c], minlength=vocab)))
    per_pos = np.asarray(per_pos, dtype=np.float64)

    return {
        "marginal_entropy_bits": h_pooled,
        "marginal_entropy_norm": h_pooled / math.log2(vocab),
        "per_pos_entropy_mean": float(per_pos.mean()),
        "per_pos_entropy_std": float(per_pos.std(ddof=1)),
        "per_pos_entropy_min": float(per_pos.min()),
        "per_pos_entropy_max": float(per_pos.max()),
        "per_pos_dispersion_norm": float(per_pos.std(ddof=1) / math.log2(vocab)),
        "codebook_used": int((pooled > 0).sum()),
        "codebook_utilization": float((pooled > 0).sum() / vocab),
    }


def _pairs(grid: np.ndarray, axis: str):
    """Adjacent (context, target) index pairs along one axis."""
    if axis == "left":
        ctx, tgt = grid[:, :, :-1], grid[:, :, 1:]
    elif axis == "up":
        ctx, tgt = grid[:, :-1, :], grid[:, 1:, :]
    else:
        raise ValueError(axis)
    return ctx.reshape(-1), tgt.reshape(-1)


def fit_bigram(train_grid: np.ndarray, vocab: int, axis: str):
    """Bigram counts from train. Fitted once, then scored at many smoothing values.

    At vocabulary 4096 the table is 16.8M cells against roughly 5M training tokens,
    so it is deliberately sparse; held-out scoring penalises that sparsity.
    """
    ctx, tgt = _pairs(train_grid, axis)
    counts = np.zeros((vocab, vocab), dtype=np.int32)
    np.add.at(counts, (ctx, tgt), 1)
    return counts, counts.sum(axis=1, dtype=np.float64)


def score_bigram(counts, row, grid: np.ndarray, vocab: int, axis: str, alpha: float) -> float:
    """Held-out cross-entropy of a split's targets under the fitted bigram, in bits.

    Probabilities are read straight out of the count matrix rather than
    materialising a normalised table, which would be 134 MB at V=4096.
    """
    ctx, tgt = _pairs(grid, axis)
    num = counts[ctx, tgt].astype(np.float64) + alpha
    den = row[ctx] + alpha * vocab
    return float(-np.mean(np.log2(num / den)))


def select_alpha_on_val(counts, row, val_grid, vocab: int, axis: str, grid: list[float]) -> float:
    """Pick the smoothing constant that minimises validation cross-entropy.

    Fixing alpha by hand is the weak point of this analysis: the pooled Spearman
    swings from +0.72 at alpha=1 to about -0.02 at alpha=0.01, which is not a
    property anyone should build a claim on. Selecting on a held-out validation
    split makes the choice a property of the data rather than of our taste, and
    mirrors the validation-based hyperparameter selection the paper already adopted
    for samplers. The test split is never consulted.

    Smoothing mass relative to data also differs sharply by vocabulary (alpha*V
    against roughly 5M tokens is 0.24x at V=1024 and 3.82x at V=4096), so a single
    hand-set alpha is not even comparable across the panel.
    """
    scores = [(score_bigram(counts, row, val_grid, vocab, axis, a), a) for a in grid]
    return min(scores)[1]


def analyse(name: str, meta: dict, tok_root: Path, alphas: list[float], grid: list[float]) -> dict:
    vocab = meta["vocab"]
    names = DIR_ALIASES.get(name, [f"chestmnist_{name}"])
    tok_dir = next((tok_root / n for n in names if (tok_root / n).is_dir()), None)
    if tok_dir is None:
        raise FileNotFoundError(f"no tokenized dir for {name}, tried {names} under {tok_root}")

    train = as_grid(load_tokens(tok_dir, "train"))
    test = as_grid(load_tokens(tok_dir, "test"))
    try:
        val = as_grid(load_tokens(tok_dir, "val"))
    except (FileNotFoundError, KeyError):
        val = None

    observed_max = int(max(train.max(), test.max()))
    if observed_max >= vocab:
        raise ValueError(f"{name}: token id {observed_max} exceeds declared vocab {vocab}")

    out = {
        "vocab": vocab,
        "dir": tok_dir.name,
        "n_train": int(train.shape[0]),
        "n_test": int(test.shape[0]),
        "n_val": int(val.shape[0]) if val is not None else 0,
        "grid": list(train.shape[1:]),
        "psnr": meta["psnr"],
        "best_fid": meta["best_fid"],
        "best_gen": meta["best_gen"],
    }
    out.update(marginal_stats(test, vocab))

    log2v = math.log2(vocab)
    h_marg = out["marginal_entropy_bits"]
    for axis in ("left", "up"):
        counts, row = fit_bigram(train, vocab, axis)

        for alpha in alphas:
            h = score_bigram(counts, row, test, vocab, axis, alpha)
            tag = f"cond_entropy_{axis}_a{alpha:g}"
            out[tag] = h
            out[f"{tag}_norm"] = h / log2v
            out[f"predictive_gain_{axis}_a{alpha:g}"] = h_marg - h

        # Full sweep, so ordering stability can be reported instead of asserted.
        out[f"sweep_{axis}"] = {
            f"{a:g}": h_marg - score_bigram(counts, row, test, vocab, axis, a) for a in grid
        }

        if val is not None:
            a_sel = select_alpha_on_val(counts, row, val, vocab, axis, grid)
            h = score_bigram(counts, row, test, vocab, axis, a_sel)
            out[f"alpha_selected_{axis}"] = a_sel
            out[f"cond_entropy_{axis}_valsel"] = h
            out[f"cond_entropy_{axis}_valsel_norm"] = h / log2v
            out[f"predictive_gain_{axis}_valsel"] = h_marg - h

        del counts, row
    return out


def _ordering_matches(rows: dict, key_fn) -> tuple[int, int]:
    """How many vocabulary levels a predictor orders exactly as best-observed FID."""
    hits = total = 0
    for vocab in (1024, 2048, 4096):
        grp = [n for n in rows if rows[n]["vocab"] == vocab]
        if len(grp) < 2:
            continue
        total += 1
        if sorted(grp, key=key_fn) == sorted(grp, key=lambda n: rows[n]["best_fid"]):
            hits += 1
    return hits, total


def group_auc(rows: dict, predictor: str, hi_group: set[str], sign: float = 1.0) -> float:
    """Rank-based separation of one quantizer family from the rest, 1.0 = perfect.

    This is the coarse question the panel can actually answer. Exact within-vocabulary
    orderings require resolving gaps like FSQ-4096 at 0.37 against LFQ-4096 at 0.41, a
    0.04 difference, while the measured seed spread for a single fixed configuration is
    std 0.024 with a range of 0.047. A criterion that hinges on separating two targets
    by less than the noise on those targets is not measuring the predictor.
    """
    vals = [(sign * rows[n][predictor], n in hi_group) for n in rows]
    pos = [v for v, g in vals if g]
    neg = [v for v, g in vals if not g]
    if not pos or not neg:
        return float("nan")
    wins = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
    return float(wins / (len(pos) * len(neg)))


def correlate(rows: dict, alphas: list[float], grid: list[float]) -> dict:
    names = list(rows.keys())
    fid = [rows[n]["best_fid"] for n in names]

    predictors = ["psnr", "marginal_entropy_norm", "per_pos_dispersion_norm"]
    for axis in ("left", "up"):
        for alpha in alphas:
            predictors.append(f"cond_entropy_{axis}_a{alpha:g}_norm")
            predictors.append(f"predictive_gain_{axis}_a{alpha:g}")
        if f"predictive_gain_{axis}_valsel" in rows[names[0]]:
            predictors.append(f"cond_entropy_{axis}_valsel_norm")
            predictors.append(f"predictive_gain_{axis}_valsel")

    # VQ is the family whose best FIDs sit clearly outside seed noise from the others,
    # so "does the statistic put VQ on the correct side" is the claim the data supports.
    vq = {n for n in names if n.startswith("VQ")}

    corr = {}
    for p in predictors:
        # PSNR predicts that higher is better, every other predictor that lower is.
        sign = -1.0 if p == "psnr" else 1.0
        hits, total = _ordering_matches(rows, lambda n, p=p, s=sign: s * rows[n][p])
        corr[p] = {
            "spearman_vs_best_fid": spearman([rows[n][p] for n in names], fid),
            "within_vocab_exact_orderings": f"{hits}/{total}",
            "vq_vs_rest_auc": group_auc(rows, p, vq, sign),
            "n": len(names),
        }

    # Within-vocabulary orderings matter more than a 9-point correlation: at matched
    # categorical rate the comparison is the one the paper's design was built for,
    # and smoothing mass is identical across the three quantizers at a fixed vocab.
    within: dict = {}
    for vocab in (1024, 2048, 4096):
        grp = [n for n in names if rows[n]["vocab"] == vocab]
        if len(grp) < 2:
            continue
        within[str(vocab)] = {
            "by_best_fid": sorted(grp, key=lambda n: rows[n]["best_fid"]),
            "by_psnr_desc": sorted(grp, key=lambda n: -rows[n]["psnr"]),
            **{f"by_{p}": sorted(grp, key=lambda n: rows[n][p]) for p in predictors if p != "psnr"},
        }

    # Stability across the whole smoothing range. A predictor whose ordering holds
    # over most of the grid is reportable; one that holds only at a hand-picked
    # alpha is not, and this is what tells the two apart.
    stability: dict = {}
    for axis in ("left", "up"):
        per_alpha = {}
        for a in grid:
            tag = f"{a:g}"
            if any(tag not in rows[n].get(f"sweep_{axis}", {}) for n in names):
                continue
            vals = [rows[n][f"sweep_{axis}"][tag] for n in names]
            hits, total = _ordering_matches(
                rows, lambda n, t=tag, ax=axis: rows[n][f"sweep_{ax}"][t]
            )
            pos = [v for v, n in zip(vals, names) if n in vq]
            neg = [v for v, n in zip(vals, names) if n not in vq]
            auc = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))
            per_alpha[tag] = {
                "spearman_vs_best_fid": spearman(vals, fid),
                "within_vocab_exact_orderings": f"{hits}/{total}",
                "vq_vs_rest_auc": float(auc),
            }
        if per_alpha:
            full = sum(1 for v in per_alpha.values() if v["within_vocab_exact_orderings"] == "3/3")
            perfect = sum(1 for v in per_alpha.values() if v["vq_vs_rest_auc"] == 1.0)
            stability[axis] = {
                "per_alpha": per_alpha,
                "n_alphas": len(per_alpha),
                "n_alphas_with_full_ordering": full,
                "fraction_full_ordering": full / len(per_alpha),
                "n_alphas_with_perfect_vq_separation": perfect,
                "fraction_perfect_vq_separation": perfect / len(per_alpha),
            }

    return {
        "predictors": corr,
        "within_vocab_orderings": within,
        "alpha_stability": stability,
    }


def smoke() -> int:
    """Check the statistics behave on fields with known structure."""
    rng = np.random.default_rng(0)
    V, N = 64, 400

    print("smoke: spearman")
    assert abs(spearman([1, 2, 3, 4], [1, 2, 3, 4]) - 1.0) < 1e-9
    assert abs(spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1.0) < 1e-9
    print("  ok")

    print("smoke: as_grid accepts flat and square inputs")
    assert as_grid(rng.integers(0, V, (10, 64))).shape == (10, 8, 8)
    assert as_grid(rng.integers(0, V, (10, 8, 8))).shape == (10, 8, 8)
    assert as_grid(rng.integers(0, V, (10, 1, 8, 8))).shape == (10, 8, 8)
    print("  ok")

    print("smoke: uniform iid field")
    iid_tr = rng.integers(0, V, (N, 8, 8))
    iid_te = rng.integers(0, V, (N, 8, 8))
    m = marginal_stats(iid_te, V)
    c_iid, r_iid = fit_bigram(iid_tr, V, "left")
    h_iid = score_bigram(c_iid, r_iid, iid_te, V, "left", 1.0)
    print(f"  marginal {m['marginal_entropy_bits']:.3f} bits (max {math.log2(V):.1f})")
    print(f"  conditional {h_iid:.3f} bits, gain {m['marginal_entropy_bits'] - h_iid:+.3f}")
    assert m["marginal_entropy_norm"] > 0.95, "uniform field should be near-maximal entropy"

    print("smoke: structured field (each row a repeated symbol)")
    base = rng.integers(0, V, (N, 8, 1))
    st_tr = np.repeat(base, 8, axis=2)
    base2 = rng.integers(0, V, (N, 8, 1))
    st_te = np.repeat(base2, 8, axis=2)
    m2 = marginal_stats(st_te, V)
    c_st, r_st = fit_bigram(st_tr, V, "left")
    h_st = score_bigram(c_st, r_st, st_te, V, "left", 1.0)
    gain = m2["marginal_entropy_bits"] - h_st
    print(f"  marginal {m2['marginal_entropy_bits']:.3f} bits")
    print(f"  conditional {h_st:.3f} bits, gain {gain:+.3f}")
    assert h_st < h_iid, "a perfectly predictable field must have lower conditional entropy"
    assert gain > 0, "structured field should show positive predictive gain"

    print("smoke: marginals match but conditionals separate them")
    print(f"  iid gain {m['marginal_entropy_bits'] - h_iid:+.3f} vs structured gain {gain:+.3f}")

    print("smoke: validation-selected alpha prefers less smoothing on structured data")
    grid = [10.0, 1.0, 0.1, 0.01, 0.001]
    a_iid = select_alpha_on_val(c_iid, r_iid, iid_te, V, "left", grid)
    a_st = select_alpha_on_val(c_st, r_st, st_te, V, "left", grid)
    print(f"  iid picks alpha={a_iid:g}, structured picks alpha={a_st:g}")
    assert a_st <= a_iid, "a predictable field should tolerate less smoothing"

    print("smoke: ok")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="self-test on synthetic fields")
    ap.add_argument("--alphas", type=float, nargs="+", default=[1.0, 0.01])
    ap.add_argument(
        "--alpha_grid",
        type=float,
        nargs="+",
        default=[10.0, 3.0, 1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001],
        help="smoothing values swept for validation selection and stability reporting",
    )
    ap.add_argument("--tokenized_root", default=None)
    ap.add_argument("--out", default=str(TABLES / "latent_modelability.json"))
    args = ap.parse_args()

    if args.smoke:
        return smoke()

    load_best_fid()

    if args.tokenized_root:
        tok_root = Path(args.tokenized_root)
    else:
        medtok = os.environ.get("MEDTOKENIZERS_ROOT")
        if not medtok:
            raise SystemExit("set MEDTOKENIZERS_ROOT or pass --tokenized_root")
        tok_root = Path(medtok) / "checkpoints" / "medmnist" / "tokenized"
    print(f"tokenized root: {tok_root}")

    rows = {}
    for name, meta in TOKENIZERS.items():
        print(f"\n{name} (V={meta['vocab']})")
        rows[name] = analyse(name, meta, tok_root, args.alphas, args.alpha_grid)
        r = rows[name]
        print(
            f"  marginal {r['marginal_entropy_bits']:.4f} bits "
            f"({r['marginal_entropy_norm']:.4f} normalised), "
            f"dispersion {r['per_pos_entropy_std']:.4f}"
        )
        for axis in ("left", "up"):
            a = args.alphas[0]
            k = f"cond_entropy_{axis}_a{a:g}"
            line = f"  H(X|{axis}) {r[k]:.4f} bits, gain {r[f'predictive_gain_{axis}_a{a:g}']:+.4f}"
            if f"alpha_selected_{axis}" in r:
                line += (
                    f"  | val-selected alpha={r[f'alpha_selected_{axis}']:g}, "
                    f"gain {r[f'predictive_gain_{axis}_valsel']:+.4f}"
                )
            print(line)

    results = {
        "_description": (
            "Generator-free latent statistics per discrete tokenizer, correlated "
            "against the best default-config generation FID-192 over the six discrete "
            "generators (results/chestmnist_factorial/eval_results.json). Conditional "
            "entropies are held-out: bigram counts fitted on train, cross-entropy scored "
            "on test. The smoothing constant is selected on the validation split, never "
            "on test. PSNR is included as the baseline predictor to beat."
        ),
        "meta": {
            "alphas": args.alphas,
            "alpha_grid": args.alpha_grid,
            "tokenized_root": str(tok_root),
            "n_tokenizers": len(rows),
        },
        "tokenizers": rows,
    }
    results.update(correlate(rows, args.alphas, args.alpha_grid))

    print(f"\nPredictors against best observed FID-192 (n={len(rows)}):")
    print(f"  {'predictor':<42} {'spearman':>9}  {'orderings':>9}  {'VQ-vs-rest AUC':>15}")
    for p, v in sorted(results["predictors"].items(), key=lambda kv: -kv[1]["vq_vs_rest_auc"]):
        print(
            f"  {p:<42} {v['spearman_vs_best_fid']:>+9.4f}  "
            f"{v['within_vocab_exact_orderings']:>9}  {v['vq_vs_rest_auc']:>15.3f}"
        )

    for axis, s in results.get("alpha_stability", {}).items():
        print(
            f"\npredictive_gain_{axis} across {s['n_alphas']} smoothing values:\n"
            f"  exact 3/3 within-vocab ordering at {s['n_alphas_with_full_ordering']} "
            f"({s['fraction_full_ordering']:.0%})\n"
            f"  perfect VQ-vs-rest separation at {s['n_alphas_with_perfect_vq_separation']} "
            f"({s['fraction_perfect_vq_separation']:.0%})"
        )

    print("\nPSNR is the baseline predictor. n=9, so treat these as directional.")
    print("Do not attach p-values.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(results, indent=2))
    tmp.replace(out)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
