# Tokenizer-Generator Coupling in Medical Image Generation

Reproduction code and result data for

> Liam Chalcroft. *Tokenizer-Generator Coupling in Medical Image Generation.*
> Advances in Neural Information Processing Systems (NeurIPS 2026).
> [arXiv:2608.07713](https://arxiv.org/abs/2608.07713)

The models themselves live in two libraries on PyPI:
[medtokenizers](https://github.com/liamchalcroft/medtokenizers) (tokenizers) and
[medlatents](https://github.com/liamchalcroft/medlatents) (latent generators, samplers,
evaluation). This repository holds only what is specific to the paper: the ChestMNIST
factorial and its supporting runs, the stored result files behind every reported number,
and the figure scripts.

## Installation

```bash
uv sync
```

This installs `medlatents` and `medtokenizers` (both 0.1.2 or later) from PyPI.
`matplotlib` is pinned to the version the paper figures were rendered with, so `figures/`
reproduces them pixel for pixel.

Scripts that train, sample or evaluate also need the tokenizer checkpoints, the tokenized
datasets and the 64 x 64 MedMNIST arrays from a medtokenizers checkout. They look for it at
`../medtokenizers` by default; set `MEDTOKENIZERS_ROOT` to point elsewhere. Generator
checkpoints are written to `checkpoints/` (override with `CHECKPOINT_ROOT`), which is not
tracked.

## Layout

```
results/
  chestmnist_factorial/      eval_results.json (main grid), memorization.json,
                             sweep_results.json (2K sampler screen), sweep_validation.json
                             (LFQ-1024 D3PM/SEDD step sweep at 10K)
  pneumoniamnist_factorial/  eval_results.json (cross-dataset replication)
  organamnist_factorial/     eval_results.json (cross-dataset replication)
  tables/                    one JSON per table or analysis in the paper
  throughput_a6000/          generation throughput measurements
  generated_tokens/          generated token indices behind stored VQ, seed, data-scale
                             and sampler-sweep results
  samples/                   fixed-seed sample matrix and memorization gallery pairs
  trajectories/              intermediate sampler states for every sample-matrix cell
scripts/                     training, sampling, sweep, evaluation and analysis entry points
figures/
  fig_*.py                   one script per paper figure, sharing style.py and tiles.py
  inputs/                    raster renders that some figures restyle
  out/                       rendered PDF and PNG figures
```

All Python scripts are run from the repository root as `uv run python scripts/<name>.py`
or `uv run python figures/<name>.py`. Scripts under `scripts/` resolve every path from
`scripts/paths.py`, and figure scripts from `figures/style.py`.

`scripts/train_medmnist2d_tokens.py` and `scripts/tokenize_medmnist.py` are copies of the
medlatents 0.1.0 files `examples/train_medmnist2d_tokens.py` and `scripts/tokenize_medmnist.py`,
which are not part of the published wheel. The only change is that `tokenize_medmnist.py`
imports medtokenizers from the installed package rather than a source checkout.

## Pipeline

These steps need GPUs, the medtokenizers checkpoints and, from step 2 onwards, trained
generator checkpoints.

1. Train the 54 discrete cells and 18 continuous reference runs: `bash scripts/run_factorial.sh`.
2. Evaluate them: `uv run python scripts/evaluate_chestmnist_generation.py`.
3. Select sampler settings at vocabulary 1024: `uv run python scripts/revalidate_on_val.py`
   (screens each grid from `scripts/sweep_sampling_hps.py` on the validation split, then
   evaluates the selected setting once on the test split).

VQ index-to-code decoding (`DiscreteTokenizer.detokenize`) applies the same codebook
normalisation as training, which requires medtokenizers 0.1.2 or later (pinned in
`pyproject.toml`).

## Paper tables

Labels are those of the extended version (arXiv); the conference version uses the short
labels in brackets.

| Table | Result file | Script |
| --- | --- | --- |
| Main results matrix (`tab:main_results`; `tab:main`) | `results/chestmnist_factorial/eval_results.json` | `scripts/evaluate_chestmnist_generation.py` |
| Training-seed variance (`tab:seed_variance`; `tab:seeds`) | `results/tables/seed_variance.json`, `results/tables/cross_gpu_calibration.json` | `scripts/run_seed_variance.sh SEED`, via `scripts/eval_seed_cell.py` |
| Tuned FID across seeds (`tab:tuned_seeds`; tuned rows of `tab:seeds`) | `results/tables/hp_revalidation.json` (seed 42), `hp_revalidation_seed43.json`, `hp_revalidation_seed44.json` | `scripts/run_tuned_seeds.sh SEED` |
| Fixed-effect decomposition (`tab:factorial_effects`; `tab:anova`) | `results/tables/factorial_effects.json` | `scripts/analyze_factorial_effects.py` |
| Sampler sweep and selected settings (`tab:hp_sweep`, `tab:selected_hps`; `tab:hps`) | `results/tables/hp_revalidation.json` | `scripts/revalidate_on_val.py` |
| Reconstruction against generation ranks (`tab:rank_failures`) | `results/chestmnist_factorial/eval_results.json`, `results/tables/recon_vs_generation.json` | `figures/fig_recon.py` |
| FID-192 against FID-2048 (`tab:fid_dual`) | `results/tables/fid_dual_2048.json`, `results/tables/fid_dual_d3pm.json` | `scripts/eval_fid_dual.py --cells default12`, `--cells d3pm_interaction` |
| Cross-dataset replication (`tab:cross_dataset`; `tab:cross`) | `results/{pneumoniamnist,organamnist}_factorial/eval_results.json`, `results/tables/cross_dataset_noise_floor.json` | `scripts/tokenize_paper_datasets.sh`, `scripts/run_paper_dataset_factorial.sh`, then `evaluate_chestmnist_generation.py` with `MEDGEN_DATASET` set; `scripts/cross_dataset_noise_floor.py` |
| Training-set-size control (`tab:data_scale`) | `results/tables/data_scale.json` | `scripts/run_data_scale.sh`, via `scripts/subsample_tokens.py` |
| Generator-free predictors (`tab:predictor`) | `results/tables/latent_modelability.json`, `results/tables/latent_modelability_alpha_sensitivity.json` | `scripts/latent_modelability.py`, then `scripts/modelability_alpha_sensitivity.py` |
| Tokenizer properties (`tab:tokenizer_props`) | `results/tables/token_entropy_summary.json`, `results/tables/benchmarks.json` | `scripts/analyze_token_entropy.py` |
| Generation throughput (`tab:throughput`) | `results/throughput_a6000/*.json` | `scripts/benchmark_throughput.py` |
| Domain-FID sanity check (`tab:domain_fid_validation`; `tab:domain`) | `results/tables/domain_fid_validation.json` | `scripts/train_chestmnist_classifier.py`, then `scripts/compute_domain_fid.py` |
| Bootstrap confidence intervals (`tab:fid_ci`; `tab:ci`) | `results/tables/fid_per_cell_ci.json`, `results/tables/fid_noise_floor.json` | `scripts/fid_bootstrap.py` |

Tokenizer reconstruction quality (`tab:tokenizers`; `tab:recon`) is produced by
medtokenizers. The remaining tables describe the design and settings and carry no results.

Numbers quoted in the text:

| Result | Result file | Script |
| --- | --- | --- |
| Classifier two-sample test (Spearman 0.86 against FID-192) | `results/tables/c2st.json` | `scripts/c2st_eval.py`, on samples saved by `scripts/eval_fid_dual.py --save_samples` |
| Memorization probe, feature space | `results/chestmnist_factorial/memorization.json` | `scripts/memorization_probe.py` |
| Memorization probe, pixel and LPIPS space | `results/tables/memorization_pixel_lpips.json` | `scripts/memorization_pixel_lpips.py` |
| Cross-GPU calibration (A6000 against L40S, six seed-42 cells) | `results/tables/cross_gpu_calibration.json` | sampling and FID functions of `scripts/evaluate_chestmnist_generation.py` (A6000, sampling seeded at 42); `scripts/eval_seed_cell.py` (L40S) |
| Per-position token entropy at vocabulary 1024 | `results/tables/token_entropy_per_position_1024.json` | `scripts/token_entropy_maps.py` |
| Compute budget | `results/tables/compute_budget.json` | `scripts/compute_budget.py` |

The VQ-1024 rows of the seed and data-scale tables were trained with the same scripts and
evaluated with the sampling and FID functions of `scripts/evaluate_chestmnist_generation.py`,
with sampling seeded at 42; the generated tokens are stored in
`results/generated_tokens/chestmnist/` (`VQ1024_<generator>_seed{43,44}_seed42_n10000.npz`
and `VQ1024_transformer_n{4700,20000,78468}_seed42_n10000.npz`), next to the tokens of the
18 VQ cells of the main table, the six seed-42 calibration cells and the cross-dataset
VQ + AR cells.

In `results/chestmnist_factorial/sweep_results.json` each entry records the reference
split it was scored against (`split`). The validation-split screens are the ones behind
`hp_revalidation.json`; their generated tokens are in
`results/generated_tokens/chestmnist/sweep/<cell>/` for the VQ-1024 generators and the
three DFM cells.

## Paper figures

Every figure regenerates from stored data without a GPU:

```bash
for f in figures/fig_*.py; do uv run python "$f"; done
```

| Figure | Script | Data |
| --- | --- | --- |
| Interaction plot (`interaction`) | `figures/fig_interaction.py` | `results/tables/seed_variance.json`, `fid_dual_2048.json`, `fid_dual_d3pm.json` |
| Sampling budget against FID (`nfe_vs_fid`) | `figures/fig_nfe.py` | `results/chestmnist_factorial/sweep_validation.json`, `results/tables/hp_revalidation.json` |
| Reconstruction against generation (`recon_vs_gen`) | `figures/fig_recon.py` | `results/chestmnist_factorial/eval_results.json` |
| Uncurated samples (`sample_grid`) | `figures/fig_samples.py` | `figures/inputs/sample_grid_original.png`, from `scripts/generate_paper_samples.py` |
| Memorization probe (`memorization_scatter`) | `figures/fig_memorization.py` | `results/chestmnist_factorial/memorization.json`, `eval_results.json` |
| Per-position token entropy (`token_entropy_1024`) | `figures/fig_entropy.py` | `results/tables/derived_per_position_entropy_1024.json`, from `figures/recover_entropy.py` |
| Memorization gallery (`memorization_gallery`) | `figures/fig_gallery.py` | `results/samples/memorization_gallery_pairs.npz`, from `scripts/memorization_gallery_pairs.py` |
| Centre inpainting (`inpaint_demo`) | `figures/fig_probes.py` | `figures/inputs/inpaint_demo_original.png` |
| Counterfactual inpainting (`anomaly_demo`) | `figures/fig_probes.py` | `figures/inputs/anomaly_demo_original.png` |

`derived_per_position_entropy_1024.json` is recovered from the colour-mapped renders in
`figures/inputs/token_entropy/` (written by `scripts/analyze_token_entropy.py`); resolution
is about 0.01 bits. The exact per-position entropies, computed from the tokenized test set,
are in `results/tables/token_entropy_per_position_1024.json` and agree with the recovered
values to within 0.03 bits.

## Samples and trajectories

`results/samples/` holds one fixed-seed batch of 16 uncurated samples for each of the 70
cells and the two tuned LFQ-1024 settings (`sample_matrix.npz`, with sampler settings,
checkpoint hashes and FID-192 per cell in `sample_matrix.json`), the first 16 real test
images (`real_test.npz`) and a contact sheet, all from `scripts/sample_matrix.py`.
`results/trajectories/` holds the intermediate sampler states for the same cells, seed and
batch, from `scripts/sample_trajectories.py`; each `<cell>.npz` stores the decoded frames as
`images` and the per-cell details are in `manifest.json`.

## Citation

```bibtex
@inproceedings{chalcroft2026coupling,
  title     = {Tokenizer-Generator Coupling in Medical Image Generation},
  author    = {Chalcroft, Liam},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS 2026)},
  year      = {2026},
  eprint    = {2608.07713},
  archivePrefix = {arXiv}
}
```

## Licence

MIT. See [LICENSE](LICENSE).
