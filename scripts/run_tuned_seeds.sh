#!/usr/bin/env bash
# Tuned re-selection across training seeds: does the D3PM/SEDD sampler-retuning
# improvement replicate across training seeds under the paper's own protocol?
#
# It runs the
# identical revalidate_on_val.py pipeline that produced the seed-42 hp_revalidation
# numbers (screen the sampling grid on the validation split at 2K, select the min-val
# config, evaluate that config once on test at 10K), but against a seed-43 or seed-44
# checkpoint. It is NOT a config-transfer test: each seed selects its own optimum, so
# a shifted optimal step count (100 at seed 42, ~500 at seed 43) still counts as
# replication as long as the improvement over the default reappears.
#
# revalidate_on_val.py resolves checkpoints as GEN_CKPT_ROOT/chestmnist_{TOK}_{MODEL},
# and parse_run_name rejects a _seed{N} suffix, so we expose the seed checkpoints
# under canonical names via a scratch CHECKPOINT_ROOT of symlinks. Nothing in the
# real checkpoint tree is moved or renamed. Sampling RNG stays at seed 42 across all
# runs, so the only thing varying between them is the trained checkpoint.
#
# Usage:
#   MEDTOKENIZERS_ROOT=../medtokenizers bash scripts/run_tuned_seeds.sh 43
set -euo pipefail

SEED="${1:?usage: run_tuned_seeds.sh SEED}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
MEDTOK="${MEDTOKENIZERS_ROOT:?set MEDTOKENIZERS_ROOT}"
SEEDS_LOGDIR="${LOGDIR:-$REPO/checkpoints/chestmnist_factorial_seeds}"
CELLS="LFQ1024_d3pm,LFQ1024_sedd"

SCRATCH="$(mktemp -d)"
GEN_ROOT="$SCRATCH/checkpoints/chestmnist_factorial"
mkdir -p "$GEN_ROOT"
cleanup() { rm -rf "$SCRATCH"; }
trap cleanup EXIT

for cell in LFQ1024_d3pm LFQ1024_sedd; do
  seed_dir="$SEEDS_LOGDIR/chestmnist_${cell}_seed${SEED}"
  [ -d "$seed_dir" ] || { echo "MISSING seed checkpoint dir: $seed_dir"; exit 1; }
  ln -sfn "$seed_dir" "$GEN_ROOT/chestmnist_${cell}"
done
echo "scratch checkpoint root:"
ls -l "$GEN_ROOT"

OUT="$REPO/results/tables/hp_revalidation_seed${SEED}.json"
echo "running val-selection -> test eval for seed $SEED cells [$CELLS]"
CHECKPOINT_ROOT="$SCRATCH/checkpoints" MEDTOKENIZERS_ROOT="$MEDTOK" \
  uv run python scripts/revalidate_on_val.py \
  --only "$CELLS" --skip_noise_floor \
  --seed 42 \
  --output "$OUT"

echo "wrote $OUT"
echo "compare test_fid_10k against seed-42: LFQ1024_d3pm 0.0898, LFQ1024_sedd 0.098"
