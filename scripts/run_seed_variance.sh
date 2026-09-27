#!/usr/bin/env bash
# Training-seed variance for the vocabulary-1024 interaction block.
#
# Trains the vocabulary-1024 interaction block {VQ,LFQ,FSQ} x {transformer,maskgit,d3pm}
# plus SEDD on LFQ, at a given seed, then evaluates each at its DEFAULT sampling
# config (10K test FID-192). This block is the interaction claim in miniature: LFQ
# is best under AR and D3PM but worst under MaskGIT in the single-seed table, so if
# that pattern survives seeds the non-separability result stops resting on point
# estimates.
#
# Per-cell training args mirror scripts/run_factorial.sh exactly (same lr, same
# per-generator extra args), with --seed threaded through and a _seed{N} name
# suffix. Resumes cleanly: cells whose checkpoint_best.pt already exists are skipped,
# so LFQ1024_d3pm_seed43 (already trained in the pilot) is not redone.
#
# SEED=42 is special: no training happens (the main-table checkpoints already exist),
# and the canonical chestmnist_factorial/ checkpoints are evaluated on THIS GPU. That
# is deliberate. Seed variance is only meaningful if every seed is evaluated on the
# same Inception feature path; the main-table seed-42 numbers were produced on an
# A6000, so re-evaluating seed 42 here puts all three seeds on one GPU.
#
# Usage:
#   MEDTOKENIZERS_ROOT=../medtokenizers bash scripts/run_seed_variance.sh 42
#   MEDTOKENIZERS_ROOT=../medtokenizers bash scripts/run_seed_variance.sh 43
#   MEDTOKENIZERS_ROOT=../medtokenizers bash scripts/run_seed_variance.sh 44
set -euo pipefail

SEED="${1:?usage: run_seed_variance.sh SEED}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
MEDTOK="${MEDTOKENIZERS_ROOT:?set MEDTOKENIZERS_ROOT}"
TOKENIZED_ROOT="$MEDTOK/checkpoints/medmnist/tokenized"
LOGDIR="${LOGDIR:-$REPO/checkpoints/chestmnist_factorial_seeds}"
EPOCHS="${EPOCHS:-100}"

# Overridable so the block can be split across pods (e.g. BLOCK_TOKS="FSQ1024" on
# one pod, "VQ1024 LFQ1024" on another). EXTRA_SEDD_TOK="" disables the SEDD cell on
# pods that are not covering LFQ.
BLOCK_TOKS="${BLOCK_TOKS:-VQ1024 LFQ1024 FSQ1024}"
BLOCK_GENS="${BLOCK_GENS:-transformer maskgit d3pm}"
EXTRA_SEDD_TOK="${EXTRA_SEDD_TOK-LFQ1024}"   # SEDD only on LFQ; unset to skip

mkdir -p "$LOGDIR"
log() { echo "[$(date '+%H:%M:%S')] $*"; }

lr_for_gen() {
  case "$1" in
    transformer|maskgit|flow) echo "3e-4" ;;
    d3pm|sedd|bayesian_flow)  echo "1e-4" ;;
  esac
}
extra_for_gen() {
  case "$1" in
    maskgit) echo "--use_curriculum --mask_schedule cosine" ;;
    d3pm)    echo "--d3pm_transition absorbing --d3pm_loss_type cross_entropy --diffusion_steps 1000 --diffusion_schedule cosine" ;;
    sedd)    echo "--d3pm_transition absorbing --diffusion_steps 1000 --diffusion_schedule cosine" ;;
    *) echo "" ;;
  esac
}

# Canonical (seed-42, main-table) checkpoint dir vs the seed-suffixed dir for 43/44.
cell_ckpt() {
  local tok="$1" gen="$2"
  if [ "$SEED" = "42" ]; then
    echo "$REPO/checkpoints/chestmnist_factorial/chestmnist_${tok}_${gen}/checkpoint_best.pt"
  else
    echo "$LOGDIR/chestmnist_${tok}_${gen}_seed${SEED}/checkpoint_best.pt"
  fi
}

train_cell() {
  local tok="$1" gen="$2"
  if [ "$SEED" = "42" ]; then return 0; fi   # main-table seed: eval only, never retrain
  local name="chestmnist_${tok}_${gen}_seed${SEED}"
  local ckpt="$LOGDIR/$name/checkpoint_best.pt"
  if [ -f "$ckpt" ]; then log "SKIP train $name (checkpoint exists)"; return 0; fi
  local tokens_dir="$TOKENIZED_ROOT/chestmnist_${tok}"
  [ -d "$tokens_dir" ] || { log "MISSING tokens $tokens_dir"; return 1; }
  log "TRAIN $name"
  uv run python scripts/train_medmnist2d_tokens.py \
    --tokens_path "$tokens_dir" \
    --epochs "$EPOCHS" --batch_size 128 \
    --hidden_size 512 --depth 12 --num_heads 8 \
    --mixed_precision bf16 --logdir "$LOGDIR" --use_ema --no-use_wandb \
    --project_name chestmnist-a4-seeds \
    --val_interval 10 --generate_samples_every 25 \
    --model "$gen" --name "$name" --lr "$(lr_for_gen "$gen")" \
    --weight_decay 0.01 --seed "$SEED" $(extra_for_gen "$gen")
  log "DONE train $name"
}

eval_cell() {
  local tok="$1" gen="$2"
  local ckpt; ckpt="$(cell_ckpt "$tok" "$gen")"
  [ -f "$ckpt" ] || { log "no ckpt to eval for chestmnist_${tok}_${gen} seed $SEED ($ckpt)"; return 1; }
  log "EVAL chestmnist_${tok}_${gen} seed $SEED (default config, 10K)"
  MEDTOKENIZERS_ROOT="$MEDTOK" uv run python scripts/eval_seed_cell.py \
    --checkpoint "$ckpt" --tokenizer "$tok" \
    --cell "chestmnist_${tok}_${gen}" --seed_label "$SEED" \
    --num_samples 10000 \
    --out "$REPO/results/tables/seed_variance.json"
}

if [ "$SEED" = "42" ]; then
  log "=== seed 42: eval-only on this GPU (no training) ==="
else
  log "=== seed $SEED: train ==="
  for tok in $BLOCK_TOKS; do
    for gen in $BLOCK_GENS; do train_cell "$tok" "$gen"; done
  done
  [ -n "$EXTRA_SEDD_TOK" ] && train_cell "$EXTRA_SEDD_TOK" sedd
fi

log "=== seed $SEED: default-config eval ==="
for tok in $BLOCK_TOKS; do
  for gen in $BLOCK_GENS; do eval_cell "$tok" "$gen" || true; done
done
[ -n "$EXTRA_SEDD_TOK" ] && { eval_cell "$EXTRA_SEDD_TOK" sedd || true; }

log "=== seed $SEED done. seed_variance.json updated. ==="
log "Next: tuned re-selection via scripts/run_tuned_seeds.sh $SEED"
