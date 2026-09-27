#!/usr/bin/env bash
# Training-set-size control: does VQ overtake LFQ at small training-set size on
# ChestMNIST itself, matching the PneumoniaMNIST inversion?
# Trains AR on LFQ-1024 and VQ-1024 at three train sizes and evaluates each.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
MEDTOK="${MEDTOKENIZERS_ROOT:?set MEDTOKENIZERS_ROOT}"
TOKROOT="$MEDTOK/checkpoints/medmnist/tokenized"
LOGDIR="$REPO/checkpoints/chestmnist_datascale"
OUT="$REPO/results/tables/data_scale.json"
SIZES="4700 20000 78468"   # Pneumonia-scale / intermediate / full
TOKS="LFQ1024 VQ1024"
mkdir -p "$LOGDIR" "$(dirname "$OUT")"
cd "$REPO"
log(){ echo "[a6 $(date -u +%H:%M:%S)] $*"; }

for tok in $TOKS; do
  for n in $SIZES; do
    name="chestmnist_${tok}_transformer_n${n}"
    ckpt="$LOGDIR/$name/checkpoint_best.pt"
    subdir="$TOKROOT/chestmnist_${tok}_n${n}"
    if [ ! -f "$ckpt" ]; then
      log "subsample $tok -> n=$n"
      uv run python scripts/subsample_tokens.py --src "$TOKROOT/chestmnist_${tok}" --n "$n" --dst "$subdir" --seed 0
      log "TRAIN $name"
      uv run python scripts/train_medmnist2d_tokens.py \
        --tokens_path "$subdir" --epochs 100 --batch_size 128 \
        --hidden_size 512 --depth 12 --num_heads 8 --mixed_precision bf16 \
        --logdir "$LOGDIR" --use_ema --no-use_wandb --project_name chestmnist-a6 \
        --val_interval 10 --generate_samples_every 0 \
        --model transformer --name "$name" --lr 3e-4 --weight_decay 0.01 --seed 42
    else
      log "SKIP train $name (exists)"
    fi
    log "EVAL $name"
    MEDTOKENIZERS_ROOT="$MEDTOK" uv run python scripts/eval_seed_cell.py \
      --checkpoint "$ckpt" --tokenizer "$tok" \
      --cell "chestmnist_${tok}_transformer" --seed_label "n${n}" \
      --num_samples 10000 --out "$OUT" || log "eval $name failed"
  done
done
log "=== A6_DONE ==="
log "interpretation: compare LFQ vs VQ at each n; if VQ<LFQ at n=4700 but LFQ<VQ at full, the Pneumonia inversion is a data-scale effect"
