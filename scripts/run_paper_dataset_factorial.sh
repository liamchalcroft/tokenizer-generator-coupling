#!/bin/bash
# Train the 6 paper-replication generator cells on a target dataset.
# Cells: LFQ-1024 + AR/D3PM, FSQ-1024 + AR, VQ-1024 + AR, VAE-c2 + LDM, AE + RF.

set -euo pipefail

DATASET="${DATASET:-pneumoniamnist}"
MEDTOKENIZERS_ROOT="${MEDTOKENIZERS_ROOT:-../medtokenizers}"
TOKENIZED_ROOT="${TOKENIZED_ROOT:-${MEDTOKENIZERS_ROOT}/checkpoints/medmnist/tokenized}"
LOGDIR="${LOGDIR:-./checkpoints/${DATASET}_factorial}"
EPOCHS="${EPOCHS:-100}"
EPOCHS_CONTINUOUS="${EPOCHS_CONTINUOUS:-200}"
BATCH_SIZE="${BATCH_SIZE:-128}"
HIDDEN_SIZE="${HIDDEN_SIZE:-512}"
DEPTH="${DEPTH:-12}"
NUM_HEADS="${NUM_HEADS:-8}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
PROJECT_NAME="${PROJECT_NAME:-${DATASET}-factorial}"
USE_WANDB="${USE_WANDB:-false}"

mkdir -p "$LOGDIR"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
wandb_flag() { [ "$USE_WANDB" = "true" ] && echo "--use_wandb" || echo "--no-use_wandb"; }

run_train() {
    local name="$1"; shift
    local checkpoint="${LOGDIR}/${name}/checkpoint_best.pt"
    if [ -f "$checkpoint" ]; then
        log "SKIP ${name} (checkpoint exists)"
        return 0
    fi
    log "Training ${name}..."
    "$@" || { log "FAILED ${name}"; return 1; }
    log "DONE ${name}"
}

# Discrete cells: LFQ + AR/D3PM, FSQ + AR, VQ + AR
DISCRETE_COMMON="--epochs $EPOCHS --batch_size $BATCH_SIZE
    --hidden_size $HIDDEN_SIZE --depth $DEPTH --num_heads $NUM_HEADS
    --mixed_precision $MIXED_PRECISION --logdir $LOGDIR --use_ema $(wandb_flag)
    --project_name $PROJECT_NAME --val_interval 10 --generate_samples_every 25"

# LFQ + AR (transformer)
run_train "${DATASET}_LFQ_transformer" \
    uv run python scripts/train_medmnist2d_tokens.py \
    --tokens_path "$TOKENIZED_ROOT/${DATASET}_LFQ" \
    --model transformer --name "${DATASET}_LFQ_transformer" --lr 3e-4 --weight_decay 0.01 \
    $DISCRETE_COMMON

# LFQ + D3PM
run_train "${DATASET}_LFQ_d3pm" \
    uv run python scripts/train_medmnist2d_tokens.py \
    --tokens_path "$TOKENIZED_ROOT/${DATASET}_LFQ" \
    --model d3pm --name "${DATASET}_LFQ_d3pm" --lr 1e-4 --weight_decay 0.01 \
    --d3pm_transition absorbing --d3pm_loss_type cross_entropy \
    --diffusion_steps 1000 --diffusion_schedule cosine \
    $DISCRETE_COMMON

# FSQ + AR
run_train "${DATASET}_FSQ_transformer" \
    uv run python scripts/train_medmnist2d_tokens.py \
    --tokens_path "$TOKENIZED_ROOT/${DATASET}_FSQ" \
    --model transformer --name "${DATASET}_FSQ_transformer" --lr 3e-4 --weight_decay 0.01 \
    $DISCRETE_COMMON

# VQ + AR
run_train "${DATASET}_VQ_transformer" \
    uv run python scripts/train_medmnist2d_tokens.py \
    --tokens_path "$TOKENIZED_ROOT/${DATASET}_VQ" \
    --model transformer --name "${DATASET}_VQ_transformer" --lr 3e-4 --weight_decay 0.01 \
    $DISCRETE_COMMON

# Continuous: VAE-c2 + LDM
run_train "${DATASET}_VAE_diffusion" \
    uv run python scripts/train_medmnist2d_tokens.py \
    --tokens_path "$TOKENIZED_ROOT/${DATASET}_VAE" \
    --model diffusion --name "${DATASET}_VAE_diffusion" --lr 1e-4 --weight_decay 0.01 \
    --epochs $EPOCHS_CONTINUOUS --batch_size $BATCH_SIZE \
    --hidden_size $HIDDEN_SIZE --depth $DEPTH --num_heads $NUM_HEADS \
    --mixed_precision $MIXED_PRECISION --logdir $LOGDIR --use_ema $(wandb_flag) \
    --project_name $PROJECT_NAME --val_interval 10 --generate_samples_every 25 \
    --diffusion_steps 1000 --diffusion_schedule cosine

# Continuous: AE + RF
run_train "${DATASET}_AE_continuous_flow" \
    uv run python scripts/train_medmnist2d_tokens.py \
    --tokens_path "$TOKENIZED_ROOT/${DATASET}_AE" \
    --model continuous_flow --name "${DATASET}_AE_continuous_flow" --lr 1e-4 --weight_decay 0.01 \
    --epochs $EPOCHS_CONTINUOUS --batch_size $BATCH_SIZE \
    --hidden_size $HIDDEN_SIZE --depth $DEPTH --num_heads $NUM_HEADS \
    --mixed_precision $MIXED_PRECISION --logdir $LOGDIR --use_ema $(wandb_flag) \
    --project_name $PROJECT_NAME --val_interval 10 --generate_samples_every 25 \
    --normalize_latents \
    --flow_num_steps 100 --base_std 1.0

log "=== ALL GENERATORS DONE for $DATASET ==="
