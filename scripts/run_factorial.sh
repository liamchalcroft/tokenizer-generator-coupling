#!/bin/bash
# ChestMNIST MedGen-Bench: 54 discrete cells + 18 continuous-latent reference runs.
# The reference runs cover 16 distinct cells: VAEc4 and VAE (lambda_KL = 1e-6) are
# the same setting under two names, one per sweep axis.
#
# Usage:
#   bash scripts/run_factorial.sh                          # full factorial
#   DISCRETE_ONLY=true bash scripts/run_factorial.sh       # discrete only
#   FILTER_TOK="FSQ1024 FSQ4096" bash scripts/run_factorial.sh  # specific tokenizers
#   DRY_RUN=true bash scripts/run_factorial.sh             # print commands only

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────
MEDTOKENIZERS_ROOT="${MEDTOKENIZERS_ROOT:-../medtokenizers}"
TOKENIZED_ROOT="${TOKENIZED_ROOT:-${MEDTOKENIZERS_ROOT}/checkpoints/medmnist/tokenized}"
LOGDIR="${LOGDIR:-./checkpoints/chestmnist_factorial}"
EPOCHS="${EPOCHS:-100}"
EPOCHS_CONTINUOUS="${EPOCHS_CONTINUOUS:-200}"
BATCH_SIZE="${BATCH_SIZE:-128}"
HIDDEN_SIZE="${HIDDEN_SIZE:-512}"
DEPTH="${DEPTH:-12}"
NUM_HEADS="${NUM_HEADS:-8}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
PROJECT_NAME="${PROJECT_NAME:-chestmnist-factorial}"
USE_WANDB="${USE_WANDB:-false}"
DRY_RUN="${DRY_RUN:-false}"
DISCRETE_ONLY="${DISCRETE_ONLY:-false}"

# ── Tokenizer inventory ──────────────────────────────────────────────
# Discrete 3×3 grid: VQ × LFQ × FSQ at 1024/2048/4096
DISCRETE_TOKS="${FILTER_TOK:-VQ1024 VQ2048 VQ4096 LFQ1024 LFQ2048 LFQ4096 FSQ1024 FSQ2048 FSQ4096}"
# LFQ4096 uses old naming convention
DISCRETE_GENERATORS="transformer maskgit flow d3pm sedd bayesian_flow"

# Continuous channel sweep + KL/no-KL reference rows
CONTINUOUS_TOKS="${FILTER_CONT_TOK:-VAEc1 VAEc2 VAEc4 VAEc8 VAEc16 VAE1e5 VAE VAE1e7 AEdisc}"
CONTINUOUS_GENERATORS="diffusion continuous_flow"

# ── Helpers ───────────────────────────────────────────────────────────
log() { echo "[$(date '+%H:%M:%S')] $*"; }
wandb_flag() { [ "$USE_WANDB" = "true" ] && echo "--use_wandb" || echo "--no-use_wandb"; }

resolve_tokens_dir() {
    local tok="$1"
    local dir="${TOKENIZED_ROOT}/chestmnist_${tok}"
    # Handle old naming for LFQ4096
    if [ ! -d "$dir" ] && [ "$tok" = "LFQ4096" ]; then
        dir="${TOKENIZED_ROOT}/chestmnist_LFQ"
    fi
    echo "$dir"
}

# Generator-specific learning rates (from literature review)
lr_for_gen() {
    case "$1" in
        transformer|maskgit|flow) echo "3e-4" ;;
        d3pm|sedd|bayesian_flow)  echo "1e-4" ;;
        diffusion|continuous_flow) echo "1e-4" ;;
    esac
}

run_train() {
    local name="$1"; shift
    local checkpoint="${LOGDIR}/${name}/checkpoint_best.pt"
    if [ -f "$checkpoint" ]; then
        log "SKIP ${name} (checkpoint exists)"
        return 0
    fi
    if [ "$DRY_RUN" = "true" ]; then
        log "DRY RUN: $*"
        return 0
    fi
    log "Training ${name}..."
    "$@" || { log "FAILED ${name}"; return 1; }
    log "DONE ${name}"
}

# ── Discrete factorial (9 tokenizers × 6 generators = 54 cells) ─────
log "========================================="
log "Discrete factorial: ${DISCRETE_TOKS}"
log "Generators: ${DISCRETE_GENERATORS}"
log "========================================="

for tok in $DISCRETE_TOKS; do
    tokens_dir=$(resolve_tokens_dir "$tok")
    if [ ! -d "$tokens_dir" ]; then
        log "SKIP ${tok} (tokenized data not found at ${tokens_dir})"
        continue
    fi

    COMMON="--tokens_path $tokens_dir
        --epochs $EPOCHS --batch_size $BATCH_SIZE
        --hidden_size $HIDDEN_SIZE --depth $DEPTH --num_heads $NUM_HEADS
        --mixed_precision $MIXED_PRECISION
        --logdir $LOGDIR --use_ema $(wandb_flag)
        --project_name $PROJECT_NAME
        --val_interval 10 --generate_samples_every 25"

    for gen in $DISCRETE_GENERATORS; do
        name="chestmnist_${tok}_${gen}"
        lr=$(lr_for_gen "$gen")
        extra_args=""

        case "$gen" in
            maskgit)
                extra_args="--use_curriculum --mask_schedule cosine"
                ;;
            flow)
                extra_args="--source_dist mask --scheduler_type polynomial --scheduler_power 3.0 --flow_loss_type cross_entropy"
                ;;
            d3pm)
                extra_args="--d3pm_transition absorbing --d3pm_loss_type cross_entropy --diffusion_steps 1000 --diffusion_schedule cosine"
                ;;
            sedd)
                extra_args="--d3pm_transition absorbing --diffusion_steps 1000 --diffusion_schedule cosine"
                ;;
        esac

        run_train "$name" \
            uv run python scripts/train_medmnist2d_tokens.py \
            $COMMON --model "$gen" --name "$name" --lr "$lr" \
            --weight_decay 0.01 $extra_args
    done
done

# ── Continuous reference panel (9 tokenizers × 2 generators = 18 cells) ───
if [ "$DISCRETE_ONLY" != "true" ]; then
    log "========================================="
    log "Continuous reference panel: ${CONTINUOUS_TOKS}"
    log "Generators: ${CONTINUOUS_GENERATORS}"
    log "========================================="

    for tok in $CONTINUOUS_TOKS; do
        tokens_dir="${TOKENIZED_ROOT}/chestmnist_${tok}"
        if [ ! -d "$tokens_dir" ]; then
            log "SKIP ${tok} (tokenized data not found)"
            continue
        fi

        # Low-KL / no-KL latents may need normalization
        normalize_flag=""
        case "$tok" in
            AE|AEdisc|VAEc1) normalize_flag="--normalize_latents" ;;
        esac

        for gen in $CONTINUOUS_GENERATORS; do
            name="chestmnist_${tok}_${gen}"
            lr=$(lr_for_gen "$gen")
            extra_args=""

            case "$gen" in
                diffusion)
                    extra_args="--diffusion_steps 1000 --diffusion_schedule cosine"
                    ;;
                continuous_flow)
                    extra_args="--flow_num_steps 100 --base_std 1.0"
                    ;;
            esac

            run_train "$name" \
                uv run python scripts/train_medmnist2d_tokens.py \
                --tokens_path "$tokens_dir" \
                --model "$gen" \
                --epochs "$EPOCHS_CONTINUOUS" --batch_size "$BATCH_SIZE" \
                --lr "$lr" --hidden_size "$HIDDEN_SIZE" --depth "$DEPTH" \
                --num_heads "$NUM_HEADS" --mixed_precision "$MIXED_PRECISION" \
                $normalize_flag \
                --logdir "$LOGDIR" --name "$name" \
                --use_ema $(wandb_flag) \
                --project_name "$PROJECT_NAME" \
                --val_interval 10 --generate_samples_every 25 \
                $extra_args
        done
    done
fi

log "========================================="
log "Factorial complete!"
log "========================================="
