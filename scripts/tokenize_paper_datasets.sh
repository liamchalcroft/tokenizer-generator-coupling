#!/bin/bash
# Tokenize a MedMNIST dataset with the 5 paper-replication tokenizers.

set -e
cd "$(dirname "$0")/.."

DATASET="${DATASET:-pneumoniamnist}"
MEDTOKENIZERS_ROOT="${MEDTOKENIZERS_ROOT:-$(pwd)/../medtokenizers}"
OUT="${MEDTOKENIZERS_ROOT}/checkpoints/medmnist/tokenized"

run_tokenize() {
    local quantizer="$1"; shift
    local ckpt="${MEDTOKENIZERS_ROOT}/checkpoints/medmnist/${DATASET}_${quantizer}_final"
    local outdir="${OUT}/${DATASET}_${quantizer}"
    if [ -f "${outdir}/metadata.json" ] && [ -f "${outdir}/test.npz" ]; then
        echo "[SKIP] ${DATASET}_${quantizer} already tokenized"
        return 0
    fi
    if [ ! -d "$ckpt" ]; then
        echo "[ERR] $ckpt not found"; return 1
    fi
    echo "=== Tokenize $DATASET with $quantizer $* ==="
    VIRTUAL_ENV=$(pwd)/.venv uv run python scripts/tokenize_medmnist.py \
        --dataset "$DATASET" --quantizer "$quantizer" \
        --checkpoint_dir "$ckpt" --output_dir "$OUT" --device cuda "$@"
}

run_tokenize LFQ --codebook_size 1024 --codebook_dim 10 --num_codebooks 1
run_tokenize FSQ --levels 4 4 4 4 4
run_tokenize VQ  --num_embeddings 1024
run_tokenize VAE --latent_channels 2
run_tokenize AE  --latent_channels 4

echo "=== ALL TOKENIZE DONE for $DATASET ==="
