#!/usr/bin/env bash
# Creates a subsampled version for training on another device
set -euo pipefail

DEST="${1:-./vqvae-batch-device}"
mkdir -p "$DEST"

cp -r \
    vqvae_batch.py \
    data.py \
    train.py \
    evaluate.py \
    requirements.txt \
    "$DEST/"

mkdir -p "$DEST/output"

echo "Exported to $DEST"
echo "Copy it: rsync -avz $DEST user@host:~/"
