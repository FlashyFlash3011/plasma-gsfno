#!/usr/bin/env bash
# Train plasma-gsfno on CUDA, then evaluate + benchmark + plot.
# Usage:  bash deploy/runpod/train_and_eval.sh [DATA_H5] [EPOCHS]
#   DATA_H5  default data/equilibria.h5
#   EPOCHS   default 200 (early stopping usually ends sooner)
set -euo pipefail

cd "$(cd "$(dirname "$0")/../.." && pwd)"   # plasma-gsfno root

DATA="${1:-data/equilibria.h5}"
EPOCHS="${2:-200}"
mkdir -p results

if [ ! -f "$DATA" ]; then
  echo "Dataset not found: $DATA  (generate locally and upload it here)"; exit 1
fi

echo "==> training on $DATA for up to $EPOCHS epochs (bf16, CUDA)"
python scripts/train.py \
  data.hdf5_path="$DATA" data.NR=65 data.NZ=65 \
  data.batch_size=64 train.epochs="$EPOCHS" train.amp_dtype=bf16

echo "==> evaluating best checkpoint on the test split"
python scripts/evaluate.py \
  --checkpoint checkpoints/best.pt --hdf5 "$DATA" --split test \
  --output results/eval_test.json

echo "==> benchmarking inference latency vs FreeGS"
python scripts/benchmark.py --checkpoint checkpoints/best.pt \
  --batch-size 1 --n-samples 100 || echo "(benchmark skipped/failed — non-fatal)"

echo "==> rendering prediction figure"
python scripts/plot_prediction.py \
  --checkpoint checkpoints/best.pt --hdf5 "$DATA" --split test \
  --out results/prediction.png || echo "(plot skipped/failed — non-fatal)"

echo "==> artifacts in results/ : eval_test.json, prediction.png ; checkpoint in checkpoints/best.pt"
echo "    pull them back with:  runpodctl send results checkpoints   (or scp)"
