#!/bin/bash
set -e
echo "===== V4 Pipeline Start ====="

echo "[1/3] Full 3-country training + F0.5 threshold tuning (Phase B+C+D)..."
python3 scratch/train_v3_full.py > scratch/train_v3_full_logs.txt 2>&1
echo "Training done. Thresholds tuned."

echo "[2/3] Parallelised inference on 64 cores (Phase A+D)..."
python3 scratch/infer_v3_parallel.py > scratch/infer_v3_parallel_logs.txt 2>&1
echo "Inference done."

echo "[3/3] Global Hungarian resolution..."
python3 scratch/apply_hungarian_resolution.py > scratch/hungarian_logs.txt 2>&1
echo "Hungarian done."

echo "===== V4 Pipeline Complete — outputs/ ready ====="
