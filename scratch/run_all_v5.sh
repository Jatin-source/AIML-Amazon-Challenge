#!/bin/bash
set -euo pipefail
echo "===== V5 Pipeline Start (resumable) ====="
echo "[1/3] Train + per-country F0.5 tuning (Phase B/C/D/F)..."
python3 scratch/train_v4_resumable.py >> scratch/train_v4_logs.txt 2>&1
echo "[2/3] Parallel resumable inference (Phase A/F)..."
python3 scratch/infer_v4_resumable.py >> scratch/infer_v4_logs.txt 2>&1
echo "[3/3] Shard assembly + Hungarian (Phase G)..."
python3 scratch/finalize_v4.py >> scratch/finalize_v4_logs.txt 2>&1
echo "===== V5 Pipeline Complete ====="
