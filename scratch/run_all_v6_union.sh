#!/bin/bash
set -euo pipefail
echo "===== V6 UNION Pipeline Start ====="
python3 -m pip install -q lightgbm==4.7.0 rapidfuzz jellyfish joblib pyarrow 2>&1 | tail -1
echo "[1/2] Union train + threshold tuning..."
python3 scratch/train_v5_union.py >> scratch/train_v5_union_logs.txt 2>&1
echo "[2/2] Union inference..."
python3 scratch/infer_v5_union.py >> scratch/infer_v5_union_logs.txt 2>&1
echo "===== V6 UNION Complete ====="
