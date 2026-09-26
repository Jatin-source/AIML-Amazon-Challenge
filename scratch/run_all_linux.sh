#!/bin/bash
echo "Starting V3 Model Training..."
python3 scratch/train_v3.py > scratch/train_v3_logs.txt

echo "Starting V3 Inference (1.7M entities)..."
python3 scratch/infer_v3.py > scratch/infer_v3_logs.txt

echo "Applying Global Hungarian Resolution..."
python3 scratch/apply_hungarian_resolution.py > scratch/hungarian_logs.txt

echo "Pipeline Finished! Files are in outputs/"
