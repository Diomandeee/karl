#!/bin/bash
# KARL V7 Training on Vast.ai RTX 4090
# 2-stage: SFT (1500 steps) + DPO (200 steps)
# Estimated: ~45min SFT + ~15min DPO = ~1hr total
# Cost: ~$0.35/hr RTX 4090

set -e

echo "=== KARL V7 Training ==="
echo "Stage 1: SFT (2284 examples, 1500 steps)"
echo "Stage 2: DPO (173 pairs, 200 steps)"
echo ""

# Install deps
pip install -q torch transformers peft bitsandbytes accelerate trl datasets sentencepiece

# Stage 1: SFT with inscription conditioning
echo ">>> Stage 1: SFT Training"
python train.py --config configs/karl_v7_qlora.yaml

# Stage 2: DPO on top of SFT adapter
echo ""
echo ">>> Stage 2: DPO Training"
python train.py --config configs/karl_v7_qlora.yaml \
    --override training.output_dir=runs/karl-v7-sft-dpo \
    --override training.max_steps=200 \
    --override training.learning_rate=5e-5 \
    --override training.per_device_train_batch_size=1 \
    --override training.gradient_accumulation_steps=8 \
    --dpo \
    --sft_checkpoint runs/karl-v7-sft/checkpoint-latest

# Stage 3: SFT-only control (no DPO) for comparison
echo ""
echo ">>> Stage 3: SFT Control (no DPO)"
python train.py --config configs/karl_v7_qlora.yaml \
    --override training.output_dir=runs/karl-v7-sft-control \
    --override anticipation.use_inscription=false \
    --override anticipation.use_gate=false

echo ""
echo "=== V7 Training Complete ==="
echo "Artifacts:"
echo "  SFT+DPO: runs/karl-v7-sft-dpo/"
echo "  SFT-only: runs/karl-v7-sft/"
echo "  Control:  runs/karl-v7-sft-control/"
