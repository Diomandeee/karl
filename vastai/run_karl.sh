#!/usr/bin/env bash
# KARL Cognitive Twin — 5-run controlled experiment on Vast.ai RTX 4090
# Same architecture as anticipation-geometry: inscription + gate + LSE
set -uo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG="configs/karl_twin_qlora.yaml"
EVAL="python3 evaluate.py"

echo "=========================================="
echo " KARL Cognitive Twin — GPU Training Suite"
echo "=========================================="

# Run 1: Control (no anticipation modules)
echo -e "\n[1/5] CONTROL — no anticipation"
python3 train.py --config $CONFIG \
    --override anticipation.use_inscription=false \
    --override anticipation.use_gate=false \
    --override anticipation.use_lse_loop=false \
    --override training.output_dir=runs/karl-twin-control
$EVAL --checkpoint runs/karl-twin-control/final --data /workspace/data/real_conv_eval.jsonl --output runs/karl-twin-control/eval.json

# Run 2: Treatment (full anticipation)
echo -e "\n[2/5] TREATMENT — full anticipation"
python3 train.py --config $CONFIG \
    --override training.output_dir=runs/karl-twin-treatment
$EVAL --checkpoint runs/karl-twin-treatment/final --data /workspace/data/real_conv_eval.jsonl --output runs/karl-twin-treatment/eval.json

# Run 3: Ablation — no inscription
echo -e "\n[3/5] ABLATION — no inscription"
python3 train.py --config $CONFIG \
    --override anticipation.use_inscription=false \
    --override training.output_dir=runs/karl-twin-no-inscription
$EVAL --checkpoint runs/karl-twin-no-inscription/final --data /workspace/data/real_conv_eval.jsonl --output runs/karl-twin-no-inscription/eval.json

# Run 4: Ablation — no gate
echo -e "\n[4/5] ABLATION — no gate"
python3 train.py --config $CONFIG \
    --override anticipation.use_gate=false \
    --override training.output_dir=runs/karl-twin-no-gate
$EVAL --checkpoint runs/karl-twin-no-gate/final --data /workspace/data/real_conv_eval.jsonl --output runs/karl-twin-no-gate/eval.json

# Run 5: Ablation — no LSE loop
echo -e "\n[5/5] ABLATION — no LSE loop"
python3 train.py --config $CONFIG \
    --override anticipation.use_lse_loop=false \
    --override training.output_dir=runs/karl-twin-no-lse
$EVAL --checkpoint runs/karl-twin-no-lse/final --data /workspace/data/real_conv_eval.jsonl --output runs/karl-twin-no-lse/eval.json

# Summary
echo -e "\n=========================================="
echo " All 5 runs complete. Results:"
echo "=========================================="
for d in runs/karl-twin-*/; do
    name=$(basename "$d")
    if [ -f "$d/eval.json" ]; then
        echo "  $name: $(python3 -c "import json; d=json.load(open('${d}eval.json')); print(f'NLL={d[\"nll\"]:.4f} scalar_mse={d.get(\"scalar_mse\",0):.4f} commit_acc={d.get(\"commitment_acc\",0):.4f}')")"
    fi
done
