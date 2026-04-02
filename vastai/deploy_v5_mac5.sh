#!/bin/bash
# Deploy KARL V5 MLX adapter to Mac5
set -e

echo "Deploying KARL V5 adapter to Mac5..."

# Upload adapter
scp -r ~/Desktop/karl/vastai/karl-v5-mlx-adapter/ mac5:~/models/karl-v5-adapter/
echo "Adapter uploaded"

# Kill existing MLX server, start with new adapter
ssh mac5 "pkill -f mlx_lm || true; sleep 2; \
  nohup python3 -m mlx_lm.server \
    --model mlx-community/Qwen3-4B-Instruct-2507-4bit \
    --adapter-path ~/models/karl-v5-adapter \
    --port 8100 \
    > ~/Desktop/mlx-v5-server.log 2>&1 & \
  sleep 5 && curl -s http://localhost:8100/v1/models | head -3"
echo "MLX server started with V5 adapter"

# Quick test
ssh mac5 'curl -s http://localhost:8100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"default\", \"messages\": [{\"role\": \"system\", \"content\": \"You are Mohamed'\''s cognitive twin.\"}, {\"role\": \"user\", \"content\": \"Rate the mesh architecture.\"}], \"max_tokens\": 100}" | python3 -c "import json,sys; print(json.load(sys.stdin)[\"choices\"][0][\"message\"][\"content\"])"'
echo "Test complete"
