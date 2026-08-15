#!/bin/bash
# Start vLLM server for Qwen 27B FP8

set -e

MODEL_PATH="./models/qwen-27b-fp8"

if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ Model not found. Run 'python scripts/download_model.py' first."
    exit 1
fi

echo "🚀 Starting vLLM server..."
echo "Model: $MODEL_PATH"
echo "Port: 8000"
echo ""

vllm serve "$MODEL_PATH" \
  --port 8000 \
  --host 0.0.0.0 \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9 \
  --quantization fp8 \
  --enable-chunked-prefill \
  --served-model-name qwen-27b-fp8
