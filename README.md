# Qwen 3.8 27B FP8 - Run Locally in 15 Minutes

**Stop paying $15K/month in API fees.** Run a 27B parameter frontier model on your MacBook Pro or consumer GPU.

This is the starter kit for running Qwen 3.8 27B FP8 locally with vLLM, optimized for development workflows.

## Why This Matters

Qwen just dropped a 27B FP8 model that punches way above its weight class. At FP8 quantization, you get:

- **~14GB VRAM requirement** (fits on RTX 4070, M3 Max, etc.)
- **Near-Claude-Code performance** on coding tasks
- **Zero API costs** after initial setup
- **Full data privacy** - nothing leaves your machine

## Quick Start

```bash
# Clone this repo
git clone https://github.com/Varritech/qwen-27b-local-starter
cd qwen-27b-local-starter

# Install dependencies (requires Python 3.10+)
pip install -r requirements.txt

# Download the model (14GB, ~10 mins on decent connection)
python scripts/download_model.py

# Start vLLM server
./scripts/start_server.sh

# Test it
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-27b-fp8",
    "messages": [{"role": "user", "content": "Write a Python function to reverse a string"}]
  }'
```

## What's Inside

- `requirements.txt` - pinned dependencies
- `scripts/download_model.py` - HuggingFace download with resume support
- `scripts/start_server.sh` - vLLM startup with optimal flags
- `docker-compose.yml` - optional containerized deployment
- `examples/` - working code samples for common tasks

## Hardware Requirements

**Minimum:**
- 16GB unified memory (Apple Silicon) OR
- 16GB VRAM (NVIDIA RTX 4070 / 3080+)
- 50GB free disk space

**Recommended:**
- 32GB+ unified memory or VRAM
- NVMe SSD for faster model loading

## Cost Comparison

| Setup | Monthly Cost | Latency | Privacy |
|-------|-------------|---------|---------|
| Claude Code API | $15-30K | 200-500ms | ❌ |
| This repo (self-hosted) | ~$800 (hardware amortized) | 50-150ms | ✅ |

## Next Steps

1. Fork this repo and customize for your stack
2. Check `examples/` for integration patterns
3. Read `GUIDE.md` for deep dive on optimization

---

**Built by [Varritech](https://varritech.com)** - We help startups ship AI products faster.

MIT License © 2026 Varritech
