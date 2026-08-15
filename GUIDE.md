# The Complete Guide to Running Qwen 3.8 27B FP8 Locally

## The Problem Nobody's Talking About

Your startup is burning $15-30K/month on Claude API fees. Your team can't ship features fast enough because every iteration costs money. You're locked into Anthropic's infrastructure, rate limits, and pricing changes.

Meanwhile, open weights models are getting scary good. Qwen just released a 27B parameter model in FP8 quantization that matches Claude Code on most coding benchmarks - and it runs on hardware you already own.

This guide walks you through everything: why FP8 matters, how to set it up, optimization tricks, and real production patterns we've learned from running this stack for clients.

## Why FP8 Changes Everything

FP8 (8-bit floating point) quantization reduces model size by ~4x compared to FP16 with minimal accuracy loss. For a 27B model:

- **FP16**: ~54GB VRAM required (A100 territory)
- **FP8**: ~14GB VRAM required (RTX 4070 / M3 Max territory)

The math: 27B parameters × 1 byte (FP8) = 27GB raw, but with KV cache overhead and vLLM optimizations, you're looking at ~14GB actual usage during inference.

## Hardware Reality Check

### Apple Silicon (Recommended for Dev)

**M3 Max / M2 Max (32GB+ unified memory)**
- ✅ Runs out of the box with llama.cpp or MLX
- ✅ Silent, no GPU drivers, just works
- ⚠️ Slower than dedicated GPU (~15-20 tokens/sec)
- Cost: Already own it or $3-4K new

**MacBook Pro M3 (16GB)**
- ✅ Barely fits with aggressive swapping
- ⚠️ Expect 5-8 tokens/sec, noticeable lag
- Cost: Free if you have one

### NVIDIA GPU (Recommended for Production)

**RTX 4070 (12GB VRAM)**
- ❌ Won't fit full model - need 4070 Ti SUPER (16GB) minimum
- Cost: ~$600-800 used

**RTX 4080/4090 (16-24GB VRAM)**
- ✅ Perfect fit, blazing fast (40-60 tokens/sec)
- Cost: $1000-2000

**RTX 3090 (24GB VRAM)**
- ✅ Best value play, used market goldmine
- ⚠️ Power hungry (350W), needs good PSU
- Cost: ~$700-900 used

### Cloud Alternative

**RunPod / Lambda Labs**
- RTX 4090 pod: ~$0.70/hour
- Monthly cost at 8hr/day: ~$170
- ✅ No upfront hardware cost
- ⚠️ Ongoing expense, data leaves your network

## Installation: The Path That Actually Works

### Step 1: Environment Setup

```bash
# Create clean Python env
python3.10 -m venv qwen-env
source qwen-env/bin/activate

# Install vLLM (optimized for Qwen)
pip install vllm==0.6.0
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install extras
pip install huggingface_hub transformers accelerate
```

### Step 2: Download the Model

Don't use `git clone` - HuggingFace Hub is smarter:

```python
# scripts/download_model.py
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3.8-27B-FP8",
    local_dir="./models/qwen-27b-fp8",
    resume_download=True,  # Critical for large models
    max_workers=8  # Parallel downloads
)
```

Run it:
```bash
python scripts/download_model.py
# Takes ~10 mins on 100Mbps connection, 14GB total
```

### Step 3: Start vLLM Server

```bash
#!/bin/bash
# scripts/start_server.sh

vllm serve ./models/qwen-27b-fp8 \
  --port 8000 \
  --host 0.0.0.0 \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9 \
  --quantization fp8 \
  --enable-chunked-prefill \
  --served-model-name qwen-27b-fp8
```

Key flags explained:
- `--gpu-memory-utilization 0.9`: Use 90% of VRAM (leave 10% for OS)
- `--quantization fp8`: Enable FP8 kernels (2x speedup on Ada Lovelace+)
- `--enable-chunked-prefill`: Better throughput for long contexts
- `--max-model-len 8192`: Context window (increase if you have VRAM)

### Step 4: Test It

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-27b-fp8",
    "messages": [
      {"role": "system", "content": "You are a helpful coding assistant."},
      {"role": "user", "content": "Write a Python function to find the longest palindromic substring"}
    ],
    "max_tokens": 500
  }'
```

Expected output: ~15-20 tokens/sec on M3 Max, ~40-50 on RTX 4090.

## Optimization Tricks We Learned the Hard Way

### 1. Paged Attention is Non-Negotiable

vLLM's paged attention gives you 2-4x throughput vs naive implementations. Don't skip it.

### 2. Batch Your Requests

If you're processing multiple files or running batch jobs:

```python
# Bad: sequential requests
for file in files:
    response = requests.post(url, json={"prompt": file})

# Good: batched
response = requests.post(url, json={
    "prompts": [f"Analyze: {f}" for f in files],
    "use_batched_api": True
})
```

### 3. KV Cache Tuning

For long-running sessions (like Cursor-style IDE integration):

```bash
--max-num-seqs 256  # Concurrent sequences
--max-num-batched-tokens 10000  # Tokens per batch
```

Adjust based on your VRAM headroom.

### 4. Cold Start Mitigation

Model loading takes 30-60 seconds. Keep the server running:

```bash
# systemd service (Linux)
[Unit]
Description=Qwen 27B vLLM Server
After=network.target

[Service]
Type=simple
User=youruser
ExecStart=/path/to/scripts/start_server.sh
Restart=always

[Install]
WantedBy=multi-user.target
```

## Production Patterns

### Pattern 1: Local Dev + Cloud Fallback

Run locally for development, failover to cloud API during peak loads:

```python
def get_completion(prompt):
    try:
        # Try local first
        return local_vllm(prompt, timeout=5)
    except TimeoutError:
        # Fallback to cloud
        return anthropic_client(prompt)
```

### Pattern 2: Multi-GPU Scaling

Two 4090s? Run tensor parallel:

```bash
--tensor-parallel-size 2
```

Doubles your throughput, nearly linear scaling.

### Pattern 3: RAG Integration

Qwen 27B + local vector DB = private ChatGPT:

```python
from chromadb import Client

chroma = Client()
collection = chroma.get_collection("docs")

def rag_query(query):
    docs = collection.query(query_texts=[query], n_results=3)
    context = "\n".join(docs["documents"][0])
    prompt = f"Context:\n{context}\n\nQuestion: {query}"
    return vllm_complete(prompt)
```

## Troubleshooting

### "CUDA out of memory"

Reduce `--gpu-memory-utilization` to 0.85 or close other GPU apps.

### Slow token generation (<10 tok/sec)

- Ensure you're using FP8 quantization
- Check GPU isn't thermal throttling
- Reduce `--max-num-seqs` if batching

### Model won't download

HuggingFace rate limiting? Use mirror:
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## The Real ROI

Let's do the math:

**Before (Claude Code API):**
- 5 developers × $200/month each = $1,000/month
- Heavy users: 3 × $600/month = $1,800/month
- **Total: $2,800/month = $33,600/year**

**After (Self-hosted Qwen 27B):**
- Hardware: RTX 4090 build = $2,500 (one-time)
- Electricity: ~$15/month
- Maintenance: ~2 hours setup time
- **Total Year 1: $2,680 | Year 2+: $180/year**

**Break-even: Month 1. Savings over 3 years: ~$98K**

And you own the infrastructure. No rate limits. No pricing surprises. Full data privacy.

## What's Next

1. **Fork this repo** - customize for your stack
2. **Integrate with your IDE** - Cursor, VSCode, Zed all support OpenAI-compatible APIs
3. **Add RAG** - connect to your codebase, docs, Slack history
4. **Scale horizontally** - add more GPUs or nodes as needed

The frontier is moving fast. Qwen 27B is just the beginning. Self-hosting isn't just about cost savings anymore - it's about velocity, privacy, and control.

---

**Built by [Varritech](https://varritech.com)**

Questions? Issues? PRs welcome. MIT License.
