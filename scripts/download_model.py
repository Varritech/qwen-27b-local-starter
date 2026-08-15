#!/usr/bin/env python3
"""Download Qwen 3.8 27B FP8 from HuggingFace with resume support."""

from huggingface_hub import snapshot_download
import os

def main():
    print("Downloading Qwen 3.8 27B FP8 (~14GB)...")
    print("This will take ~10 minutes on a decent connection.")
    
    snapshot_download(
        repo_id="Qwen/Qwen3.8-27B-FP8",
        local_dir="./models/qwen-27b-fp8",
        resume_download=True,
        max_workers=8,
        local_dir_use_symlinks=False
    )
    
    print("\n✅ Download complete!")
    print("Model saved to ./models/qwen-27b-fp8")
    print("\nNext step: Run ./scripts/start_server.sh")

if __name__ == "__main__":
    main()
