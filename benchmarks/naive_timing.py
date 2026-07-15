"""Naive baseline timing for pi0 (Phase 0).

Honest wall-clock latency of one action-chunk prediction on the GPU:
vision encode + backbone prefill + 10-step flow denoise. No optimization.
CUDA-synchronized around each call so the numbers are real (not async launch time).

Usage:
    python benchmarks/naive_timing.py --dtype bfloat16 --runs 10 --warmup 2
"""
from __future__ import annotations

import argparse
import statistics
import time

import torch

from lerobot.policies.pi0 import PI0Policy
from lerobot.utils.constants import (
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

MODEL_ID = "lerobot/pi0_base"


def build_batch(policy: PI0Policy, device: str, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    cfg = policy.config
    model_dtype = next(policy.parameters()).dtype
    b = 1
    batch: dict = {}
    # one random image per camera the checkpoint expects, [B,3,H,W] in [0,1]
    # (images are internally forced to float32 by the vision preprocess)
    h, w = cfg.image_resolution
    for key in cfg.image_features:
        batch[key] = torch.rand(b, 3, h, w, device=device)
    # padded state vector — must match model weight dtype
    batch[OBS_STATE] = torch.rand(b, cfg.max_state_dim, device=device, dtype=model_dtype)
    # pre-tokenized language (tokenization is trivial/CPU; time the model, not the tokenizer)
    L = cfg.tokenizer_max_length
    batch[OBS_LANGUAGE_TOKENS] = torch.zeros(b, L, dtype=torch.long, device=device)
    batch[OBS_LANGUAGE_ATTENTION_MASK] = torch.ones(b, L, dtype=torch.bool, device=device)
    return batch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA not available"
    device = "cuda"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    torch.cuda.reset_peak_memory_stats()
    print(f"loading {MODEL_ID} ({args.dtype}) ...")
    policy = PI0Policy.from_pretrained(MODEL_ID)
    policy.to(device=device, dtype=dtype)
    policy.eval()

    load_vram = torch.cuda.memory_allocated() / 1e9
    print(f"VRAM after load: {load_vram:.2f} GB")

    batch = build_batch(policy, device, args.seed)

    # first inference (correctness smoke)
    with torch.no_grad():
        out = policy.predict_action_chunk(batch)
    torch.cuda.synchronize()
    print(f"first inference OK — action chunk shape {tuple(out.shape)} "
          f"(expect [1, {policy.config.chunk_size}, action_dim])")

    # timed runs, CUDA-synchronized
    lat_ms = []
    with torch.no_grad():
        for _ in range(args.runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            policy.predict_action_chunk(batch)
            torch.cuda.synchronize()
            lat_ms.append((time.perf_counter() - t0) * 1000)

    timed = lat_ms[args.warmup:]
    peak_vram = torch.cuda.max_memory_allocated() / 1e9
    mean = statistics.mean(timed)
    worst = max(timed)
    p50 = statistics.median(timed)

    print("\n==== NAIVE BASELINE (pi0_base, RTX 4090) ====")
    print(f"dtype             : {args.dtype}")
    print(f"denoise steps     : {policy.config.num_inference_steps}")
    print(f"chunk_size        : {policy.config.chunk_size}")
    print(f"VRAM after load   : {load_vram:.2f} GB")
    print(f"VRAM peak (infer) : {peak_vram:.2f} GB")
    print(f"runs (kept)       : {len(timed)} (dropped {args.warmup} warmup)")
    print(f"latency mean      : {mean:.1f} ms")
    print(f"latency p50       : {p50:.1f} ms")
    print(f"latency WORST     : {worst:.1f} ms")
    print(f"effective rate    : {policy.config.n_action_steps / (mean/1000):.1f} Hz "
          f"(chunk {policy.config.n_action_steps} / mean latency)")


if __name__ == "__main__":
    main()
