"""Latency waterfall for pi0 (Phase 1, coarse).

Splits the ~217ms of one action-chunk prediction into its real stages by
wrapping the hot-path methods with CUDA events (no Nsight needed):

    preprocess  -> resize/pad/normalize images on GPU
    embed_prefix-> SigLIP vision encode (all cameras) + language embedding
    prefill     -> one PaliGemma forward over the prefix, builds the KV cache
    denoise     -> the 10-step flow loop (each step = embed_suffix + KV clone +
                   gemma_expert forward + action out-proj)
    overhead    -> everything not captured above (Python, launches, transfers)

This is method-level attribution, not kernel-level. It answers "which of the
three engines owns the budget?" — the Phase 1 question. Kernel-level (Nsight
Compute) comes after we know where to point it.

Usage:
    python benchmarks/latency_waterfall.py --runs 10 --warmup 2
"""
from __future__ import annotations

import argparse
import functools
import statistics

import torch

from lerobot.policies.pi0 import PI0Policy
from lerobot.utils.constants import (
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

MODEL_ID = "lerobot/pi0_base"

# name -> list[(start_event, end_event)] recorded during the current run
_events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {}


def _timed(name, fn):
    """Wrap a method so each call records a CUDA start/end event pair."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn(*args, **kwargs)
        end.record()
        _events.setdefault(name, []).append((start, end))
        return out

    return wrapper


def _elapsed(name) -> float:
    """Total ms across every recorded call for `name` (after a sync)."""
    return sum(s.elapsed_time(e) for s, e in _events.get(name, []))


def _count(name) -> int:
    return len(_events.get(name, []))


def build_batch(policy: PI0Policy, device: str, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    cfg = policy.config
    model_dtype = next(policy.parameters()).dtype
    b = 1
    batch: dict = {}
    h, w = cfg.image_resolution
    for key in cfg.image_features:
        batch[key] = torch.rand(b, 3, h, w, device=device)
    batch[OBS_STATE] = torch.rand(b, cfg.max_state_dim, device=device, dtype=model_dtype)
    L = cfg.tokenizer_max_length
    batch[OBS_LANGUAGE_TOKENS] = torch.zeros(b, L, dtype=torch.long, device=device)
    batch[OBS_LANGUAGE_ATTENTION_MASK] = torch.ones(b, L, dtype=torch.bool, device=device)
    return batch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["bfloat16", "float32"], default="float32")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA not available"
    device = "cuda"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    print(f"loading {MODEL_ID} ({args.dtype}) ...")
    policy = PI0Policy.from_pretrained(MODEL_ID)
    policy.to(device=device, dtype=dtype)
    policy.eval()

    model = policy.model
    pgw = model.paligemma_with_expert

    # --- instrument the hot path (wrap the bound methods on the instances) ---
    policy._preprocess_images = _timed("preprocess", policy._preprocess_images)
    model.embed_prefix = _timed("embed_prefix", model.embed_prefix)
    model.denoise_step = _timed("denoise_step", model.denoise_step)
    # every PaliGemmaWithExpert.forward: call [0] is the prefill, [1..] are the
    # per-step expert forwards inside denoise_step.
    pgw.forward = _timed("pgw_forward", pgw.forward)

    batch = build_batch(policy, device, args.seed)

    # warmup + correctness smoke (not timed)
    with torch.no_grad():
        out = policy.predict_action_chunk(batch)
    torch.cuda.synchronize()
    print(f"first inference OK — action chunk {tuple(out.shape)}")

    buckets = ["preprocess", "embed_prefix", "prefill", "denoise", "overhead"]
    per_run: dict[str, list[float]] = {k: [] for k in buckets}
    totals: list[float] = []
    denoise_steps_seen = 0

    with torch.no_grad():
        for r in range(args.runs):
            _events.clear()
            torch.cuda.synchronize()
            total_start = torch.cuda.Event(enable_timing=True)
            total_end = torch.cuda.Event(enable_timing=True)
            total_start.record()
            policy.predict_action_chunk(batch)
            total_end.record()
            torch.cuda.synchronize()

            if r < args.warmup:
                continue

            total = total_start.elapsed_time(total_end)
            pre = _elapsed("preprocess")
            emb = _elapsed("embed_prefix")
            pgw_calls = _events.get("pgw_forward", [])
            prefill = pgw_calls[0][0].elapsed_time(pgw_calls[0][1]) if pgw_calls else 0.0
            denoise = _elapsed("denoise_step")
            overhead = total - (pre + emb + prefill + denoise)
            denoise_steps_seen = _count("denoise_step")

            per_run["preprocess"].append(pre)
            per_run["embed_prefix"].append(emb)
            per_run["prefill"].append(prefill)
            per_run["denoise"].append(denoise)
            per_run["overhead"].append(overhead)
            totals.append(total)

    mean_total = statistics.mean(totals)
    print("\n==== LATENCY WATERFALL (pi0_base, RTX 4090) ====")
    print(f"dtype           : {args.dtype}")
    print(f"denoise steps   : {denoise_steps_seen} (per prediction)")
    print(f"runs (kept)     : {len(totals)} (dropped {args.warmup} warmup)")
    print(f"total mean      : {mean_total:.1f} ms\n")
    print(f"{'stage':<14}{'mean ms':>10}{'% total':>10}")
    print("-" * 34)
    for k in buckets:
        m = statistics.mean(per_run[k])
        print(f"{k:<14}{m:>10.1f}{100 * m / mean_total:>9.1f}%")
    print("-" * 34)
    per_step = statistics.mean(per_run["denoise"]) / max(denoise_steps_seen, 1)
    print(f"\nper denoise step: {per_step:.1f} ms  x{denoise_steps_seen} steps")
    print(f"=> the flow loop alone is {100 * statistics.mean(per_run['denoise']) / mean_total:.0f}% of the budget")


if __name__ == "__main__":
    main()
