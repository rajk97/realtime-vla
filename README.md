# Real-Time π0

### The full latency anatomy of a VLA, and what it takes to hit 30Hz+ on one consumer GPU

Building — in public — a rigorous latency study and systems-level optimization of the
[π0](https://www.physicalintelligence.company/blog/pi0) vision-language-action (VLA) model,
targeting **≥3× end-to-end speedup** and **≥30Hz sustained control** on a single **RTX 4090**,
with **≤2% task-success degradation** validated on LIBERO.

> **Status:** Day 0 — baseline setup. Follow progress in **[LOG.md](LOG.md)** (updated every session).

---

## Why this exists

Text-LLM serving is commoditized; real-time **VLA** inference is still embryonic. This project treats
π0 as three engines sharing one hard deadline — a SigLIP vision encoder, a Gemma 2B backbone, and a
flow-matching action expert — and asks the honest question: *where does the time actually go, and how
fast can it go without breaking the policy?*

Every speed claim here is paired with a correctness number. Speed without correctness is fraud in this
domain.

## Success criteria (definition of done)

1. **≥3×** end-to-end speedup over the naive PyTorch baseline on the 4090.
2. Sustained **≥30Hz** effective control rate (reported alongside time-to-first-action).
3. **≤2% absolute** task-success degradation on LIBERO vs. baseline.
4. Public deliverables: reproducible repo + writeup (latency waterfall, Hz-vs-success Pareto) + ≥1 upstream PR.

## Benchmark rigor (non-negotiable)

- Report **time-to-first-action (TTFA)** and **effective control frequency** as separate headline numbers.
- Report **p99 / worst-case** latency, not just the mean. Robots care about tails.
- CUDA events + explicit sync; warmup runs discarded; clock settings disclosed.
- Every optimization level carries a LIBERO task-success number — including the ones that failed.

## Roadmap

| Phase | Focus |
|---|---|
| **0** | Baseline on the table — env, clones, one honest inference, naive timing |
| **1** | The anatomy — Nsight profiles, latency waterfall, top-5 bottlenecks by evidence |
| **2** | The war — CUDA graphs, kernel fusion, FP8/INT8, token pruning, multi-stream concurrency |
| **3** | The proof — full Pareto sweep, final numbers, polished repo, upstream PR, writeup |

Full plan: **[REALTIME_PI0_PROJECT.md](REALTIME_PI0_PROJECT.md)**.

## Reproducing this

This repo does **not** vendor the upstream models — it pins them. See **[REPRODUCE.md](REPRODUCE.md)**
for exact commit hashes of `openpi` (JAX reference / correctness oracle) and `lerobot` (PyTorch port,
the optimization target) plus environment setup.

## Hardware

Single **NVIDIA RTX 4090** (24GB), CUDA 12.4. The 4090 is the entire lab — validation is in simulation
(LIBERO), no physical robot required.

---

*Not affiliated with Physical Intelligence or Hugging Face. An independent inference-systems study on
frozen, publicly released weights.*
