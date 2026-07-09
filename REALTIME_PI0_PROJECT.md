# Real-Time π0
### The full latency anatomy of a VLA, and what it takes to hit 30Hz+ on one consumer GPU

**Owner:** Raj
**Hardware:** RTX 4090 (24GB), local
**Duration:** ~16 weeks, serious part-time
**Status:** Day 0 — baseline setup

---

## 1. One-paragraph summary

Take an open vision-language-action (VLA) model — π0 (Physical Intelligence) — and (a) profile its inference latency honestly and completely, (b) optimize it aggressively at the systems/kernel level, and (c) prove the optimized policy still works, via task-success validation in simulation. Publish the whole thing: reproducible repo, rigorous writeup with latency waterfall and speed-vs-success Pareto charts, and at least one upstreamed PR. This is a credibility-building artifact for positioning in physical-AI / real-time multimodal inference.

## 2. Why this project (strategic rationale — do not relitigate mid-project)

- Text LLM serving is commoditized (vLLM/SGLang); VLA real-time inference is embryonic — the runtime ecosystem (vla.cpp, TensorRT-Edge-LLM, VLA-Perf-style benchmarks) appeared only in the last ~year. Early window.
- Robotics/physical AI raised record capital in 2025–2026 and is talent-constrained. Engineers are paid from funding, not robot revenue (AV precedent: engineers were paid top-of-market through a decade-long pre-commercial period).
- The skillset is a strict superset of text LLM inference: quantization, KV cache, prefill optimization, kernel fusion, CUDA graphs — PLUS heterogeneous pipelines, hard real-time deadlines, and downstream-task correctness validation. Downside case ≈ "unusually strong multimodal inference portfolio," which still wins text-inference interviews.
- Maybe a dozen rigorous public VLA latency studies exist. Early enough to be cited and findable.

## 3. Why π0 is the right target

π0 = three engines with different performance characters bolted together, sharing one deadline:

| Component | What it does | Bottleneck character |
|---|---|---|
| SigLIP vision encoder (ViT) | 16×16 patches → ~256 tokens per camera | Compute-bound |
| Gemma 2B backbone | Prefill over vision tokens + instruction + joint state | Compute + memory bandwidth (classic LLM prefill) |
| Flow-matching action expert | Iteratively refines an action chunk, ~10 denoising steps, cross-attends to backbone output | Latency-critical inner loop |

Robot control needs 30–100Hz with **bounded** latency. Production pattern in the field: slow planner (5–10Hz) + fast action expert (50–100Hz), asynchronous — i.e., a real-time concurrent-systems problem, not a config-tuning problem.

## 4. Success criteria (definition of done)

1. **≥3×** end-to-end speedup over naive PyTorch baseline on the 4090
2. Sustained **≥30Hz effective control rate** (account for action chunking: effective rate = chunk_size / inference_latency, but report time-to-first-action separately)
3. **≤2% absolute** task-success degradation on LIBERO benchmark vs. baseline policy
4. Public deliverables: reproducible repo + arXiv-quality writeup (latency waterfall, Hz-vs-success Pareto curve) + ≥1 accepted upstream PR (openpi, LeRobot, or vla.cpp ecosystem)

## 5. Benchmark methodology (non-negotiable rigor rules)

- Report **time-to-first-action (TTFA)** and **effective control frequency** as separate headline numbers. Never report only average throughput.
- Report **worst-case / p99** latency, not just mean. Robots care about tails.
- All timing with CUDA events + explicit synchronization (after Day 0's naive perf_counter baseline). Warmup runs discarded. Fixed clocks noted (`nvidia-smi -lgc` if used — disclose it).
- Every optimization level gets a LIBERO task-success number. Speed without correctness is fraud in this domain. Publish the full Pareto curve, including the points that failed.
- Everything reproducible: pinned deps, seeds, exact checkpoint hashes, one-command benchmark script.

## 6. Phases and task breakdown

### Phase 0 — Baseline on the table (Week 0) ← CURRENT
- [ ] Env verified: driver/CUDA versions logged, ~50GB disk free
- [ ] Clone `Physical-Intelligence/openpi` AND `huggingface/lerobot` (PyTorch π0 port)
- [ ] π0 base checkpoint downloaded; VRAM footprint at bf16 recorded
- [ ] One successful end-to-end inference
- [ ] Naive timing: 10 runs, drop 2 warmup, record mean + worst in LOG.md
- [ ] **Stack decision:** openpi (JAX, official) vs LeRobot (PyTorch port). Criteria: does the PyTorch port load official weights faithfully and include the flow-matching expert? Prior: PyTorch for CUDA kernel + Nsight workflow — but verify fidelity first. Decide with evidence, then commit.

### Phase 1 — The anatomy (Weeks 1–3)
- [ ] Nsight Systems + Nsight Compute profiles of the full pipeline
- [ ] Latency waterfall: vision encode / prefill / each flow step / H2D-D2H transfers / Python & launch overhead — in ms and % of budget
- [ ] TTFA vs. throughput decomposition
- [ ] Identify top 5 bottlenecks by profiler evidence (not by intuition)
- [ ] LIBERO harness running; baseline task-success recorded
- [ ] Deliverable: "Anatomy" writeup section — publishable on its own

### Phase 2 — The war (Weeks 4–12), in profiler-evidence order
- [ ] CUDA graphs: kill launch overhead across the whole pipeline (esp. the 10-step flow loop)
- [ ] Kernel fusion where the vision encoder bleeds (custom CUDA where torch.compile falls short)
- [ ] Quantization: FP8 / INT8 (Ada has FP8 support) — per-component, with success-rate check at each step
- [ ] Visual token pruning (live research direction — engineer against the literature, e.g. LightVLA-style)
- [ ] **The signature move — concurrency:** multi-stream pipelining. Encode frame t+1 on stream A while the action expert denoises for frame t on stream B; exploit action-chunk open-loop execution to hide latency. Async System1/System2 split if time allows. This is where HPC/concurrent-programming skill separates this work from everyone else's.
- [ ] Stretch (only if ahead of schedule): speculative/early-exit ideas for the flow loop; multimodal spec decoding is underexplored
- Expect to be stuck around week 6. That is the plan working, not failing.

### Phase 3 — The proof + shipping (Weeks 12–16)
- [ ] Full Pareto sweep: every optimization level × LIBERO success rate
- [ ] Final headline numbers vs. success criteria
- [ ] Repo polished: one-command repro, README, pinned env
- [ ] Writeup: arXiv-style; charts: waterfall, Pareto, TTFA-vs-Hz
- [ ] ≥1 PR upstreamed (openpi / LeRobot / vla.cpp ecosystem) — name in commit history of tools the field will standardize on
- [ ] Distribution: arXiv or long-form blog + repo link; consider workshop submission

## 7. Stack & tools

- **Model:** π0 base checkpoint (openpi); fallback: OpenVLA-OFT
- **Framework:** decide Day 1 — LeRobot PyTorch port (preferred for kernel work) vs openpi JAX (official reference; keep as correctness oracle regardless)
- **Profiling:** Nsight Systems, Nsight Compute, torch.profiler, CUDA events
- **Optimization:** torch.compile, CUDA graphs, custom CUDA kernels, TensorRT where it earns its keep, FP8/INT8
- **Validation:** LIBERO (simulation — no robot required; the 4090 is the entire lab)
- **Logging:** `LOG.md` in repo root — every session: date, what moved, what broke, the number. Raw material for the writeup.

## 8. Risks and pre-committed responses

| Risk | Response |
|---|---|
| LeRobot π0 port is unfaithful / incomplete | Use openpi JAX as correctness oracle; optimize the PyTorch port and validate outputs against reference; worst case switch to OpenVLA-OFT |
| Quantization tanks task success | That's a *finding*, not a failure — the Pareto curve including failures is the paper |
| Stuck >2 weeks on one bottleneck | Timebox: ship the partial result, move to next bottleneck, note it as future work |
| Someone publishes similar work mid-project | Differentiate on rigor (TTFA/p99/Pareto) and the concurrency angle; cite them, don't compete on their axis |
| Scope creep toward training/finetuning | Out of scope. This is an inference systems project. Frozen weights only. |

## 9. Working agreements (for Claude Code sessions)

- Profiler evidence before optimization. No speculative optimization.
- Every speed change is paired with a correctness check before it's called a win.
- Naive-but-correct first, fast second. Keep the reference implementation runnable at all times.
- Update LOG.md at the end of every session with the current headline number.
- The deliverable is public artifacts, not private cleverness. Bias toward writing things down.
