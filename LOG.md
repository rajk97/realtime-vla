7/9/26
Check	Status
GPU	RTX 4090, 24GB (21.7GB free now) ✓
Driver / CUDA	550.144.03 / CUDA 12.4 ✓
Disk free	92GB free (1.4T used, 94% full) ⚠️ enough for now, but tight — π0 checkpoints + two repos + deps will eat 40–60GB
Python / conda	miniconda 24.11.3, base Python 3.12 ✓
git	2.34.1 ✓

Using vla conda environment

## Stack-decision evidence (Phase 0, §6)
- Repos cloned: openpi (JAX, official) + lerobot (PyTorch port).
- Check 1 — Faithful weight loading? YES. LeRobot ships parity tests
  (tests/policies/pi0_pi05/test_pi0_original_vs_lerobot.py) that run the
  PyTorch port vs original openpi and assert_close on both forward + sample_actions
  (eager and torch.compile). Built-in correctness oracle.
- Check 2 — Flow-matching action expert present? YES. modeling_pi0.py::sample_actions
  has the 10-step denoise loop (x_t = x_t + dt*v_t); separate gemma_expert alongside
  paligemma backbone (PaliGemmaWithExpertModel, "almost exact copy of openpi").
- Why openpi is JAX: built on Google Gemma/PaliGemma (JAX heritage), TPU-first,
  training-oriented (jit/vmap/pmap). Not tuned for single-GPU low-latency inference —
  which is exactly why the PyTorch port exists and is the right base for this project.
- Leaning: optimize LeRobot (PyTorch); keep openpi as correctness oracle. Not yet committed.

## Went public (build-in-public)
- Repo live: https://github.com/rajk97/realtime-vla (public).
- Scaffolding: README.md, REPRODUCE.md (pinned upstream commits), .gitignore (excludes openpi/, lerobot/, checkpoints, profiles).
- Accountability cadence: end every session by appending the current headline number here, then commit + push. 

---

## TODO — 7/14 (Phase 0, ~90 min)
Theme: get LeRobot π0 running end-to-end once and capture the first HONEST naive latency.
Rule: naive-but-correct first. NO torch.compile / CUDA graphs / Nsight / quantization (that's Phase 1).

- [ ] Install LeRobot π0 deps in `vla` env; download π0 checkpoint
- [ ] Record VRAM at bf16 (nvidia-smi) here
- [ ] One successful end-to-end inference (LeRobot PyTorch) — confirm action-chunk shape
- [ ] Naive timing: `time.perf_counter`, 10 runs, drop 2 warmup → record MEAN + WORST here
- [ ] Commit + push

Definition of done: LOG has VRAM@bf16 + "first inference ✓" + naive mean/worst; pushed.

### Results (fill in)
- VRAM @ bf16: 8.09 GB weights (bf16 load). fp32 load 16.13 GB, peak 16.39 GB.
  NOTE: bf16 end-to-end inference errors in the LeRobot port (internal noise/time
  tensors are float32 → dtype mismatch). Naive baseline taken in float32 (native
  checkpoint dtype). bf16 fix deferred to Phase 2.
- First inference: ✅ action chunk shape (1, 50, 32)
- Naive latency (float32, seed 0, CUDA-synced, 10 runs drop 2):
  mean 216.6 ms · p50 216.1 ms · WORST 221.2 ms
- Effective rate: 230.8 Hz (50-action chunk / 216.6 ms). TTFA ≈ 217 ms (the number to beat).
- Config: chunk_size 50, num_inference_steps 10, image 224², state/action dim 32.
- Checkpoint + seed: lerobot/pi0_base @ snapshot 25c379b52ba2…, seed 0.
- Bench script: benchmarks/naive_timing.py

### HEADLINE (7/15): pi0_base naive = 216.6 ms mean / 221.2 ms worst (fp32, 4090). TTFA ~217 ms.

### Env notes (7/14)
- Env: fresh conda `vla` @ Python 3.12 (lerobot 0.6.1 requires >=3.12; old 3.10 env removed).
- lerobot installed editable with `.[pi]` extra.
- Checkpoint: `lerobot/pi0_base`, snapshot `25c379b52ba2…`, 14GB on disk (fp32), 4.028B params.
- torch gotcha: `.[pi]` pulled torch cu130 (CUDA 13) → incompatible with driver 550.144 (CUDA 12.4), fell back to CPU. Fixed: reinstalled `torch==2.11.0+cu126` / `torchvision 0.26.0+cu126` from cu126 index → GPU (RTX 4090) confirmed working.
