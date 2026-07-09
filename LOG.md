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