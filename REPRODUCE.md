# Reproducing Real-Time π0

The upstream models are **pinned, not vendored**. Clone them at the exact commits below.

## Hardware / driver baseline (Day 0)

- GPU: NVIDIA RTX 4090 (24GB)
- Driver: 550.144.03
- CUDA toolkit: 12.4
- OS: Linux

## Pinned upstream commits

| Repo | Role | Commit |
|---|---|---|
| [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | JAX reference — official weights + correctness oracle | `15a9616a00943ada6c20a0f158e3adb39df2ccac` |
| [huggingface/lerobot](https://github.com/huggingface/lerobot) | PyTorch port — the optimization target | `e40b58a8dfa9e7b86918c374791599d070518d11` |

```bash
git clone https://github.com/Physical-Intelligence/openpi.git
git -C openpi checkout 15a9616a00943ada6c20a0f158e3adb39df2ccac

git clone https://github.com/huggingface/lerobot.git
git -C lerobot checkout e40b58a8dfa9e7b86918c374791599d070518d11
```

## Environment

```bash
conda create -n vla python=3.10 -y
conda activate vla
# install steps TBD — pinned in this file as they are verified
```

## Fidelity check (why the PyTorch port is trusted)

LeRobot ships parity tests that assert its PyTorch π0 matches the original openpi model
numerically (forward pass and action sampling, eager and torch.compile):

```
lerobot/tests/policies/pi0_pi05/test_pi0_original_vs_lerobot.py
```

These serve as the built-in correctness oracle for all optimization work in this repo.
