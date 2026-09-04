# Exact reference environment

- Host kernel: `4.18.0-2.4.3.3.kwai.x86_64`
- GPUs: 4 x NVIDIA A800-SXM4-80GB
- NVIDIA driver: `535.129.03`
- CUDA reported by PyTorch: `12.4`
- Python: `3.11`
- LLaMA-Factory commit: `01398eb18dd475a6e27c36f15b970aeacf0d4a60`

The pinned Python packages are in `requirements-lock.txt`. The exact
LLaMA-Factory source used by the experiment is vendored under
`vendor/LLaMA-Factory/src`; the launcher places it first on `PYTHONPATH`.

Byte-identical repetition requires the same GPU model/count, driver, CUDA,
PyTorch, FlashAttention, Transformers, FSDP/NCCL topology, environment
variables, base-model bytes and input-data bytes. A different stack can still
reproduce the procedure but is not promised to reproduce every floating-point
bit.
