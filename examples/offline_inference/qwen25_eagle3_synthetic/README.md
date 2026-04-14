# Qwen Eagle3 Synthetic Acceptance Experiments

This directory contains two helper scripts for engineering-side speculative
decoding experiments on Qwen-family verifier models such as:

- `Qwen/Qwen2.5-0.5B-Instruct`
- `Qwen/Qwen3-1.7B`

- `generate_fake_eagle3.py`
  Creates a minimal Speculators-format EAGLE3 draft checkpoint that pairs with
  the verifier model but does not require any training.
- `benchmark_synthetic_eagle3.py`
  Runs offline speculative decoding sweeps with
  `rejection_sample_method="synthetic"` so acceptance becomes a controlled
  experiment variable instead of a learned model property.

## Why this setup exists

For a small or mid-sized verifier like Qwen2.5-0.5B or Qwen3-1.7B, it is often
unclear whether EAGLE3 is worth training at all. This workflow lets you:

1. Build a structurally valid EAGLE3 checkpoint with near-zero preparation.
2. Keep the real drafter + verifier + scheduler execution path in vLLM.
3. Sweep synthetic acceptance rates such as `0.2, 0.4, 0.6, 0.8`.
4. Measure latency / throughput / observed acceptance before investing in
   training a real speculator.

## Environment expectations

These scripts are intentionally not executed by CI here. They are meant to be
run on a machine that has:

- a working CUDA / ROCm environment for vLLM
- `uv` and a project environment for vLLM
- `transformers`, `torch`, `safetensors`
- the `speculators` Python package available

The generator script will fail fast with a clear message if `speculators` is
missing.

## Typical flow

Generate a fake EAGLE3 checkpoint:

```bash
uv run python examples/offline_inference/qwen25_eagle3_synthetic/generate_fake_eagle3.py \
  --output-dir /path/to/qwen25-05b-fake-eagle3 \
  --verifier Qwen/Qwen2.5-0.5B-Instruct \
  --num-layers 1 \
  --init-strategy gaussian
```

Run a speculative sweep:

```bash
uv run python examples/offline_inference/qwen25_eagle3_synthetic/benchmark_synthetic_eagle3.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --draft-model /path/to/qwen25-05b-fake-eagle3 \
  --acceptance-rates 0.2 0.4 0.6 0.8 \
  --num-speculative-tokens 1 2 4 6 \
  --output-dir /path/to/results
```

Generate a fake EAGLE3 checkpoint for Qwen3-1.7B:

```bash
uv run python examples/offline_inference/qwen25_eagle3_synthetic/generate_fake_eagle3.py \
  --output-dir /path/to/qwen3-17b-fake-eagle3 \
  --verifier Qwen/Qwen3-1.7B \
  --num-layers 1 \
  --init-strategy gaussian
```

Run a speculative sweep on Qwen3-1.7B:

```bash
uv run python examples/offline_inference/qwen25_eagle3_synthetic/benchmark_synthetic_eagle3.py \
  --model Qwen/Qwen3-1.7B \
  --draft-model /path/to/qwen3-17b-fake-eagle3 \
  --acceptance-rates 0.2 0.4 0.6 0.8 \
  --num-speculative-tokens 1 2 4 6 \
  --output-dir /path/to/results-qwen3
```

## Notes

- The generated draft checkpoint is intended for systems experiments, not
  quality evaluation.
- The benchmark script defaults to synthetic rejection sampling so the target
  acceptance rate is controlled by config rather than draft quality.
- Keeping the draft vocabulary equal to the verifier vocabulary simplifies the
  checkpoint and avoids unnecessary tokenizer mapping issues.
