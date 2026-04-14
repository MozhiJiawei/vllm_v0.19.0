#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import gc
import json
import statistics
import time
from pathlib import Path
from typing import Iterable

import torch
from vllm import LLM, SamplingParams
from vllm.v1.metrics.reader import Counter, Vector

DEFAULT_PROMPTS = [
    "Write a short explanation of speculative decoding for a systems engineer.",
    "Summarize the tradeoffs between latency and throughput in one paragraph.",
    "List three ways to benchmark LLM inference in production.",
    "Explain why acceptance rate matters for EAGLE3 in simple terms.",
    "Give a compact overview of a 1B to 2B Qwen-family model and its likely serving profile.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark a Qwen-family verifier such as Qwen2.5-0.5B-Instruct "
            "or Qwen3-1.7B with a fake or real EAGLE3 draft model using "
            "synthetic rejection sampling."
        )
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help=(
            "Verifier model name or path. Examples: "
            "Qwen/Qwen2.5-0.5B-Instruct, Qwen/Qwen3-1.7B."
        ),
    )
    parser.add_argument(
        "--draft-model",
        required=True,
        help="Speculators-format Eagle3 draft model path.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to store JSONL / CSV results.",
    )
    parser.add_argument(
        "--acceptance-rates",
        type=float,
        nargs="+",
        default=[0.2, 0.4, 0.6, 0.8],
        help="Synthetic average acceptance rates to sweep.",
    )
    parser.add_argument(
        "--num-speculative-tokens",
        type=int,
        nargs="+",
        default=[1, 2, 4, 6],
        help="Speculative depths to sweep.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Verifier tensor parallel size.",
    )
    parser.add_argument(
        "--draft-tensor-parallel-size",
        type=int,
        default=1,
        help="Draft tensor parallel size.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="vLLM gpu_memory_utilization.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="vLLM max_model_len.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum generation length per prompt.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. Use 0.0 for more stable experiments.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Sampling top_p.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Global seed for vLLM.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Pass through vLLM enforce_eager.",
    )
    parser.add_argument(
        "--enable-chunked-prefill",
        action="store_true",
        help="Enable chunked prefill in vLLM.",
    )
    parser.add_argument(
        "--prompts-jsonl",
        type=str,
        default=None,
        help=(
            "Optional JSONL file with one prompt per line. Supported keys: "
            "'prompt', 'text', or 'messages'."
        ),
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=None,
        help="Optional cap on the number of prompts loaded from file/defaults.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of repeated runs per configuration.",
    )
    return parser.parse_args()


def load_prompts(args: argparse.Namespace) -> list[str | list[dict]]:
    prompts: list[str | list[dict]] = []
    if args.prompts_jsonl:
        path = Path(args.prompts_jsonl)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if "messages" in record:
                    prompts.append(record["messages"])
                elif "prompt" in record:
                    prompts.append(record["prompt"])
                elif "text" in record:
                    prompts.append(record["text"])
                else:
                    raise ValueError(
                        f"Unsupported prompt JSONL record keys: {sorted(record.keys())}"
                    )
    else:
        prompts = DEFAULT_PROMPTS.copy()

    if args.num_prompts is not None:
        prompts = prompts[: args.num_prompts]
    if not prompts:
        raise ValueError("No prompts available for benchmarking.")

    return prompts


def collect_spec_metrics(metrics, num_spec_tokens: int) -> dict:
    result = {
        "num_drafts": 0,
        "num_draft_tokens": 0,
        "num_accepted_tokens": 0,
        "acceptance_per_pos": [0] * num_spec_tokens,
    }
    for metric in metrics:
        if metric.name == "vllm:spec_decode_num_drafts":
            assert isinstance(metric, Counter)
            result["num_drafts"] += int(metric.value)
        elif metric.name == "vllm:spec_decode_num_draft_tokens":
            assert isinstance(metric, Counter)
            result["num_draft_tokens"] += int(metric.value)
        elif metric.name == "vllm:spec_decode_num_accepted_tokens":
            assert isinstance(metric, Counter)
            result["num_accepted_tokens"] += int(metric.value)
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            assert isinstance(metric, Vector)
            for idx, value in enumerate(metric.values[:num_spec_tokens]):
                result["acceptance_per_pos"][idx] += int(value)

    num_drafts = result["num_drafts"]
    num_draft_tokens = result["num_draft_tokens"]
    num_accepted = result["num_accepted_tokens"]
    result["mean_acceptance_length"] = (
        1.0 + (num_accepted / num_drafts) if num_drafts else 1.0
    )
    result["observed_acceptance_rate"] = (
        num_accepted / num_draft_tokens if num_draft_tokens else 0.0
    )
    result["observed_acceptance_per_pos"] = [
        (value / num_drafts if num_drafts else 0.0)
        for value in result["acceptance_per_pos"]
    ]
    return result


def output_token_count(outputs) -> int:
    return sum(len(output.outputs[0].token_ids) for output in outputs)


def is_chat_prompt(prompt: str | list[dict]) -> bool:
    return isinstance(prompt, list)


def run_inference(
    llm: LLM,
    prompts: list[str | list[dict]],
    sampling_params: SamplingParams,
):
    has_chat = any(is_chat_prompt(prompt) for prompt in prompts)
    has_text = any(not is_chat_prompt(prompt) for prompt in prompts)
    if has_chat and has_text:
        raise ValueError(
            "Do not mix raw text prompts and chat messages in the same benchmark run."
        )
    if has_chat:
        return llm.chat(prompts, sampling_params=sampling_params)
    return llm.generate(prompts, sampling_params=sampling_params)


def build_sampling_params(args: argparse.Namespace) -> SamplingParams:
    return SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )


def build_speculative_config(
    args: argparse.Namespace,
    acceptance_rate: float,
    num_spec_tokens: int,
) -> dict:
    return {
        "method": "eagle3",
        "model": args.draft_model,
        "draft_tensor_parallel_size": args.draft_tensor_parallel_size,
        "num_speculative_tokens": num_spec_tokens,
        "rejection_sample_method": "synthetic",
        "synthetic_acceptance_rate": acceptance_rate,
    }


def instantiate_llm(
    args: argparse.Namespace,
    speculative_config: dict | None,
) -> LLM:
    return LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        speculative_config=speculative_config,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
        enable_chunked_prefill=args.enable_chunked_prefill,
        disable_log_stats=False,
        seed=args.seed,
    )


def release_llm(llm: LLM | None) -> None:
    if llm is None:
        return
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_once(
    args: argparse.Namespace,
    prompts: list[str | list[dict]],
    speculative_config: dict | None,
    label: str,
) -> dict:
    llm: LLM | None = None
    started = time.perf_counter()
    try:
        llm = instantiate_llm(args, speculative_config)
        outputs = run_inference(llm, prompts, build_sampling_params(args))
        elapsed = time.perf_counter() - started
        total_output_tokens = output_token_count(outputs)
        metrics = llm.get_metrics()
        spec_metrics = (
            collect_spec_metrics(
                metrics,
                speculative_config["num_speculative_tokens"],
            )
            if speculative_config is not None
            else {}
        )
        return {
            "label": label,
            "elapsed_seconds": elapsed,
            "num_prompts": len(prompts),
            "total_output_tokens": total_output_tokens,
            "throughput_toks_per_s": (
                total_output_tokens / elapsed if elapsed > 0 else 0.0
            ),
            "tokens_per_prompt": (
                total_output_tokens / len(prompts) if prompts else 0.0
            ),
            "speculative": speculative_config is not None,
            "speculative_config": speculative_config,
            **spec_metrics,
        }
    finally:
        release_llm(llm)


def run_repeated(
    args: argparse.Namespace,
    prompts: list[str | list[dict]],
    speculative_config: dict | None,
    label: str,
) -> dict:
    runs = []
    for repeat_idx in range(args.repeats):
        run_label = f"{label}-repeat-{repeat_idx + 1}"
        runs.append(run_once(args, prompts, speculative_config, run_label))

    elapsed_values = [run["elapsed_seconds"] for run in runs]
    throughput_values = [run["throughput_toks_per_s"] for run in runs]
    observed_acceptance_values = [
        run.get("observed_acceptance_rate", 0.0) for run in runs
    ]
    mean_acceptance_length_values = [
        run.get("mean_acceptance_length", 1.0) for run in runs
    ]

    summary = dict(runs[-1])
    summary["label"] = label
    summary["repeat_runs"] = runs
    summary["elapsed_seconds_mean"] = statistics.mean(elapsed_values)
    summary["elapsed_seconds_min"] = min(elapsed_values)
    summary["elapsed_seconds_max"] = max(elapsed_values)
    summary["throughput_toks_per_s_mean"] = statistics.mean(throughput_values)
    summary["throughput_toks_per_s_min"] = min(throughput_values)
    summary["throughput_toks_per_s_max"] = max(throughput_values)
    summary["observed_acceptance_rate_mean"] = statistics.mean(
        observed_acceptance_values
    )
    summary["mean_acceptance_length_mean"] = statistics.mean(
        mean_acceptance_length_values
    )
    return summary


def compute_speedup(baseline: dict, candidate: dict) -> float | None:
    baseline_elapsed = baseline.get("elapsed_seconds_mean", baseline["elapsed_seconds"])
    candidate_elapsed = candidate.get(
        "elapsed_seconds_mean", candidate["elapsed_seconds"]
    )
    if baseline_elapsed <= 0 or candidate_elapsed <= 0:
        return None
    return baseline_elapsed / candidate_elapsed


def save_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def save_csv(path: Path, rows: list[dict]) -> None:
    flat_rows = []
    for row in rows:
        flat_rows.append(
            {
                "label": row["label"],
                "speculative": row["speculative"],
                "target_acceptance_rate": (
                    row.get("speculative_config", {}) or {}
                ).get("synthetic_acceptance_rate"),
                "num_speculative_tokens": (
                    row.get("speculative_config", {}) or {}
                ).get("num_speculative_tokens"),
                "elapsed_seconds_mean": row.get(
                    "elapsed_seconds_mean", row["elapsed_seconds"]
                ),
                "throughput_toks_per_s_mean": row.get(
                    "throughput_toks_per_s_mean", row["throughput_toks_per_s"]
                ),
                "observed_acceptance_rate_mean": row.get(
                    "observed_acceptance_rate_mean",
                    row.get("observed_acceptance_rate"),
                ),
                "mean_acceptance_length_mean": row.get(
                    "mean_acceptance_length_mean",
                    row.get("mean_acceptance_length"),
                ),
                "speedup_vs_baseline": row.get("speedup_vs_baseline"),
            }
        )

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(args)

    baseline = run_repeated(
        args=args,
        prompts=prompts,
        speculative_config=None,
        label="baseline",
    )

    rows = [baseline]
    for num_spec_tokens in args.num_speculative_tokens:
        for acceptance_rate in args.acceptance_rates:
            config = build_speculative_config(args, acceptance_rate, num_spec_tokens)
            label = f"synthetic-r{acceptance_rate:.3f}-k{num_spec_tokens}"
            result = run_repeated(
                args=args,
                prompts=prompts,
                speculative_config=config,
                label=label,
            )
            result["speedup_vs_baseline"] = compute_speedup(baseline, result)
            rows.append(result)

    run_manifest = {
        "model": args.model,
        "draft_model": args.draft_model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "draft_tensor_parallel_size": args.draft_tensor_parallel_size,
        "acceptance_rates": args.acceptance_rates,
        "num_speculative_tokens": args.num_speculative_tokens,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repeats": args.repeats,
        "num_prompts": len(prompts),
        "prompt_source": args.prompts_jsonl or "built_in_defaults",
    }

    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8"
    )
    save_jsonl(output_dir / "results.jsonl", rows)
    save_csv(output_dir / "results.csv", rows)

    print(f"Saved benchmark results to: {output_dir}")
    print(f"Baseline throughput: {baseline.get('throughput_toks_per_s_mean'):.4f} tok/s")
    for row in rows[1:]:
        print(
            f"{row['label']}: speedup_vs_baseline={row.get('speedup_vs_baseline')}, "
            f"observed_acceptance_rate_mean={row.get('observed_acceptance_rate_mean'):.4f}, "
            f"mean_acceptance_length_mean={row.get('mean_acceptance_length_mean'):.4f}"
        )


if __name__ == "__main__":
    main()
