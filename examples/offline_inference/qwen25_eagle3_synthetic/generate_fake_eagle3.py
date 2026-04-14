#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoConfig, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a minimal Speculators-format EAGLE3 checkpoint for a "
            "Qwen-family verifier such as Qwen2.5-0.5B-Instruct or "
            "Qwen3-1.7B without training."
        )
    )
    parser.add_argument(
        "--verifier",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help=(
            "Verifier model name or path. Examples: "
            "Qwen/Qwen2.5-0.5B-Instruct, Qwen/Qwen3-1.7B."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write the fake EAGLE3 checkpoint into.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=1,
        help="Number of EAGLE3 decoder layers to keep in the draft model.",
    )
    parser.add_argument(
        "--draft-vocab-size",
        type=int,
        default=None,
        help=(
            "Draft vocabulary size. Defaults to verifier vocab size to keep "
            "token mapping simple and robust."
        ),
    )
    parser.add_argument(
        "--ttt-steps",
        type=int,
        default=4,
        help="Default speculative depth embedded into the generated config.",
    )
    parser.add_argument(
        "--aux-layer-ids",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Auxiliary hidden-state layer ids to embed in the config. "
            "Defaults to the last verifier layer."
        ),
    )
    parser.add_argument(
        "--init-strategy",
        choices=["gaussian", "zeros", "xavier_uniform"],
        default="gaussian",
        help="How to initialize trainable draft-only parameters.",
    )
    parser.add_argument(
        "--init-std",
        type=float,
        default=0.02,
        help="Standard deviation for gaussian initialization.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="Checkpoint dtype for stored tensors.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed used for initialization.",
    )
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def infer_aux_layer_ids(config, provided: list[int] | None) -> list[int]:
    if provided:
        return provided

    num_layers = getattr(config, "num_hidden_layers", None)
    if not isinstance(num_layers, int) or num_layers <= 0:
        return [0]
    return [max(0, num_layers - 1)]


def build_vocab_mappings(
    target_vocab_size: int,
    draft_vocab_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if draft_vocab_size <= 0:
        raise ValueError("draft_vocab_size must be positive.")
    if draft_vocab_size > target_vocab_size:
        raise ValueError(
            "draft_vocab_size cannot exceed target_vocab_size for this helper script."
        )

    # Speculators expects:
    # - t2d: a boolean mask over the target vocab indicating which target
    #   tokens are present in the draft vocab.
    # - d2t: a dense mapping from draft vocab index -> target vocab index.
    #
    # For this helper we expose the first `draft_vocab_size` target tokens as
    # the draft vocabulary. In the full-vocab case this becomes an all-True
    # mask plus an identity d2t mapping.
    visible = torch.arange(draft_vocab_size, dtype=torch.long)
    t2d = torch.zeros(target_vocab_size, dtype=torch.bool)
    t2d[visible] = True
    d2t = visible.clone()
    return t2d, d2t


def initialize_parameter(
    tensor: torch.Tensor,
    strategy: str,
    std: float,
) -> None:
    if strategy == "zeros":
        torch.nn.init.zeros_(tensor)
    elif strategy == "xavier_uniform":
        if tensor.ndim < 2:
            bound = 1.0 / math.sqrt(max(1, tensor.numel()))
            torch.nn.init.uniform_(tensor, -bound, bound)
        else:
            torch.nn.init.xavier_uniform_(tensor)
    elif strategy == "gaussian":
        torch.nn.init.normal_(tensor, mean=0.0, std=std)
    else:
        raise ValueError(f"Unsupported init strategy: {strategy}")


def maybe_import_speculators():
    try:
        from speculators.models.eagle3 import Eagle3DraftModel
    except ImportError as exc:  # pragma: no cover - helper script
        raise SystemExit(
            "The 'speculators' package is required to generate an EAGLE3 "
            "checkpoint. Install it in the environment used to run this script."
        ) from exc
    return Eagle3DraftModel


def trainable_parameter_names(model: torch.nn.Module) -> set[str]:
    names: set[str] = set()
    for name, param in model.named_parameters():
        if param.requires_grad:
            names.add(name)
    return names


def cast_state_dict(
    state_dict: dict[str, torch.Tensor], dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    converted: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if torch.is_floating_point(value):
            converted[key] = value.detach().cpu().to(dtype)
        else:
            converted[key] = value.detach().cpu()
    return converted


def save_metadata(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    Eagle3DraftModel = maybe_import_speculators()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    verifier_config = AutoConfig.from_pretrained(args.verifier, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.verifier, trust_remote_code=True)

    verifier_vocab_size = int(getattr(verifier_config, "vocab_size"))
    draft_vocab_size = args.draft_vocab_size or verifier_vocab_size
    aux_layer_ids = infer_aux_layer_ids(verifier_config, args.aux_layer_ids)
    t2d, d2t = build_vocab_mappings(verifier_vocab_size, draft_vocab_size)

    model = Eagle3DraftModel.from_training_args(
        verifier_config=verifier_config,
        verifier_name_or_path=args.verifier,
        num_layers=args.num_layers,
        draft_vocab_size=draft_vocab_size,
        norm_before_residual=True,
        norm_before_fc=False,
        embed_requires_grad=False,
        ttt_steps=args.ttt_steps,
        t2d=t2d,
        d2t=d2t,
        eagle_aux_hidden_state_layer_ids=aux_layer_ids,
    )

    trainable_names = trainable_parameter_names(model)
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        initialize_parameter(param.data, args.init_strategy, args.init_std)

    config = model.config.to_dict()
    config["architectures"] = ["Eagle3LlamaForCausalLM"]
    config["draft_vocab_size"] = draft_vocab_size
    config["eagle_aux_hidden_state_layer_ids"] = aux_layer_ids
    config["transformer_layer_config"] = verifier_config.to_dict()
    config["speculators_model_type"] = "eagle3"
    config["speculators_config"] = {
        "algorithm": "eagle3",
        "default_proposal_method": "greedy",
        "proposal_methods": [
            {
                "name": "greedy",
                "speculative_tokens": args.ttt_steps,
            }
        ],
        "verifier": {
            "name_or_path": args.verifier,
        },
    }

    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )

    state_dict = cast_state_dict(model.state_dict(), resolve_dtype(args.dtype))
    save_file(state_dict, str(output_dir / "model.safetensors"))
    tokenizer.save_pretrained(output_dir)

    save_metadata(
        output_dir / "fake_eagle3_metadata.json",
        {
            "verifier": args.verifier,
            "num_layers": args.num_layers,
            "draft_vocab_size": draft_vocab_size,
            "full_vocab_identity_mapping": draft_vocab_size == verifier_vocab_size,
            "ttt_steps": args.ttt_steps,
            "aux_layer_ids": aux_layer_ids,
            "init_strategy": args.init_strategy,
            "init_std": args.init_std,
            "dtype": args.dtype,
            "seed": args.seed,
            "trainable_parameter_count": sum(
                model.get_parameter(name).numel() for name in trainable_names
            ),
            "trainable_parameters": sorted(trainable_names),
            "notes": [
                "This checkpoint is intentionally untrained.",
                "Use it with synthetic rejection sampling for systems experiments.",
                "The verifier tokenizer/config come from the source model.",
            ],
        },
    )

    print(f"Wrote fake EAGLE3 checkpoint to: {output_dir}")
    print(f"Verifier: {args.verifier}")
    print(f"Draft layers: {args.num_layers}")
    print(f"Draft vocab size: {draft_vocab_size}")
    print(f"Aux layer ids: {aux_layer_ids}")


if __name__ == "__main__":
    main()
