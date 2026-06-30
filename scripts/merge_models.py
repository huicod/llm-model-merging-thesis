#!/usr/bin/env python
"""Merge task-specialized causal language models with MaskRankMean."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mask_rankmean_core import MergeConfig, merge_state_dicts


DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True, help="Path or Hugging Face id of the base model.")
    parser.add_argument("--models", nargs="+", required=True, help="Fine-tuned model directories to merge.")
    parser.add_argument("--output-dir", required=True, help="Directory where the merged model is saved.")
    parser.add_argument("--base-mask-rate", type=float, default=0.9)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--mask-strategy", choices=["magnitude", "random"], default="magnitude")
    parser.add_argument("--no-rescale", action="store_true", help="Disable 1/(1-mask_rate) rescaling.")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def load_model(path: str, dtype: torch.dtype, trust_remote_code: bool):
    return AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
        device_map="cpu",
    )


def main() -> None:
    args = parse_args()
    dtype = DTYPES[args.dtype]

    base_model = load_model(args.base_model, dtype=dtype, trust_remote_code=args.trust_remote_code)
    tuned_models = [load_model(path, dtype=dtype, trust_remote_code=args.trust_remote_code) for path in args.models]

    config = MergeConfig(
        base_mask_rate=args.base_mask_rate,
        alpha=args.alpha,
        mask_strategy=args.mask_strategy,
        use_rescale=not args.no_rescale,
    )
    merged_state = merge_state_dicts(
        base_state=base_model.state_dict(),
        tuned_states=[model.state_dict() for model in tuned_models],
        config=config,
    )

    missing, unexpected = base_model.load_state_dict(merged_state, strict=False)
    if missing:
        print(f"Missing keys while loading merged state: {missing}")
    if unexpected:
        print(f"Unexpected keys while loading merged state: {unexpected}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_model.save_pretrained(output_dir, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(args.models[0], trust_remote_code=args.trust_remote_code)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved merged model to {output_dir}")


if __name__ == "__main__":
    main()
