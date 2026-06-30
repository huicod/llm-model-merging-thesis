#!/usr/bin/env python
"""Plot average layer-wise L1 parameter deltas against a base model."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM

from mask_rankmean_core import layer_id_from_name


def parse_model_arg(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Model arguments must use name=/path/to/model format.")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("Model name and path must be non-empty.")
    return name, path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--model", action="append", type=parse_model_arg, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def load_model(path: str, dtype: torch.dtype, trust_remote_code: bool):
    return AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
        device_map="cpu",
    )


def layer_l1_average(base_model, tuned_model) -> list[float]:
    base_state = base_model.state_dict()
    tuned_state = tuned_model.state_dict()
    sums: dict[int, float] = {}
    counts: dict[int, int] = {}

    for name, base_tensor in base_state.items():
        layer_id = layer_id_from_name(name)
        if layer_id is None or name not in tuned_state:
            continue
        if not torch.is_floating_point(base_tensor):
            continue
        delta = tuned_state[name].float() - base_tensor.float()
        sums[layer_id] = sums.get(layer_id, 0.0) + delta.abs().mean().item()
        counts[layer_id] = counts.get(layer_id, 0) + 1

    max_layer = max(sums)
    return [sums.get(layer, 0.0) / max(counts.get(layer, 1), 1) for layer in range(max_layer + 1)]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = dtype_from_name(args.dtype)
    base_model = load_model(args.base_model, dtype=dtype, trust_remote_code=args.trust_remote_code)

    series = []
    for name, path in args.model:
        tuned_model = load_model(path, dtype=dtype, trust_remote_code=args.trust_remote_code)
        values = layer_l1_average(base_model, tuned_model)
        series.append((name, values))
        del tuned_model

    csv_path = output_dir / "layer_l1_deltas.csv"
    max_len = max(len(values) for _, values in series)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["layer", *[name for name, _ in series]])
        for layer in range(max_len):
            writer.writerow([layer, *[values[layer] if layer < len(values) else "" for _, values in series]])

    layers = np.arange(max_len)
    width = min(0.8 / max(len(series), 1), 0.25)
    plt.figure(figsize=(14, 5))
    for index, (name, values) in enumerate(series):
        offset = (index - (len(series) - 1) / 2) * width
        plt.bar(layers + offset, values, width=width, label=name)
    plt.xlabel("Transformer layer")
    plt.ylabel("Average L1 parameter delta")
    plt.title("Layer-wise parameter change against base model")
    plt.legend()
    plt.tight_layout()
    figure_path = output_dir / "layer_l1_deltas.png"
    plt.savefig(figure_path, dpi=200)
    print(f"Wrote {csv_path}")
    print(f"Wrote {figure_path}")


if __name__ == "__main__":
    main()
