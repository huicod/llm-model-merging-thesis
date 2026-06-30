#!/usr/bin/env python
"""Create a bounded MathInstruct JSON subset for SFT experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to MathInstruct.json.")
    parser.add_argument("--output", required=True, help="Path to the subset JSON file.")
    parser.add_argument("--limit", type=int, default=22500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    subset = data[: args.limit]
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(subset, handle, indent=2, ensure_ascii=False)

    print(f"Wrote {len(subset)} examples to {output_path}")


if __name__ == "__main__":
    main()
