#!/usr/bin/env python
"""Convert a Parquet dataset shard into JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input Parquet file.")
    parser.add_argument("--output", required=True, help="Output JSONL file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data_frame = pd.read_parquet(args.input)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in data_frame.to_dict(orient="records"):
            json.dump(record, handle, ensure_ascii=False)
            handle.write("\n")

    print(f"Wrote {len(data_frame)} records to {output_path}")


if __name__ == "__main__":
    main()
