"""Command-line interface: python -m brain.data.build."""
from __future__ import annotations

import argparse
import json

from brain.config import BrainConfig
from brain.data.pipeline import DatasetConfig, DatasetError, build_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build reproducible D-DAMS text dataset splits from a local file.")
    parser.add_argument("--input", required=True, help="Local source file; raw inputs should live under datasets/raw/")
    parser.add_argument("--output", required=True, help="Output directory; use datasets/processed/<dataset-version>/")
    parser.add_argument("--config", required=True, help="Dataset processing config JSON")
    parser.add_argument("--brain-config", default="configs/duo_v0.json", help="Model config for tokenizer compatibility")
    args = parser.parse_args()
    try:
        dataset_config = DatasetConfig.load(args.config)
        brain_config = BrainConfig.load(args.brain_config)
        result = build_dataset(args.input, args.output, dataset_config, brain_config=brain_config)
    except (DatasetError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({"manifest": result["manifest"], "stats": result["stats"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
