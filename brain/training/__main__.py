"""Command-line entry point: ``python -m brain.training train|evaluate``."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .evaluation import evaluate_checkpoint
from .run_config import TrainingRunConfig
from .trainer import train_run


def _parse_override(value: str) -> tuple[str, Any]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("runtime override must be KEY=JSON_VALUE")
    key, raw = value.split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError("runtime override key must not be empty")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = raw
    return key, parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m brain.training")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="train Duo with resumable checkpoints")
    train.add_argument("--config", required=True, help="strict JSON TrainingRunConfig")
    train.add_argument("--resume", help="training-resume checkpoint to continue")
    train.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=JSON_VALUE",
        help="runtime override; repeat as needed, e.g. --set max_steps=25",
    )

    evaluate = commands.add_parser("evaluate", help="evaluate an inference checkpoint on validation or test")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--split", required=True, choices=("validation", "test"))
    evaluate.add_argument("--data", required=True, help="explicit JSONL path for the selected split")
    evaluate.add_argument("--manifest", help="optional dataset manifest to verify and record")
    evaluate.add_argument("--batch-size", type=int, default=16)
    evaluate.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:<index>")
    evaluate.add_argument("--output", help="JSON report path; defaults to a timestamped file beside the checkpoint")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "train":
            configured = TrainingRunConfig.load(args.config)
            overrides: dict[str, Any] = {}
            for item in args.overrides:
                key, value = _parse_override(item)
                if key in overrides:
                    raise ValueError(f"runtime override {key!r} was supplied more than once")
                overrides[key] = value
            effective = configured.with_overrides(overrides)
            result = train_run(
                effective,
                resume_path=args.resume,
                config_file_path=args.config,
                configured_defaults=configured.to_dict(),
                runtime_overrides=overrides,
            )
            print(json.dumps(result, sort_keys=True, allow_nan=False))
            return 0

        checkpoint = Path(args.checkpoint).resolve()
        output = Path(args.output).resolve() if args.output else checkpoint.parent / (
            f"evaluation-{args.split}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')}.json"
        )
        report = evaluate_checkpoint(
            checkpoint,
            split=args.split,
            data_path=args.data,
            batch_size=args.batch_size,
            device=args.device,
            manifest_path=args.manifest,
            output_path=output,
        )
        metrics = report["metrics"]
        print(
            f"MEASURED split={args.split} nll={metrics['nll']:.8f} "
            f"perplexity={metrics['perplexity']:.8f} target_tokens={metrics['target_tokens']} "
            f"checkpoint_sha256={report['checkpoint']['sha256']} report={report['report_path']}"
        )
        return 0
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
