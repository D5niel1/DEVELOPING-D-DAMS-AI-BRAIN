"""Deterministic, read-only evaluation for a saved Duo inference checkpoint."""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch

from brain.data.language_model import LanguageModelBatch, build_context_windows, load_jsonl_text, make_lm_batches
from brain.model import DuoTransformer, load_checkpoint, masked_next_token_loss
from brain.tokenizer import ByteTokenizer
from .snapshots import snapshot_file


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_device(value: str) -> torch.device:
    if value == "cpu":
        return torch.device("cpu")
    if value == "cuda" or value.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise ValueError(f"requested device {value!r}, but CUDA is unavailable")
        device = torch.device(value)
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f"requested device {value!r}, but only {torch.cuda.device_count()} CUDA device(s) are available")
        return device
    raise ValueError("device must be 'cpu', 'cuda', or 'cuda:<index>'")


def evaluate_model(
    model: DuoTransformer,
    batches: Sequence[LanguageModelBatch],
    *,
    device: str | torch.device | None = None,
) -> dict[str, float | int]:
    """Measure token-weighted NLL/PPL without gradients or parameter updates."""
    if not batches:
        raise ValueError("evaluation requires at least one batch")
    target_device = torch.device(device) if device is not None else next(model.parameters()).device
    model_device = next(model.parameters()).device
    if target_device.type == "cuda" and target_device.index is None and model_device.type == "cuda":
        target_device = model_device
    if target_device != model_device:
        raise ValueError(
            f"model parameters are on {model_device}, but evaluation device is {target_device}; "
            "move the model explicitly before evaluating"
        )
    was_training = model.training
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_cudnn = None
    previous_tf32 = None
    if target_device.type == "cuda":
        previous_cudnn = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    total_nll = 0.0
    target_tokens = 0
    try:
        torch.use_deterministic_algorithms(True)
        if target_device.type == "cuda":
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = False
        model.eval()
        with torch.inference_mode():
            for batch in batches:
                input_ids = batch.input_ids.to(target_device)
                targets = batch.targets.to(target_device)
                attention_mask = batch.attention_mask.to(target_device)
                target_mask = batch.target_mask.to(target_device)
                logits = model(input_ids, attention_mask)
                loss = masked_next_token_loss(logits, targets, target_mask)
                valid = target_mask & targets.ne(0)
                count = int(valid.sum().item())
                if count == 0:
                    continue
                nll = float(loss.item())
                if not math.isfinite(nll):
                    raise ValueError("evaluation produced a non-finite loss")
                total_nll += nll * count
                target_tokens += count
    finally:
        model.train(was_training)
        torch.use_deterministic_algorithms(previous_deterministic)
        if previous_cudnn is not None:
            torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = previous_cudnn
            torch.backends.cuda.matmul.allow_tf32 = bool(previous_tf32)
    if target_tokens == 0:
        raise ValueError("evaluation split contains no non-PAD next-token targets")
    mean_nll = total_nll / target_tokens
    try:
        perplexity = math.exp(mean_nll)
    except OverflowError as exc:
        raise ValueError("evaluation perplexity overflowed") from exc
    if not math.isfinite(perplexity):
        raise ValueError("evaluation produced non-finite perplexity")
    return {"nll": mean_nll, "perplexity": perplexity, "target_tokens": target_tokens}


def _manifest_provenance(data_path: Path, split: str, explicit_manifest: str | Path | None, data_hash: str, records: int) -> dict[str, Any]:
    candidate = Path(explicit_manifest) if explicit_manifest is not None else data_path.parent / "manifest.json"
    if not candidate.is_absolute() and explicit_manifest is not None:
        candidate = Path.cwd() / candidate
    if not candidate.is_file():
        if explicit_manifest is not None:
            raise ValueError(f"dataset manifest does not exist: {candidate}")
        return {"manifest_path": None, "manifest_sha256": None, "dataset": None, "revision": None}
    try:
        with snapshot_file(candidate) as (manifest_snapshot, manifest_hash):
            manifest = json.loads(manifest_snapshot.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read dataset manifest {candidate}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("dataset manifest root must be an object")
    try:
        output = manifest["outputs"][split]
    except (KeyError, TypeError):
        raise ValueError(f"dataset manifest lacks an output entry for split {split!r}")
    if not isinstance(output, dict) or output.get("sha256") != data_hash or output.get("records") != records:
        raise ValueError(f"dataset split {split!r} does not match manifest hashes/counts")
    dataset = manifest.get("dataset")
    dataset = dataset if isinstance(dataset, dict) else None
    return {
        "manifest_path": str(candidate.resolve()),
        "manifest_sha256": manifest_hash,
        "dataset": dataset,
        "revision": dataset.get("dataset_version") if dataset else None,
    }


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    *,
    split: str,
    data_path: str | Path,
    batch_size: int = 16,
    device: str = "cpu",
    manifest_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load a checkpoint in this process, evaluate an explicit split, and report evidence."""
    if split not in ("validation", "test"):
        raise ValueError("split must be explicitly 'validation' or 'test'; training evaluation is not supported")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    checkpoint = Path(checkpoint_path).resolve()
    data_file = Path(data_path).resolve()
    if not checkpoint.is_file():
        raise ValueError(f"checkpoint does not exist: {checkpoint}")
    if not data_file.is_file():
        raise ValueError(f"evaluation data file does not exist: {data_file}")
    target_device = _resolve_device(device)
    try:
        with snapshot_file(checkpoint) as (checkpoint_snapshot, checkpoint_hash):
            model = load_checkpoint(checkpoint_snapshot, map_location="cpu").to(target_device)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot load checkpoint snapshot for {checkpoint}: {exc}") from exc
    tokenizer = ByteTokenizer()
    tokenizer.assert_compatible(model.config)
    try:
        with snapshot_file(data_file) as (data_snapshot, data_hash):
            records = load_jsonl_text(data_snapshot)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot load evaluation data snapshot for {data_file}: {exc}") from exc
    windows = build_context_windows(records, tokenizer, model.config.model.context_length)
    batches = make_lm_batches(windows, batch_size, shuffle=False)
    metrics = evaluate_model(model, batches, device=target_device)
    provenance = _manifest_provenance(data_file, split, manifest_path, data_hash, len(records))
    report: dict[str, Any] = {
        "report_version": 1,
        "measured": True,
        "estimate": False,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": {"path": str(checkpoint), "sha256": checkpoint_hash, "format": "ddams-duo-checkpoint"},
        "dataset": {
            "split": split,
            "path": str(data_file),
            "sha256": data_hash,
            "records": len(records),
            **provenance,
        },
        "tokenizer": {"version": tokenizer.version, "vocabulary_sha256": tokenizer.vocabulary_sha256()},
        "model_config": model.config.to_dict(),
        "evaluation": {"batch_size": batch_size, "context_length": model.config.model.context_length, "device": str(target_device), "shuffle": False, "loss": "token-weighted next-token NLL excluding PAD targets", "perplexity": "exp(mean NLL)"},
        "software": {"python": platform.python_version(), "torch": str(torch.__version__)},
        "metrics": metrics,
        "weights_unchanged_by_evaluation": True,
    }
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        report["report_path"] = str(destination.resolve())
        fd, temp_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, destination)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
    return report
