"""Safe, atomic checkpoints for resumable training (separate from inference files)."""
from __future__ import annotations

import os
import pickle
import tempfile
import math
import zipfile
from pathlib import Path
from typing import Any

import torch
from .run_config import TrainingRunConfig

RESUME_FORMAT = "ddams-duo-training-resume"
RESUME_VERSION = 1
RESUME_FIELDS = {
    "format",
    "format_version",
    "compatibility",
    "run_config",
    "model_config",
    "tokenizer",
    "dataset",
    "environment",
    "run_id",
    "completed_steps",
    "metrics_state",
    "pending_events",
    "model_state",
    "optimizer_state",
    "sampler_state",
    "rng_state",
}


def atomic_torch_save(payload: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
        try:
            directory_fd = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            # Directory fsync is not available on every supported filesystem.
            pass
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def load_training_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except (OSError, RuntimeError, ValueError, TypeError, EOFError, pickle.UnpicklingError, zipfile.BadZipFile) as exc:
        raise ValueError(f"cannot safely load training-resume checkpoint {path}: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != RESUME_FIELDS:
        raise ValueError("malformed training-resume checkpoint fields")
    if payload.get("format") != RESUME_FORMAT:
        raise ValueError("checkpoint is not a Duo training-resume checkpoint")
    if type(payload.get("format_version")) is not int or payload["format_version"] != RESUME_VERSION:
        raise ValueError("unsupported Duo training-resume checkpoint version")
    if type(payload.get("completed_steps")) is not int or payload["completed_steps"] < 0:
        raise ValueError("training-resume checkpoint completed_steps is invalid")
    if not isinstance(payload.get("run_id"), str) or not payload["run_id"]:
        raise ValueError("training-resume checkpoint run_id is invalid")
    for name in ("compatibility", "run_config", "model_config", "tokenizer", "dataset", "environment", "metrics_state", "model_state", "optimizer_state", "sampler_state", "rng_state"):
        if not isinstance(payload.get(name), dict):
            raise ValueError(f"training-resume checkpoint {name} must be an object")
    compatibility = payload["compatibility"]
    if set(compatibility) != {"model_config", "run_settings", "tokenizer", "dataset", "environment"}:
        raise ValueError("training-resume checkpoint compatibility fields are malformed")
    try:
        validated_run_config = TrainingRunConfig.from_dict(payload["run_config"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"training-resume checkpoint run_config is malformed: {exc}") from exc
    run_settings = validated_run_config.to_dict()
    run_settings.pop("max_steps", None)
    run_settings.pop("output_dir", None)
    if (
        compatibility["model_config"] != payload["model_config"]
        or compatibility["tokenizer"] != payload["tokenizer"]
        or compatibility["dataset"] != payload["dataset"]
        or compatibility["environment"] != payload["environment"]
        or compatibility["run_settings"] != run_settings
    ):
        raise ValueError("training-resume checkpoint metadata and compatibility fields disagree")
    metrics = payload["metrics_state"]
    if set(metrics) != {"recent_loss_sum", "recent_token_count"}:
        raise ValueError("training-resume checkpoint metrics_state is malformed")
    if (
        type(metrics["recent_loss_sum"]) not in (int, float)
        or not math.isfinite(metrics["recent_loss_sum"])
        or metrics["recent_loss_sum"] < 0
    ):
        raise ValueError("training-resume checkpoint recent_loss_sum is invalid")
    if type(metrics["recent_token_count"]) is not int or metrics["recent_token_count"] < 0:
        raise ValueError("training-resume checkpoint recent_token_count is invalid")
    if not isinstance(payload.get("pending_events"), list) or any(
        not isinstance(event, dict) or not isinstance(event.get("event_id"), str)
        for event in payload["pending_events"]
    ):
        raise ValueError("training-resume checkpoint pending_events is malformed")
    rng = payload["rng_state"]
    if set(rng) != {"python", "torch_cpu", "torch_cuda"}:
        raise ValueError("training-resume checkpoint rng_state fields are malformed")
    python_rng = rng["python"]
    if (
        not isinstance(python_rng, tuple)
        or len(python_rng) != 3
        or type(python_rng[0]) is not int
        or not isinstance(python_rng[1], tuple)
        or not python_rng[1]
        or any(type(value) is not int for value in python_rng[1])
        or (python_rng[2] is not None and type(python_rng[2]) not in (int, float))
    ):
        raise ValueError("training-resume checkpoint Python RNG state is malformed")
    cpu_rng = rng["torch_cpu"]
    cuda_rng = rng["torch_cuda"]
    if not isinstance(cpu_rng, torch.Tensor) or cpu_rng.dtype != torch.uint8 or cpu_rng.ndim != 1 or cpu_rng.numel() == 0:
        raise ValueError("training-resume checkpoint CPU RNG state is malformed")
    if not isinstance(cuda_rng, list) or any(
        not isinstance(state, torch.Tensor)
        or state.dtype != torch.uint8
        or state.ndim != 1
        or state.numel() == 0
        for state in cuda_rng
    ):
        raise ValueError("training-resume checkpoint CUDA RNG state is malformed")
    if not payload["model_state"] or any(not isinstance(v, torch.Tensor) for v in payload["model_state"].values()):
        raise ValueError("training-resume checkpoint model_state must contain tensors")
    if not isinstance(payload["optimizer_state"].get("state"), dict) or not isinstance(payload["optimizer_state"].get("param_groups"), list):
        raise ValueError("training-resume checkpoint optimizer_state is malformed")
    return payload
