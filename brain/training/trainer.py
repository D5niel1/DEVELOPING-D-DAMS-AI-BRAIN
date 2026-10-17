"""Reusable deterministic Duo training with separate resume/inference artifacts."""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch
from torch.nn.utils import clip_grad_norm_

from brain.config import BrainConfig
from brain.data.language_model import LanguageModelBatch, build_context_windows, load_jsonl_text, make_lm_batches
from brain.model import DuoTransformer, masked_next_token_loss, save_checkpoint
from brain.tokenizer import ByteTokenizer
from .checkpoint import RESUME_FORMAT, RESUME_VERSION, atomic_torch_save, load_training_checkpoint
from .evaluation import evaluate_model
from .run_config import TrainingRunConfig
from .sampler import DeterministicBatchSampler
from .snapshots import snapshot_file

ROOT = Path(__file__).resolve().parents[2]
IMPLEMENTATION_FILES = (
    "requirements.txt",
    "brain/config.py",
    "brain/data/language_model.py",
    "brain/model/__init__.py",
    "brain/model/duo.py",
    "brain/tokenizer/__init__.py",
    "brain/tokenizer/byte_tokenizer.py",
    "brain/training/__init__.py",
    "brain/training/__main__.py",
    "brain/training/checkpoint.py",
    "brain/training/evaluation.py",
    "brain/training/run_config.py",
    "brain/training/sampler.py",
    "brain/training/snapshots.py",
    "brain/training/trainer.py",
)


class TrainingError(RuntimeError):
    """Raised when training or resume validation cannot proceed safely."""


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _implementation_fingerprint() -> dict[str, Any]:
    digest = hashlib.sha256()
    for relative in IMPLEMENTATION_FILES:
        source = ROOT / relative
        if not source.is_file():
            raise TrainingError(f"cannot fingerprint missing training implementation file: {relative}")
        content = source.read_bytes()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return {"sha256": digest.hexdigest(), "files": list(IMPLEMENTATION_FILES)}


def _json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _jsonl_append(path: Path, event: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _commit_pending_events(
    path: Path,
    pending: list[dict[str, Any]],
    *,
    completed_step: int | None = None,
) -> None:
    """Atomically reconcile only events backed by the supplied durable checkpoint."""
    events: list[dict[str, Any]] = []
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                if index == len(lines) - 1:
                    break  # A crash may have left only the final append incomplete.
                raise TrainingError(f"metrics log has malformed JSON before its final line: {path}:{index + 1}")
            if not isinstance(event, dict) or not isinstance(event.get("event_id"), str):
                raise TrainingError(f"metrics log event lacks an event_id: {path}:{index + 1}")
            kind = event.get("type")
            event_step = event.get("step")
            if completed_step is not None:
                if kind in ("training_step", "validation") and type(event_step) is int and event_step > completed_step:
                    continue
                if kind == "resume" and type(event.get("completed_steps")) is int and event["completed_steps"] > completed_step:
                    continue
            events.append(event)
    event_ids = {event["event_id"] for event in events}
    for event in pending:
        event_id = event.get("event_id")
        if not isinstance(event_id, str):
            raise TrainingError("pending metrics event is missing an event_id")
        if event_id not in event_ids:
            events.append(dict(event))
            event_ids.add(event_id)
    serialized = "".join(json.dumps(event, sort_keys=True, allow_nan=False) + "\n" for event in events)
    _text_atomic(path, serialized)


def _load_split(path: Path, split: str) -> tuple[list[str], str]:
    if not path.is_file():
        raise TrainingError(f"configured {split} dataset file does not exist: {path}")
    try:
        with snapshot_file(path) as (snapshot, content_hash):
            records = load_jsonl_text(snapshot)
    except ValueError as exc:
        raise TrainingError(f"cannot load {split} dataset: {exc}") from exc
    if not records:
        raise TrainingError(f"configured {split} dataset is empty: {path}")
    return records, content_hash


def _dataset_metadata(
    run_config: TrainingRunConfig,
    train_path: Path,
    validation_path: Path,
    train_hash: str,
    validation_hash: str,
    train_records: list[str],
    validation_records: list[str],
) -> dict[str, Any]:
    manifest_path = (
        resolve_repo_path(run_config.dataset_manifest_path)
        if run_config.dataset_manifest_path is not None
        else train_path.parent / "manifest.json"
    )
    manifest: dict[str, Any] | None = None
    manifest_hash: str | None = None
    if manifest_path.is_file():
        try:
            with snapshot_file(manifest_path) as (manifest_snapshot, manifest_hash):
                manifest = json.loads(manifest_snapshot.read_bytes())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TrainingError(f"cannot load dataset manifest {manifest_path}: {exc}") from exc
        if not isinstance(manifest, dict) or not isinstance(manifest.get("outputs"), dict):
            raise TrainingError("dataset manifest must contain an outputs object")
        for split, actual_hash, records in (
            ("train", train_hash, train_records),
            ("validation", validation_hash, validation_records),
        ):
            try:
                output = manifest["outputs"][split]
            except (KeyError, TypeError) as exc:
                raise TrainingError(f"dataset manifest lacks {split} output metadata") from exc
            if not isinstance(output, dict) or output.get("sha256") != actual_hash or output.get("records") != len(records):
                raise TrainingError(f"loaded {split} split does not match dataset manifest hashes/counts")
    elif run_config.dataset_manifest_path is not None:
        raise TrainingError(f"configured dataset manifest does not exist: {manifest_path}")
    dataset = manifest.get("dataset") if isinstance(manifest, dict) else None
    return {
        "manifest_path": str(manifest_path.resolve()) if manifest is not None else None,
        "manifest_sha256": manifest_hash,
        "dataset": dataset if isinstance(dataset, dict) else None,
        "splits": {
            "train": {"path": str(train_path), "sha256": train_hash, "records": len(train_records)},
            "validation": {"path": str(validation_path), "sha256": validation_hash, "records": len(validation_records)},
        },
    }


def _environment(device: torch.device, num_threads: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "device": str(device),
        "device_type": device.type,
        "num_threads": num_threads,
        "num_interop_threads": torch.get_num_interop_threads(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "thread_environment": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        "deterministic_algorithms": True,
    }
    if device.type == "cuda":
        result["cuda_version"] = torch.version.cuda
        result["cuda_device_count"] = torch.cuda.device_count()
        result["device_name"] = torch.cuda.get_device_name(device)
    return result


def _device(value: str) -> torch.device:
    if value == "cpu":
        return torch.device("cpu")
    if value == "cuda" or value.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise TrainingError(f"requested device {value!r}, but CUDA is unavailable")
        result = torch.device(value)
        if result.index is not None and result.index >= torch.cuda.device_count():
            raise TrainingError(f"requested device {value!r}, but only {torch.cuda.device_count()} CUDA device(s) are available")
        return result
    raise TrainingError("device must be 'cpu', 'cuda', or 'cuda:<index>'")


@contextmanager
def _deterministic_runtime(device: torch.device, num_threads: int) -> Iterator[None]:
    previous_python_rng = random.getstate()
    previous_cpu_rng = torch.get_rng_state()
    previous_cuda_rng = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_cudnn = None
    previous_tf32 = None
    if device.type == "cuda":
        previous_cudnn = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.set_num_threads(num_threads)
        torch.use_deterministic_algorithms(True)
        if device.type == "cuda":
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = False
        yield
    finally:
        random.setstate(previous_python_rng)
        torch.set_rng_state(previous_cpu_rng)
        if previous_cuda_rng is not None:
            torch.cuda.set_rng_state_all(previous_cuda_rng)
        torch.set_num_threads(previous_threads)
        torch.use_deterministic_algorithms(previous_deterministic)
        if previous_cudnn is not None:
            torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = previous_cudnn
            torch.backends.cuda.matmul.allow_tf32 = bool(previous_tf32)


def _seed_all(seed: int, device: torch.device) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _rng_state(device: torch.device) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state().cpu(),
        "torch_cuda": [state.cpu() for state in torch.cuda.get_rng_state_all()] if device.type == "cuda" else [],
    }


def _restore_rng_state(state: Mapping[str, Any], device: torch.device) -> None:
    if set(state) != {"python", "torch_cpu", "torch_cuda"}:
        raise TrainingError("resume checkpoint has malformed RNG state")
    try:
        random.setstate(state["python"])
        torch.set_rng_state(state["torch_cpu"].cpu())
        cuda_states = state["torch_cuda"]
        if device.type == "cuda":
            if not isinstance(cuda_states, list) or len(cuda_states) != torch.cuda.device_count():
                raise TrainingError("resume checkpoint CUDA RNG-state count is incompatible with this device")
            torch.cuda.set_rng_state_all(cuda_states)
        elif cuda_states:
            raise TrainingError("resume checkpoint was created with CUDA RNG state but CPU was requested")
    except (AttributeError, KeyError, IndexError, TypeError, ValueError, RuntimeError) as exc:
        raise TrainingError(f"cannot restore checkpoint RNG state: {exc}") from exc


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {_cpu_tree(key): _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def _compatibility(
    run_config: TrainingRunConfig,
    model_config: BrainConfig,
    tokenizer: ByteTokenizer,
    dataset: dict[str, Any],
    environment: dict[str, Any],
) -> dict[str, Any]:
    settings = run_config.to_dict()
    # Extending the total target does not change the mathematical continuation;
    # output_dir is excluded from math settings but separately tied to checkpoint parent.
    settings.pop("max_steps")
    settings.pop("output_dir")
    return {
        "model_config": model_config.to_dict(),
        "run_settings": settings,
        "tokenizer": {"version": tokenizer.version, "vocabulary_sha256": tokenizer.vocabulary_sha256()},
        "dataset": dataset,
        "environment": environment,
    }


def _build_batch(windows: list[list[int]], indices: list[int], seed: int) -> LanguageModelBatch:
    batches = make_lm_batches([windows[index] for index in indices], len(indices), seed=seed, shuffle=False)
    if len(batches) != 1:
        raise TrainingError("internal batching error: sampler indices did not produce one batch")
    return batches[0]


def _resume_payload(
    model: DuoTransformer,
    optimizer: torch.optim.Optimizer,
    sampler: DeterministicBatchSampler,
    completed_steps: int,
    run_config: TrainingRunConfig,
    model_config: BrainConfig,
    tokenizer: ByteTokenizer,
    dataset: dict[str, Any],
    environment: dict[str, Any],
    recent_loss_sum: float,
    recent_token_count: int,
    pending_events: list[dict[str, Any]],
    run_id: str,
) -> dict[str, Any]:
    return {
        "format": RESUME_FORMAT,
        "format_version": RESUME_VERSION,
        "compatibility": _compatibility(run_config, model_config, tokenizer, dataset, environment),
        "run_config": run_config.to_dict(),
        "model_config": model_config.to_dict(),
        "tokenizer": {"version": tokenizer.version, "vocabulary_sha256": tokenizer.vocabulary_sha256()},
        "dataset": dataset,
        "environment": environment,
        "run_id": run_id,
        "completed_steps": completed_steps,
        "metrics_state": {"recent_loss_sum": recent_loss_sum, "recent_token_count": recent_token_count},
        "pending_events": list(pending_events),
        "model_state": _cpu_tree(model.state_dict()),
        "optimizer_state": _cpu_tree(optimizer.state_dict()),
        "sampler_state": sampler.state_dict(),
        "rng_state": _rng_state(torch.device(run_config.device)),
    }


def _check_resume_compatibility(payload: dict[str, Any], current: dict[str, Any], target_steps: int) -> None:
    saved = payload["compatibility"]
    if saved != current:
        differences = sorted(key for key in set(saved) | set(current) if saved.get(key) != current.get(key))
        raise TrainingError(
            "resume checkpoint is incompatible in: " + ", ".join(differences)
            + "; keep model, data, tokenizer, optimizer, batching, seed, device, and software settings unchanged"
        )
    if target_steps < payload["completed_steps"]:
        raise TrainingError(
            f"max_steps ({target_steps}) is below checkpoint completed_steps ({payload['completed_steps']})"
        )


def _set_up_optimizer(config: TrainingRunConfig, model: DuoTransformer) -> torch.optim.Optimizer:
    if config.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2),
            eps=config.epsilon,
        )
    raise TrainingError(f"unsupported optimizer {config.optimizer!r}")


def train_run(
    run_config: TrainingRunConfig,
    *,
    resume_path: str | Path | None = None,
    config_file_path: str | Path | None = None,
    configured_defaults: Mapping[str, Any] | None = None,
    runtime_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Train to the configured total step count, optionally restoring exact state."""
    run_config.validate()
    device = _device(run_config.device)
    model_config_path = resolve_repo_path(run_config.model_config_path)
    if not model_config_path.is_file():
        raise TrainingError(f"model config does not exist: {model_config_path}")
    try:
        with snapshot_file(model_config_path) as (model_config_snapshot, model_config_hash):
            model_config = BrainConfig.load(model_config_snapshot)
    except (OSError, ValueError) as exc:
        raise TrainingError(f"cannot read model config {model_config_path}: {exc}") from exc
    if run_config.context_length != model_config.model.context_length:
        raise TrainingError(
            "context_length must equal model_config.model.context_length; edit the model config to choose another architecture context"
        )
    tokenizer = ByteTokenizer()
    tokenizer.assert_compatible(model_config)

    train_path = resolve_repo_path(run_config.train_path).resolve()
    validation_path = resolve_repo_path(run_config.validation_path).resolve()
    train_records, train_hash = _load_split(train_path, "train")
    validation_records, validation_hash = _load_split(validation_path, "validation")
    dataset = _dataset_metadata(
        run_config,
        train_path,
        validation_path,
        train_hash,
        validation_hash,
        train_records,
        validation_records,
    )
    train_windows = build_context_windows(train_records, tokenizer, run_config.context_length)
    validation_windows = build_context_windows(validation_records, tokenizer, run_config.context_length)
    if not train_windows or not validation_windows:
        raise TrainingError("train and validation splits must both produce at least one context window")
    if run_config.drop_last and len(train_windows) < run_config.batch_size:
        raise TrainingError("drop_last would discard every training window; lower batch_size or disable drop_last")
    validation_batches = make_lm_batches(
        validation_windows, run_config.batch_size, seed=run_config.seed, shuffle=False
    )
    environment = _environment(device, run_config.num_threads)
    environment["implementation"] = _implementation_fingerprint()
    compatibility = _compatibility(run_config, model_config, tokenizer, dataset, environment)
    output_dir = resolve_repo_path(run_config.output_dir).resolve()
    resuming = resume_path is not None
    if output_dir.exists() and not output_dir.is_dir():
        raise TrainingError(f"output_dir exists but is not a directory: {output_dir}")
    if not resuming and output_dir.exists() and any(output_dir.iterdir()):
        raise TrainingError(f"output directory is not empty; choose a new path or use --resume: {output_dir}")
    run_metadata_path = output_dir / "run_metadata.json"
    existing_metadata: dict[str, Any] | None = None
    if resuming:
        resume_directory = Path(resume_path).resolve().parent
        if output_dir != resume_directory:
            raise TrainingError(
                "resume output_dir must be the parent directory of the checkpoint; refusing to merge or overwrite another run"
            )
    if resuming and output_dir.exists():
        entries = list(output_dir.iterdir())
        if entries and not run_metadata_path.is_file():
            raise TrainingError(f"resume output directory contains files but no run metadata; refusing to overwrite: {output_dir}")
        if run_metadata_path.is_file():
            try:
                existing_metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
                if not isinstance(existing_metadata, dict):
                    raise TrainingError("existing run metadata must be a JSON object")
                resume_invocations = existing_metadata.get("resume_invocations", [])
                if not isinstance(resume_invocations, list) or any(
                    not isinstance(invocation, dict) for invocation in resume_invocations
                ):
                    raise TrainingError("existing run metadata resume_invocations must be a list of objects")
                existing_metadata["resume_invocations"] = resume_invocations
                prior_config = TrainingRunConfig.from_dict(existing_metadata["effective_run_config"])
                prior_compatibility = _compatibility(prior_config, model_config, tokenizer, dataset, environment)
            except (OSError, AttributeError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise TrainingError(f"existing run metadata is malformed or incompatible: {exc}") from exc
            expected_metadata = {
                "model_config_sha256": model_config_hash,
                "model_config": model_config.to_dict(),
                "tokenizer": {"version": tokenizer.version, "vocabulary_sha256": tokenizer.vocabulary_sha256()},
                "dataset": dataset,
                "environment": environment,
                "implementation": environment["implementation"],
            }
            if prior_compatibility != compatibility or any(
                existing_metadata.get(key) != value for key, value in expected_metadata.items()
            ):
                raise TrainingError("existing run metadata does not match the resume checkpoint configuration")
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    resume_checkpoint_path = output_dir / "latest.resume.pt"
    inference_checkpoint_path = output_dir / "final.inference.pt"
    started_utc = datetime.now(timezone.utc).isoformat()
    run_id = started_utc
    with _deterministic_runtime(device, run_config.num_threads):
        _seed_all(run_config.seed, device)
        model = DuoTransformer(model_config, seed=run_config.seed).to(device)
        optimizer = _set_up_optimizer(run_config, model)
        sampler = DeterministicBatchSampler(
            len(train_windows), run_config.batch_size, run_config.seed, drop_last=run_config.drop_last
        )
        completed_steps = 0
        resume_hash: str | None = None
        recent_loss_sum = 0.0
        recent_token_count = 0
        pending_events: list[dict[str, Any]] = []
        if resuming:
            resume_file = Path(resume_path).resolve()
            try:
                with snapshot_file(resume_file) as (resume_snapshot, resume_hash):
                    payload = load_training_checkpoint(resume_snapshot, map_location="cpu")
            except (OSError, ValueError) as exc:
                raise TrainingError(f"cannot load resume checkpoint {resume_file}: {exc}") from exc
            run_id = payload["run_id"]
            if existing_metadata is not None and existing_metadata.get("created_at_utc") != run_id:
                raise TrainingError("resume checkpoint run ID does not match the existing output directory")
            current_compatibility = _compatibility(run_config, model_config, tokenizer, dataset, environment)
            _check_resume_compatibility(payload, current_compatibility, run_config.max_steps)
            try:
                model.load_state_dict(payload["model_state"], strict=True)
                optimizer.load_state_dict(payload["optimizer_state"])
                sampler.load_state_dict(payload["sampler_state"])
            except (RuntimeError, TypeError, ValueError) as exc:
                raise TrainingError(f"resume checkpoint state is incompatible: {exc}") from exc
            completed_steps = payload["completed_steps"]
            recent_loss_sum = float(payload["metrics_state"]["recent_loss_sum"])
            recent_token_count = payload["metrics_state"]["recent_token_count"]
            pending_events = list(payload["pending_events"])
            _restore_rng_state(payload["rng_state"], device)
            _commit_pending_events(metrics_path, pending_events, completed_step=completed_steps)
            pending_events.clear()
            _jsonl_append(
                metrics_path,
                {
                    "event_id": f"resume:{completed_steps}:{started_utc}",
                    "type": "resume",
                    "completed_steps": completed_steps,
                    "checkpoint_sha256": resume_hash,
                    "target_max_steps": run_config.max_steps,
                },
            )
        else:
            if not metrics_path.exists():
                metrics_path.touch()
            baseline = evaluate_model(model, validation_batches, device=device)
            _jsonl_append(metrics_path, {"event_id": "baseline_validation:0", "type": "baseline_validation", "step": 0, **baseline})

        if existing_metadata is not None:
            if not resuming:
                raise TrainingError("run metadata already exists; refusing to overwrite a previous run")
            run_metadata = existing_metadata
            run_metadata["resume_invocations"].append(
                {
                    "started_at_utc": started_utc,
                    "target_max_steps": run_config.max_steps,
                    "checkpoint_sha256": resume_hash,
                    "config_file": str(Path(config_file_path).resolve()) if config_file_path else None,
                    "configured_defaults": dict(configured_defaults or run_config.to_dict()),
                    "effective_run_config": run_config.to_dict(),
                    "runtime_overrides": dict(runtime_overrides or {}),
                }
            )
        else:
            if run_metadata_path.exists():
                raise TrainingError("run metadata appeared after preflight; refusing to overwrite it")
            run_metadata = {
                "metadata_version": 1,
                "created_at_utc": run_id,
                "config_file": str(Path(config_file_path).resolve()) if config_file_path else None,
                "configured_defaults": dict(configured_defaults or run_config.to_dict()),
                "runtime_overrides": dict(runtime_overrides or {}),
                "effective_run_config": run_config.to_dict(),
                "model_config_path": str(model_config_path.resolve()),
                "model_config_sha256": model_config_hash,
                "model_config": model_config.to_dict(),
                "tokenizer": {"version": tokenizer.version, "vocabulary_sha256": tokenizer.vocabulary_sha256()},
                "dataset": dataset,
                "environment": environment,
                "implementation": environment["implementation"],
                "resume_invocations": [],
            }
        run_metadata["effective_run_config"] = run_config.to_dict()
        _json_atomic(run_metadata_path, run_metadata)

        last_validation: dict[str, float | int] | None = None
        for step in range(completed_steps + 1, run_config.max_steps + 1):
            indices = sampler.next_indices()
            batch = _build_batch(train_windows, indices, run_config.seed)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch.input_ids.to(device), batch.attention_mask.to(device))
            loss = masked_next_token_loss(logits, batch.targets.to(device), batch.target_mask.to(device))
            if not bool(torch.isfinite(loss).item()):
                raise TrainingError(f"non-finite loss at optimizer step {step}; no new checkpoint was written")
            valid_targets = batch.target_mask & batch.targets.ne(0)
            target_count = int(valid_targets.sum().item())
            loss_value = float(loss.detach().item())
            loss.backward()
            gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
            if not gradients or any(not bool(torch.isfinite(gradient).all().item()) for gradient in gradients):
                raise TrainingError(f"missing or non-finite gradients at optimizer step {step}; no new checkpoint was written")
            try:
                gradient_norm = float(
                    clip_grad_norm_(model.parameters(), run_config.gradient_clip_norm, error_if_nonfinite=True).item()
                )
            except RuntimeError as exc:
                raise TrainingError(f"gradient clipping failed at optimizer step {step}: {exc}") from exc
            if not math.isfinite(gradient_norm):
                raise TrainingError(f"non-finite gradient norm at optimizer step {step}; no new checkpoint was written")
            optimizer.step()
            if any(not bool(torch.isfinite(parameter).all().item()) for parameter in model.parameters()):
                raise TrainingError(f"non-finite model weights after optimizer step {step}; no new checkpoint was written")

            recent_loss_sum += loss_value * target_count
            recent_token_count += target_count
            event = {
                "type": "training_step",
                "event_id": f"training_step:{step}",
                "step": step,
                "batch_nll": loss_value,
                "target_tokens": target_count,
                "gradient_norm_pre_clip": gradient_norm,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "sampler_epoch": sampler.epoch,
                "sampler_cursor": sampler.cursor,
            }
            pending_events.append(event)
            print(
                f"STEP {step}/{run_config.max_steps} loss={loss_value:.6f} tokens={target_count} "
                f"grad_norm={gradient_norm:.6f} lr={optimizer.param_groups[0]['lr']:.8g}",
                flush=True,
            )

            if step % run_config.evaluation_interval == 0 or step == run_config.max_steps:
                last_validation = evaluate_model(model, validation_batches, device=device)
                if recent_token_count:
                    recent_train_nll = recent_loss_sum / recent_token_count
                else:
                    recent_train_nll = loss_value
                eval_event = {
                    "event_id": f"validation:{step}",
                    "type": "validation",
                    "step": step,
                    "training_nll_since_previous_evaluation": recent_train_nll,
                    "training_target_tokens_since_previous_evaluation": recent_token_count,
                    "validation": last_validation,
                }
                pending_events.append(eval_event)
                print(
                    f"EVAL step={step} train_nll={recent_train_nll:.6f} "
                    f"validation_nll={last_validation['nll']:.6f} perplexity={last_validation['perplexity']:.6f}",
                    flush=True,
                )
                recent_loss_sum = 0.0
                recent_token_count = 0

            if step % run_config.checkpoint_interval == 0 or step == run_config.max_steps:
                payload = _resume_payload(
                    model,
                    optimizer,
                    sampler,
                    step,
                    run_config,
                    model_config,
                    tokenizer,
                    dataset,
                    environment,
                    recent_loss_sum,
                    recent_token_count,
                    pending_events,
                    run_id,
                )
                atomic_torch_save(payload, resume_checkpoint_path)
                _commit_pending_events(metrics_path, pending_events, completed_step=step)
                pending_events.clear()
                print(f"CHECKPOINT step={step} path={resume_checkpoint_path}", flush=True)

        if completed_steps > run_config.max_steps:
            raise TrainingError("internal error: completed step count exceeded configured maximum")
        if completed_steps == run_config.max_steps:
            # An already-complete resumed state can be finalized without another update.
            pass
        if last_validation is None:
            last_validation = evaluate_model(model, validation_batches, device=device)
        completed_steps = run_config.max_steps
        if not resume_checkpoint_path.is_file() or completed_steps == 0:
            payload = _resume_payload(
                model,
                optimizer,
                sampler,
                completed_steps,
                run_config,
                model_config,
                tokenizer,
                dataset,
                environment,
                recent_loss_sum,
                recent_token_count,
                pending_events,
                run_id,
            )
            atomic_torch_save(payload, resume_checkpoint_path)
            _commit_pending_events(metrics_path, pending_events, completed_step=completed_steps)
            pending_events.clear()
        save_checkpoint(model, inference_checkpoint_path)
        summary = {
            "status": "completed",
            "completed_steps": completed_steps,
            "target_max_steps": run_config.max_steps,
            "resume_checkpoint": str(resume_checkpoint_path),
            "resume_checkpoint_sha256": _sha256(resume_checkpoint_path),
            "inference_checkpoint": str(inference_checkpoint_path),
            "inference_checkpoint_sha256": _sha256(inference_checkpoint_path),
            "metrics_log": str(metrics_path),
            "run_metadata": str(run_metadata_path),
            "training_summary": str(output_dir / "training_summary.json"),
            "last_validation": last_validation,
            "device": str(device),
        }
        _json_atomic(output_dir / "training_summary.json", summary)
        print(
            f"DONE steps={completed_steps} inference_checkpoint={inference_checkpoint_path} "
            f"resume_checkpoint={resume_checkpoint_path}",
            flush=True,
        )
        return summary
