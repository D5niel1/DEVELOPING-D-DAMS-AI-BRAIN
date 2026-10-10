"""One-run Duo v0 CPU smoke training experiment (not a general trainer)."""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.nn.utils import clip_grad_norm_

from brain.config import BrainConfig
from brain.data.language_model import (
    LanguageModelBatch,
    build_context_windows,
    load_jsonl_text,
    make_lm_batches,
)
from brain.model import DuoTransformer, masked_next_token_loss, save_checkpoint
from brain.tokenizer import ByteTokenizer

ROOT = Path(__file__).resolve().parents[2]
MAX_OPTIMIZER_STEPS = 100
MAX_WALL_SECONDS = 120.0
EVAL_INTERVAL = 10
BATCH_SIZE = 4
GRADIENT_CLIP_NORM = 1.0
WEIGHT_DECAY = 0.01
BETAS = (0.9, 0.999)
ADAM_EPSILON = 1e-8
DIVERGENCE_RATIO = 1.10
DIVERGENCE_CHECKS = 2


class SmokeTrainingError(RuntimeError):
    """Raised if this bounded engineering smoke run cannot be completed safely."""


def _json_write_atomic(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _jsonl_append(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_data_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _verify_local_splits(config: BrainConfig) -> tuple[dict[str, Path], dict[str, list[str]], dict[str, Any]]:
    paths = {
        "train": _resolve_data_path(config.data.train_path),
        "validation": _resolve_data_path(config.data.validation_path),
        "test": _resolve_data_path(config.data.test_path),
    }
    if len({path.parent.resolve() for path in paths.values()}) != 1:
        raise SmokeTrainingError("configured train/validation/test files must share one processed dataset directory")
    manifest_path = next(iter(paths.values())).parent / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SmokeTrainingError(f"cannot read dataset manifest {manifest_path}: {exc}") from exc
    dataset = manifest.get("dataset")
    if not isinstance(dataset, dict) or dataset.get("dataset_name") != "openai/gsm8k-main-train-first128-question-answer":
        raise SmokeTrainingError("dataset manifest is not the approved local GSM8K sample")
    if not str(dataset.get("source_url", "")).startswith("https://huggingface.co/datasets/openai/gsm8k/"):
        raise SmokeTrainingError("dataset manifest does not identify the verified GSM8K source")
    if dataset.get("license_verified") is not True:
        raise SmokeTrainingError("dataset manifest does not record a verified license")
    records: dict[str, list[str]] = {}
    split_provenance: dict[str, Any] = {}
    for split, path in paths.items():
        try:
            advertised = manifest["outputs"][split]
            expected_hash = advertised["sha256"]
            expected_count = advertised["records"]
        except (KeyError, TypeError) as exc:
            raise SmokeTrainingError(f"dataset manifest lacks valid {split} output metadata") from exc
        if split != "test":
            actual_hash = _sha256(path)
            if actual_hash != expected_hash:
                raise SmokeTrainingError(f"configured {split} split does not match its manifest")
            values = load_jsonl_text(path)
            if not values:
                raise SmokeTrainingError(f"configured {split} split is empty: {path}")
            if len(values) != expected_count:
                raise SmokeTrainingError(f"configured {split} record count does not match its manifest")
            records[split] = values
        else:
            if not path.is_file():
                raise SmokeTrainingError(f"configured test split does not exist: {path}")
            # Do not read/hash test contents until the single final evaluation.
            actual_hash = None
        split_provenance[split] = {
            "path": str(path),
            "sha256": actual_hash,
            "manifest_sha256_expected": expected_hash if split == "test" else None,
            "records": expected_count,
            "manifest_sha256": _sha256(manifest_path),
        }
    return paths, records, {"manifest_path": str(manifest_path), "manifest_sha256": _sha256(manifest_path), "splits": split_provenance, "dataset": dataset}


def evaluate_batches(model: DuoTransformer, batches: Sequence[LanguageModelBatch]) -> dict[str, float | int]:
    """Token-weighted next-token NLL/perplexity, excluding padded targets."""
    model.eval()
    total_nll = 0.0
    target_count = 0
    with torch.inference_mode():
        for batch in batches:
            logits = model(batch.input_ids, batch.attention_mask)
            loss = masked_next_token_loss(logits, batch.targets, batch.target_mask)
            valid = batch.target_mask & batch.targets.ne(0)
            count = int(valid.sum().item())
            if count == 0:
                continue
            loss_value = float(loss.item())
            if not math.isfinite(loss_value):
                raise SmokeTrainingError("non-finite evaluation loss")
            total_nll += loss_value * count
            target_count += count
    if target_count == 0:
        raise SmokeTrainingError("evaluation split contains no scored next-token targets")
    mean_nll = total_nll / target_count
    try:
        perplexity = math.exp(mean_nll)
    except OverflowError:
        perplexity = math.inf
    if not math.isfinite(perplexity):
        raise SmokeTrainingError("non-finite evaluation perplexity")
    return {"nll": mean_nll, "perplexity": perplexity, "target_tokens": target_count}


def _verify_checkpoint_in_fresh_process(model: DuoTransformer, checkpoint: Path) -> dict[str, Any]:
    """Use fixed synthetic IDs so this integrity check never reads the test split."""
    probe = [[2, 71, 108, 3]]
    model.eval()
    input_ids = torch.tensor(probe, dtype=torch.long)
    with torch.inference_mode():
        expected = model(input_ids).cpu()
    script = r'''import json, sys, torch
from brain.model import load_checkpoint
model=load_checkpoint(sys.argv[1]).cpu().eval()
ids=torch.tensor(json.loads(sys.argv[2]),dtype=torch.long)
with torch.inference_mode(): logits=model(ids)
print(json.dumps(logits.tolist(),allow_nan=False))
'''
    completed = subprocess.run(
        [sys.executable, "-c", script, str(checkpoint), json.dumps(probe)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    actual = torch.tensor(json.loads(completed.stdout), dtype=torch.float32)
    if not torch.allclose(actual, expected, rtol=0.0, atol=1e-6):
        raise SmokeTrainingError("fresh-process checkpoint logits differ by more than 1e-6")
    return {"verified": True, "process": "separate Python process", "input_ids": probe, "atol": 1e-6, "max_abs_error": float((actual - expected).abs().max().item())}


def _all_finite_tensors(tensors: Sequence[torch.Tensor]) -> bool:
    return all(bool(torch.isfinite(tensor).all().item()) for tensor in tensors)


def _write_status(output_dir: Path, status: dict[str, Any]) -> None:
    _json_write_atomic(output_dir / "experiment.json", status)


def run() -> int:
    started = time.monotonic()
    start_utc = datetime.now(timezone.utc).isoformat()
    config_path = ROOT / "configs" / "duo_v0.json"
    config = BrainConfig.load(config_path)
    if config.model.context_length != 128:
        raise SmokeTrainingError("smoke experiment requires context_length=128")
    seed = config.training.seed
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    tokenizer = ByteTokenizer()
    tokenizer.assert_compatible(config)
    paths, records, data_provenance = _verify_local_splits(config)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output_dir = ROOT / config.checkpoint_dir / "smoke" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    metrics_path = output_dir / "metrics.jsonl"
    checkpoint_path = output_dir / "duo-smoke-final.pt"
    train_windows = build_context_windows(records["train"], tokenizer, 128)
    validation_windows = build_context_windows(records["validation"], tokenizer, 128)
    train_eval_batches = make_lm_batches(train_windows, BATCH_SIZE, seed=seed, shuffle=False)
    validation_batches = make_lm_batches(validation_windows, BATCH_SIZE, seed=seed, shuffle=False)
    train_batches = make_lm_batches(train_windows, BATCH_SIZE, seed=seed, shuffle=True)

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        commit = None
    status: dict[str, Any] = {
        "run_id": run_id,
        "status": "running",
        "started_utc": start_utc,
        "git_commit": commit,
        "backend": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu", "cuda_available": torch.cuda.is_available(), "threads": torch.get_num_threads(), "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()},
        "config_path": str(config_path),
        "config": config.to_dict(),
        "model": {"init_seed": seed, "parameters": None},
        "optimizer": {"class": "AdamW", "lr": config.training.learning_rate, "weight_decay": WEIGHT_DECAY, "betas": list(BETAS), "eps": ADAM_EPSILON, "gradient_clip_norm": GRADIENT_CLIP_NORM},
        "batching": {"batch_size": BATCH_SIZE, "context_length": 128, "train_windows": len(train_windows), "validation_windows": len(validation_windows), "available_train_batches": len(train_batches), "shuffle": "one deterministic random.Random(seed) permutation of all training windows; first batches consumed once without wrapping", "shuffle_seed": seed, "drop_last": False, "optimizer_steps_max": MAX_OPTIMIZER_STEPS, "wall_time_max_seconds": MAX_WALL_SECONDS, "evaluation_interval_steps": EVAL_INTERVAL},
        "smoke_runner_sha256": _sha256(Path(__file__).resolve()),
        "data": data_provenance,
        "artifact_directory": str(output_dir),
        "checkpoint_path": str(checkpoint_path),
        "steps_completed": 0,
        "stop_reason": None,
        "events": [],
    }
    _write_status(output_dir, status)

    model = DuoTransformer(config, seed=seed).cpu()
    status["model"]["parameters"] = model.parameter_count
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate, weight_decay=WEIGHT_DECAY, betas=BETAS, eps=ADAM_EPSILON)

    # Recompute only train/validation pre-training baselines. The test split is
    # hashed for provenance but its records are not loaded until the final step.
    baseline_train = evaluate_batches(model, train_eval_batches)
    baseline_validation = evaluate_batches(model, validation_batches)
    baseline_event = {"type": "baseline", "step": 0, "train": baseline_train, "validation": baseline_validation, "test": "held out"}
    status["events"].append(baseline_event)
    _jsonl_append(metrics_path, baseline_event)
    print("BASELINE " + json.dumps(baseline_event, sort_keys=True), flush=True)
    _write_status(output_dir, status)

    train_step_batches = iter(train_batches)
    recent_loss_sum = 0.0
    recent_token_count = 0
    consecutive_high_validation = 0
    stop_reason: str | None = None
    steps_completed = 0
    training_finished = False
    optimizer_started = time.monotonic()

    for step in range(1, MAX_OPTIMIZER_STEPS + 1):
        if time.monotonic() - started >= MAX_WALL_SECONDS:
            stop_reason = "120_second_wall_time_limit"
            break
        try:
            batch = next(train_step_batches)
        except StopIteration:
            stop_reason = "training_batches_exhausted_before_100_steps"
            break
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch.input_ids, batch.attention_mask)
        loss = masked_next_token_loss(logits, batch.targets, batch.target_mask)
        if not bool(torch.isfinite(loss).item()):
            stop_reason = f"non_finite_loss_at_step_{step}"
            break
        valid = batch.target_mask & batch.targets.ne(0)
        token_count = int(valid.sum().item())
        loss_value = float(loss.detach().item())
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        if not gradients or not _all_finite_tensors(gradients):
            stop_reason = f"non_finite_or_missing_gradients_at_step_{step}"
            break
        grad_norm = float(clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM).item())
        if not math.isfinite(grad_norm) or not _all_finite_tensors(gradients):
            stop_reason = f"non_finite_clipped_gradients_at_step_{step}"
            break
        try:
            optimizer.step()
        except Exception as exc:
            stop_reason = f"optimizer_step_failed_at_step_{step}: {type(exc).__name__}: {exc}"
            break
        steps_completed = step
        status["steps_completed"] = steps_completed
        weights_finite = _all_finite_tensors([parameter.detach() for parameter in model.parameters()])
        step_event = {"type": "training_step", "step": step, "batch_nll": loss_value, "target_tokens": token_count, "gradient_norm_pre_clip": grad_norm, "learning_rate": optimizer.param_groups[0]["lr"], "weights_finite_after_step": weights_finite}
        _jsonl_append(metrics_path, step_event)
        if not weights_finite:
            stop_reason = f"non_finite_weights_after_step_{step}"
            break
        recent_loss_sum += loss_value * token_count
        recent_token_count += token_count

        if step % EVAL_INTERVAL == 0:
            validation = evaluate_batches(model, validation_batches)
            training_nll = recent_loss_sum / recent_token_count
            evaluation_event = {"type": "scheduled_evaluation", "step": step, "training_nll_last_10_steps": training_nll, "training_target_tokens_last_10_steps": recent_token_count, "validation": validation}
            status["events"].append(evaluation_event)
            _jsonl_append(metrics_path, evaluation_event)
            print("EVAL " + json.dumps(evaluation_event, sort_keys=True), flush=True)
            recent_loss_sum = 0.0
            recent_token_count = 0
            if validation["nll"] > DIVERGENCE_RATIO * baseline_validation["nll"]:
                consecutive_high_validation += 1
            else:
                consecutive_high_validation = 0
            if consecutive_high_validation >= DIVERGENCE_CHECKS:
                stop_reason = f"validation_nll_above_110_percent_of_baseline_at_{DIVERGENCE_CHECKS}_consecutive_checks"
                break
            _write_status(output_dir, status)
        if time.monotonic() - started >= MAX_WALL_SECONDS:
            stop_reason = "120_second_wall_time_limit"
            break
    else:
        stop_reason = "100_optimizer_step_limit"
        training_finished = True

    # If the step-100 validation event ended the for-loop with break, the
    # experiment is still complete only when step 100 was reached without a
    # divergence/time stop.
    if steps_completed == MAX_OPTIMIZER_STEPS and stop_reason is None:
        stop_reason = "100_optimizer_step_limit"
        training_finished = True
    if stop_reason is None:
        stop_reason = "training_stopped_without_explicit_reason"
    status["steps_completed"] = steps_completed
    status["optimizer_elapsed_seconds"] = time.monotonic() - optimizer_started
    status["stop_reason"] = stop_reason
    status["training_completed_to_step_cap"] = training_finished
    _write_status(output_dir, status)

    # Non-finite state is never serialized as a purported final checkpoint.
    if stop_reason.startswith("non_finite"):
        status["status"] = "stopped_non_finite"
    elif stop_reason.startswith("optimizer_step_failed"):
        status["status"] = "failed_optimizer_step"
    else:
        try:
            model.eval()
            save_checkpoint(model, checkpoint_path)
            reload_result = _verify_checkpoint_in_fresh_process(model, checkpoint_path)
            status["checkpoint_reload"] = reload_result
            status["events"].append({"type": "checkpoint_reload", **reload_result})
            _jsonl_append(metrics_path, {"type": "checkpoint_reload", **reload_result})
            status["status"] = "completed" if training_finished else "stopped_early"
        except Exception as exc:
            status["status"] = "failed_checkpoint_verification"
            status["stop_reason"] = f"checkpoint_save_or_fresh_process_reload_failed: {type(exc).__name__}: {exc}"
            status["checkpoint_reload"] = {"verified": False, "error": f"{type(exc).__name__}: {exc}"}
            _write_status(output_dir, status)
            print("STOP " + status["stop_reason"], flush=True)
            return 2

    if training_finished and status["status"] == "completed":
        # First and only read/evaluation of test records in this run; no tuning follows.
        final_test_sha256 = _sha256(paths["test"])
        if final_test_sha256 != data_provenance["splits"]["test"]["manifest_sha256_expected"]:
            status["status"] = "failed_test_manifest_validation"
            status["stop_reason"] = "final_test_sha256_does_not_match_manifest"
            status["final_test"] = None
            status["finished_utc"] = datetime.now(timezone.utc).isoformat()
            status["elapsed_seconds_total"] = time.monotonic() - started
            _write_status(output_dir, status)
            print("STOP " + status["stop_reason"], flush=True)
            return 2
        status["data"]["splits"]["test"]["sha256"] = final_test_sha256
        status["data"]["splits"]["test"]["hash_verified_at_final_evaluation"] = True
        test_records = load_jsonl_text(paths["test"])
        if len(test_records) != data_provenance["splits"]["test"]["records"]:
            status["status"] = "failed_test_manifest_validation"
            status["stop_reason"] = "final_test_record_count_does_not_match_manifest"
            status["final_test"] = None
            status["finished_utc"] = datetime.now(timezone.utc).isoformat()
            status["elapsed_seconds_total"] = time.monotonic() - started
            _write_status(output_dir, status)
            print("STOP " + status["stop_reason"], flush=True)
            return 2
        test_windows = build_context_windows(test_records, tokenizer, 128)
        test_batches = make_lm_batches(test_windows, BATCH_SIZE, seed=seed, shuffle=False)
        final_test = evaluate_batches(model, test_batches)
        test_event = {"type": "final_test_evaluation", "step": steps_completed, "test": final_test, "tuning_after_test": False}
        status["final_test"] = final_test
        status["events"].append(test_event)
        _jsonl_append(metrics_path, test_event)
    else:
        status["final_test"] = None
        status["test_evaluation_skipped_reason"] = "run did not complete 100 optimizer steps under all stop conditions"

    status["finished_utc"] = datetime.now(timezone.utc).isoformat()
    status["elapsed_seconds_total"] = time.monotonic() - started
    _write_status(output_dir, status)
    print("RESULT " + json.dumps(status, sort_keys=True), flush=True)
    return 0 if status["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(run())
