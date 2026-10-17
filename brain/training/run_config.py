"""Validated run-time configuration for reusable Duo training."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class TrainingRunConfig:
    model_config_path: str
    train_path: str
    validation_path: str
    output_dir: str
    dataset_manifest_path: str | None
    batch_size: int
    context_length: int
    learning_rate: float
    weight_decay: float
    optimizer: str
    beta1: float
    beta2: float
    epsilon: float
    seed: int
    evaluation_interval: int
    checkpoint_interval: int
    max_steps: int
    device: str
    gradient_clip_norm: float
    drop_last: bool
    num_threads: int

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrainingRunConfig":
        if not isinstance(value, Mapping):
            raise ValueError("training run config must be a JSON object")
        expected = set(cls.__dataclass_fields__)
        missing, extra = expected - set(value), set(value) - expected
        if missing:
            raise ValueError(f"training run config missing fields: {', '.join(sorted(missing))}")
        if extra:
            raise ValueError(f"training run config has unknown fields: {', '.join(sorted(extra))}")
        config = cls(**dict(value))
        config.validate()
        return config

    @classmethod
    def load(cls, path: str | Path) -> "TrainingRunConfig":
        try:
            value = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot load training run config {path}: {exc}") from exc
        return cls.from_dict(value)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def with_overrides(self, overrides: Mapping[str, Any]) -> "TrainingRunConfig":
        expected = set(self.__dataclass_fields__)
        unknown = set(overrides) - expected
        if unknown:
            raise ValueError(f"unknown runtime override(s): {', '.join(sorted(unknown))}")
        updated = replace(self, **dict(overrides))
        updated.validate()
        return updated

    def validate(self) -> None:
        for name in ("model_config_path", "train_path", "validation_path", "output_dir"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.dataset_manifest_path is not None and (
            not isinstance(self.dataset_manifest_path, str) or not self.dataset_manifest_path.strip()
        ):
            raise ValueError("dataset_manifest_path must be null or a non-empty string")
        for name in ("batch_size", "context_length", "evaluation_interval", "checkpoint_interval", "max_steps", "num_threads"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.context_length > 128:
            raise ValueError("context_length must not exceed 128")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        for name in ("learning_rate", "weight_decay", "beta1", "beta2", "epsilon", "gradient_clip_norm"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("beta1 and beta2 must be in [0, 1)")
        if self.epsilon <= 0 or self.gradient_clip_norm <= 0:
            raise ValueError("epsilon and gradient_clip_norm must be positive")
        if self.optimizer != "adamw":
            raise ValueError("optimizer must be 'adamw'")
        if self.device not in ("cpu", "cuda") and not (
            isinstance(self.device, str) and self.device.startswith("cuda:") and self.device[5:].isdigit()
        ):
            raise ValueError("device must be 'cpu', 'cuda', or 'cuda:<index>'")
        if type(self.drop_last) is not bool:
            raise ValueError("drop_last must be a boolean")
