"""Validated, dependency-free configuration for the D-DAMS AI Brain foundation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping, TypeVar


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    context_length: int
    embedding_dim: int
    num_layers: int
    num_heads: int
    feed_forward_dim: int
    dropout: float = 0.0

    def validate(self) -> None:
        for name in ("vocab_size", "context_length", "embedding_dim", "num_layers", "num_heads", "feed_forward_dim"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"model.{name} must be a positive integer")
        if self.embedding_dim % self.num_heads:
            raise ValueError("model.embedding_dim must be divisible by model.num_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("model.dropout must be in [0, 1)")


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float
    batch_size: int
    max_steps: int
    seed: int
    optimizer: str = "adamw"
    validation_interval: int = 100

    def validate(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("training.learning_rate must be positive")
        for name in ("batch_size", "max_steps", "validation_interval"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"training.{name} must be a positive integer")
        if not isinstance(self.optimizer, str) or not self.optimizer.strip():
            raise ValueError("training.optimizer must not be empty")


@dataclass(frozen=True)
class DataConfig:
    train_path: str
    validation_path: str
    test_path: str

    def validate(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"data.{field.name} must not be empty")


@dataclass(frozen=True)
class BrainConfig:
    version: str
    tokenizer_version: str
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    checkpoint_dir: str

    def validate(self) -> None:
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("version must not be empty")
        if not isinstance(self.tokenizer_version, str) or not self.tokenizer_version.strip():
            raise ValueError("tokenizer_version must not be empty")
        if not isinstance(self.checkpoint_dir, str) or not self.checkpoint_dir.strip():
            raise ValueError("checkpoint_dir must not be empty")
        self.model.validate()
        self.training.validate()
        self.data.validate()
        if self.model.vocab_size < 4:
            raise ValueError("model.vocab_size must reserve at least four special-token IDs")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping with stable field ordering."""
        return asdict(self)

    def save(self, path: str | Path) -> None:
        self.validate()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BrainConfig":
        expected = {"version", "tokenizer_version", "model", "training", "data", "checkpoint_dir"}
        _check_keys(value, expected, "config")
        config = cls(
            version=_as_string(value["version"], "version"),
            tokenizer_version=_as_string(value["tokenizer_version"], "tokenizer_version"),
            model=_from_section(ModelConfig, value["model"], "model"),
            training=_from_section(TrainingConfig, value["training"], "training"),
            data=_from_section(DataConfig, value["data"], "data"),
            checkpoint_dir=_as_string(value["checkpoint_dir"], "checkpoint_dir"),
        )
        config.validate()
        return config

    @classmethod
    def load(cls, path: str | Path) -> "BrainConfig":
        try:
            value = json.loads(Path(path).read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON configuration: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError("configuration root must be a JSON object")
        return cls.from_dict(value)


T = TypeVar("T")


def _check_keys(value: Any, expected: set[str], section: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{section} must be a JSON object")
    missing = expected - value.keys()
    unknown = value.keys() - expected
    if missing:
        raise ValueError(f"{section} missing fields: {', '.join(sorted(missing))}")
    if unknown:
        raise ValueError(f"{section} has unknown fields: {', '.join(sorted(unknown))}")


def _from_section(cls: type[T], value: Any, section: str) -> T:
    expected = {item.name for item in fields(cls)}
    _check_keys(value, expected, section)
    try:
        return cls(**value)
    except TypeError as exc:
        raise ValueError(f"invalid fields in {section}: {exc}") from exc


def _as_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    return value
