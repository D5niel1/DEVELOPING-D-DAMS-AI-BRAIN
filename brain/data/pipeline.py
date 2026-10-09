"""Reproducible local text dataset ingestion and splitting (standard library only)."""
from __future__ import annotations

import hashlib
import json
import math
import platform
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from brain.config import BrainConfig
from brain.tokenizer import ByteTokenizer


PIPELINE_VERSION = "ddams-text-pipeline-v1"
SUPPORTED_FORMATS = {"jsonl", "txt-lines"}


class DatasetError(ValueError):
    """Raised when dataset configuration, source or processing invariants fail."""


@dataclass(frozen=True)
class DatasetConfig:
    dataset_name: str
    input_format: str = "jsonl"
    text_field: str = "text"
    seed: int = 1337
    train_fraction: float = 0.8
    validation_fraction: float = 0.1
    test_fraction: float = 0.1
    source_url: str | None = None
    license: str | None = None
    dataset_version: str | None = None
    retrieval_date: str | None = None
    license_verified: bool = False

    def validate(self) -> None:
        if not isinstance(self.dataset_name, str) or not self.dataset_name.strip():
            raise DatasetError("dataset_name must be a non-empty string")
        if not isinstance(self.input_format, str) or self.input_format not in SUPPORTED_FORMATS:
            raise DatasetError(f"input_format must be one of {sorted(SUPPORTED_FORMATS)}")
        if not isinstance(self.text_field, str) or not self.text_field:
            raise DatasetError("text_field must be a non-empty string")
        if type(self.seed) is not int:
            raise DatasetError("seed must be an integer")
        fractions = (self.train_fraction, self.validation_fraction, self.test_fraction)
        if any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in fractions):
            raise DatasetError("split fractions must be finite, positive numbers")
        if not math.isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise DatasetError("train, validation and test fractions must sum to 1")
        for name in ("source_url", "license", "dataset_version", "retrieval_date"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise DatasetError(f"{name} must be a non-empty string or null")
        if type(self.license_verified) is not bool:
            raise DatasetError("license_verified must be a boolean")
        if self.license_verified and not self.license:
            raise DatasetError("license_verified requires a documented license value")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Any) -> "DatasetConfig":
        if not isinstance(value, dict):
            raise DatasetError("dataset config root must be a JSON object")
        expected = {field for field in cls.__dataclass_fields__}
        missing, unknown = expected - value.keys(), value.keys() - expected
        if missing or unknown:
            raise DatasetError(f"invalid dataset config fields; missing={sorted(missing)}, unknown={sorted(unknown)}")
        config = cls(**value)
        config.validate()
        return config

    @classmethod
    def load(cls, path: str | Path) -> "DatasetConfig":
        try:
            value = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DatasetError(f"cannot read dataset config: {exc}") from exc
        return cls.from_dict(value)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_records(config: DatasetConfig, raw_bytes: bytes) -> tuple[list[tuple[int, Any]], int, dict[str, int]]:
    try:
        text = raw_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise DatasetError(f"input file is not valid UTF-8 at byte {exc.start}") from exc
    lines = text.splitlines()
    records: list[tuple[int, Any]] = []
    rejected: dict[str, int] = {}
    for line_number, line in enumerate(lines, start=1):
        if config.input_format == "txt-lines":
            records.append((line_number, line))
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            rejected["malformed_json"] = rejected.get("malformed_json", 0) + 1
            continue
        if not isinstance(parsed, dict):
            rejected["record_not_object"] = rejected.get("record_not_object", 0) + 1
            continue
        if config.text_field not in parsed:
            rejected["missing_text_field"] = rejected.get("missing_text_field", 0) + 1
            continue
        records.append((line_number, parsed[config.text_field]))
    return records, len(lines), rejected


def _split_counts(size: int, fractions: tuple[float, float, float]) -> tuple[int, int, int]:
    quotas = [size * part for part in fractions]
    counts = [math.floor(quota) for quota in quotas]
    remainder = size - sum(counts)
    order = sorted(range(3), key=lambda i: (-(quotas[i] - counts[i]), i))
    for idx in order[:remainder]:
        counts[idx] += 1
    # When there are at least three records, avoid a nominal split being empty.
    # Move one from the largest donor; totals remain exact.
    if size >= 3:
        for empty_idx in [i for i, count in enumerate(counts) if count == 0]:
            donors = [i for i, count in enumerate(counts) if count > 1]
            if donors:
                donor = max(donors, key=lambda i: (counts[i], fractions[i], -i))
                counts[donor] -= 1
                counts[empty_idx] += 1
    return tuple(counts)  # type: ignore[return-value]


def _text_stats(texts: list[str], tokenizer: ByteTokenizer) -> dict[str, Any]:
    lengths = [len(text) for text in texts]
    token_counts = [len(tokenizer.encode(text)) for text in texts]
    return {
        "records": len(texts),
        "characters": sum(lengths),
        "characters_min": min(lengths) if lengths else 0,
        "characters_max": max(lengths) if lengths else 0,
        "characters_mean": (sum(lengths) / len(lengths)) if lengths else 0.0,
        "byte_token_count": sum(token_counts),
        "byte_tokens_min_per_record": min(token_counts) if token_counts else 0,
        "byte_tokens_max_per_record": max(token_counts) if token_counts else 0,
        "byte_tokens_mean_per_record": (sum(token_counts) / len(token_counts)) if token_counts else 0.0,
    }


def build_dataset(
    input_path: str | Path,
    output_dir: str | Path,
    config: DatasetConfig,
    *,
    tokenizer: ByteTokenizer | None = None,
    brain_config: BrainConfig | None = None,
) -> dict[str, Any]:
    """Build deterministic train/validation/test JSONL files and provenance.

    Cleaning trims leading/trailing Unicode whitespace only. Internal spaces,
    punctuation and Unicode code points are otherwise unchanged; no Unicode
    normalization is performed. Duplicates are exact matches after this trim,
    keeping the first source occurrence. Invalid JSONL rows are rejected and
    counted. A fixed seed determines the partition permutation.
    """
    config.validate()
    source = Path(input_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if not source.is_file():
        raise DatasetError(f"input file does not exist or is not a file: {source}")
    # Refuse output directories that contain the source, preventing accidental raw-data overwrite.
    if destination == source or destination in source.parents:
        raise DatasetError("output directory must not equal or contain the input source file")

    tokenizer = tokenizer or ByteTokenizer()
    if brain_config is not None:
        tokenizer.assert_compatible(brain_config)
    try:
        raw_bytes = source.read_bytes()
    except OSError as exc:
        raise DatasetError(f"cannot read input file {source}: {exc}") from exc
    candidate_records, input_count, rejected = _read_records(config, raw_bytes)
    accepted: list[tuple[str, int, str]] = []
    seen_text: set[str] = set()
    duplicates = 0
    empty_records = 0
    wrong_type_records = 0
    for line_number, value in candidate_records:
        if not isinstance(value, str):
            rejected["text_not_string"] = rejected.get("text_not_string", 0) + 1
            wrong_type_records += 1
            continue
        cleaned = value.strip()
        if not cleaned:
            rejected["empty_after_trim"] = rejected.get("empty_after_trim", 0) + 1
            empty_records += 1
            continue
        digest = _sha256(cleaned.encode("utf-8"))
        if cleaned in seen_text:
            duplicates += 1
            continue
        seen_text.add(cleaned)
        accepted.append((cleaned, line_number, digest))

    accepted.sort(key=lambda item: item[1])
    if len(accepted) + duplicates + sum(rejected.values()) != input_count:
        raise DatasetError("input record accounting invariant failed")
    fractions = (float(config.train_fraction), float(config.validation_fraction), float(config.test_fraction))
    counts = _split_counts(len(accepted), fractions)
    order = list(range(len(accepted)))
    random.Random(config.seed).shuffle(order)
    split_names = ("train", "validation", "test")
    splits: dict[str, list[str]] = {}
    cursor = 0
    for name, count in zip(split_names, counts):
        selected = order[cursor:cursor + count]
        splits[name] = [accepted[index][0] for index in selected]
        cursor += count
    if cursor != len(accepted) or sum(map(len, splits.values())) != len(accepted):
        raise DatasetError("split records do not account for all accepted records")
    split_texts = [set(texts) for texts in splits.values()]
    if any(split_texts[i] & split_texts[j] for i in range(3) for j in range(i + 1, 3)):
        raise DatasetError("duplicate leakage detected across dataset splits")

    destination.mkdir(parents=True, exist_ok=True)
    split_files: dict[str, dict[str, Any]] = {}
    for name in split_names:
        output_path = destination / f"{name}.jsonl"
        output_bytes = "".join(
            json.dumps({"text": text}, ensure_ascii=False, separators=(",", ":")) + "\n"
            for text in splits[name]
        ).encode("utf-8")
        output_path.write_bytes(output_bytes)
        split_files[name] = {
            "path": output_path.name,
            "records": len(splits[name]),
            "sha256": _sha256(output_bytes),
        }

    stats: dict[str, Any] = {
        "pipeline_version": PIPELINE_VERSION,
        "input_records": input_count,
        "valid_text_records_before_deduplication": sum(
            1 for line_no, value in candidate_records if isinstance(value, str) and value.strip()
        ),
        "accepted_records": len(accepted),
        "rejected_records": sum(rejected.values()),
        "rejected_by_reason": dict(sorted(rejected.items())),
        "empty_records": empty_records,
        "wrong_type_records": wrong_type_records,
        "duplicate_records": duplicates,
        "split_sizes": {name: len(splits[name]) for name in split_names},
        "split_proportions_exactly_representable": all(
            math.isclose(len(splits[name]) / len(accepted), fractions[i], abs_tol=1e-12)
            for i, name in enumerate(split_names)
        ) if accepted else False,
        "text_statistics": _text_stats([item[0] for item in accepted], tokenizer),
        "split_text_statistics": {name: _text_stats(splits[name], tokenizer) for name in split_names},
    }
    manifest = {
        "manifest_version": 1,
        "pipeline_version": PIPELINE_VERSION,
        "dataset": config.to_dict(),
        "input": {
            "file_name": source.name,
            "format": config.input_format,
            "size_bytes": len(raw_bytes),
            "sha256": _sha256(raw_bytes),
            "records": input_count,
        },
        "processing": {
            "cleaning": "trim leading/trailing Unicode whitespace; preserve internal whitespace; no Unicode normalization",
            "deduplication": "exact cleaned UTF-8 text equality, first source occurrence retained",
            "split_algorithm": "Python random.Random(seed).shuffle over accepted source-order records; largest-remainder sizes; nonempty partitions when at least three records",
            "seed": config.seed,
            "python_version": platform.python_version(),
            "split_fractions": {name: fractions[i] for i, name in enumerate(split_names)},
            "tokenizer_version": tokenizer.version,
            "tokenizer_vocab_size": tokenizer.vocab_size,
        },
        "counts": {
            "input_records": input_count,
            "accepted_records": len(accepted),
            "rejected_records": sum(rejected.values()),
            "duplicate_records": duplicates,
            "split_sizes": stats["split_sizes"],
        },
        "outputs": split_files,
    }
    (destination / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"manifest": manifest, "stats": stats}
