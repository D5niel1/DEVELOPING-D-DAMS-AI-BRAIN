"""JSONL loading and deterministic, record-isolated language-model batches."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch

from brain.tokenizer import ByteTokenizer

MAX_CONTEXT_LENGTH = 128
PAD_ID = 0
BOS_ID = 2
EOS_ID = 3


class TextDataError(ValueError):
    """Raised when a JSONL text dataset is missing or malformed."""


@dataclass(frozen=True)
class LanguageModelBatch:
    """Right-padded next-token batch; input and target tensors are aligned."""

    input_ids: torch.Tensor
    targets: torch.Tensor
    attention_mask: torch.Tensor
    target_mask: torch.Tensor


def load_jsonl_text(path: str | Path, *, text_field: str = "text") -> list[str]:
    """Load all records from JSONL without silently discarding or cleaning text."""
    source = Path(path)
    if not isinstance(text_field, str) or not text_field:
        raise ValueError("text_field must be a non-empty string")
    try:
        content = source.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise TextDataError(f"dataset file does not exist: {source}") from exc
    except UnicodeDecodeError as exc:
        raise TextDataError(f"dataset is not valid UTF-8: {source}:{exc.start}") from exc
    except OSError as exc:
        raise TextDataError(f"cannot read dataset file {source}: {exc}") from exc

    records: list[str] = []
    for line_number, line in enumerate(content.splitlines(), start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TextDataError(f"malformed JSONL at {source}:{line_number}: {exc.msg}") from exc
        if not isinstance(row, dict):
            raise TextDataError(f"JSONL record at {source}:{line_number} must be an object")
        if text_field not in row:
            raise TextDataError(f"missing text field {text_field!r} at {source}:{line_number}")
        text = row[text_field]
        if not isinstance(text, str):
            raise TextDataError(f"text field {text_field!r} must be a string at {source}:{line_number}")
        records.append(text)
    return records


def build_context_windows(
    records: Iterable[str], tokenizer: ByteTokenizer, context_length: int = MAX_CONTEXT_LENGTH
) -> list[list[int]]:
    """Encode each record independently and split into overlapping next-token windows.

    BOS and EOS are inserted once per source record. Records are never concatenated,
    so no next-token target crosses a record boundary. Each returned token window
    contains at most ``context_length + 1`` IDs: after the one-token input/target
    shift, model input length is at most ``context_length``. Adjacent windows from
    a long record overlap by one token so every within-record next-token pair is
    represented exactly once. No text/token is silently truncated or dropped.
    """
    if type(context_length) is not int or not 1 <= context_length <= MAX_CONTEXT_LENGTH:
        raise ValueError(f"context_length must be an integer in [1, {MAX_CONTEXT_LENGTH}]")
    windows: list[list[int]] = []
    for index, text in enumerate(records):
        if not isinstance(text, str):
            raise TypeError(f"record {index} must be a string")
        ids = [BOS_ID, *tokenizer.encode(text), EOS_ID]
        # There is at least one transition (BOS -> EOS for an empty text).
        for start in range(0, len(ids) - 1, context_length):
            window = ids[start : start + context_length + 1]
            if len(window) >= 2:
                windows.append(window)
    return windows


def shift_tokens(
    token_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return inputs, next-token targets, input mask, and target/loss mask."""
    if not isinstance(token_ids, torch.Tensor) or token_ids.ndim != 2:
        raise ValueError("token_ids must be a rank-2 [batch, sequence] tensor")
    if token_ids.shape[0] == 0 or token_ids.shape[1] < 2:
        raise ValueError("token_ids must have a non-empty batch and sequence length >= 2")
    if attention_mask is None:
        valid = torch.ones_like(token_ids, dtype=torch.bool)
    else:
        if not isinstance(attention_mask, torch.Tensor) or attention_mask.shape != token_ids.shape:
            raise ValueError("attention_mask must have the same [batch, sequence] shape as token_ids")
        if attention_mask.dtype not in (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("attention_mask must have a boolean or integer dtype")
        valid = attention_mask.to(device=token_ids.device, dtype=torch.bool)
    return token_ids[:, :-1], token_ids[:, 1:], valid[:, :-1], valid[:, 1:]


def make_lm_batches(
    windows: Sequence[Sequence[int]],
    batch_size: int,
    *,
    seed: int = 1337,
    shuffle: bool = False,
) -> list[LanguageModelBatch]:
    """Create deterministic right-padded batches and shift them for next-token LM."""
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if type(shuffle) is not bool:
        raise ValueError("shuffle must be a boolean")
    checked: list[list[int]] = []
    for index, window in enumerate(windows):
        ids = list(window)
        if len(ids) < 2:
            raise ValueError(f"window {index} must contain at least two token IDs")
        if len(ids) > MAX_CONTEXT_LENGTH + 1:
            raise ValueError(f"window {index} exceeds context_length + 1 ({MAX_CONTEXT_LENGTH + 1})")
        if any(type(token_id) is not int or not 0 <= token_id < 260 for token_id in ids):
            raise ValueError(f"window {index} contains an invalid token ID")
        checked.append(ids)
    order = list(range(len(checked)))
    if shuffle:
        random.Random(seed).shuffle(order)

    batches: list[LanguageModelBatch] = []
    for start in range(0, len(order), batch_size):
        selected = [checked[i] for i in order[start : start + batch_size]]
        if not selected:
            continue
        width = max(map(len, selected))
        padded = torch.full((len(selected), width), PAD_ID, dtype=torch.long)
        mask = torch.zeros((len(selected), width), dtype=torch.bool)
        for row, ids in enumerate(selected):
            padded[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            mask[row, : len(ids)] = True
        inputs, targets, input_mask, target_mask = shift_tokens(padded, mask)
        batches.append(LanguageModelBatch(inputs, targets, input_mask, target_mask))
    return batches


def load_lm_batches(
    path: str | Path,
    tokenizer: ByteTokenizer,
    context_length: int = MAX_CONTEXT_LENGTH,
    batch_size: int = 16,
    *,
    seed: int = 1337,
    shuffle: bool = False,
    text_field: str = "text",
) -> list[LanguageModelBatch]:
    """Load JSONL and make deterministic, record-isolated next-token batches."""
    records = load_jsonl_text(path, text_field=text_field)
    windows = build_context_windows(records, tokenizer, context_length)
    return make_lm_batches(windows, batch_size, seed=seed, shuffle=shuffle)
