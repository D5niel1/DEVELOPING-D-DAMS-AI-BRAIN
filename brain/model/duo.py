"""Duo v0: a small, local-only, decoder-only causal Transformer."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from brain.config import BrainConfig
from brain.tokenizer import ByteTokenizer

CHECKPOINT_FORMAT = "ddams-duo-checkpoint"
CHECKPOINT_VERSION = 1


def build_causal_allowed_mask(
    attention_mask: torch.Tensor | None,
    *,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
) -> torch.Tensor:
    """Build `[B,1,T,T]` mask where True means a key may be attended to.

    Rows are query positions and columns are key positions. The lower-triangular
    causal condition blocks future keys; `attention_mask` additionally blocks pad
    keys. Padded query rows are safely zeroed in model outputs after each block.
    """
    if sequence_length < 1 or batch_size < 1:
        raise ValueError("batch_size and sequence_length must be positive")
    causal = torch.ones((sequence_length, sequence_length), dtype=torch.bool, device=device).tril()
    allowed = causal.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, sequence_length, sequence_length)
    if attention_mask is not None:
        if not isinstance(attention_mask, torch.Tensor) or attention_mask.shape != (batch_size, sequence_length):
            raise ValueError("attention_mask must have shape [batch_size, sequence_length]")
        if attention_mask.dtype not in (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("attention_mask must have a boolean or integer dtype")
        valid_keys = attention_mask.to(device=device, dtype=torch.bool)
        allowed = allowed & valid_keys[:, None, None, :]
    return allowed


class CausalSelfAttention(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.qkv = nn.Linear(embedding_dim, 3 * embedding_dim, bias=True)
        self.output = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
        batch, sequence, width = x.shape
        qkv = self.qkv(x).view(batch, sequence, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        scores = (query @ key.transpose(-2, -1)) * (self.head_dim ** -0.5)
        # Use a finite sentinel then explicitly zero and renormalize disallowed
        # entries. This avoids NaNs for all-padding rows whose allowed set is empty.
        scores = scores.masked_fill(~allowed, -torch.finfo(scores.dtype).max)
        weights = torch.softmax(scores, dim=-1).masked_fill(~allowed, 0.0)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)
        attended = self.dropout(weights) @ value
        attended = attended.transpose(1, 2).contiguous().view(batch, sequence, width)
        return self.output(attended)


class TransformerBlock(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embedding_dim, eps=config.norm_eps)
        self.attention = CausalSelfAttention(config.embedding_dim, config.num_heads, config.dropout)
        self.norm2 = nn.LayerNorm(config.embedding_dim, eps=config.norm_eps)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.embedding_dim, config.feed_forward_dim),
            nn.GELU(),
            nn.Linear(config.feed_forward_dim, config.embedding_dim),
        )
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, allowed: torch.Tensor, query_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.residual_dropout(self.attention(self.norm1(x), allowed))
        x = x * query_mask.unsqueeze(-1)
        x = x + self.residual_dropout(self.feed_forward(self.norm2(x)))
        return x * query_mask.unsqueeze(-1)


class DuoTransformer(nn.Module):
    """Configurable Duo v0 decoder; no hosted model/API or network use."""

    def __init__(self, config: BrainConfig, *, seed: int | None = None) -> None:
        super().__init__()
        config.validate()
        ByteTokenizer().assert_compatible(config)
        self.config = config
        self.model_config = config.model
        self.init_seed = config.training.seed if seed is None else seed
        if type(self.init_seed) is not int:
            raise ValueError("seed must be an integer")

        # Isolate model initialization from the caller's global RNG while making
        # the complete initialization reproducible from the documented seed.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.init_seed)
            m = config.model
            self.token_embedding = nn.Embedding(m.vocab_size, m.embedding_dim, padding_idx=0)
            self.position_embedding = nn.Embedding(m.context_length, m.embedding_dim)
            self.embedding_dropout = nn.Dropout(m.dropout)
            self.blocks = nn.ModuleList([TransformerBlock(m) for _ in range(m.num_layers)])
            self.final_norm = nn.LayerNorm(m.embedding_dim, eps=m.norm_eps)
            self.output_projection = None if m.tie_embeddings else nn.Linear(m.embedding_dim, m.vocab_size, bias=True)
            self.apply(self._initialize_module)

    @staticmethod
    def _initialize_module(module: nn.Module) -> None:
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
            raise ValueError("input_ids must be a rank-2 [batch, sequence] tensor")
        batch, sequence = input_ids.shape
        if batch < 1 or sequence < 1:
            raise ValueError("input_ids must have non-empty batch and sequence dimensions")
        if sequence > self.model_config.context_length:
            raise ValueError(
                f"sequence length {sequence} exceeds configured context_length {self.model_config.context_length}"
            )
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("input_ids must use torch.int32 or torch.int64 token IDs")
        if torch.any(input_ids < 0) or torch.any(input_ids >= self.model_config.vocab_size):
            raise ValueError(f"input_ids must be in [0, {self.model_config.vocab_size})")
        if attention_mask is None:
            query_mask = torch.ones((batch, sequence), dtype=torch.bool, device=input_ids.device)
        else:
            if not isinstance(attention_mask, torch.Tensor) or attention_mask.shape != input_ids.shape:
                raise ValueError("attention_mask must have the same [batch, sequence] shape as input_ids")
            if attention_mask.dtype not in (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
                raise ValueError("attention_mask must have a boolean or integer dtype")
            query_mask = attention_mask.to(device=input_ids.device, dtype=torch.bool)
        allowed = build_causal_allowed_mask(
            query_mask, batch_size=batch, sequence_length=sequence, device=input_ids.device
        )
        positions = torch.arange(sequence, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids.long()) + self.position_embedding(positions)
        x = self.embedding_dropout(x) * query_mask.unsqueeze(-1)
        for block in self.blocks:
            x = block(x, allowed, query_mask)
        x = self.final_norm(x) * query_mask.unsqueeze(-1)
        if self.output_projection is None:
            logits = F.linear(x, self.token_embedding.weight)
        else:
            logits = self.output_projection(x)
        return logits * query_mask.unsqueeze(-1)


def masked_next_token_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor | None = None,
    *,
    pad_id: int = 0,
) -> torch.Tensor:
    """Cross-entropy excluding padding targets; zero-valid batches return zero."""
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise ValueError("logits must have shape [batch, sequence, vocabulary]")
    if not isinstance(targets, torch.Tensor) or targets.shape != logits.shape[:2]:
        raise ValueError("targets must have shape [batch, sequence] matching logits")
    if not 0 <= pad_id < logits.shape[-1]:
        raise ValueError("pad_id must be a valid vocabulary ID")
    if target_mask is None:
        valid = targets.ne(pad_id)
    else:
        if not isinstance(target_mask, torch.Tensor) or target_mask.shape != targets.shape:
            raise ValueError("target_mask must have the same [batch, sequence] shape as targets")
        valid = target_mask.to(device=targets.device, dtype=torch.bool) & targets.ne(pad_id)
    count = valid.sum()
    if count.item() == 0:
        return logits.sum() * 0.0
    per_token = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none")
    selected = valid.reshape(-1).to(device=per_token.device)
    return per_token[selected].mean()


def save_checkpoint(model: DuoTransformer, path: str | Path) -> None:
    """Atomically save config, seed, version and tensor state to a local checkpoint."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "format_version": CHECKPOINT_VERSION,
        "config": model.config.to_dict(),
        "tokenizer_version": ByteTokenizer.version,
        "tokenizer_vocabulary_sha256": ByteTokenizer().vocabulary_sha256(),
        "init_seed": model.init_seed,
        "state_dict": model.state_dict(),
    }
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(fd)
    try:
        torch.save(payload, temporary_name)
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def load_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> DuoTransformer:
    """Load a data-only checkpoint with PyTorch's restricted weights-only loader."""
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"cannot load Duo checkpoint: {exc}") from exc
    expected_fields = {
        "format", "format_version", "config", "tokenizer_version",
        "tokenizer_vocabulary_sha256", "init_seed", "state_dict",
    }
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise ValueError("malformed Duo checkpoint fields")
    if (
        payload["format"] != CHECKPOINT_FORMAT
        or type(payload["format_version"]) is not int
        or payload["format_version"] != CHECKPOINT_VERSION
    ):
        raise ValueError("unsupported Duo checkpoint format/version")
    config_data = payload["config"]
    if not isinstance(config_data, dict):
        raise ValueError("checkpoint config must be a JSON-compatible object")
    config = BrainConfig.from_dict(config_data)
    tokenizer = ByteTokenizer()
    tokenizer.assert_compatible(config)
    if payload["tokenizer_version"] != tokenizer.version:
        raise ValueError("checkpoint tokenizer version does not match installed tokenizer")
    if payload["tokenizer_vocabulary_sha256"] != tokenizer.vocabulary_sha256():
        raise ValueError("checkpoint tokenizer vocabulary hash does not match installed tokenizer")
    seed = payload["init_seed"]
    if type(seed) is not int:
        raise ValueError("checkpoint init_seed must be an integer")
    model = DuoTransformer(config, seed=seed)
    if not isinstance(payload["state_dict"], dict):
        raise ValueError("checkpoint state_dict must be a tensor mapping")
    try:
        model.load_state_dict(payload["state_dict"], strict=True)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(f"checkpoint state_dict is incompatible: {exc}") from exc
    model.eval()
    return model
