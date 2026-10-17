"""Deterministic window ordering with an explicit, checkpointable cursor."""
from __future__ import annotations

import random
from typing import Any


class DeterministicBatchSampler:
    """Yield index batches, retaining incomplete batches unless drop_last is set."""

    STATE_VERSION = 1

    def __init__(self, size: int, batch_size: int, seed: int, *, drop_last: bool = False) -> None:
        if type(size) is not int or size <= 0:
            raise ValueError("sampler size must be a positive integer")
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if type(drop_last) is not bool:
            raise ValueError("drop_last must be a boolean")
        if drop_last and size < batch_size:
            raise ValueError("drop_last would produce no batches for this dataset")
        self.size = size
        self.batch_size = batch_size
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        self.cursor = 0
        self.order: list[int] = []
        self._new_epoch_order()

    def _new_epoch_order(self) -> None:
        self.order = list(range(self.size))
        random.Random(self.seed + self.epoch).shuffle(self.order)
        self.cursor = 0

    def next_indices(self) -> list[int]:
        while True:
            remaining = self.size - self.cursor
            if remaining == 0:
                self.epoch += 1
                self._new_epoch_order()
                remaining = self.size
            if self.drop_last and remaining < self.batch_size:
                self.cursor = self.size
                continue
            end = min(self.size, self.cursor + self.batch_size)
            indices = self.order[self.cursor:end]
            self.cursor = end
            if indices:
                return indices

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": self.STATE_VERSION,
            "size": self.size,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "drop_last": self.drop_last,
            "epoch": self.epoch,
            "cursor": self.cursor,
            "order": list(self.order),
        }

    def load_state_dict(self, state: Any) -> None:
        expected = {"version", "size", "batch_size", "seed", "drop_last", "epoch", "cursor", "order"}
        if not isinstance(state, dict) or set(state) != expected:
            raise ValueError("malformed sampler state fields")
        for name, current in (("size", self.size), ("batch_size", self.batch_size), ("seed", self.seed), ("drop_last", self.drop_last)):
            if state[name] != current or type(state[name]) is not type(current):
                raise ValueError(f"sampler checkpoint {name} is incompatible")
        if type(state["version"]) is not int or state["version"] != self.STATE_VERSION:
            raise ValueError("unsupported sampler state version")
        epoch, cursor, order = state["epoch"], state["cursor"], state["order"]
        if type(epoch) is not int or epoch < 0:
            raise ValueError("sampler checkpoint epoch is invalid")
        if type(cursor) is not int or not 0 <= cursor <= self.size:
            raise ValueError("sampler checkpoint cursor is invalid")
        if not isinstance(order, list) or len(order) != self.size or any(type(i) is not int for i in order):
            raise ValueError("sampler checkpoint order is invalid")
        if sorted(order) != list(range(self.size)):
            raise ValueError("sampler checkpoint order is not a permutation")
        expected_order = list(range(self.size))
        random.Random(self.seed + epoch).shuffle(expected_order)
        if order != expected_order:
            raise ValueError("sampler checkpoint order does not match its seed and epoch")
        if self.drop_last and cursor < self.size and cursor % self.batch_size:
            raise ValueError("sampler checkpoint cursor is not on a batch boundary")
        self.epoch, self.cursor, self.order = epoch, cursor, list(order)
