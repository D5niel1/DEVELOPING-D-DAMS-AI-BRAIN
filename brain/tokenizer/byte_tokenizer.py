"""Fixed-vocabulary UTF-8 byte tokenizer with versioned JSON serialization."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


FORMAT_VERSION = 1
TOKENIZER_VERSION = "duo-byte-v1"
SPECIAL_TOKENS = ("<pad>", "<unk>", "<bos>", "<eos>")
SPECIAL_TOKEN_IDS = {token: i for i, token in enumerate(SPECIAL_TOKENS)}
BYTE_OFFSET = len(SPECIAL_TOKENS)
VOCAB_SIZE = BYTE_OFFSET + 256
_BYTE_TOKENS = tuple(f"<0x{i:02X}>" for i in range(256))
EXPECTED_VOCABULARY = SPECIAL_TOKENS + _BYTE_TOKENS


class TokenizerError(ValueError):
    """Raised when input or a tokenizer artifact is invalid."""


class ByteTokenizer:
    """Encode Unicode text as UTF-8 bytes using stable IDs 4..259.

    IDs 0..3 are reserved for pad, unknown, beginning-of-sequence, and
    end-of-sequence. Unknown *vocabulary spellings* map to ID 1 through
    ``token_id``; arbitrary Unicode input is representable as UTF-8 and hence
    never maps to unknown. Encoding does not implicitly add BOS/EOS.
    """

    format_version = FORMAT_VERSION
    version = TOKENIZER_VERSION
    special_tokens = SPECIAL_TOKENS
    special_token_ids = SPECIAL_TOKEN_IDS
    vocab_size = VOCAB_SIZE
    vocabulary = EXPECTED_VOCABULARY
    _token_to_id = {token: idx for idx, token in enumerate(EXPECTED_VOCABULARY)}

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if type(add_special_tokens) is not bool:
            raise TypeError("add_special_tokens must be a bool")
        ids = [BYTE_OFFSET + byte for byte in text.encode("utf-8")]
        if add_special_tokens:
            return [SPECIAL_TOKEN_IDS["<bos>"]] + ids + [SPECIAL_TOKEN_IDS["<eos>"]]
        return ids

    def decode(self, ids: list[int] | tuple[int, ...], *, skip_special_tokens: bool = True) -> str:
        if not isinstance(ids, (list, tuple)):
            raise TypeError("ids must be a list or tuple of integers")
        if type(skip_special_tokens) is not bool:
            raise TypeError("skip_special_tokens must be a bool")
        output: list[str] = []
        byte_buffer = bytearray()

        def flush_bytes() -> None:
            if byte_buffer:
                try:
                    output.append(byte_buffer.decode("utf-8", errors="strict"))
                except UnicodeDecodeError as exc:
                    raise TokenizerError(f"token IDs contain invalid UTF-8 at byte offset {exc.start}") from exc
                byte_buffer.clear()

        for position, token_id in enumerate(ids):
            if type(token_id) is not int or not 0 <= token_id < VOCAB_SIZE:
                raise TokenizerError(f"invalid token ID at position {position}: {token_id!r}")
            if token_id >= BYTE_OFFSET:
                byte_buffer.append(token_id - BYTE_OFFSET)
            else:
                flush_bytes()
                if token_id == SPECIAL_TOKEN_IDS["<unk>"]:
                    output.append("<unk>")
                elif not skip_special_tokens:
                    output.append(SPECIAL_TOKENS[token_id])
        flush_bytes()
        return "".join(output)

    def token_id(self, token: str) -> int:
        """Look up a vocabulary spelling; unknown spellings map to UNK ID 1."""
        if not isinstance(token, str):
            raise TypeError("token must be a string")
        return self._token_to_id.get(token, SPECIAL_TOKEN_IDS["<unk>"])

    def token(self, token_id: int) -> str:
        """Return a vocabulary spelling for a valid ID, otherwise raise."""
        if type(token_id) is not int or not 0 <= token_id < VOCAB_SIZE:
            raise TokenizerError(f"invalid token ID: {token_id!r}")
        return self.vocabulary[token_id]

    def assert_compatible(self, config: Any) -> None:
        """Require config tokenizer version and model vocabulary size to match."""
        model = getattr(config, "model", None)
        configured_size = getattr(model, "vocab_size", None)
        configured_version = getattr(config, "tokenizer_version", None)
        if configured_version != self.version:
            raise TokenizerError(
                f"tokenizer version {configured_version!r} does not match {self.version!r}"
            )
        if configured_size != self.vocab_size:
            raise TokenizerError(
                f"model vocab_size {configured_size!r} does not match tokenizer vocab_size {self.vocab_size}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "ddams-byte-tokenizer",
            "format_version": FORMAT_VERSION,
            "tokenizer_version": TOKENIZER_VERSION,
            "encoding": "utf-8",
            "unknown_policy": "unknown vocabulary spellings map to <unk>; input text uses UTF-8 bytes",
            "special_tokens": [{"token": token, "id": idx} for idx, token in enumerate(SPECIAL_TOKENS)],
            "vocabulary": list(self.vocabulary),
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, value: Any) -> "ByteTokenizer":
        if not isinstance(value, dict):
            raise TokenizerError("tokenizer artifact root must be a JSON object")
        expected_keys = {"format", "format_version", "tokenizer_version", "encoding", "unknown_policy", "special_tokens", "vocabulary"}
        if set(value) != expected_keys:
            missing, unknown = expected_keys - set(value), set(value) - expected_keys
            raise TokenizerError(f"invalid artifact fields; missing={sorted(missing)}, unknown={sorted(unknown)}")
        if value["format"] != "ddams-byte-tokenizer":
            raise TokenizerError("unsupported tokenizer artifact format")
        if type(value["format_version"]) is not int or value["format_version"] != FORMAT_VERSION:
            raise TokenizerError(f"unsupported tokenizer format_version: {value['format_version']!r}")
        if value["tokenizer_version"] != TOKENIZER_VERSION or value["encoding"] != "utf-8":
            raise TokenizerError("incompatible tokenizer version or encoding")
        if value["unknown_policy"] != "unknown vocabulary spellings map to <unk>; input text uses UTF-8 bytes":
            raise TokenizerError("incompatible unknown-token policy")
        special = value["special_tokens"]
        required_special = [{"token": token, "id": idx} for idx, token in enumerate(SPECIAL_TOKENS)]
        if special != required_special:
            raise TokenizerError("special tokens or their stable IDs do not match this tokenizer version")
        vocabulary = value["vocabulary"]
        if not isinstance(vocabulary, list) or any(not isinstance(token, str) for token in vocabulary):
            raise TokenizerError("vocabulary must be a list of strings")
        if len(vocabulary) != VOCAB_SIZE:
            raise TokenizerError(f"vocabulary must contain exactly {VOCAB_SIZE} tokens")
        if len(set(vocabulary)) != len(vocabulary):
            raise TokenizerError("vocabulary contains duplicate tokens")
        if tuple(vocabulary) != EXPECTED_VOCABULARY:
            raise TokenizerError("vocabulary ordering or token spellings do not match this tokenizer version")
        return cls()

    @classmethod
    def load(cls, path: str | Path) -> "ByteTokenizer":
        try:
            value = json.loads(Path(path).read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise TokenizerError(f"malformed tokenizer JSON artifact: {exc}") from exc
        return cls.from_dict(value)

    def vocabulary_sha256(self) -> str:
        payload = json.dumps(list(self.vocabulary), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
