# D-DAMS Byte Tokenizer v1

**Status:** Implemented and covered by tests; not production-qualified. This component does not imply that a language model or training system exists.

## Format and stable vocabulary

Tokenizer version: `duo-byte-v1`; serialization format: `ddams-byte-tokenizer`, format version `1`. The vocabulary is fixed, ordered and independent of training data:

| ID | Token | Meaning |
|---:|---|---|
| 0 | `<pad>` | Padding; removed from decoded output by default |
| 1 | `<unk>` | Unknown vocabulary spelling; decodes to the literal `<unk>` |
| 2 | `<bos>` | Beginning marker; not inserted unless requested |
| 3 | `<eos>` | End marker; not inserted unless requested |
| 4–259 | `<0x00>` … `<0xFF>` | One UTF-8 byte per token; ID is byte value + 4 |

The vocabulary size is **260**. It is fixed: there is no learned vocabulary construction, merge table, corpus dependency, or Unicode normalization. `encode(text)` converts text to UTF-8 and maps each byte to `byte + 4`. Consequently, all valid Python Unicode strings are representable; unknown handling applies to vocabulary-spelling lookup (`token_id`), which returns ID 1 for an absent spelling. Input must be `str`; bytes and other types raise `TypeError`. Empty text encodes to an empty list.

`encode(..., add_special_tokens=True)` wraps the byte IDs with BOS/EOS. `decode(..., skip_special_tokens=True)` skips PAD/BOS/EOS, renders UNK as `<unk>`, and strictly decodes the remaining byte runs as UTF-8. With `skip_special_tokens=False`, PAD/BOS/EOS are emitted as their literal spellings. Invalid IDs, booleans-as-IDs, and byte sequences that are not valid UTF-8 raise `TokenizerError`.

## Artifact and compatibility

`ByteTokenizer.save(path)` emits readable UTF-8 JSON containing format/version, encoding, the unknown policy, explicit special-token IDs, and the complete vocabulary. Loading validates exact fields, version, special IDs, vocabulary size, uniqueness, spelling, and order; malformed or incompatible files fail closed. The format contains no executable data. The vocabulary SHA-256 helper is available for reproducibility records.

`assert_compatible(config)` checks both `config.tokenizer_version == "duo-byte-v1"` and `config.model.vocab_size == 260`. The current `configs/duo_v0.json` is still a provisional example and is not evidence of a model. A future model must use a compatible configuration.

## Minimal use

```python
from brain.tokenizer import ByteTokenizer

tokenizer = ByteTokenizer()
ids = tokenizer.encode("Duo — café 🌍")
assert tokenizer.decode(ids) == "Duo — café 🌍"
tokenizer.save("tokenizer.json")
loaded = ByteTokenizer.load("tokenizer.json")
```

No third-party dependency is required (Python 3.10+). Verify with `python -m unittest discover -s tests -v`.
