# D-DAMS AI Brain

D-DAMS AI Brain is the independent research codebase for the Duo language-model project. The repository has a validated configuration system, deterministic `duo-byte-v1` tokenizer, reproducible local dataset pipeline, and a small **implemented and tested but untrained** Duo v0 decoder-only Transformer. No conversational capability, training result, production inference service, API, or agent is claimed.

## Current implemented milestone

`brain.model.DuoTransformer` is a local PyTorch CPU model with a causal attention mask, learned positional embeddings, pre-LayerNorm blocks, GELU MLPs, tied token/output weights, masked next-token loss, and versioned local checkpoint save/load. Its default 260-token, 128-context configuration contains **124,928 trainable parameters**. Architecture and test evidence are in [`docs/DUO_V0.md`](docs/DUO_V0.md).

The configuration API is `brain.config`; the tokenizer API is `brain.tokenizer.ByteTokenizer`. The fixed tokenizer vocabulary has 260 entries (special IDs 0–3 and UTF-8 byte IDs 4–259); see [`docs/TOKENIZER.md`](docs/TOKENIZER.md).

The dataset pipeline is available through `python -m brain.data.build`; it processes local JSONL or one-record-per-line UTF-8 text, writes deterministic splits, and records statistics and hashes. See [`docs/DATA_PIPELINE.md`](docs/DATA_PIPELINE.md). The bounded GSM8K sample's provenance and MIT license evidence are recorded in [`docs/DATASET_GSM8K_SAMPLE.md`](docs/DATASET_GSM8K_SAMPLE.md). Raw and processed rows remain Git-ignored and must exist locally before the sample paths in `configs/duo_v0.json` can be loaded. The Hugging Face connector remains disabled.

## Setup and verification

Verified on CPython 3.12.3 with PyTorch 2.7.1+cpu. Install the pinned CPU backend with:

```sh
python -m pip install -r requirements.txt
```

Run the complete CPU-first suite and basic checks:

```sh
python -m unittest discover -s tests -v
python -m compileall -q brain tests
git diff --check
```

Tests exercise forward shapes and finite logits, causal invariance, padding and target masks, a real finite backward pass, deterministic initialization, parameter count, and a fresh-Python-process checkpoint reload. The tests use temporary checkpoints; no checkpoint or dataset is committed.

## Repository layout

- `brain/config.py` — strict model/data/training configuration
- `brain/tokenizer/` — deterministic versioned byte tokenizer
- `brain/data/` — local dataset pipeline plus JSONL loader, record-isolated context windows, and batches
- `brain/model/duo.py` — local Duo v0 Transformer, masked loss, checkpoint helpers
- `brain/training/`, `brain/inference/`, `brain/agent/`, `brain/memory/` — reserved; not implemented
- `api/` — reserved for a later, gated API phase; not implemented
- `configs/` — model and dataset configuration examples
- `datasets/raw/` and `datasets/processed/` — local, Git-ignored data only
- `tests/` — standard-library and model/backend acceptance tests
- `docs/` — engineering records

## Roadmap and gates

Duo v0's **architecture acceptance gate** covers implementation, forward/backward behavior, causal masking, and fresh-process checkpoint reload. Passing it establishes a functioning testable model implementation only; it does not establish conversational capability. Training has not started. A separate controlled CPU smoke-training proposal is documented in [`docs/DUO_V0.md`](docs/DUO_V0.md) and requires explicit approval before any optimizer step. Training, reproducible evaluation, and standalone generation must be implemented and verified before any API. Manus is not a runtime dependency.

See [`docs/FOUNDATION_BASELINE.md`](docs/FOUNDATION_BASELINE.md) for the initial repository audit and current foundation status.
