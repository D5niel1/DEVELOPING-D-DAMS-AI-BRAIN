# D-DAMS AI Brain

D-DAMS AI Brain is the research codebase for the Duo language-model project. It has a validated configuration system, deterministic `duo-byte-v1` tokenizer, reproducible local dataset pipeline, and an implemented/tested small Duo v0 decoder-only Transformer. One bounded 100-step CPU smoke run was completed earlier; its trained checkpoint is local and Git-ignored. Reusable training/resume, deterministic standalone evaluation, and a minimal local greedy-ID decoder are now implemented and tested using synthetic fixtures only. This does not establish conversational capability. No production inference service, API, or agents are claimed.

## Current implemented milestone

`brain.model.DuoTransformer` is a local PyTorch CPU model with a causal attention mask, learned positional embeddings, pre-LayerNorm blocks, GELU MLPs, tied token/output weights, masked next-token loss, and versioned local checkpoint save/load. Its default 260-token, 128-context configuration contains **124,928 trainable parameters**. Architecture and test evidence are in [`docs/DUO_V0.md`](docs/DUO_V0.md).

The model package also has `generate_greedy` for deterministic local token-ID generation; it is not a chat service. The separate CLI `python -m brain.training` provides configurable AdamW training, atomic optimizer/RNG/sampler resume checkpoints, and validation/test JSON evaluation reports. The example run config is `configs/training.duo_v0.example.json`. The training CLI was exercised only on temporary synthetic data in unit tests; it has not run a new GSM8K experiment. See [`docs/TRAINING_INFRASTRUCTURE.md`](docs/TRAINING_INFRASTRUCTURE.md) before any future experiment. Running the training command starts training and requires separate approval.

The configuration API is `brain.config`; the tokenizer API is `brain.tokenizer.ByteTokenizer`. The fixed tokenizer vocabulary has 260 entries (special IDs 0–3 and UTF-8 byte IDs 4–259); see [`docs/TOKENIZER.md`](docs/TOKENIZER.md).

The dataset pipeline is available through `python -m brain.data.build`; it processes local JSONL or one-record-per-line UTF-8 text, writes deterministic splits, and records statistics and hashes. See [`docs/DATA_PIPELINE.md`](docs/DATA_PIPELINE.md). The bounded GSM8K sample's provenance and MIT license evidence are recorded in [`docs/DATASET_GSM8K_SAMPLE.md`](docs/DATASET_GSM8K_SAMPLE.md). Raw and processed rows remain Git-ignored and must exist locally before the sample paths in the example configs can be loaded. The Hugging Face connector remains disabled.

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

Tests exercise forward shapes and finite logits, causal invariance, padding and target masks, a real finite backward pass, deterministic initialization, parameter count, fresh-process checkpoint/reload and generation, synthetic optimizer-state resume equivalence, and evaluation-report behavior. Test checkpoints, synthetic data, and all default training outputs are kept outside tracked source or under Git-ignored paths.

## Repository layout

- `brain/config.py` — strict model/data/training configuration
- `brain/tokenizer/` — deterministic versioned byte tokenizer
- `brain/data/` — local dataset pipeline plus JSONL loader, record-isolated context windows, and batches
- `brain/model/duo.py` — local Duo v0 Transformer, masked loss, model checkpoint helpers, and greedy token decoder
- `brain/training/` — strict run config, deterministic sampler, reusable trainer/resume checkpoints, standalone evaluator and CLI
- `brain/training/smoke.py` — historical bounded CPU smoke runner; unchanged in this milestone
- `brain/inference/`, `brain/agent/`, `brain/memory/` — reserved; not implemented
- `api/` — reserved for a later, gated API phase; not implemented
- `configs/` — model, data, and training configuration examples
- `datasets/raw/` and `datasets/processed/` — local, Git-ignored data only
- `tests/` — standard-library and model/backend acceptance tests
- `docs/` — engineering records

## Roadmap and gates

Duo v0's architecture gate covers implementation, forward/backward behavior, causal masking, and fresh-process checkpoint reload. The historic smoke run verified 100 optimizer steps on the small local GSM8K sample; the new trainer's resume/evaluation/generation infrastructure has been verified with synthetic fixtures only. Neither establishes conversational capability or useful language quality. Production inference, API, keys, agents, and deployment remain unimplemented. The smoke command is `python -m brain.training.smoke`; the reusable trainer is `python -m brain.training train`. Running either again is a new experiment and requires separate approval. Manus is not a runtime dependency.

See [`docs/FOUNDATION_BASELINE.md`](docs/FOUNDATION_BASELINE.md) for the initial repository audit and current foundation status.
