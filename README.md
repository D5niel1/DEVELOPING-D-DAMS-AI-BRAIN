# D-DAMS AI Brain

D-DAMS AI Brain is the independent research codebase for the Duo language-model project. The repository is currently at the **Phase 1 foundation** stage. A deterministic, fixed-vocabulary UTF-8 byte tokenizer and validated configuration foundation are implemented and tested. No model, dataset, training pipeline, inference engine, checkpoint, or API has been implemented or verified.

## Current implemented milestone

A dependency-free configuration foundation is available through `brain.config`. The tokenizer API is `brain.tokenizer.ByteTokenizer`; its fixed vocabulary has 260 entries (special IDs 0–3 and UTF-8 byte IDs 4–259). See [`docs/TOKENIZER.md`](docs/TOKENIZER.md) for its rules and artifact format. The example at `configs/duo_v0.json` is a **proposed configuration only**: it does not establish that Duo v0 exists or has been trained.

Run the tests with Python's standard library (Python 3.10+):

```sh
python -m unittest discover -s tests -v
```

## Repository layout

- `brain/` — configuration and tokenizer implemented; model, training, inference, memory and agent modules remain unimplemented
- `api/` — reserved for a later, gated API phase; not implemented
- `configs/` — reproducible configuration files
- `datasets/raw/` and `datasets/processed/` — reserved; no dataset included
- `checkpoints/` — reserved; no model artifacts included
- `tests/` — standard-library tests
- `docs/` — engineering records

## Roadmap and gates

Implement and verify Phase 1 (configuration, tokenizer, data, model, training, tests) before creating an official Duo v0 checkpoint. Verify training, fresh-process reload, deterministic evaluation and standalone inference before beginning an API. No Manus-specific runtime dependency is intended.

See [`docs/FOUNDATION_BASELINE.md`](docs/FOUNDATION_BASELINE.md) for the initial repository audit and exact current status.
