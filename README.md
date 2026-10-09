# D-DAMS AI Brain

D-DAMS AI Brain is the independent research codebase for the Duo language-model project. The repository is currently at the **Phase 1 foundation** stage. A deterministic, fixed-vocabulary UTF-8 byte tokenizer, validated configuration foundation, and local reproducible dataset pipeline are implemented and tested. A bounded 128-record GSM8K sample is imported locally for pipeline and task-specific experiment readiness; the data is not committed. No model, training loop, inference engine, checkpoint, or API has been implemented or verified.

## Current implemented milestone

A dependency-free configuration foundation is available through `brain.config`. The tokenizer API is `brain.tokenizer.ByteTokenizer`; its fixed vocabulary has 260 entries (special IDs 0–3 and UTF-8 byte IDs 4–259). See [`docs/TOKENIZER.md`](docs/TOKENIZER.md) for its rules and artifact format. The example at `configs/duo_v0.json` is a **proposed configuration only**: it does not establish that Duo v0 exists or has been trained.

The dataset pipeline is available through `python -m brain.data.build`; it processes local JSONL or one-record-per-line UTF-8 text, writes deterministic splits, and records statistics and file hashes. See [`docs/DATA_PIPELINE.md`](docs/DATA_PIPELINE.md). The Hugging Face connector remains disabled; the later GSM8K sample used public unauthenticated endpoints and has separately documented license evidence.

The Phase 1D GSM8K sample's source revision, MIT license evidence, retrieval hashes, processing results and limitations are recorded in [`docs/DATASET_GSM8K_SAMPLE.md`](docs/DATASET_GSM8K_SAMPLE.md). The sample-specific import config is `configs/dataset.gsm8k.sample.json`; raw and processed rows remain Git-ignored. Public unauthenticated Hugging Face API access worked, but the Hugging Face connector itself remains disabled.

Run the tests with Python's standard library (Python 3.10+):

```sh
python -m unittest discover -s tests -v
```

## Repository layout

- `brain/` — configuration and tokenizer implemented; model, training, inference, memory and agent modules remain unimplemented
- `api/` — reserved for a later, gated API phase; not implemented
- `configs/` — reproducible configuration files
- `brain/data/` — local data validation, cleaning, deduplication, splitting, statistics and provenance
- `datasets/raw/` and `datasets/processed/` — reserved; no dataset included
- `checkpoints/` — reserved; no model artifacts included
- `tests/` — standard-library tests
- `docs/` — engineering records

## Roadmap and gates

Implement and verify Phase 1 (configuration, tokenizer, data, model, training, tests) before creating an official Duo v0 checkpoint. Verify training, fresh-process reload, deterministic evaluation and standalone inference before beginning an API. No Manus-specific runtime dependency is intended.

See [`docs/FOUNDATION_BASELINE.md`](docs/FOUNDATION_BASELINE.md) for the initial repository audit and exact current status.
