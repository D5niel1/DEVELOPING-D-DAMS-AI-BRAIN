# D-DAMS AI Brain — Foundation Baseline

**Initial audit date:** 2026-10-08
**Initially audited revision:** `ad895e2eb501a5a3e33133c7bc24d07e4af08068` (`Initial commit`)

## Historical initial audit

The repository initially held only a README placeholder and generic `.gitignore`. No model, tokenizer, dataset, training or inference system, API, checkpoint, dependencies, or tests were present.

At the initial audit, `python -m pytest -q` could not run because pytest was not installed. The repository then had no tests. The foundation tests use Python's `unittest`; the model milestone adds the declared PyTorch CPU backend.

## Current implemented foundation

- `brain.config`: validated configuration system with Duo architecture constraints.
- `brain.tokenizer.ByteTokenizer`: deterministic fixed UTF-8 byte vocabulary, stable special IDs, versioned save/load, malformed-artifact checks and config compatibility (see [`TOKENIZER.md`](TOKENIZER.md)).
- `brain.data`: local JSONL/text ingestion, cleaning, exact deduplication, deterministic splits and provenance hashes (see [`DATA_PIPELINE.md`](DATA_PIPELINE.md)).
- `brain.data.language_model`: JSONL `text` loader, explicit per-record BOS/EOS policy, deterministic overlapping context windows, right-padded batches, input/target shift and masks.
- `brain.model.DuoTransformer`: small local pre-LayerNorm causal decoder with learned positions, 2 layers, 4 heads, width 64, FFN 256, 260-token vocabulary and tied output weights. Default trainable parameter count is 124,928.
- Versioned model checkpoint save/load includes validated config, tokenizer version/hash, initialization seed and state dict. Fresh-process output equivalence is tested.
- A bounded 128-record GSM8K question/answer sample remains locally stored under ignored raw/processed paths. Its provenance and MIT license are documented in [`DATASET_GSM8K_SAMPLE.md`](DATASET_GSM8K_SAMPLE.md); dataset files are not committed.
- `brain.training.smoke`: a deterministic, CPU-only runner capped at 100 optimizer steps or 120 seconds, with validation guards, finite-state checks, ignored logs/checkpoints, and fresh-process checkpoint verification. One such smoke run completed; detailed results are in [`DUO_V0.md`](DUO_V0.md).

## Current status and limitations

Duo v0 is **implemented and tested as an architecture; one 100-step smoke-training run is complete**. The smoke-trained checkpoint remains local and Git-ignored; the committed model source/config still initialize the base model. Passing architecture tests and the bounded loss decrease establish a functioning test/training path only; they are not evidence of conversational capability or useful language quality. No full training, optimizer-state resume, production inference, or another training run has been started.

The current configured sample paths name `train.jsonl`, `validation.jsonl`, and `test.jsonl` under the Git-ignored GSM8K processed directory. Those files must be built/present locally; a clean clone does not contain the dataset. The 128-record subset is only for pipeline validation and limited smoke work, not a general language-training corpus. General training/resume, independent evaluation reports, generation controls, API, keys, agents, and deployment remain unimplemented.

## Environment and verification

The initial environment check found CPython 3.12.3 and no PyTorch. For the model milestone, PyTorch **2.7.1+cpu** is declared in `requirements.txt` and installed in the active user environment. Verification commands and acceptance coverage are recorded in [`DUO_V0.md`](DUO_V0.md). No Manus-specific runtime dependency is introduced.
