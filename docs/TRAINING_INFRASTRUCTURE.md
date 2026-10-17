# Duo v0 — Reusable Training and Evaluation Infrastructure

**Status:** Implemented and verified with synthetic fixtures only. This adds a reusable CPU-first training runner, deterministic resume state, a distinct inference artifact, and standalone validation/test evaluation. It is not a new GSM8K training experiment, a full-scale training claim, a production inference service, or evidence of conversational capability.

## Configuration and invocation

`configs/training.duo_v0.example.json` is an explicit example run configuration, separate from `configs/duo_v0.json` (the model/data architecture configuration). It records the model config path; train and validation paths; optional dataset manifest; output directory; batch and context sizes; AdamW learning rate, weight decay, betas and epsilon; seed; validation and checkpoint intervals; target maximum optimizer steps; device; gradient clipping; incomplete-batch policy; and thread count. The example points at the existing Git-ignored local GSM8K sample. Its presence is not permission to start training, and it has **not** been run in this milestone.

Run relative paths from the repository root. A training invocation requires an explicit config. The CLI distinguishes the original JSON values from each invocation's `--set` runtime overrides and records both plus the effective configuration in `run_metadata.json`.

```sh
# Starts training; do this only after a separate experiment approval.
python -m brain.training train --config configs/training.duo_v0.example.json

# Runtime overrides are explicit and recorded; max_steps is a total target.
python -m brain.training train \
  --config configs/training.duo_v0.example.json \
  --set max_steps=120 \
  --set batch_size=4

# Resume from the most recent atomic training checkpoint, extending the total target.
python -m brain.training train \
  --config configs/training.duo_v0.example.json \
  --set max_steps=120 \
  --resume checkpoints/duo_v0/training/default/latest.resume.pt
```

The selected `context_length` must equal `model_config.model.context_length`; change the model config to choose another architecture context. Supported run devices are CPU and CUDA (the current verified environment is CPU-only). The runner enables deterministic backend operations and restores process RNG/backend flags after it returns. Its training sampler uses a reproducible per-epoch permutation, records the exact order/cursor, and retains the final partial batch unless `drop_last` is selected. The only implemented optimizer is AdamW; there is no scheduler, mixed precision, distributed training, automatic stopping heuristic, or hyperparameter search.

## Checkpoints and resume compatibility

The existing versioned inference checkpoint format remains unchanged. A completed run produces:

- `latest.resume.pt`: a separate `ddams-duo-training-resume` version-1 payload, written atomically at each configured interval and at the target step. It contains model weights, AdamW state, originating run ID, completed step count, sampler permutation/epoch/cursor, Python and PyTorch CPU (and, when applicable, CUDA) RNG states, recent training-metric aggregation, pending metrics events, model/tokenizer metadata, dataset hashes/manifest identity, effective settings, and Python/PyTorch/device metadata.
- `final.inference.pt`: the existing portable Duo inference checkpoint, without optimizer state.
- `metrics.jsonl`, `run_metadata.json`, and `training_summary.json`: checkpoint-backed step/evaluation records, provenance/configuration, and final artifact hashes. Pending step events are stored in the resume checkpoint and published atomically only after the corresponding checkpoint is durable; resume repairs missing committed events and removes future/uncommitted entries. These outputs live under the ignored `checkpoints/` directory by default.

Resume loading uses PyTorch's restricted `weights_only=True` mode. Before restoring state, it checks architecture/configuration, tokenizer version and vocabulary hash, exact train/validation snapshot hashes and manifest, batch/optimizer/seed/device/software/thread settings, a SHA-256 fingerprint of the training/model/data implementation files, run identity, plus sampler consistency. Input files are streamed into temporary snapshots; the recorded hash is for the exact bytes parsed or loaded. An increased total `max_steps` is permitted. The configured output directory must resolve to the resume checkpoint's parent directory so logs and run identity cannot be mixed with another run; moving the complete run directory together is supported. Incompatible code, data, or math/runtime settings are rejected. It does not silently load arbitrary pickled objects.

## Independent evaluation and local generation

Training reads only the configured train and validation splits; it never selects or evaluates the test split automatically. A separate command requires the split name and data file explicitly, loads the inference checkpoint in that process, checks a supplied or adjacent manifest against the selected split when available, and writes a JSON report. With no explicit output path it creates a timestamped report beside the checkpoint.

```sh
python -m brain.training evaluate \
  --checkpoint checkpoints/duo_v0/training/default/final.inference.pt \
  --split validation \
  --data datasets/processed/gsm8k-main-train-first128/validation.jsonl \
  --manifest datasets/processed/gsm8k-main-train-first128/manifest.json
```

The report records whether results are measured (not estimated), the exact checkpoint snapshot SHA-256, selected split and exact data snapshot SHA-256, record count, manifest SHA-256 and dataset identity when available, tokenizer version/vocabulary hash, model config, context/batch/device/loss method, Python/PyTorch versions, token-weighted next-token NLL, perplexity, and scored target-token count. PAD targets are excluded. Evaluation runs in inference mode, does not compute gradients, and preserves the model's parameter values and prior train/eval mode; a model/device mismatch is rejected clearly. The model package also now has a minimal deterministic greedy decoder (`generate_greedy`) for local ID generation and checkpoint verification; it is not a chat service or HTTP API.

## Verification and scope

`tests/test_training_infrastructure.py` uses temporary synthetic JSONL files only. Its fourteen tests cover strict run config and runtime overrides, sampler ordering/cursor restoration and partial batches, immutable snapshot hashes, token-weighted PAD-safe read-only evaluation and device mismatch, a synthetic manifest check, restricted resume checkpoint contents, implementation/config incompatibility, malformed saved run-config/nested RNG/checkpoint/metadata handling, output-path safety, checkpoint-backed metrics repair after a simulated crash, exact interrupted/resumed versus uninterrupted CPU weights with non-zero dropout, standalone JSON-report behavior, explicit fresh-process test-split CLI summary, training CLI invocation, and separate-process inference-logit/greedy-generation equivalence. The pre-existing 54 tests cover the original foundation/model/data acceptance behaviors.

The full repository suite, Python compilation, both CLI help paths, example-config validation and `git diff --check` are run as milestone checks. No real dataset training, benchmark, hyperparameter selection, or real GSM8K test-split evaluation is part of this infrastructure milestone. A synthetic test split is evaluated only by a unit test of the explicit evaluation command. The only optimizer steps exercised by the new infrastructure are tiny synthetic fixtures. The historic 100-step smoke run documented in [`DUO_V0.md`](DUO_V0.md) is unchanged and remains the only prior GSM8K subset run.

**Next milestone:** define and request separate approval for one controlled training experiment using these interfaces; do not execute it until approved.
