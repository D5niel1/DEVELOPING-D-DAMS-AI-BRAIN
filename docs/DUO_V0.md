# Duo v0 — Implemented Architecture and Verification

**Status:** Implemented and acceptance-tested as a local decoder-only Transformer. **Untrained.** Not evidence of conversational capability, language quality, or fitness as a product.

## Backend and setup

- Tensor/autograd backend: **PyTorch 2.7.1+cpu**, pinned in `requirements.txt`.
- Verified environment: CPython **3.12.3**, CPU-only. Other Python/PyTorch combinations have not been tested.
- Install in the supported environment with `python -m pip install -r requirements.txt`.
- Run CPU-first verification with `python -m unittest discover -s tests -v`.
- No hosted model, provider SDK, API, or key is used. No GPU or external inference endpoint is required.

## Architecture

The implementation is in `brain/model/duo.py` (`DuoTransformer`). The default configuration in `configs/duo_v0.json` sets:

| Setting | Value |
|---|---:|
| Tokenizer / vocabulary | `duo-byte-v1` / 260 |
| Context length | 128 tokens maximum |
| Embedding width | 64 |
| Decoder blocks | 2 |
| Attention heads | 4, head width 16 |
| Feed-forward width | 256 |
| Positional encoding | Learned absolute positions |
| Block style | Pre-LayerNorm residual |
| Activation | GELU |
| Normalization epsilon | 1e-5 |
| Dropout | 0.0 |
| Output | Vocabulary logits, tied token-embedding weights |
| Initialization | Embedding weights normal(0, 0.02), padding row zero; linear Xavier-uniform/zero bias; LayerNorm ones/zeros; seeded initialization isolated from caller's global RNG |

Forward contract: integer token IDs `[B,T]` with `1 ≤ B`, `1 ≤ T ≤ 128`, valid IDs `0..259` → finite next-token logits `[B,T,260]`. Invalid rank, empty dimensions, wrong dtype, out-of-range IDs, or excess context raise a clear `ValueError`.

For the default tied-weight configuration, actual trainable parameter count is **124,928**, verified against `sum(p.numel() for p in model.parameters() if p.requires_grad)`. With `tie_embeddings=false`, the optional biased output projection adds 16,900 parameters, for **141,828** total.

## JSONL input and sequence contract

The pipeline-produced sample is one UTF-8 JSON object per line with a string `text` field. The loader (`brain/data/language_model.py`) has no implicit cleaning or record dropping; missing files, malformed JSON, non-object rows, missing fields, and non-string text raise `TextDataError` with a path/line.

The example config points to the locally generated ignored files:

- `datasets/processed/gsm8k-main-train-first128/train.jsonl`
- `datasets/processed/gsm8k-main-train-first128/validation.jsonl`
- `datasets/processed/gsm8k-main-train-first128/test.jsonl`

They must first exist locally; raw/processed data are intentionally excluded from Git. Each record is tokenized independently as `[BOS] + UTF8-byte tokens + [EOS]`. Empty text becomes `[BOS, EOS]`. Records are never concatenated, preventing next-token prediction across unrelated examples.

Long records are segmented into deterministic windows of at most 129 IDs, with `context_length=128`. Adjacent windows overlap at one token: this preserves each within-record next-token transition exactly once; nothing is silently truncated or discarded. After shifting, model input and target lengths are at most 128.

`make_lm_batches` right-pads each batch with PAD ID 0, gives an attention mask for valid input positions, shifts `input_ids=tokens[:, :-1]` and `targets=tokens[:, 1:]`, and gives a target/loss mask so padded targets are excluded. Optional shuffling uses a local `random.Random(seed)`; it does not alter global RNG state. Model attention blocks future keys and all pad keys; padded query outputs are zeroed. Empty attention rows are renormalized safely rather than allowed to produce NaNs. `masked_next_token_loss` excludes masked/PAD targets and returns a differentiable zero for an all-padding batch.

## Checkpoints

`save_checkpoint(model, path)` writes a local atomic file containing checkpoint format/version, complete validated model config, tokenizer version and vocabulary SHA-256, initialization seed, and model state dict. `load_checkpoint(path, map_location="cpu")` uses PyTorch's restricted `weights_only=True` loader, verifies the payload/config/tokenizer metadata, reconstructs the model, strictly loads state, and returns it in evaluation mode for repeatable inference. The checkpoint is for model inference reproduction; it contains no optimizer/training-resume state. Test checkpoints were created only in temporary directories and were not committed.

## Acceptance verification

The tests in `tests/test_duo_model.py` and `tests/test_language_model_data.py` cover:

- `[B,T,260]` output at lengths 1, short, and 128; rejection above 128 and invalid token input.
- Lower-triangular mask orientation and padding-key/query behavior, including an all-padding batch without NaNs.
- Future-token invariance: changing tokens after position `k` leaves logits through `k` unchanged within absolute tolerance 1e-6; later logits change in the test fixture.
- Record boundaries, BOS/EOS policy, deterministic overlapping windows, no lost next-token pairs, JSONL errors, batch reproducibility, exact shift, and target-mask padding exclusion.
- Real masked loss/backward with finite loss and nonzero finite gradients for every expected trainable parameter.
- Deterministic parameter initialization under a fixed seed, without consuming caller RNG state.
- Actual parameter-count and tokenizer/model vocabulary compatibility.
- Save/load in a **separate Python process**: same inputs produce logits equal within absolute tolerance 1e-6.

Run the full acceptance suite and basic repository checks:

```sh
python -m unittest discover -s tests -v
python -m compileall -q brain tests
git diff --check
```

## Controlled training experiment proposal — not started

This proposal is separate from implementation. **No training has been run; explicit user approval is required before starting.**

- **Objective:** validate a tiny, deterministic training loop and checkpoint-resume path; check whether train next-token cross-entropy can decrease without claiming general language ability.
- **Dataset:** only the existing local GSM8K sample: 102 train / 13 validation / 13 test records. This is a tiny pipeline/smoke-test subset, not a capable conversational corpus.
- **Compute:** CPU-only on the current environment (reported 6 CPUs, 23 GiB RAM total, about 21 GiB available); no GPU/cloud service. Proposed cap: one CPU thread, batch size 4, context 128, at most 100 optimizer steps, no more than 2 minutes wall time. Resource readings are environment observations, not a guarantee of future allocation.
- **Measured pre-training baseline:** CPU, PyTorch 2.7.1+cpu, one thread, config/init seed 1337, no optimizer steps, deterministic non-shuffled batches of 4. Cross-entropy is averaged over non-PAD next-token targets; perplexity is `exp(mean NLL)`.

  | Split | Records | Windows | Scored next-token targets | Random-init NLL | Random-init perplexity |
  |---|---:|---:|---:|---:|---:|
  | Train | 102 | 487 | 56,224 | 5.536343772 | 253.748538708 |
  | Validation | 13 | 72 | 8,540 | 5.536363717 | 253.753599869 |
  | Test | 13 | 66 | 7,677 | 5.539546504 | 254.562530173 |

  These are actual random-initialized model measurements on this tiny sample, not trained-model results or capability metrics. Recompute and record the same deterministic baseline at experiment start. The test baseline has been measured once; do not tune against it, and use the test split only for one final post-training comparison.
- **Evaluation:** report step-wise training loss; fixed-seed validation loss/perplexity every 10 steps; compare final train/validation/test metrics with their pre-training baseline. Also save/reload a temporary checkpoint and confirm finite/equivalent outputs.
- **Stop criteria:** stop at 100 optimizer steps or 2 minutes; stop immediately on non-finite loss/gradient/weights, failed checkpoint reload, or a validation loss more than 10% above its measured baseline at two consecutive validation checks. Do not tune against the test split. Regardless of outcome, report the experiment as a smoke test on a small arithmetic dataset, not as evidence of conversational capability.

This experiment remains **not approved and not run**. Full training is a separate decision after the user reviews the proposal.
