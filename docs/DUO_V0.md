# Duo v0 — Implemented Architecture and Verification

**Status:** The local decoder-only Transformer is implemented and acceptance-tested; one bounded 100-step smoke run is complete, with its trained checkpoint kept locally and Git-ignored. The tracked config still initializes the base model. Neither the architecture gate nor this smoke run establishes conversational capability, language quality, or fitness as a product.

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

## Controlled smoke-training experiment — completed

**Status:** one bounded engineering experiment was explicitly requested and completed on 2026-10-10. This was not a capability evaluation, a full training run, or approval for another run.

- **Objective:** verify optimizer execution, deterministic loss reporting, scheduled validation, finite state, and portable checkpoint reload.
- **Data:** pinned local GSM8K main/train rows 0–127, transformed into 102 train / 13 validation / 13 test records. Training used train only; validation was used for monitoring and the divergence stop; test bytes were not read for the new baseline and the test split was evaluated once after step 100 and checkpoint verification.
- **Runtime:** CPython 3.12.3, PyTorch 2.7.1+cpu, CPU only, one thread, deterministic algorithms enabled. Model config/init seed 1337; 124,928 parameters; `duo-byte-v1`; context 128; batch size 4; one deterministic seeded permutation of 487 train windows; first 100 of 122 batches consumed once without wrapping.
- **Optimizer:** AdamW, learning rate 0.0003 from the existing config, weight decay 0.01, betas `(0.9, 0.999)`, epsilon `1e-8`, gradient-norm clip 1.0; no scheduler, mixed precision, or hyperparameter search. The model's configured initialization was retained. Cap: 100 optimizer steps or 120 seconds, whichever came first; validation every 10 steps.
- **Metric method:** token-weighted next-token NLL over non-PAD targets; perplexity `exp(NLL)`. Pre-training train/validation baselines were recomputed deterministically. Baseline targets: 56,224 train and 8,540 validation.

| Step | Training NLL (token-weighted mean over the latest 10 optimizer batches) | Validation NLL | Validation perplexity |
|---:|---:|---:|---:|
| 0 baseline | 5.536344 (full train split) | 5.536364 | 253.753598 |
| 10 | 5.341129 | 5.171812 | 176.233857 |
| 20 | 5.071839 | 4.935765 | 139.179546 |
| 30 | 4.840838 | 4.699457 | 109.887489 |
| 40 | 4.599408 | 4.473043 | 87.622962 |
| 50 | 4.424301 | 4.262616 | 70.995480 |
| 60 | 4.221397 | 4.082603 | 59.299643 |
| 70 | 4.014588 | 3.919227 | 50.361516 |
| 80 | 3.911514 | 3.784574 | 44.016904 |
| 90 | 3.734431 | 3.673338 | 39.383139 |
| 100 | 3.702470 | 3.591931 | 36.304117 |

- **Outcome:** 100/100 steps completed; exact stop reason `100_optimizer_step_limit`; total elapsed 7.3182 seconds, including baseline, final verification, and test evaluation (optimizer loop: 2.6617 seconds). No loss, gradient, or post-step weights were non-finite. Validation NLL stayed below its 110%-of-baseline guard at every check.
- **Test:** final single evaluation after the run: 13 records, 7,677 scored targets, NLL 3.620738, perplexity 37.365116. The earlier random-init test reference remains the pre-run value documented above (NLL 5.539546504, PPL 254.562530173); it was not recomputed or used for tuning in this run. No changes were made after viewing test metrics.
- **Checkpoint:** saved locally and Git-ignored at `checkpoints/duo_v0/smoke/20261010T201807.659648Z/duo-smoke-final.pt` (511,897 bytes; SHA-256 `f8066d50e6c891c0e19e55b383eabebb75f6160019f7e698fcfcaa8ae97b49ce`). A separate Python process reloaded it and produced logits for synthetic IDs `[[2, 71, 108, 3]]` with maximum absolute difference 0.0 (tolerance `1e-6`). The checkpoint contains smoke-trained weights; it is not committed.
- **Artifacts:** ignored `experiment.json` and `metrics.jsonl` in the same run directory preserve settings, hashes, all 100 step records, ten scheduled evaluation events, stop reason, and final metrics.

This establishes a functioning **bounded smoke-training path**, not general training/resume infrastructure or useful conversational ability. The dataset is only a tiny arithmetic subset and the 13-record test result is not a generalization claim. No optimizer-state resume, multi-seed evaluation, or full training was implemented or attempted.

**One next milestone:** implement a reusable training/resume and deterministic evaluation-reporting path with portable optimizer-state checkpoints; do not launch another experiment without explicit approval.
