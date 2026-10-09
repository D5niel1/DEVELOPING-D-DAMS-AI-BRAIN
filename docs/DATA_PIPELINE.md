# D-DAMS Dataset Pipeline (Phase 1C)

**Status:** local ingestion, cleaning, exact deduplication, deterministic train/validation/test splitting, tokenizer-byte statistics, hashes and manifests are implemented. Synthetic tests exercise the pipeline. A separate, bounded 128-row GSM8K sample has been imported locally; its provenance and results are recorded in [`DATASET_GSM8K_SAMPLE.md`](DATASET_GSM8K_SAMPLE.md). No model has been trained.

## Input formats

`brain.data.pipeline` is dependency-free (Python 3.10+). It accepts:

1. `jsonl` (recommended): one UTF-8 JSON object per line, with the configured text field (default `text`) containing a string. Invalid JSON, non-object rows, missing fields and non-string text are rejected and counted by reason.
2. `txt-lines`: one record per physical line in UTF-8 text. This format is only suitable when each line is a separate training example; it cannot represent embedded newlines within one example.

Inputs should be kept in `datasets/raw/`. Processed outputs belong in `datasets/processed/`; both directories' contents are git-ignored. The builder refuses output locations that would contain the source file.

## Cleaning and deduplication

Text must be a string. The pipeline trims leading/trailing Unicode whitespace and rejects records that are empty after trimming. It preserves internal whitespace, punctuation, and Unicode code points; **it performs no Unicode normalization**. Duplicate detection compares the exact cleaned text, keeps the first occurrence in source order, and counts later exact duplicates separately from rejected records. It does not attempt fuzzy, case-folded, or semantic deduplication.

## Splits and reproducibility

Accepted unique records are shuffled using `random.Random(seed).shuffle` and then sliced according to configurable positive fractions (default 0.8 / 0.1 / 0.1). Integer sizes use largest-remainder allocation. When there are at least three records, empty split counts are avoided by transferring records from the largest split; for fewer than three records, empty splits are permitted. Statistics report whether the final proportions exactly match the requested fractions. Rebuilding the same input bytes with the same config and Python behavior produces byte-identical split files and reports; no build timestamp is inserted.

The manifest records the exact input file name, raw SHA-256 and byte size, processing configuration and seed, Python version, tokenizer version/vocabulary size, record counts, output paths and per-output SHA-256 hashes. `stats.json` records input, accepted unique, valid pre-deduplication, rejected and per-reason counts; duplicate counts; split sizes; character and UTF-8 byte-token totals/min/max/means for the full accepted corpus and each split. Byte-token counts use `duo-byte-v1` without adding BOS/EOS. An invariant check requires accepted + duplicate + rejected records to equal input physical lines.

Provenance/license fields are copied from the supplied dataset config and are not independently asserted by the pipeline. Null means unknown/not supplied. `license_verified` is a caller-provided provenance flag; set it true only after reviewing the authoritative source terms yourself. The pipeline does not fetch or verify licenses.

## Configuration and run

Start with `configs/dataset.local-import.example.json`; replace `dataset_name` and provenance fields based on your source review. Keep `source_url`, `license`, version/date null when unknown; do not mark a license verified without evidence.

```sh
python -m brain.data.build \
  --input datasets/raw/my-approved-sample.jsonl \
  --output datasets/processed/my-approved-sample-v1 \
  --config configs/dataset.local-import.example.json \
  --brain-config configs/duo_v0.json
```

The command validates tokenizer version and vocabulary size against the model configuration before building. The sample model config is only provisional; no model exists yet. To test without an external corpus:

```sh
python -m unittest discover -s tests -v
python -m compileall -q brain tests
```

## Dataset access and import readiness

The session's Hugging Face connector is listed but **disabled**. Public anonymous access to the Hugging Face Hub metadata API and dataset-server rows API was separately verified for the GSM8K sample; this does not verify authenticated or gated access and does not mean the connector works. The selected sample, license evidence, candidate comparison, source revision, retrieval hashes and import results are documented in [`DATASET_GSM8K_SAMPLE.md`](DATASET_GSM8K_SAMPLE.md). No other corpus was downloaded.

For future datasets: open the canonical card and linked source/license; verify that terms cover the intended use and redistribution; pin a revision; select a bounded sample; record dataset name, revision, source URL, license, retrieval date and selection method; preserve source bytes under `datasets/raw/`; then run the pipeline. Do not proceed if license or provenance is unclear. Enable and separately verify the Hugging Face connector before relying on connector-based or authenticated access.

## Limitations

The implementation reads the input into memory and is intended for bounded initial experiments, not large corpora. Split assignment depends on Python's `random.Random` behavior; record Python version with experiment records for long-term reproduction. It does not shard, stream, tokenize into training tensors, make stratified/group splits, remove near-duplicates, or download datasets. The generated splits and reports are ignored by Git; the config/code/tests/docs are versioned.
