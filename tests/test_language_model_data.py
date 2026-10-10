import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from brain.config import BrainConfig
from brain.data.language_model import (
    TextDataError,
    build_context_windows,
    load_jsonl_text,
    load_lm_batches,
    make_lm_batches,
    shift_tokens,
)
from brain.tokenizer import ByteTokenizer

ROOT = Path(__file__).resolve().parents[1]


class LanguageModelDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.tokenizer = ByteTokenizer()

    def tearDown(self):
        self.temp.cleanup()

    def test_loader_reads_jsonl_text_records(self):
        path = self.root / "data.jsonl"
        path.write_text('{"text":"alpha"}\n{"text":"beta\\nline"}\n', encoding="utf-8")
        self.assertEqual(load_jsonl_text(path), ["alpha", "beta\nline"])

    def test_loader_reports_missing_file_malformed_json_and_missing_field(self):
        with self.assertRaisesRegex(TextDataError, "does not exist"):
            load_jsonl_text(self.root / "missing.jsonl")
        malformed = self.root / "malformed.jsonl"
        malformed.write_text('{"text":"valid"}\n{bad json}\n', encoding="utf-8")
        with self.assertRaisesRegex(TextDataError, r"malformed JSONL.*:2"):
            load_jsonl_text(malformed)
        missing_field = self.root / "missing-field.jsonl"
        missing_field.write_text('{"question":"no text field"}\n', encoding="utf-8")
        with self.assertRaisesRegex(TextDataError, "missing text field 'text'.*:1"):
            load_jsonl_text(missing_field)

    def test_loader_rejects_non_object_and_non_string_text(self):
        for payload, message in [("[]", "must be an object"), ('{"text":3}', "must be a string")]:
            path = self.root / "bad.jsonl"
            path.write_text(payload + "\n", encoding="utf-8")
            with self.subTest(payload=payload), self.assertRaisesRegex(TextDataError, message):
                load_jsonl_text(path)

    def test_record_boundaries_bos_eos_and_empty_record(self):
        windows = build_context_windows(["A", "", "B"], self.tokenizer, context_length=8)
        self.assertEqual(len(windows), 3)
        self.assertEqual(windows[0], [2, *self.tokenizer.encode("A"), 3])
        self.assertEqual(windows[1], [2, 3])
        self.assertEqual(windows[2], [2, *self.tokenizer.encode("B"), 3])
        # No window combines two source records or creates a cross-record target.
        self.assertTrue(all(window[0] == 2 and window[-1] == 3 for window in windows))

    def test_long_record_uses_deterministic_overlapping_windows_without_losing_pairs(self):
        text = "x" * 260
        first = build_context_windows([text], self.tokenizer, context_length=128)
        second = build_context_windows([text], self.tokenizer, context_length=128)
        self.assertEqual(first, second)
        self.assertEqual([len(window) for window in first], [129, 129, 6])
        ids = [2, *self.tokenizer.encode(text), 3]
        transitions = []
        for window in first:
            transitions.extend(zip(window, window[1:]))
        self.assertEqual(transitions, list(zip(ids, ids[1:])))

    def test_context_length_validation_and_tokenizer_config(self):
        for length in (0, 129, True):
            with self.subTest(length=length), self.assertRaises(ValueError):
                build_context_windows(["x"], self.tokenizer, length)
        config = BrainConfig.load(ROOT / "configs" / "duo_v0.json")
        self.tokenizer.assert_compatible(config)

    def test_base_data_package_does_not_eagerly_import_torch(self):
        script = "import sys, brain.data; assert 'torch' not in sys.modules"
        subprocess.run([sys.executable, "-c", script], cwd=ROOT, check=True, capture_output=True, text=True)

    def test_shifted_inputs_targets_and_masks(self):
        tokens = torch.tensor([[2, 5, 3, 0], [2, 6, 7, 3]])
        mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool)
        inputs, targets, input_mask, target_mask = shift_tokens(tokens, mask)
        self.assertTrue(torch.equal(inputs, tokens[:, :-1]))
        self.assertTrue(torch.equal(targets, tokens[:, 1:]))
        self.assertTrue(torch.equal(input_mask, mask[:, :-1]))
        self.assertTrue(torch.equal(target_mask, mask[:, 1:]))

    def test_right_padding_masks_and_deterministic_batch_shuffle(self):
        windows = [[2, 8, 3], [2, 9, 10, 3], [2, 11, 3]]
        batches_a = make_lm_batches(windows, 2, shuffle=True, seed=11)
        batches_b = make_lm_batches(windows, 2, shuffle=True, seed=11)
        self.assertEqual(len(batches_a), 2)
        for a, b in zip(batches_a, batches_b):
            self.assertTrue(torch.equal(a.input_ids, b.input_ids))
            self.assertTrue(torch.equal(a.targets, b.targets))
            self.assertTrue(torch.equal(a.attention_mask, b.attention_mask))
            self.assertTrue(torch.equal(a.target_mask, b.target_mask))
        plain = make_lm_batches(windows[:2], 2)[0]
        self.assertEqual(tuple(plain.input_ids.shape), (2, 3))
        self.assertEqual(plain.input_ids[0].tolist(), [2, 8, 3])
        self.assertEqual(plain.attention_mask[0].tolist(), [True, True, True])
        self.assertEqual(plain.targets[0].tolist(), [8, 3, 0])
        self.assertEqual(plain.target_mask[0].tolist(), [True, True, False])

    def test_end_to_end_loader_uses_actual_configured_split_path(self):
        config = BrainConfig.load(ROOT / "configs" / "duo_v0.json")
        self.assertTrue(Path(config.data.train_path).is_file())
        records = load_jsonl_text(config.data.train_path)
        batches = load_lm_batches(config.data.train_path, self.tokenizer, context_length=128, batch_size=4)
        self.assertEqual(len(records), 102)
        self.assertTrue(batches)
        self.assertLessEqual(max(batch.input_ids.shape[1] for batch in batches), 128)
        self.assertEqual(sum(int(batch.target_mask.sum()) for batch in batches), sum(len(w) - 1 for w in build_context_windows(records, self.tokenizer)))

    def test_all_configured_splits_exist_and_have_text_field(self):
        config = BrainConfig.load(ROOT / "configs" / "duo_v0.json")
        for split, path in (
            ("train", config.data.train_path),
            ("validation", config.data.validation_path),
            ("test", config.data.test_path),
        ):
            with self.subTest(split=split):
                records = load_jsonl_text(path)
                self.assertTrue(records)
                self.assertTrue(all(isinstance(record, str) for record in records))


if __name__ == "__main__":
    unittest.main()
