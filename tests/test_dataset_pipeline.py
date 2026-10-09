import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from brain.config import BrainConfig
from brain.data import DatasetConfig, DatasetError, build_dataset
from brain.tokenizer import ByteTokenizer

ROOT = Path(__file__).resolve().parents[1]


class DatasetPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.raw = self.root / "raw.jsonl"
        self.output = self.root / "processed"
        self.brain_config = BrainConfig.load(ROOT / "configs" / "duo_v0.json")

    def tearDown(self):
        self.temp.cleanup()

    def write_jsonl(self, records):
        self.raw.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")

    def read_split(self, folder, name):
        path = folder / f"{name}.jsonl"
        return [json.loads(line)["text"] for line in path.read_text(encoding="utf-8").splitlines()]

    def base_config(self, **kwargs):
        value = {"dataset_name": "test-fixture", **kwargs}
        return DatasetConfig(**value)

    def test_valid_records_cleaning_duplicates_rejections_and_stats(self):
        self.raw.write_text(
            '\n'.join([
                json.dumps({"text": "  first  keeps   inner   "}),
                json.dumps({"text": "second"}),
                json.dumps({"text": " first  keeps   inner "}),
                json.dumps({"text": " \t "}),
                json.dumps({"text": 12}),
                "{malformed",
                json.dumps(["not", "an", "object"]),
                json.dumps({"body": "missing text"}),
            ]) + "\n",
            encoding="utf-8",
        )
        result = build_dataset(self.raw, self.output, self.base_config(), brain_config=self.brain_config)
        stats = result["stats"]
        self.assertEqual(stats["input_records"], 8)
        self.assertEqual(stats["accepted_records"], 2)
        self.assertEqual(stats["duplicate_records"], 1)
        self.assertEqual(stats["rejected_records"], 5)
        self.assertEqual(stats["rejected_by_reason"], {
            "empty_after_trim": 1, "malformed_json": 1, "missing_text_field": 1,
            "record_not_object": 1, "text_not_string": 1,
        })
        texts = sum((self.read_split(self.output, name) for name in ("train", "validation", "test")), [])
        self.assertIn("first  keeps   inner", texts)
        self.assertNotIn("first keeps inner", texts)
        self.assertEqual(stats["text_statistics"]["byte_token_count"], sum(len(ByteTokenizer().encode(x)) for x in texts))
        self.assertEqual(result["manifest"]["input"]["sha256"], hashlib.sha256(self.raw.read_bytes()).hexdigest())
        self.assertTrue((self.output / "manifest.json").is_file())
        self.assertTrue((self.output / "stats.json").is_file())

    def test_repeated_builds_are_byte_for_byte_deterministic(self):
        self.write_jsonl([{"text": f"sample {i}"} for i in range(20)])
        config = self.base_config(seed=2026)
        first = self.root / "first"
        second = self.root / "second"
        a = build_dataset(self.raw, first, config, brain_config=self.brain_config)
        b = build_dataset(self.raw, second, config, brain_config=self.brain_config)
        for name in ("train.jsonl", "validation.jsonl", "test.jsonl", "manifest.json", "stats.json"):
            self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())
        self.assertEqual(a, b)

    def test_splits_are_disjoint_and_cover_unique_accepted_texts(self):
        self.write_jsonl([{"text": f"record {i}"} for i in range(13)] + [{"text": "record 2"}])
        build_dataset(self.raw, self.output, self.base_config(), brain_config=self.brain_config)
        parts = [set(self.read_split(self.output, name)) for name in ("train", "validation", "test")]
        self.assertFalse(parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2])
        self.assertEqual(len(set.union(*parts)), 13)

    def test_fixed_seed_controls_split_assignment(self):
        self.write_jsonl([{"text": f"record {i}"} for i in range(30)])
        a = self.root / "seed-a"
        b = self.root / "seed-b"
        c = self.root / "seed-c"
        build_dataset(self.raw, a, self.base_config(seed=10), brain_config=self.brain_config)
        build_dataset(self.raw, b, self.base_config(seed=10), brain_config=self.brain_config)
        build_dataset(self.raw, c, self.base_config(seed=11), brain_config=self.brain_config)
        self.assertEqual((a / "train.jsonl").read_bytes(), (b / "train.jsonl").read_bytes())
        self.assertNotEqual((a / "train.jsonl").read_bytes(), (c / "train.jsonl").read_bytes())

    def test_empty_input_and_small_datasets_are_safe(self):
        self.raw.write_bytes(b"")
        result = build_dataset(self.raw, self.output, self.base_config(), brain_config=self.brain_config)
        self.assertEqual(result["stats"]["split_sizes"], {"train": 0, "validation": 0, "test": 0})
        self.assertFalse(result["stats"]["split_proportions_exactly_representable"])
        self.write_jsonl([{"text": "only one"}])
        result = build_dataset(self.raw, self.output, self.base_config(), brain_config=self.brain_config)
        self.assertEqual(sum(result["stats"]["split_sizes"].values()), 1)
        self.assertEqual(result["stats"]["accepted_records"], 1)
        self.assertFalse(result["stats"]["split_proportions_exactly_representable"])

    def test_text_line_input_format(self):
        self.raw.write_text("  alpha  \n\n beta\nalpha\n", encoding="utf-8")
        config = self.base_config(input_format="txt-lines")
        result = build_dataset(self.raw, self.output, config, brain_config=self.brain_config)
        self.assertEqual(result["stats"]["input_records"], 4)
        self.assertEqual(result["stats"]["duplicate_records"], 1)
        self.assertEqual(result["stats"]["accepted_records"], 2)

    def test_bad_config_missing_file_bad_utf8_and_output_overlap_fail(self):
        with self.assertRaises(DatasetError):
            self.base_config(train_fraction=0.7).validate()
        with self.assertRaises(DatasetError):
            build_dataset(self.root / "missing.jsonl", self.output, self.base_config())
        self.raw.write_bytes(b'{"text":"\xff"}\n')
        with self.assertRaisesRegex(DatasetError, "valid UTF-8"):
            build_dataset(self.raw, self.output, self.base_config())
        with self.assertRaisesRegex(DatasetError, "contain the input source"):
            build_dataset(self.raw, self.root, self.base_config())

    def test_tokenizer_config_mismatch_is_rejected(self):
        self.write_jsonl([{"text": "data"}])
        wrong = self.brain_config.to_dict()
        wrong["tokenizer_version"] = "other-tokenizer"
        incompatible = BrainConfig.from_dict(wrong)
        with self.assertRaisesRegex(ValueError, "tokenizer version"):
            build_dataset(self.raw, self.output, self.base_config(), brain_config=incompatible)

    def test_provenance_and_hashes_are_recorded_without_inventing_license(self):
        self.write_jsonl([{"text": "sample"}])
        cfg = self.base_config(source_url=None, license=None, license_verified=False, dataset_version=None)
        result = build_dataset(self.raw, self.output, cfg, brain_config=self.brain_config)
        manifest = result["manifest"]
        self.assertIsNone(manifest["dataset"]["license"])
        self.assertFalse(manifest["dataset"]["license_verified"])
        self.assertEqual(len(manifest["input"]["sha256"]), 64)
        for split in manifest["outputs"].values():
            output_bytes = (self.output / split["path"]).read_bytes()
            self.assertEqual(split["sha256"], hashlib.sha256(output_bytes).hexdigest())
        self.assertEqual(manifest["processing"]["tokenizer_version"], "duo-byte-v1")

    def test_unicode_normalization_is_not_applied(self):
        self.write_jsonl([{"text": "é"}, {"text": "e\u0301"}])
        result = build_dataset(self.raw, self.output, self.base_config(), brain_config=self.brain_config)
        self.assertEqual(result["stats"]["accepted_records"], 2)
        self.assertEqual(result["stats"]["duplicate_records"], 0)

    def test_verified_license_requires_value(self):
        with self.assertRaisesRegex(DatasetError, "requires a documented license"):
            self.base_config(license_verified=True).validate()


if __name__ == "__main__":
    unittest.main()
