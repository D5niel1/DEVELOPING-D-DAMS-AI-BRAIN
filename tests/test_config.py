import json
import math
import tempfile
import unittest
from pathlib import Path

from brain.config import BrainConfig


ROOT = Path(__file__).resolve().parents[1]


class BrainConfigTests(unittest.TestCase):
    def test_example_config_loads_and_round_trips(self):
        config = BrainConfig.load(ROOT / "configs" / "duo_v0.json")
        self.assertEqual(config.version, "duo-v0")
        self.assertEqual(config.model.embedding_dim, 64)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "config.json"
            config.save(path)
            self.assertEqual(BrainConfig.load(path), config)
            self.assertTrue(path.read_text(encoding="utf-8").endswith("\n"))

    def test_invalid_head_dimension_is_rejected(self):
        value = json.loads((ROOT / "configs" / "duo_v0.json").read_text(encoding="utf-8"))
        value["model"]["embedding_dim"] = 63
        with self.assertRaisesRegex(ValueError, "divisible"):
            BrainConfig.from_dict(value)

    def test_invalid_architecture_fields_are_rejected(self):
        original = json.loads((ROOT / "configs" / "duo_v0.json").read_text(encoding="utf-8"))
        for field, bad, message in (
            ("context_length", 129, "context_length"),
            ("position_encoding", "rotary", "position_encoding"),
            ("activation", "relu", "activation"),
            ("norm_eps", math.inf, "norm_eps"),
            ("tie_embeddings", 1, "tie_embeddings"),
            ("initialization", "unknown", "initialization"),
        ):
            value = json.loads(json.dumps(original))
            value["model"][field] = bad
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, message):
                BrainConfig.from_dict(value)

    def test_unknown_keys_are_rejected(self):
        value = json.loads((ROOT / "configs" / "duo_v0.json").read_text(encoding="utf-8"))
        value["unexpected"] = True
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            BrainConfig.from_dict(value)

    def test_nonpositive_training_value_is_rejected(self):
        value = json.loads((ROOT / "configs" / "duo_v0.json").read_text(encoding="utf-8"))
        value["training"]["batch_size"] = 0
        with self.assertRaisesRegex(ValueError, "batch_size"):
            BrainConfig.from_dict(value)

    def test_invalid_json_reports_clear_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text("{broken", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid JSON"):
                BrainConfig.load(path)


if __name__ == "__main__":
    unittest.main()
