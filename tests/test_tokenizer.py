import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from brain.config import BrainConfig
from brain.tokenizer import ByteTokenizer, TokenizerError
from brain.tokenizer.byte_tokenizer import SPECIAL_TOKEN_IDS, VOCAB_SIZE

ROOT = Path(__file__).resolve().parents[1]


class ByteTokenizerTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = ByteTokenizer()

    def test_repeated_encoding_is_deterministic(self):
        text = "Duo — café 🌍"
        self.assertEqual(self.tokenizer.encode(text), self.tokenizer.encode(text))
        self.assertEqual(self.tokenizer.decode(self.tokenizer.encode(text)), text)

    def test_special_tokens_have_stable_ids(self):
        self.assertEqual(SPECIAL_TOKEN_IDS, {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3})
        self.assertEqual([self.tokenizer.token_id(x) for x in ("<pad>", "<unk>", "<bos>", "<eos>")], [0, 1, 2, 3])
        self.assertEqual(self.tokenizer.encode("A", add_special_tokens=True), [2, 69, 3])
        self.assertEqual(self.tokenizer.decode([2, 69, 3]), "A")
        self.assertEqual(self.tokenizer.decode([2, 69, 3], skip_special_tokens=False), "<bos>A<eos>")

    def test_unknown_vocabulary_spellings_map_to_unk(self):
        self.assertEqual(self.tokenizer.token_id("<not-in-vocab>"), SPECIAL_TOKEN_IDS["<unk>"])
        self.assertEqual(self.tokenizer.decode([1]), "<unk>")
        self.assertEqual(self.tokenizer.decode([1], skip_special_tokens=False), "<unk>")
        self.assertEqual(self.tokenizer.decode(self.tokenizer.encode("🧪")), "🧪")

    def test_empty_input_is_supported(self):
        self.assertEqual(self.tokenizer.encode(""), [])
        self.assertEqual(self.tokenizer.decode([]), "")
        self.assertEqual(self.tokenizer.encode("", add_special_tokens=True), [2, 3])

    def test_unsupported_input_types_are_rejected(self):
        with self.assertRaises(TypeError):
            self.tokenizer.encode(b"bytes")
        with self.assertRaises(TypeError):
            self.tokenizer.encode("x", add_special_tokens=1)

    def test_invalid_token_ids_are_rejected(self):
        for value in (-1, VOCAB_SIZE, True, 1.5, "1"):
            with self.subTest(value=value), self.assertRaises(TokenizerError):
                self.tokenizer.decode([value])
        with self.assertRaises(TokenizerError):
            self.tokenizer.token(VOCAB_SIZE)
        with self.assertRaises(TokenizerError):
            self.tokenizer.decode([4 + 0x80])  # UTF-8 continuation byte without a leading byte.

    def test_malformed_and_incompatible_artifacts_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            path.write_text("{bad", encoding="utf-8")
            with self.assertRaisesRegex(TokenizerError, "malformed"):
                ByteTokenizer.load(path)
            artifact = self.tokenizer.to_dict()
            artifact["format_version"] = 99
            path.write_text(json.dumps(artifact), encoding="utf-8")
            with self.assertRaisesRegex(TokenizerError, "format_version"):
                ByteTokenizer.load(path)
            artifact = self.tokenizer.to_dict()
            artifact["special_tokens"][0]["id"] = 8
            with self.assertRaisesRegex(TokenizerError, "stable IDs"):
                ByteTokenizer.from_dict(artifact)
            artifact = self.tokenizer.to_dict()
            artifact["vocabulary"][10] = artifact["vocabulary"][11]
            with self.assertRaisesRegex(TokenizerError, "duplicate"):
                ByteTokenizer.from_dict(artifact)

    def test_save_load_preserves_vocabulary_and_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifacts" / "tokenizer.json"
            self.tokenizer.save(path)
            loaded = ByteTokenizer.load(path)
            self.assertEqual(loaded.vocabulary, self.tokenizer.vocabulary)
            self.assertEqual(loaded.vocabulary_sha256(), self.tokenizer.vocabulary_sha256())
            for text in ("", "plain text", "é漢字🙂"):
                self.assertEqual(loaded.encode(text), self.tokenizer.encode(text))
                self.assertEqual(loaded.decode(loaded.encode(text)), text)
            self.assertEqual(loaded.special_token_ids, self.tokenizer.special_token_ids)

    def test_fresh_process_reload_preserves_exact_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            self.tokenizer.save(path)
            script = (
                "import json, sys; from brain.tokenizer import ByteTokenizer; "
                "t=ByteTokenizer.load(sys.argv[1]); "
                "print(json.dumps({'ids':t.encode('Duo — café 🌍'), 'vocab':list(t.vocabulary), "
                "'sha256':t.vocabulary_sha256(), 'special':t.special_token_ids}, ensure_ascii=False))"
            )
            result = subprocess.run(
                [sys.executable, "-c", script, str(path)],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            loaded = json.loads(result.stdout)
            self.assertEqual(loaded["ids"], self.tokenizer.encode("Duo — café 🌍"))
            self.assertEqual(loaded["vocab"], list(self.tokenizer.vocabulary))
            self.assertEqual(loaded["sha256"], self.tokenizer.vocabulary_sha256())
            self.assertEqual(loaded["special"], self.tokenizer.special_token_ids)

    def test_configuration_vocabulary_compatibility(self):
        config = BrainConfig.load(ROOT / "configs" / "duo_v0.json")
        self.tokenizer.assert_compatible(config)
        value = config.to_dict()
        value["model"]["vocab_size"] = 256
        incompatible_config = BrainConfig.from_dict(value)
        with self.assertRaisesRegex(TokenizerError, "does not match"):
            self.tokenizer.assert_compatible(incompatible_config)
        value = config.to_dict()
        value["tokenizer_version"] = "future-version"
        incompatible_version = BrainConfig.from_dict(value)
        with self.assertRaisesRegex(TokenizerError, "tokenizer version"):
            self.tokenizer.assert_compatible(incompatible_version)


if __name__ == "__main__":
    unittest.main()
