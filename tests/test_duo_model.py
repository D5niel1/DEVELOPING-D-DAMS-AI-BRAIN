import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

from brain.config import BrainConfig
from brain.model import (
    DuoTransformer,
    build_causal_allowed_mask,
    load_checkpoint,
    masked_next_token_loss,
    save_checkpoint,
)
from brain.tokenizer import ByteTokenizer

ROOT = Path(__file__).resolve().parents[1]


class DuoModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.config = BrainConfig.load(ROOT / "configs" / "duo_v0.json")

    def test_forward_shapes_for_single_short_and_max_context(self):
        model = DuoTransformer(self.config, seed=1337).eval()
        for length in (1, 7, 128):
            ids = torch.randint(4, 260, (2, length), dtype=torch.long)
            with self.subTest(length=length), torch.no_grad():
                logits = model(ids)
                self.assertEqual(tuple(logits.shape), (2, length, 260))
                self.assertTrue(torch.isfinite(logits).all().item())

    def test_rejects_invalid_and_over_context_inputs(self):
        model = DuoTransformer(self.config)
        with self.assertRaisesRegex(ValueError, "exceeds configured"):
            model(torch.zeros((1, 129), dtype=torch.long))
        with self.assertRaisesRegex(ValueError, "rank-2"):
            model(torch.zeros((1, 2, 3), dtype=torch.long))
        with self.assertRaisesRegex(ValueError, "in \\[0, 260\\)"):
            model(torch.tensor([[260]], dtype=torch.long))
        with self.assertRaisesRegex(ValueError, "torch.int32 or torch.int64"):
            model(torch.ones((1, 2), dtype=torch.float32))

    def test_tokenizer_compatibility_and_vocabulary_dimension(self):
        model = DuoTransformer(self.config)
        ByteTokenizer().assert_compatible(self.config)
        self.assertEqual(model.token_embedding.num_embeddings, 260)
        self.assertTrue(model.config.model.tie_embeddings)
        self.assertIsNone(model.output_projection)
        self.assertNotIn("output_projection.weight", dict(model.named_parameters()))
        self.assertEqual(model.output_projection if model.output_projection is not None else model.token_embedding.weight.shape, (260, 64))

    def test_causal_mask_orientation(self):
        allowed = build_causal_allowed_mask(None, batch_size=1, sequence_length=4, device=torch.device("cpu"))
        self.assertEqual(allowed.shape, (1, 1, 4, 4))
        self.assertEqual(
            allowed[0, 0].to(torch.int64).tolist(),
            [[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]],
        )

    def test_future_tokens_do_not_change_earlier_logits(self):
        model = DuoTransformer(self.config, seed=43).eval()
        original = torch.tensor([[2, 70, 71, 72, 73, 74, 75]])
        changed = original.clone()
        changed[:, 4:] = torch.tensor([[111, 112, 113]])
        with torch.no_grad():
            logits_a = model(original)
            logits_b = model(changed)
        self.assertTrue(torch.allclose(logits_a[:, :4], logits_b[:, :4], rtol=0.0, atol=1e-6))
        self.assertGreater(float((logits_a[:, 4:] - logits_b[:, 4:]).abs().max()), 0.0)

    def test_padding_keys_queries_are_safe_and_padded_logits_zero(self):
        model = DuoTransformer(self.config, seed=19).eval()
        ids = torch.tensor([[2, 66, 3, 0, 0], [2, 67, 68, 3, 0]])
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.bool)
        with torch.no_grad():
            logits = model(ids, mask)
        self.assertTrue(torch.isfinite(logits).all().item())
        self.assertTrue(torch.equal(logits[~mask], torch.zeros_like(logits[~mask])))

    def test_all_padding_rows_remain_finite_and_zero(self):
        model = DuoTransformer(self.config, seed=23).eval()
        ids = torch.zeros((2, 5), dtype=torch.long)
        mask = torch.zeros_like(ids, dtype=torch.bool)
        logits = model(ids, mask)
        self.assertTrue(torch.isfinite(logits).all().item())
        self.assertEqual(float(logits.abs().sum()), 0.0)

    def test_padding_mask_blocks_pad_keys_from_valid_queries(self):
        allowed = build_causal_allowed_mask(
            torch.tensor([[True, True, False, False]]),
            batch_size=1,
            sequence_length=4,
            device=torch.device("cpu"),
        )
        self.assertEqual(allowed[0, 0, 1].tolist(), [True, True, False, False])

    def test_next_token_loss_and_padding_exclusion(self):
        logits = torch.tensor([[[2.0, 0.0, -1.0], [0.0, 2.0, -1.0], [1.0, 0.0, -1.0]]])
        targets = torch.tensor([[1, 0, 2]])
        target_mask = torch.tensor([[True, False, True]])
        loss = masked_next_token_loss(logits, targets, target_mask)
        expected = F.cross_entropy(logits[0, [0, 2]], targets[0, [0, 2]])
        self.assertTrue(torch.allclose(loss, expected))

    def test_all_padding_targets_return_differentiable_finite_zero_loss(self):
        logits = torch.randn((2, 3, 260), requires_grad=True)
        targets = torch.zeros((2, 3), dtype=torch.long)
        loss = masked_next_token_loss(logits, targets)
        self.assertTrue(torch.isfinite(loss).item())
        self.assertEqual(float(loss), 0.0)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all().item())

    def test_real_backward_pass_finite_gradients_for_all_trainable_parameters(self):
        model = DuoTransformer(self.config, seed=31).train()
        ids = torch.tensor([[2, 72, 73, 3, 0], [2, 74, 75, 76, 3]])
        mask = ids.ne(0)
        inputs = ids[:, :-1]
        targets = ids[:, 1:]
        target_mask = mask[:, 1:]
        logits = model(inputs, mask[:, :-1])
        loss = masked_next_token_loss(logits, targets, target_mask)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        missing = [name for name, param in model.named_parameters() if param.requires_grad and param.grad is None]
        self.assertEqual(missing, [])
        for name, param in model.named_parameters():
            with self.subTest(parameter=name):
                self.assertTrue(torch.isfinite(param.grad).all().item())
                self.assertGreater(float(param.grad.abs().sum()), 0.0, msg=f"no gradient for {name}")

    def test_fixed_seed_initialization_is_deterministic_and_rng_isolated(self):
        torch.manual_seed(555)
        before = torch.random.get_rng_state().clone()
        first = DuoTransformer(self.config, seed=99)
        after = torch.random.get_rng_state()
        second = DuoTransformer(self.config, seed=99)
        self.assertTrue(torch.equal(before, after))
        for a, b in zip(first.state_dict().values(), second.state_dict().values()):
            self.assertTrue(torch.equal(a, b))

    def test_parameter_count_matches_declared_architecture_and_tied_policy(self):
        model = DuoTransformer(self.config, seed=5)
        expected = 124_928
        actual = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.assertEqual(model.parameter_count, expected)
        self.assertEqual(actual, expected)
        self.assertIs(model.output_projection, None)
        self.assertEqual(model.token_embedding.weight.shape, (260, 64))

    def test_untied_output_projection_is_configurable(self):
        value = self.config.to_dict()
        value["model"]["tie_embeddings"] = False
        untied_config = BrainConfig.from_dict(value)
        model = DuoTransformer(untied_config, seed=5)
        self.assertIsNotNone(model.output_projection)
        self.assertEqual(tuple(model.output_projection.weight.shape), (260, 64))
        self.assertEqual(model.parameter_count, 141_828)

    def test_checkpoint_fresh_process_reproduces_logits(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "duo-v0.pt"
            model = DuoTransformer(self.config, seed=2026).eval()
            ids = torch.tensor([[2, 80, 81, 3], [2, 82, 3, 0]], dtype=torch.long)
            mask = ids.ne(0)
            with torch.no_grad():
                expected = model(ids, mask)
            save_checkpoint(model, checkpoint)
            script = r'''
import json, sys, torch
from brain.model import load_checkpoint
m=load_checkpoint(sys.argv[1]).eval()
ids=torch.tensor([[2,80,81,3],[2,82,3,0]],dtype=torch.long)
mask=ids.ne(0)
with torch.no_grad(): out=m(ids,mask)
print(json.dumps(out.tolist()))
'''
            result = subprocess.run(
                [sys.executable, "-c", script, str(checkpoint)],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            actual = torch.tensor(json.loads(result.stdout))
            self.assertTrue(torch.allclose(actual, expected, rtol=0.0, atol=1e-6))

    def test_checkpoint_in_current_process_preserves_config_seed_and_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            original = DuoTransformer(self.config, seed=123).eval()
            save_checkpoint(original, path)
            restored = load_checkpoint(path)
            self.assertEqual(restored.config, original.config)
            self.assertEqual(restored.init_seed, 123)
            self.assertFalse(restored.training)
            for name, tensor in original.state_dict().items():
                self.assertTrue(torch.equal(tensor, restored.state_dict()[name]), name)


if __name__ == "__main__":
    unittest.main()
