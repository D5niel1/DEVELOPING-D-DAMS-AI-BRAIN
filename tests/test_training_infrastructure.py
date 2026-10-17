import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch
from torch.nn import functional as F

from brain.config import BrainConfig
from brain.data.language_model import LanguageModelBatch, build_context_windows, make_lm_batches
from brain.model import DuoTransformer, generate_greedy, load_checkpoint, save_checkpoint
from brain.tokenizer import ByteTokenizer
from brain.training import (
    DeterministicBatchSampler,
    TrainingRunConfig,
    evaluate_checkpoint,
    evaluate_model,
    train_run,
)
from brain.training.checkpoint import atomic_torch_save, load_training_checkpoint
from brain.training.evaluation import sha256_file
from brain.training.snapshots import snapshot_file
from brain.training.trainer import TrainingError

ROOT = Path(__file__).resolve().parents[1]


class TrainingInfrastructureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.train_path = self._write_jsonl("train.jsonl", ["alpha", "beta", "gamma"])
        self.validation_path = self._write_jsonl("validation.jsonl", ["delta", "epsilon"])
        self.test_path = self._write_jsonl("test.jsonl", ["zeta", "eta"])
        base = BrainConfig.load(ROOT / "configs" / "duo_v0.json").to_dict()
        base["model"]["context_length"] = 16
        base["model"]["dropout"] = 0.15
        base["training"]["seed"] = 37
        base["data"] = {
            "train_path": str(self.train_path),
            "validation_path": str(self.validation_path),
            "test_path": str(self.test_path),
        }
        self.model_config_path = self.root / "model.json"
        self.model_config_path.write_text(json.dumps(base), encoding="utf-8")
        self.run_config = TrainingRunConfig(
            model_config_path=str(self.model_config_path),
            train_path=str(self.train_path),
            validation_path=str(self.validation_path),
            output_dir=str(self.root / "run-a"),
            dataset_manifest_path=None,
            batch_size=2,
            context_length=16,
            learning_rate=0.001,
            weight_decay=0.01,
            optimizer="adamw",
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            seed=37,
            evaluation_interval=3,
            checkpoint_interval=2,
            max_steps=2,
            device="cpu",
            gradient_clip_norm=1.0,
            drop_last=False,
            num_threads=1,
        )
        torch.set_num_threads(1)

    def tearDown(self):
        self.temp.cleanup()

    def _write_jsonl(self, name, texts):
        path = self.root / name
        path.write_text("".join(json.dumps({"text": text}) + "\n" for text in texts), encoding="utf-8")
        return path

    def test_training_config_validation_and_runtime_overrides(self):
        value = self.run_config.to_dict()
        loaded = TrainingRunConfig.from_dict(value)
        self.assertEqual(loaded, self.run_config)
        override = loaded.with_overrides({"max_steps": 9, "device": "cpu"})
        self.assertEqual(override.max_steps, 9)
        self.assertEqual(loaded.max_steps, 2)
        for key, invalid in (("max_steps", 0), ("batch_size", True), ("learning_rate", float("nan")), ("optimizer", "other"), ("drop_last", 0), ("device", "mps")):
            bad = dict(value)
            bad[key] = invalid
            with self.subTest(key=key), self.assertRaises(ValueError):
                TrainingRunConfig.from_dict(bad)
        with self.assertRaisesRegex(ValueError, "unknown runtime override"):
            loaded.with_overrides({"secret_mode": True})
        not_a_directory = self.root / "output-is-a-file"
        not_a_directory.write_text("preserve", encoding="utf-8")
        with self.assertRaisesRegex(TrainingError, "exists but is not a directory"):
            train_run(replace(self.run_config, output_dir=str(not_a_directory)))

    def test_train_cli_records_defaults_overrides_and_manifest(self):
        manifest_path = self.root / "manifest.json"
        manifest = {
            "manifest_version": 1,
            "dataset": {"dataset_name": "synthetic-fixture", "dataset_version": "fixture-r1", "license": "CC0-1.0"},
            "outputs": {
                "train": {"sha256": sha256_file(self.train_path), "records": 3},
                "validation": {"sha256": sha256_file(self.validation_path), "records": 2},
            },
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        cli_output = self.root / "cli-run"
        configured = replace(
            self.run_config,
            output_dir=str(cli_output),
            dataset_manifest_path=str(manifest_path),
        )
        config_path = self.root / "run.json"
        config_path.write_text(json.dumps(configured.to_dict()), encoding="utf-8")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "brain.training",
                "train",
                "--config",
                str(config_path),
                "--set",
                "max_steps=1",
                "--set",
                "learning_rate=0.0005",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        metadata = json.loads((cli_output / "run_metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["configured_defaults"]["max_steps"], 2)
        self.assertEqual(metadata["configured_defaults"]["learning_rate"], 0.001)
        self.assertEqual(metadata["runtime_overrides"], {"max_steps": 1, "learning_rate": 0.0005})
        self.assertEqual(metadata["effective_run_config"]["max_steps"], 1)
        self.assertEqual(metadata["effective_run_config"]["learning_rate"], 0.0005)
        self.assertEqual(metadata["dataset"]["dataset"]["dataset_version"], "fixture-r1")
        self.assertEqual(json.loads((cli_output / "training_summary.json").read_text())["completed_steps"], 1)
        self.assertIn("DONE steps=1", result.stdout)

    def test_sampler_order_incomplete_batches_and_restored_cursor(self):
        first = DeterministicBatchSampler(5, 2, 123)
        second = DeterministicBatchSampler(5, 2, 123)
        first_batches = [first.next_indices() for _ in range(3)]
        second_batches = [second.next_indices() for _ in range(3)]
        self.assertEqual(first_batches, second_batches)
        self.assertEqual(sorted(i for batch in first_batches for i in batch), list(range(5)))
        self.assertEqual([len(batch) for batch in first_batches], [2, 2, 1])
        state = first.state_dict()
        restored = DeterministicBatchSampler(5, 2, 123)
        restored.load_state_dict(state)
        self.assertEqual(restored.next_indices(), first.next_indices())
        with self.assertRaisesRegex(ValueError, "would produce no batches"):
            DeterministicBatchSampler(2, 4, 123, drop_last=True)

    def test_snapshot_hash_is_for_exact_bytes_consumed(self):
        source = self.root / "mutable.jsonl"
        original = b'{"text":"before"}\n'
        source.write_bytes(original)
        with snapshot_file(source) as (snapshot, digest):
            source.write_bytes(b'{"text":"after"}\n')
            self.assertEqual(snapshot.read_bytes(), original)
            self.assertEqual(sha256_file(snapshot), digest)
            self.assertNotEqual(sha256_file(source), digest)

    def test_evaluation_is_token_weighted_pad_safe_and_does_not_mutate_weights(self):
        config = BrainConfig.load(self.model_config_path)
        model = DuoTransformer(config, seed=7).train()
        tokenizer = ByteTokenizer()
        windows = build_context_windows(["a", "long"], tokenizer, 16)
        batches = make_lm_batches(windows, 2, shuffle=False)
        batch = batches[0]
        deliberately_bad_mask = batch.target_mask.clone()
        deliberately_bad_mask[batch.targets.eq(0)] = True
        batches = [LanguageModelBatch(batch.input_ids, batch.targets, batch.attention_mask, deliberately_bad_mask)]
        before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
        result = evaluate_model(model, batches, device="cpu")
        self.assertEqual(model.training, True)
        expected_nll_sum = 0.0
        expected_count = 0
        model.eval()
        with torch.inference_mode():
            for batch in batches:
                logits = model(batch.input_ids, batch.attention_mask)
                losses = F.cross_entropy(logits.reshape(-1, 260), batch.targets.reshape(-1), reduction="none")
                valid = (batch.target_mask & batch.targets.ne(0)).reshape(-1)
                expected_nll_sum += float(losses[valid].sum())
                expected_count += int(valid.sum())
        model.train()
        self.assertEqual(result["target_tokens"], expected_count)
        self.assertAlmostEqual(result["nll"], expected_nll_sum / expected_count, places=6)
        for name, tensor in model.state_dict().items():
            self.assertTrue(torch.equal(tensor, before[name]), name)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        with self.assertRaisesRegex(ValueError, "model parameters are on cpu"):
            evaluate_model(model, batches, device="cuda")

    def test_resume_checkpoint_contains_optimizer_step_rng_and_sampler(self):
        train_run(self.run_config)
        path = Path(self.run_config.output_dir) / "latest.resume.pt"
        payload = load_training_checkpoint(path)
        self.assertEqual(payload["completed_steps"], 2)
        self.assertEqual(payload["format"], "ddams-duo-training-resume")
        self.assertTrue(payload["optimizer_state"]["state"])
        self.assertIn("exp_avg", next(iter(payload["optimizer_state"]["state"].values())))
        self.assertIn("exp_avg_sq", next(iter(payload["optimizer_state"]["state"].values())))
        self.assertEqual(payload["sampler_state"]["size"], 3)
        self.assertEqual(payload["sampler_state"]["cursor"], 3)
        self.assertEqual(payload["rng_state"]["torch_cpu"].dtype, torch.uint8)
        resumed = train_run(replace(self.run_config, max_steps=3), resume_path=path)
        self.assertEqual(resumed["completed_steps"], 3)
        self.assertEqual(resumed["last_validation"]["target_tokens"] > 0, True)
        with self.assertRaisesRegex(TrainingError, "does not match the resume checkpoint configuration"):
            train_run(replace(self.run_config, learning_rate=0.002, max_steps=4), resume_path=path)
        with self.assertRaisesRegex(TrainingError, "resume output_dir must be the parent directory"):
            train_run(replace(self.run_config, output_dir=str(self.root / "other-run"), max_steps=4), resume_path=path)
        changed_implementation = Path(self.run_config.output_dir) / "changed-implementation.resume.pt"
        payload["environment"]["implementation"]["sha256"] = "0" * 64
        payload["compatibility"]["environment"] = payload["environment"]
        atomic_torch_save(payload, changed_implementation)
        with self.assertRaisesRegex(TrainingError, "incompatible in: environment"):
            train_run(replace(self.run_config, max_steps=4), resume_path=changed_implementation)
        metadata_path = Path(self.run_config.output_dir) / "run_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["resume_invocations"] = "malformed"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaisesRegex(TrainingError, "resume_invocations must be a list of objects"):
            train_run(replace(self.run_config, max_steps=4), resume_path=path)

    def test_interrupted_resume_matches_uninterrupted_training_exactly(self):
        interrupted = train_run(self.run_config)
        resume_path = Path(interrupted["resume_checkpoint"])
        resumed_config = replace(self.run_config, max_steps=4)
        resumed = train_run(resumed_config, resume_path=resume_path)
        continuous_config = replace(self.run_config, max_steps=4, output_dir=str(self.root / "run-b"))
        continuous = train_run(continuous_config)
        resumed_model = load_checkpoint(resumed["inference_checkpoint"])
        continuous_model = load_checkpoint(continuous["inference_checkpoint"])
        self.assertEqual(resumed["completed_steps"], 4)
        self.assertEqual(continuous["completed_steps"], 4)
        self.assertEqual(resumed_model.init_seed, continuous_model.init_seed)
        for name, tensor in resumed_model.state_dict().items():
            self.assertTrue(torch.equal(tensor, continuous_model.state_dict()[name]), name)

    def test_invalid_resume_checkpoint_is_rejected_without_training(self):
        invalid = self.root / "invalid.resume.pt"
        invalid.write_bytes(b"not a torch checkpoint")
        with self.assertRaisesRegex(ValueError, "cannot safely load training-resume checkpoint"):
            load_training_checkpoint(invalid)
        with self.assertRaisesRegex(ValueError, "cannot load Duo checkpoint"):
            load_checkpoint(invalid)

    def test_malformed_nested_rng_state_is_rejected_before_resume(self):
        result = train_run(self.run_config)
        payload = load_training_checkpoint(result["resume_checkpoint"])
        payload["rng_state"]["torch_cpu"] = 3
        malformed = Path(self.run_config.output_dir) / "malformed-rng.resume.pt"
        atomic_torch_save(payload, malformed)
        with self.assertRaisesRegex(ValueError, "CPU RNG state is malformed"):
            load_training_checkpoint(malformed)
        malformed_config_payload = load_training_checkpoint(result["resume_checkpoint"])
        malformed_config_payload["run_config"].pop("max_steps")
        malformed_config_payload["run_config"].pop("output_dir")
        malformed_config = Path(self.run_config.output_dir) / "malformed-run-config.resume.pt"
        atomic_torch_save(malformed_config_payload, malformed_config)
        with self.assertRaisesRegex(ValueError, "run_config is malformed"):
            load_training_checkpoint(malformed_config)

    def test_resume_repairs_metrics_from_durable_checkpoint_and_drops_future_events(self):
        first = train_run(self.run_config)
        metrics_path = Path(self.run_config.output_dir) / "metrics.jsonl"
        events = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
        # Simulate a crash immediately after the checkpoint commit but before
        # its pending training/validation events were durably published.
        events = [event for event in events if event.get("event_id") not in ("training_step:2", "validation:2")]
        events.append({"event_id": "training_step:99", "type": "training_step", "step": 99, "batch_nll": 1.0})
        metrics_path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
        resumed = train_run(replace(self.run_config, max_steps=3), resume_path=first["resume_checkpoint"])
        result = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
        step_events = [event for event in result if event.get("type") == "training_step"]
        validation_events = [event for event in result if event.get("type") == "validation"]
        self.assertEqual([event["step"] for event in step_events], [1, 2, 3])
        self.assertEqual([event["step"] for event in validation_events], [2, 3])
        self.assertEqual(len({event["event_id"] for event in result}), len(result))
        self.assertFalse(any(event.get("step") == 99 for event in result))
        self.assertEqual(resumed["completed_steps"], 3)

    def test_standalone_evaluation_writes_report_and_keeps_checkpoint_unchanged(self):
        result = train_run(self.run_config)
        checkpoint = Path(result["inference_checkpoint"])
        before_hash = sha256_file(checkpoint)
        output = self.root / "eval" / "report.json"
        report = evaluate_checkpoint(
            checkpoint,
            split="validation",
            data_path=self.validation_path,
            batch_size=2,
            device="cpu",
            output_path=output,
        )
        self.assertTrue(output.is_file())
        stored = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(stored["report_path"], str(output.resolve()))
        self.assertEqual(stored["checkpoint"]["sha256"], before_hash)
        self.assertEqual(stored["dataset"]["split"], "validation")
        self.assertEqual(stored["metrics"], report["metrics"])
        self.assertTrue(stored["measured"])
        self.assertFalse(stored["estimate"])
        self.assertEqual(sha256_file(checkpoint), before_hash)
        with self.assertRaisesRegex(ValueError, "explicitly 'validation' or 'test'"):
            evaluate_checkpoint(checkpoint, split="train", data_path=self.train_path)

    def test_evaluation_cli_uses_explicit_test_split_and_prints_measured_summary(self):
        result = train_run(self.run_config)
        checkpoint = Path(result["inference_checkpoint"])
        checkpoint_hash = sha256_file(checkpoint)
        report_path = self.root / "fresh-process-eval.json"
        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "brain.training",
                "evaluate",
                "--checkpoint",
                str(checkpoint),
                "--split",
                "test",
                "--data",
                str(self.test_path),
                "--batch-size",
                "2",
                "--output",
                str(report_path),
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["dataset"]["split"], "test")
        self.assertTrue(report["measured"])
        self.assertIn("MEASURED split=test", process.stdout)
        self.assertIn("target_tokens=", process.stdout)
        self.assertEqual(report["checkpoint"]["sha256"], checkpoint_hash)
        self.assertEqual(sha256_file(checkpoint), checkpoint_hash)

    def test_fresh_process_loads_inference_checkpoint_and_matches_greedy_generation(self):
        config = BrainConfig.load(self.model_config_path)
        model = DuoTransformer(config, seed=91).eval()
        checkpoint = self.root / "fresh-process.inference.pt"
        save_checkpoint(model, checkpoint)
        prompt = torch.tensor([[2, 71, 72]], dtype=torch.long)
        with torch.inference_mode():
            expected_logits = model(prompt)
        expected_generated = generate_greedy(model, prompt, 5)
        code = r'''import json,sys,torch
from brain.model import generate_greedy,load_checkpoint
m=load_checkpoint(sys.argv[1]).cpu().eval()
x=torch.tensor([[2,71,72]],dtype=torch.long)
with torch.inference_mode(): logits=m(x)
generated=generate_greedy(m,x,5)
print(json.dumps({"logits":logits.tolist(),"generated":generated.tolist()},allow_nan=False))
'''
        output = subprocess.run(
            [sys.executable, "-c", code, str(checkpoint)],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        actual = json.loads(output.stdout)
        self.assertTrue(torch.allclose(torch.tensor(actual["logits"]), expected_logits, rtol=0.0, atol=1e-6))
        self.assertTrue(torch.equal(torch.tensor(actual["generated"]), expected_generated))

    def test_cli_evaluation_runs_in_fresh_process_and_records_explicit_test_split(self):
        result = train_run(self.run_config)
        checkpoint = result["inference_checkpoint"]
        output = self.root / "fresh-evaluation.json"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "brain.training",
                "evaluate",
                "--checkpoint",
                checkpoint,
                "--split",
                "test",
                "--data",
                str(self.test_path),
                "--batch-size",
                "2",
                "--output",
                str(output),
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(report["dataset"]["split"], "test")
        self.assertEqual(report["dataset"]["path"], str(self.test_path.resolve()))
        self.assertEqual(report["checkpoint"]["sha256"], sha256_file(checkpoint))
        self.assertEqual(report["evaluation"]["device"], "cpu")
        self.assertGreater(report["metrics"]["target_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
