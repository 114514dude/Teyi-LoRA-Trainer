import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib
import unittest


PROJECT = Path(__file__).resolve().parents[1]
BRIDGE = PROJECT / "backend" / "backend_bridge.py"


class BackendBridgeIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        core = self.root / "app" / "backend" / "core"
        core.mkdir(parents=True)
        (core / "train.py").write_text(
            "import argparse, tomllib\n"
            "p=argparse.ArgumentParser(); p.add_argument('--config'); p.add_argument('--deepspeed', action='store_true'); a=p.parse_args()\n"
            "with open(a.config, 'rb') as f: c=tomllib.load(f)\n"
            "config=c\n"
            "class Value:\n"
            "  def __init__(self, value): self.value=value\n"
            "  def item(self): return self.value\n"
            "class Engine:\n"
            "  def train_batch(self, iterator): return Value(0.25 / iterator)\n"
            "model_engine=Engine(); iterator=1; optimizer=type('Optimizer', (), {'param_groups':[{'lr': 0.0001}]})()\n"
            "steps_per_epoch = 1\n\n"
            "step=1; epoch=1\n"
            "while step <= 2:\n"
            "  loss = model_engine.train_batch(iterator).item()\n"
            "  print(f'Started new epoch: {epoch}', flush=True)\n"
            "  step += 1; epoch += 1\n"
            "print('TRAIN_STUB_OK', c['model']['type'], flush=True)\n",
            encoding="utf-8",
        )
        dataset = self.root / "dataset"
        dataset.mkdir()
        (dataset / "one.png").write_bytes(b"image")
        (dataset / "one.txt").write_text("anima test", encoding="utf-8")
        validation = dataset / "validation"
        validation.mkdir()
        (validation / "eval.png").write_bytes(b"image")
        (validation / "eval.txt").write_text("validation", encoding="utf-8")
        model = self.root / "models"
        model.mkdir()
        for name in ("transformer.safetensors", "vae.safetensors", "t5.safetensors"):
            (model / name).write_bytes(b"weights")
        (model / "llm.safetensors").write_bytes(b"weights")
        self.config = {
            "dataset": {
                "input_path": str(dataset), "resolutions": "[[768, 1024]]",
                "enable_ar_bucket": True, "min_ar": 0.5, "max_ar": 2,
                "num_ar_buckets": 7, "num_repeats": 2, "skip_empty_caption": True,
                "eval_subdir": "validation",
            },
            "model": {
                "backend_type": "anima", "dtype": "bfloat16", "transformer_dtype": "bfloat16",
                "timestep_sample_method": "logit_normal",
                "paths": {
                    "transformer_path": str(model / "transformer.safetensors"),
                    "vae_path": str(model / "vae.safetensors"),
                    "t5_path": str(model / "t5.safetensors"), "llm_path": str(model / "llm.safetensors"),
                },
                "required_paths": [
                    {"key": "transformer_path", "kind": "file"}, {"key": "vae_path", "kind": "file"},
                    {"key": "t5_path", "kind": "file"}, {"key": "llm_path", "kind": "file"},
                ],
            },
            "adapter": {"type": "LoRA", "rank": 32, "dropout": 0.05, "lr": "1e-4", "text_encoder_lr": "5e-5"},
            "training": {
                "output_folder_name": "anima_lora", "output_path": str(self.root / "output"),
                "epochs": 2, "micro_batch_size_per_gpu": 1, "gradient_accumulation_steps": 2,
                "warmup_steps": 10, "save_every_n_steps": 100, "steps_per_print": 5,
                "optimizer_config": json.dumps({"type": "AdamW", "scheduler": "Cosine"}),
            },
            "start_args": {},
        }

    def tearDown(self):
        self.temp.cleanup()

    def invoke(self, action, extra_env=None):
        return subprocess.run(
            [sys.executable, str(BRIDGE), "--action", action, "--trainer_root", str(self.root),
             "--full_config", json.dumps(self.config)],
            text=True, capture_output=True, encoding="utf-8", timeout=20,
            env={**os.environ, "LORA_TRAINER_DIRECT": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", **(extra_env or {})},
        )

    def payload(self, output):
        raw = output.split("__JSON_START__")[-1].split("__JSON_END__")[0]
        return json.loads(raw)

    def test_generate_real_toml_files(self):
        result = self.invoke("generate_config")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = self.payload(result.stdout)
        with open(payload["config_path"], "rb") as stream:
            train = tomllib.load(stream)
        with open(payload["dataset_config"], "rb") as stream:
            dataset = tomllib.load(stream)
        self.assertEqual(train["model"]["type"], "anima")
        self.assertEqual(Path(train["model"]["t5_path"]), Path(self.config["model"]["paths"]["t5_path"]))
        self.assertEqual(train["optimizer"]["type"], "adamw_optimi")
        self.assertEqual(dataset["resolutions"], [[768, 1024]])
        self.assertTrue(payload["eval_config"])

    def test_start_executes_original_train_entry(self):
        result = self.invoke("start_training")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("TRAIN_STUB_OK anima", result.stdout)
        self.assertIn('"status": "TRAINING_PLAN"', result.stdout)
        self.assertIn('"totalSteps": 2', result.stdout)
        self.assertIn('"status": "PROGRESS"', result.stdout)
        self.assertIn('"loss": 0.25', result.stdout)

    def test_nonzero_rank_does_not_duplicate_core_events(self):
        result = self.invoke("start_training", {"RANK": "1", "LOCAL_RANK": "1"})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("TRAIN_STUB_OK anima", result.stdout)
        self.assertNotIn('"status": "TRAINING_PLAN"', result.stdout)
        self.assertNotIn('"status": "PROGRESS"', result.stdout)

    def test_missing_required_model_path_fails_before_launch(self):
        self.config["model"]["paths"]["llm_path"] = str(self.root / "missing.safetensors")
        result = self.invoke("validate_config")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("模型路径无效", self.payload(result.stdout)["message"])

    def test_nested_dataset_is_emitted_as_core_directory(self):
        nested = Path(self.config["dataset"]["input_path"]) / "character_a"
        nested.mkdir()
        (nested / "two.png").write_bytes(b"image")
        (nested / "two.txt").write_text("nested caption", encoding="utf-8")
        result = self.invoke("generate_config")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = self.payload(result.stdout)
        with open(payload["dataset_config"], "rb") as stream:
            dataset = tomllib.load(stream)
        emitted = {Path(item["path"]) for item in dataset["directory"]}
        self.assertIn(nested, emitted)
        self.assertNotIn(Path(self.config["dataset"]["input_path"]) / "validation", emitted)

    def test_unsupported_model_omits_block_swap(self):
        checkpoint = self.root / "models" / "sdxl.safetensors"
        checkpoint.write_bytes(b"weights")
        self.config["model"] = {
            "backend_type": "sdxl",
            "supports_block_swap": True,
            "dtype": "bfloat16",
            "transformer_dtype": "bfloat16",
            "timestep_sample_method": "logit_normal",
            "paths": {"checkpoint_path": str(checkpoint)},
            "required_paths": [{"key": "checkpoint_path", "kind": "file"}],
        }
        self.config["training"]["blocks_to_swap"] = 20
        result = self.invoke("generate_config")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = self.payload(result.stdout)
        with open(payload["config_path"], "rb") as stream:
            train = tomllib.load(stream)
        self.assertNotIn("blocks_to_swap", train)

    def test_supported_model_keeps_block_swap(self):
        self.config["model"]["supports_block_swap"] = True
        self.config["training"]["blocks_to_swap"] = 12
        result = self.invoke("generate_config")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = self.payload(result.stdout)
        with open(payload["config_path"], "rb") as stream:
            train = tomllib.load(stream)
        self.assertEqual(train["blocks_to_swap"], 12)


if __name__ == "__main__":
    unittest.main()
