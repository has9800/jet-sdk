# tests/test_engine_hf.py
import os
from pathlib import Path
from jet.engine_hf import train

# Plain tokenizer stub (no MagicMock) to avoid mock internals issues. [web:1555]
class T:
    def __init__(self):
        self.pad_token = None
        self.eos_token = "</s>"
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.padding_side = "right"

    def save_pretrained(self, out_dir: str):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        # Minimal artifact so downstream code that inspects tokenizer dir won't fail
        with open(Path(out_dir) / "tokenizer.json", "w", encoding="utf-8") as f:
            f.write("{}")

# Plain model stub to safely attach generation_config without MagicMock attribute guards. [web:1555]
class M:
    def __init__(self):
        self.generation_config = None  # set by engine_hf.train
        self.config = object()

    def save_pretrained(self, out_dir: str):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(out_dir) / "config.json", "w", encoding="utf-8") as f:
            f.write("{}")

# Plain trainer stub compatible with TRL SFTTrainer signature that trains and saves. [web:1600]
class Trainer:
    def __init__(self, model=None, train_dataset=None, eval_dataset=None, args=None, **kwargs):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.args = args

    def train(self):
        # No-op for unit test
        return

    def save_model(self, out_dir: str):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        # Simulate model save
        with open(Path(out_dir) / "pytorch_model.bin", "wb") as f:
            f.write(b"\x00")

def test_engine_hf_persists_generation_config(monkeypatch, tmp_path):
    # Patch HF factory functions to return our plain stubs. [web:1600]
    monkeypatch.setattr("jet.engine_hf.AutoTokenizer.from_pretrained", lambda *a, **k: T())
    monkeypatch.setattr("jet.engine_hf.AutoModelForCausalLM.from_pretrained", lambda *a, **k: M())
    monkeypatch.setattr("jet.engine_hf.SFTTrainer", Trainer)

    # Minimal options namespace expected by engine_hf.train. [web:1600]
    Opts = type("Opts", (), {})
    opts = Opts()
    opts.model = "stub-model"
    opts.output_dir = str(tmp_path / "out")
    opts.per_device_batch = 1
    opts.grad_accum = 1
    opts.lr = 1e-4
    opts.epochs = 1
    opts.text_field = "text"
    opts.seed = 0
    # Use decoding preset to exercise generation_config path
    opts.decoding_preset = "balanced"

    # Minimal train/eval datasets
    train_ds = [{"text": "hello world"}]
    eval_ds = None

    job = train(opts, train_ds, eval_ds)
    assert hasattr(job, "model_dir") and os.path.isdir(job.model_dir)

    # Assert the engine persisted generation_config.json as part of artifacts. [web:1600]
    gc_path = Path(opts.output_dir) / "generation_config.json"
    assert gc_path.exists(), f"generation_config.json not found at {gc_path}"