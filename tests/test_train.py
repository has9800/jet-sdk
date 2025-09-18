# tests/test_train.py
from unittest.mock import MagicMock
from easyllm.train import FineTuner
from easyllm.options import TrainOptions

def test_finetuner_offline(monkeypatch, tmp_path):
    # Mock Unsloth model + tokenizer
    fake_tok = MagicMock()
    fake_tok.pad_token = None
    fake_tok.eos_token = "<eos>"
    def fake_from_pretrained(model_name, **kw):
        fake_model = MagicMock()
        fake_model.config = MagicMock()
        return fake_model, fake_tok
    monkeypatch.setattr("easyllm.train.FastLanguageModel.from_pretrained", fake_from_pretrained)
    monkeypatch.setattr("easyllm.train.FastLanguageModel.get_peft_model", lambda m, **kw: m)

    # Mock TRL SFTTrainer
    calls = {"trained": False, "saved": False}
    class FakeTrainer:
        def __init__(self, **kw): pass
        def add_callback(self, cb): pass
        def train(self): calls["trained"] = True
        def save_model(self, out): 
            (tmp_path / "model").mkdir(parents=True, exist_ok=True)
            calls["saved"] = True
    monkeypatch.setattr("easyllm.train.SFTTrainer", FakeTrainer)

    # Mock MLflow
    class FakeRun:
        def __enter__(self): return self
        def __exit__(self, *a): pass
    monkeypatch.setattr("easyllm.train.mlflow.set_experiment", lambda *_: None)
    monkeypatch.setattr("easyllm.train.mlflow.start_run", lambda: FakeRun())
    monkeypatch.setattr("easyllm.train.mlflow.log_params", lambda *_: None)
    monkeypatch.setattr("easyllm.train.mlflow.log_artifacts", lambda *_: None)

    # Minimal Dataset-like
    fake_ds = type("D", (), {"__len__": lambda self: 2, "__iter__": lambda self: iter([{"text":"a"},{"text":"b"}])})()

    opts = TrainOptions(output_dir=str(tmp_path), epochs=1, use_4bit=False, bf16=False, flash_attn2=False)
    job = FineTuner("any/model", opts).train(fake_ds, eval_ds=None)
    assert calls["trained"] and calls["saved"]
    assert hasattr(job, "model_dir")