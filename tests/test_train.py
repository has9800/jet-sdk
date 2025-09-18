from unittest.mock import MagicMock
import os
from jet.train import train_with_options  # current API entrypoint [web:1216]
from jet.options import TrainOptions
from jet.dataset import DatasetBuilder

def test_finetuner_offline(monkeypatch, tmp_path):
    # Create a tiny local dataset
    p = tmp_path / "sample.txt"
    p.write_text("hello world\nsecond sample\n")
    ds = DatasetBuilder(f"text:{p}", split="train", text_field="text").load()

    # Fake Unsloth loader/tokenizer
    fake_tok = MagicMock()
    fake_tok.pad_token = None
    fake_tok.eos_token = "<eos>"

    def fake_from_pretrained(model_name, **kw):
        fake_model = MagicMock()
        fake_model.config = MagicMock()
        return fake_model, fake_tok

    def fake_get_peft_model(model, **kw):
        return model  # no-op for offline test

    class FakeTrainer:
        def __init__(self, model=None, tokenizer=None, processing_class=None, train_dataset=None, eval_dataset=None, args=None):
            self.args = args
        def train(self):  # no-op training
            pass
        def save_model(self, outdir):
            os.makedirs(outdir, exist_ok=True)
            # create a sentinel file
            with open(os.path.join(outdir, "adapter_config.json"), "w") as f:
                f.write("{}")

    # IMPORTANT: patch the actual module used by our code
    monkeypatch.setattr("jet.engine_unsloth.FastLanguageModel.from_pretrained", fake_from_pretrained, raising=True)  # [web:393]
    monkeypatch.setattr("jet.engine_unsloth.FastLanguageModel.get_peft_model", fake_get_peft_model, raising=True)    # [web:393]
    monkeypatch.setattr("jet.engine_unsloth.SFTTrainer", FakeTrainer, raising=True)                                  # [web:1216]

    out = tmp_path / "model"
    opts = TrainOptions(model="dummy", engine="unsloth", epochs=1, max_seq=64, output_dir=str(out))
    job = train_with_options(opts, ds)
    assert job.model_dir == str(out)
    assert (out / "adapter_config.json").exists()
