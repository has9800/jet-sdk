# tests/test_eval.py (patch the targets to current module path)
from unittest.mock import MagicMock
from jet.eval import Evaluator  # updated import path [web:1106]

def test_evaluator_offline(monkeypatch, tmp_path):
    # Fake tokenizer: mimic __call__ returning tensors and decode returning a string [web:1106]
    class FakeTok(MagicMock):
        pad_token = "<|pad|>"
        eos_token = "</s>"
        def __call__(self, x, return_tensors=None):
            return {"input_ids": [[1, 2]], "attention_mask": [[1, 1]]}
        def decode(self, x, skip_special_tokens=True):
            return "PROMPT_1 response"

    # Fake model: mimic generate returning token ids [web:1106]
    class FakeModel(MagicMock):
        def generate(self, input_ids=None, attention_mask=None, max_new_tokens=16):
            return [[1, 2, 3]]

    # Patch the correct symbols under jet.eval so the code uses fakes offline [web:1106]
    monkeypatch.setattr("jet.eval.AutoTokenizer.from_pretrained", lambda *a, **k: FakeTok())
    monkeypatch.setattr("jet.eval.AutoModelForCausalLM.from_pretrained", lambda *a, **k: FakeModel())

    # Run evaluator end-to-end without network/files [web:1106]
    ev = Evaluator(str(tmp_path), bf16=False)
    rep = ev.evaluate(["PROMPT_1"], ["REF_1"])
    assert isinstance(rep, dict) and rep.get("count") == 1 and rep["preds"][0].startswith("PROMPT_1")  # smoke [web:1106]
