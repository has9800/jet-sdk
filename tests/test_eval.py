# tests/test_eval.py
import sys
import types
from unittest.mock import MagicMock
from jet.eval import Evaluator

def test_evaluator_offline(monkeypatch, tmp_path):
    # Fake tokenizer returns tensors and decodes to a deterministic string
    class FakeTok(MagicMock):
        pad_token_id = 0
        eos_token_id = 1
        def __call__(self, x, return_tensors=None):
            return {"input_ids": [[1, 2]], "attention_mask": [[1, 1]]}
        def decode(self, x, skip_special_tokens=True):
            return "PROMPT response"

    # Fake model supports minimal generate signature only (to trigger fallback)
    class FakeModel(MagicMock):
        def generate(self, input_ids=None, attention_mask=None, max_new_tokens=64):
            return [[1, 2, 3]]

    monkeypatch.setattr("jet.eval.AutoTokenizer.from_pretrained", lambda *a, **k: FakeTok())
    monkeypatch.setattr("jet.eval.AutoModelForCausalLM.from_pretrained", lambda *a, **k: FakeModel())

    ev = Evaluator(str(tmp_path), do_sample=True, temperature=0.7, top_p=0.9, no_repeat_ngram_size=2)
    out = ev.evaluate(["PROMPT"], ["REF"])
    assert out["count"] == 1 and out["preds"][0].startswith("PROMPT"), out  # smoke with fallback [web:1361]

def test_evaluator_rouge(monkeypatch, tmp_path):
    # Patch tokenizer/model as before
    class FakeTok(MagicMock):
        pad_token_id = 0
        eos_token_id = 1
        def __call__(self, x, return_tensors=None): return {"input_ids": [[1,2]], "attention_mask": [[1,1]]}
        def decode(self, x, skip_special_tokens=True): return "hello world"
    class FakeModel(MagicMock):
        def generate(self, **kw): return [[1,2,3]]

    monkeypatch.setattr("jet.eval.AutoTokenizer.from_pretrained", lambda *a, **k: FakeTok())
    monkeypatch.setattr("jet.eval.AutoModelForCausalLM.from_pretrained", lambda *a, **k: FakeModel())

    # Mock evaluate.load("rouge") -> object with compute(...)
    def fake_load(name):
        assert name == "rouge"
        m = MagicMock()
        m.compute.return_value = {"rouge1": 0.5, "rouge2": 0.2, "rougeL": 0.4, "rougeLsum": 0.4}
        return m
    fake_evaluate = types.SimpleNamespace(load=fake_load)
    monkeypatch.setitem(sys.modules, "evaluate", fake_evaluate)

    ev = Evaluator(str(tmp_path))
    rep = ev.evaluate(["hi"], ["hi"])
    assert "metrics" in rep and rep["metrics"].get("rouge1") == 0.5, rep  # ROUGE reported [web:1487]

def test_evaluator_perplexity_hook(monkeypatch, tmp_path):
    class FakeTok(MagicMock):
        pad_token_id = 0
        eos_token_id = 1
        def __call__(self, x, return_tensors=None): return {"input_ids": [[1,2,3]]}
        def decode(self, x, skip_special_tokens=True): return "x"
    class FakeModel(MagicMock):
        def generate(self, **kw): return [[1,2,3]]

    monkeypatch.setattr("jet.eval.AutoTokenizer.from_pretrained", lambda *a, **k: FakeTok())
    monkeypatch.setattr("jet.eval.AutoModelForCausalLM.from_pretrained", lambda *a, **k: FakeModel())
    # Stub perplexity to avoid torch dependency in this test
    monkeypatch.setattr("jet.eval._compute_perplexity", lambda texts, model, tok: {"perplexity": 42.0})

    ev = Evaluator(str(tmp_path))
    rep = ev.evaluate(["a"], ["a"], perplexity_texts=["holdout"])
    assert rep["metrics"].get("perplexity") == 42.0, rep  # PPL included [web:1503]
