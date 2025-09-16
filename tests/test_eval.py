# tests/test_eval.py
from unittest.mock import MagicMock
from easyllm.eval import Evaluator

def test_evaluator_offline(monkeypatch, tmp_path):
    # Fake tokenizer & model
    monkeypatch.setattr("easyllm.eval.AutoTokenizer.from_pretrained", lambda *a, **k: MagicMock(
        return_tensors="pt", 
        **{"__call__": lambda self, x, return_tensors=None: {"input_ids":[[1,2]], "attention_mask":[[1,1]]},
           "decode": lambda self, x, skip_special_tokens=True: "PROMPT_1 response"}
    ))
    class FakeModel(MagicMock):
        def generate(self, **kw): return [[1,2,3]]
        def __call__(self, **kw):
            out = MagicMock()
            out.loss = 0.0  # perplexity = exp(0) = 1
            return out
        def eval(self): return self
        def to(self, *a): return self
    monkeypatch.setattr("easyllm.eval.AutoModelForCausalLM.from_pretrained", lambda *a, **k: FakeModel())

    # Fake rouge
    monkeypatch.setattr("easyllm.eval.hf_evaluate.load", lambda name: MagicMock(compute=lambda **kw: {"rouge1":0.5,"rouge2":0.3,"rougeL":0.4}))

    # Mock MLflow logging
    monkeypatch.setattr("easyllm.eval.mlflow.log_dict", lambda *_: None)

    rep = Evaluator(str(tmp_path)).evaluate(["PROMPT_1","P2","P3"], ["ref1","ref2","ref3"])
    assert "rouge" in rep and "mean_ppl" in rep
