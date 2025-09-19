# tests/test_decoding.py
from types import SimpleNamespace
from jet.decoding import make_generation_config

def test_balanced_preset_sets_expected_fields():
    tok = SimpleNamespace(pad_token_id=0, eos_token_id=1)
    class Opts(SimpleNamespace): pass
    opts = Opts(decoding_preset="balanced")
    gc = make_generation_config(tok, opts)
    assert gc.do_sample is True and gc.temperature == 0.7 and gc.top_p == 0.9 and gc.top_k == 50, gc  # balanced defaults [web:1361]
    assert gc.pad_token_id == 0 and gc.eos_token_id == 1, gc  # safe IDs [web:1361]

def test_explicit_overrides_take_precedence():
    tok = SimpleNamespace(pad_token_id=5, eos_token_id=6)
    class Opts(SimpleNamespace): pass
    opts = Opts(decoding_preset=None, do_sample=False, temperature=0.0, top_p=None, top_k=None,
                repetition_penalty=1.0, no_repeat_ngram_size=0, max_new_tokens=16)
    gc = make_generation_config(tok, opts)
    assert gc.do_sample is False and gc.temperature == 0.0 and gc.max_new_tokens == 16, gc  # deterministic [web:1361]
