# src/jet/decoding.py
from transformers import GenerationConfig  # HF decoding config [web:1361]

def _safe_int_id(v):
    return v if isinstance(v, int) else None  # avoid mocks/objects in configs [web:1361]

_PRESETS = {
    "deterministic": dict(do_sample=False, temperature=0.0, top_p=None, top_k=None,
                          repetition_penalty=1.0, no_repeat_ngram_size=0, max_new_tokens=64),
    "balanced": dict(do_sample=True, temperature=0.7, top_p=0.9, top_k=50,
                     repetition_penalty=1.1, no_repeat_ngram_size=2, max_new_tokens=64),
    "creative": dict(do_sample=True, temperature=1.0, top_p=0.95, top_k=100,
                     repetition_penalty=1.05, no_repeat_ngram_size=2, max_new_tokens=128),
}  # curated bundles for common needs [web:1361]

def make_generation_config(tok, opts):
    if getattr(opts, "decoding_preset", None):
        cfg = {**_PRESETS[opts.decoding_preset]}
    else:
        cfg = dict(do_sample=opts.do_sample, temperature=opts.temperature,
                   top_p=opts.top_p, top_k=opts.top_k,
                   repetition_penalty=opts.repetition_penalty,
                   no_repeat_ngram_size=opts.no_repeat_ngram_size,
                   max_new_tokens=opts.max_new_tokens)
    pad_id = _safe_int_id(getattr(tok, "pad_token_id", None)) or _safe_int_id(getattr(tok, "eos_token_id", None))
    eos_id = _safe_int_id(getattr(tok, "eos_token_id", None))
    return GenerationConfig(pad_token_id=pad_id, eos_token_id=eos_id, **cfg)  # persisted with model [web:1361]
