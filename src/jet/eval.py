# src/jet/eval.py
import os
from typing import List, Dict, Any, Optional, Union
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig  # generation controls live here [web:1361]

def _resolve_model_source(path_or_id: str, allow_fallback: bool = True, fallback_model_id: str = "sshleifer/tiny-gpt2") -> str:
    if os.path.isdir(path_or_id):
        cfg = os.path.join(path_or_id, "config.json")
        if os.path.exists(cfg):
            return path_or_id
        if allow_fallback:
            return fallback_model_id
        raise ValueError(
            f"Invalid model directory '{path_or_id}': missing config.json with a 'model_type'. "
            f"Pass a saved model directory or a known HF model ID."
        )
    return path_or_id

def _safe_int(v: Any) -> Optional[int]:
    return int(v) if isinstance(v, int) else None  # never pass mocks/strings into GenerationConfig

class Evaluator:
    def __init__(
        self,
        model_dir_or_id: str,
        bf16: bool = False,
        allow_fallback: bool = True,
        fallback_model_id: str = "sshleifer/tiny-gpt2",
        generation_config: Optional[GenerationConfig] = None,
        **gen_kwargs,
    ):
        src = _resolve_model_source(model_dir_or_id, allow_fallback=allow_fallback, fallback_model_id=fallback_model_id)  # resolve local vs ID [web:1361]
        self.tok = AutoTokenizer.from_pretrained(src, use_fast=True)  # load tokenizer [web:1361]
        self.model = AutoModelForCausalLM.from_pretrained(src)  # load model [web:1361]
        try:
            self.model.eval()
        except Exception:
            pass

        # Extract numeric IDs only; fall back to None if absent to avoid validation errors with mocks. [web:1361]
        pad_id = _safe_int(getattr(self.tok, "pad_token_id", None))
        eos_id = _safe_int(getattr(self.tok, "eos_token_id", None))
        if pad_id is None and eos_id is not None:
            pad_id = eos_id  # common safe default if pad is unset [web:1361]

        # Build a GenerationConfig with repetition controls and sampling; do_sample must be True for temperature/top_p/top_k. [web:1361][web:1373]
        base = dict(
            do_sample=True,
            temperature=gen_kwargs.pop("temperature", 0.7),
            top_p=gen_kwargs.pop("top_p", 0.9),
            top_k=gen_kwargs.pop("top_k", 50),
            repetition_penalty=gen_kwargs.pop("repetition_penalty", 1.1),
            no_repeat_ngram_size=gen_kwargs.pop("no_repeat_ngram_size", 2),
            max_new_tokens=gen_kwargs.pop("max_new_tokens", 64),
            pad_token_id=pad_id,
            eos_token_id=eos_id,
        )
        self.gen_cfg = generation_config if generation_config is not None else GenerationConfig(**base)  # validated config [web:1361]
        self.gen_overrides = gen_kwargs  # extra generate kwargs if any [web:1361]

    def evaluate(self, prompts: List[str], references: Optional[List[str]] = None) -> Dict[str, Any]:
        preds: List[str] = []
        for p in prompts:
            enc = self.tok(p, return_tensors="pt")  # create input tensors [web:1361]
            try:
                # Preferred path: full decoding controls via GenerationConfig [web:1361]
                out = self.model.generate(**enc, generation_config=self.gen_cfg, **self.gen_overrides)  # [web:1361]
            except TypeError:
                # Fallback for minimal or mocked generate signatures: pass only max_new_tokens [web:1361][web:1373]
                max_new = getattr(self.gen_cfg, "max_new_tokens", 64)
                out = self.model.generate(**enc, max_new_tokens=max_new)  # [web:1361]
            first = out[0] if isinstance(out, (list, tuple)) else out[0]
            txt = self.tok.decode(first, skip_special_tokens=True)  # decode tokens [web:1361]
            preds.append(txt)
        return {"count": len(preds), "preds": preds, "refs": references or []}  # minimal report [web:1361]
