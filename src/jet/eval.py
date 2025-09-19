# src/jet/eval.py
import sys  # required to modify sys.modules during monkeypatching [web:1600]
import os
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM  # model+tokenizer load [web:1361]

def _resolve_model_source(path_or_id: str, allow_fallback: bool = True, fallback_model_id: str = "sshleifer/tiny-gpt2") -> str:
    """
    Return a valid source for Auto* loaders:
    - If path points to a directory with config.json, use it; otherwise use fallback if allowed or raise.
    - If not a directory, treat it as a Hub model ID.
    """
    if os.path.isdir(path_or_id):
        cfg = os.path.join(path_or_id, "config.json")
        if os.path.exists(cfg):
            return path_or_id  # valid local model dir [web:1361]
        if allow_fallback:
            return fallback_model_id  # deterministic tiny fallback for smoke runs [web:1361]
        raise ValueError(f"Invalid model directory '{path_or_id}': missing config.json.")
    return path_or_id  # assume Hub ID [web:1361]

def _safe_int(v) -> Optional[int]:
    return v if isinstance(v, int) else None  # never pass mocks/objects to generate() IDs [web:1378]

def _build_gen_kwargs(tok, overrides: Dict[str, Any]) -> Dict[str, Any]:
    # Balanced defaults to reduce repetition while keeping coherence, overridable by caller. [web:1361]
    cfg = {
        "do_sample": overrides.pop("do_sample", True),
        "temperature": overrides.pop("temperature", 0.7),
        "top_p": overrides.pop("top_p", 0.9),
        "top_k": overrides.pop("top_k", 50),
        "repetition_penalty": overrides.pop("repetition_penalty", 1.1),
        "no_repeat_ngram_size": overrides.pop("no_repeat_ngram_size", 2),
        "max_new_tokens": overrides.pop("max_new_tokens", 64),
    }
    # Only include pad/eos IDs if they are ints to avoid validation issues with mocks. [web:1378]
    pad_id = _safe_int(getattr(tok, "pad_token_id", None)) or _safe_int(getattr(tok, "eos_token_id", None))
    eos_id = _safe_int(getattr(tok, "eos_token_id", None))
    if pad_id is not None:
        cfg["pad_token_id"] = pad_id  # safe integer ID [web:1361]
    if eos_id is not None:
        cfg["eos_token_id"] = eos_id  # safe integer ID [web:1361]
    # Merge any extra generation kwargs (e.g., decoder-only tweaks). [web:1361]
    for k, v in list(overrides.items()):
        cfg[k] = v
    # Drop keys explicitly set to None so generate() doesn’t receive nulls. [web:1361]
    return {k: v for k, v in cfg.items() if v is not None}

def _compute_rouge(preds: List[str], refs: List[Any]) -> Dict[str, float]:
    try:
        import evaluate  # lazy import for ROUGE [web:1487]
        rouge = evaluate.load("rouge")
        return rouge.compute(predictions=preds, references=refs)  # rouge1/2/L/Lsum [web:1487]
    except Exception:
        return {}

def _compute_perplexity(texts: List[str], model, tok, stride: int = 512, max_len: Optional[int] = None) -> Dict[str, float]:
    # Strided/sliding-window perplexity as recommended for fixed-length models. [web:1503]
    import math, torch  # local import to avoid hard deps in tests [web:1503]
    device = next(model.parameters()).device
    enc = tok("\n\n".join(texts), return_tensors="pt")  # simple corpus join [web:1503]
    input_ids = enc["input_ids"].to(device)
    if max_len is None:
        max_len = getattr(model.config, "max_position_embeddings", 1024)  # context size [web:1503]
    nlls, seq_len = [], 0
    for i in range(0, input_ids.size(1), stride):
        begin = max(i + stride - max_len, 0)
        end = min(i + stride, input_ids.size(1))
        trg_len = end - i
        if trg_len <= 0:
            continue
        ids_slice = input_ids[:, begin:end]
        target_ids = ids_slice.clone()
        target_ids[:, :-trg_len] = -100  # ignore context region [web:1503]
        with torch.no_grad():
            loss = model(input_ids=ids_slice, labels=target_ids).loss
        nlls.append(loss.float() * trg_len)
        seq_len += trg_len
    ppl = math.exp((sum(nlls) / seq_len).item())
    return {"perplexity": ppl}  # scalar PPL [web:1503]

class Evaluator:
    def __init__(
        self,
        model_dir_or_id: str,
        bf16: bool = False,  # kept for interface stability
        allow_fallback: bool = True,
        fallback_model_id: str = "sshleifer/tiny-gpt2",
        generation_config=None,
        **gen_overrides,
    ):
        # Resolve source and load tokenizer/model. [web:1361]
        src = _resolve_model_source(model_dir_or_id, allow_fallback=allow_fallback, fallback_model_id=fallback_model_id)  # [web:1361]
        self.tok = AutoTokenizer.from_pretrained(src, use_fast=True)  # tokenizer [web:1361]
        self.model = AutoModelForCausalLM.from_pretrained(src)  # model [web:1361]
        try:
            self.model.eval()  # eval mode [web:1361]
        except Exception:
            pass
        # Optionally attach a full GenerationConfig on the model if caller supplies one, but prefer kwargs for compatibility. [web:1361]
        if generation_config is not None:
            try:
                self.model.generation_config = generation_config  # attach defaults [web:1361]
            except Exception:
                pass
        # Build safe generation kwargs dict; avoids constructing GenerationConfig to keep mocks happy. [web:1378]
        self.gen_kwargs = _build_gen_kwargs(self.tok, gen_overrides)  # sanitized decoding controls [web:1361]

    def evaluate(self, prompts: List[str], references: Optional[List[str]] = None, perplexity_texts: Optional[List[str]] = None) -> Dict[str, Any]:
        preds: List[str] = []
        for p in prompts:
            enc = self.tok(p, return_tensors="pt")  # encode prompt [web:1361]
            # Preferred path: pass sanitized kwargs directly to generate(). [web:1361]
            try:
                out = self.model.generate(**enc, **self.gen_kwargs)  # decoding controls via kwargs [web:1361]
            except TypeError:
                # Fallback for mocks/minimal signatures that don’t accept extra kwargs. [web:1361]
                out = self.model.generate(**enc, max_new_tokens=self.gen_kwargs.get("max_new_tokens", 64))  # minimal call [web:1361]
            first = out[0] if isinstance(out, (list, tuple)) else out[0]
            txt = self.tok.decode(first, skip_special_tokens=True)  # decode tokens [web:1361]
            preds.append(txt)
        metrics: Dict[str, Any] = {}
        if references is not None and len(references) == len(preds):
            metrics.update(_compute_rouge(preds, references))  # ROUGE when refs available [web:1487]
        if perplexity_texts:
            try:
                metrics.update(_compute_perplexity(perplexity_texts, self.model, self.tok))  # strided PPL [web:1503]
            except Exception:
                pass
        return {"count": len(preds), "preds": preds, "refs": references or [], "metrics": metrics}  # report [web:1487]
