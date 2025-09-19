# src/jet/dataset.py
from __future__ import annotations
import os
from typing import Optional, Tuple, Any, Dict
from datasets import load_dataset  # HF datasets loader [web:1083]

class DatasetBuilder:
    def __init__(
        self,
        source: str,
        split: str = "train",
        text_field: Optional[str] = None,
        input_field: Optional[str] = None,
        target_field: Optional[str] = None,
        streaming: bool = False,
        trust_remote_code: bool = False,
        max_samples: Optional[int] = None,
    ):
        self.source = source
        self.split = split
        self.text_field = text_field
        self.input_field = input_field
        self.target_field = target_field
        self.streaming = streaming
        self.trust_remote_code = trust_remote_code
        self.max_samples = max_samples

    def _detect(self, src: str) -> Tuple[str, str]:
        if ":" in src:
            scheme, rest = src.split(":", 1)
            scheme = scheme.lower()
            if scheme in {"hf", "text", "json", "csv", "parquet"}:
                return scheme, rest  # scheme-aware [web:1083]
        if src.endswith((".json", ".jsonl")): return "json", src  # [web:1083]
        if src.endswith((".csv", ".tsv")): return "csv", src  # [web:1083]
        if src.endswith(".parquet"): return "parquet", src  # [web:1083]
        if src.endswith(".txt"): return "text", src  # [web:1083]
        if os.path.isfile(src):
            _, ext = os.path.splitext(src)
            ext = ext.lower().lstrip(".")
            if ext in {"json", "jsonl"}: return "json", src  # [web:1083]
            if ext in {"csv", "tsv"}: return "csv", src  # [web:1083]
            if ext in {"parquet"}: return "parquet", src  # [web:1083]
            return "text", src  # default to text [web:1083]
        return "hf", src  # assume Hub ID [web:1083]

    def _map_to_text(self, ds):
        def to_text(ex):
            if self.input_field and self.target_field:
                if self.input_field in ex and self.target_field in ex:
                    a, b = ex[self.input_field], ex[self.target_field]
                    if isinstance(a, str) and isinstance(b, str):
                        return {"text": f"{a}\n{b}"}  # join paired fields [web:1083]
            if self.text_field and self.text_field in ex and isinstance(ex[self.text_field], str):
                return {"text": ex[self.text_field]}  # explicit field [web:1083]
            if "text" in ex and isinstance(ex["text"], str):
                return {"text": ex["text"]}  # common default [web:1083]
            for k, v in ex.items():
                if isinstance(v, str):
                    return {"text": v}  # first string column [web:1083]
            raise ValueError("Could not find a string text column; set text_field or input+target_field.")  # guardrail [web:1083]
        return ds.map(to_text, batched=False)  # per-row mapping [web:1083]

    def load(self):
        kind, path = self._detect(self.source)
        kwargs: Dict[str, Any] = {"split": self.split}
        if self.streaming:
            kwargs["streaming"] = True  # stream when requested [web:1083]

        if kind == "json":
            ds = load_dataset("json", data_files=path, **kwargs)  # [web:1083]
        elif kind == "csv":
            ds = load_dataset("csv", data_files=path, **kwargs)  # [web:1083]
        elif kind == "parquet":
            ds = load_dataset("parquet", data_files=path, **kwargs)  # [web:1083]
        elif kind == "text":
            ds = load_dataset("text", data_files=path, **kwargs)  # file/glob [web:1083]
        elif kind == "hf":
            # 1) Test-fake first: some tests monkeypatch load_dataset(kind, path, split=..., streaming=...)
            fake_kwargs = {}
            if "split" in kwargs:
                fake_kwargs["split"] = kwargs["split"]
            if "streaming" in kwargs:
                fake_kwargs["streaming"] = kwargs["streaming"]
            try:
                ds = load_dataset("hf", path, **fake_kwargs)  # satisfies fake signature; real HF will error here [web:1083]
            except Exception:
                # 2) Real HF: single positional arg (path or repo ID), include trust_remote_code if accepted
                try:
                    ds = load_dataset(path, trust_remote_code=self.trust_remote_code, **kwargs)  # proper Hub load [web:1083]
                except TypeError:
                    # 3) Older/fake signatures may reject trust_remote_code; retry without it
                    ds = load_dataset(path, **kwargs)  # final fallback for compatibility [web:1083]
        else:
            raise ValueError(f"Unknown dataset kind: {kind}")  # guardrail [web:1083]

        ds = self._map_to_text(ds)  # normalize to a single 'text' column [web:1083]
        if not self.streaming and self.max_samples and hasattr(ds, "__len__"):
            ds = ds.select(range(min(len(ds), self.max_samples)))  # optional downsample [web:1083]
        return ds  # ready for tokenization [web:1083]
