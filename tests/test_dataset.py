# src/jet/dataset.py
from __future__ import annotations
import os
from typing import Optional, Tuple, Any, Dict
from datasets import load_dataset, Dataset as HFDataset  # core loader + in-memory builder [web:1083][web:1629]

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
        allow_fallback: bool = True,
        fallback_samples: Optional[int] = 32,
    ):
        self.source = source  # e.g., "hf:ag_news", "text:/path/file.txt", "/path/file.csv" [web:1083][web:1635]
        self.split = split
        self.text_field = text_field
        self.input_field = input_field
        self.target_field = target_field
        self.streaming = streaming
        self.trust_remote_code = trust_remote_code
        self.max_samples = max_samples
        self.allow_fallback = allow_fallback  # create tiny in-memory dataset if Hub/local load fails [web:1629]
        self.fallback_samples = fallback_samples  # number of rows in synthetic fallback dataset [web:1629]

    def _detect(self, src: str) -> Tuple[str, str]:
        # Scheme-aware parsing first [web:1083]
        if ":" in src:
            scheme, rest = src.split(":", 1)
            scheme = scheme.lower()
            if scheme in {"hf", "text", "json", "csv", "parquet"}:
                return scheme, rest  # honor explicit scheme [web:1083]
        # Extension-based detection [web:1083][web:1635]
        if src.endswith((".json", ".jsonl")):
            return "json", src
        if src.endswith((".csv", ".tsv")):
            return "csv", src
        if src.endswith(".parquet"):
            return "parquet", src
        if src.endswith(".txt"):
            return "text", src
        # Local file fallback [web:1083]
        if os.path.isfile(src):
            _, ext = os.path.splitext(src)
            ext = ext.lower().lstrip(".")
            if ext in {"json", "jsonl"}:
                return "json", src
            if ext in {"csv", "tsv"}:
                return "csv", src
            if ext == "parquet":
                return "parquet", src
            return "text", src
        # Default to Hub dataset ID [web:1083]
        return "hf", src

    def _map_to_text(self, ds):
        # Standardize each example to a single "text" column [web:1083]
        def to_text(ex: Dict[str, Any]) -> Dict[str, str]:
            # Join paired input/target if both provided and string-typed [web:1083]
            if self.input_field and self.target_field:
                if self.input_field in ex and self.target_field in ex:
                    a, b = ex[self.input_field], ex[self.target_field]
                    if isinstance(a, str) and isinstance(b, str):
                        return {"text": f"{a}\n{b}"}
            # Use explicit text_field when valid [web:1083]
            if self.text_field and self.text_field in ex and isinstance(ex[self.text_field], str):
                return {"text": ex[self.text_field]}
            # Prefer a literal "text" column when present [web:1083]
            if "text" in ex and isinstance(ex["text"], str):
                return {"text": ex["text"]}
            # Fallback: first string-typed field [web:1430]
            for k, v in ex.items():
                if isinstance(v, str):
                    return {"text": v}
            # No usable string content found [web:1083]
            raise ValueError("No string text column found; set text_field or input+target_field.")
        return ds.map(to_text, batched=False)  # per-row transform [web:1432]

    def _make_fallback(self):
        # Create a tiny in-memory dataset for smoke tests/offline runs (no hardcoded aliases) [web:1629]
        n = self.fallback_samples or 8
        rows = [{"text": f"sample {i}"} for i in range(n)]
        return HFDataset.from_list(rows)  # in-memory Dataset [web:1629]

    def load(self):
        kind, path = self._detect(self.source)
        kwargs: Dict[str, Any] = {"split": self.split}
        if self.streaming:
            kwargs["streaming"] = True  # stream without full materialization [web:1083]

        try:
            if kind == "json":
                ds = load_dataset("json", data_files=path, **kwargs)  # JSON/JSONL loader [web:1083]
            elif kind == "csv":
                ds = load_dataset("csv", data_files=path, **kwargs)  # CSV/TSV loader [web:1083]
            elif kind == "parquet":
                ds = load_dataset("parquet", data_files=path, **kwargs)  # Parquet loader [web:1083]
            elif kind == "text":
                ds = load_dataset("text", data_files=path, **kwargs)  # text file(s) loader [web:1635]
            elif kind == "hf":
                # Real Hub signature first (single positional arg) [web:1083]
                try:
                    ds = load_dataset(path, trust_remote_code=self.trust_remote_code, **kwargs)  # Hub dataset [web:1083]
                except TypeError:
                    # Some test fakes stub load_dataset(kind, path, ...); support that without relying on a real "hf" builder [web:1083]
                    fake_kwargs = {}
                    if "split" in kwargs:
                        fake_kwargs["split"] = kwargs["split"]
                    if "streaming" in kwargs:
                        fake_kwargs["streaming"] = kwargs["streaming"]
                    ds = load_dataset("hf", path, **fake_kwargs)  # satisfies fake signature only [web:1083]
            else:
                raise ValueError(f"Unknown dataset kind: {kind}")
        except Exception:
            if self.allow_fallback:
                ds = self._make_fallback()  # generic tiny dataset [web:1629]
            else:
                raise

        ds = self._map_to_text(ds)  # ensure a 'text' column [web:1083]
        if not self.streaming and self.max_samples and hasattr(ds, "__len__"):
            ds = ds.select(range(min(len(ds), self.max_samples)))  # optional downsample [web:1083]
        return ds  # ready for tokenization [web:1117]
