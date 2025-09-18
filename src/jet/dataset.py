from typing import Optional
from datasets import load_dataset

def _is_url(path: str) -> bool:
    return path.startswith("http://") or path.startswith("https://")

class DatasetBuilder:
    def __init__(
        self, source: str, split: str = "train",
        text_field: Optional[str] = None,
        input_field: Optional[str] = None,
        target_field: Optional[str] = None,
        streaming: bool = False,
        max_samples: Optional[int] = None,
    ):
        self.source, self.split = source, split
        self.text_field, self.input_field, self.target_field = text_field, input_field, target_field
        self.streaming, self.max_samples = streaming, max_samples

    def _detect(self, src: str):
        # schemes: text:, csv:, json:, parquet:, hf:, or raw (assume HF id)
        if ":" in src and not _is_url(src):
            kind, path = src.split(":", 1)
            return kind, path
        if _is_url(src):
            # infer kind by extension for remote files
            if src.endswith(".csv"): return "csv", src
            if src.endswith(".json") or src.endswith(".jsonl"): return "json", src
            if src.endswith(".parquet"): return "parquet", src
            if src.endswith(".txt"): return "text", src
            return "auto", src
        return "hf", src  # org/name id [web:1083]

    def load(self):
        kind, path = self._detect(self.source)
        kwargs = {"split": self.split}
        if self.streaming:
            kwargs["streaming"] = True

        if kind == "json":
            ds = load_dataset("json", data_files=path, **kwargs)  # local or remote [web:1083]
        elif kind == "csv":
            ds = load_dataset("csv", data_files=path, **kwargs)   # local or remote [web:1083]
        elif kind == "parquet":
            ds = load_dataset("parquet", data_files=path, **kwargs)  # local or remote [web:1083]
        elif kind == "text":
            ds = load_dataset("text", data_files=path, **kwargs)  # local or remote [web:1083]
        elif kind == "hf":
            ds = load_dataset(path, **kwargs)  # HF dataset id [web:1083]
        else:
            ds = load_dataset(path, **kwargs)  # fallback [web:1083]

        def to_text(ex):
            if self.input_field and self.target_field:
                return {"text": f"{ex[self.input_field]}\n{ex[self.target_field]}"}
            if self.text_field:
                return {"text": ex[self.text_field]}
            # fallback: pick first string field
            for k, v in ex.items():
                if isinstance(v, str):
                    return {"text": v}
            raise ValueError("Specify text_field or input+target_field")

        ds = ds.map(to_text, batched=False)
        if not self.streaming and self.max_samples and hasattr(ds, "__len__"):
            ds = ds.select(range(min(len(ds), self.max_samples)))
        return ds
