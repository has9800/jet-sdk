# easyllm/dataset.py
from datasets import load_dataset, Dataset
from typing import Optional, Callable
import os

class DatasetBuilder:
    def __init__(self,
                 source: str,
                 split: str = "train",
                 text_field: Optional[str] = None,
                 input_field: Optional[str] = None,
                 target_field: Optional[str] = None,
                 streaming: bool = False,
                 max_samples: Optional[int] = None):
        self.source = source
        self.split = split
        self.text_field = text_field
        self.input_field = input_field
        self.target_field = target_field
        self.streaming = streaming
        self.max_samples = max_samples

    def _detect(self, path: str):
        if ":" in path and os.path.sep in path:
            kind, loc = path.split(":", 1)
            return kind, loc
        return "auto", path

    def load(self) -> Dataset:
        kind, loc = self._detect(self.source)
        kwargs = {"split": self.split}
        if self.streaming:
            kwargs["streaming"] = True

        if kind == "json":
            ds = load_dataset("json", data_files=loc, **kwargs)
        elif kind == "csv":
            ds = load_dataset("csv", data_files=loc, **kwargs)
        elif kind == "parquet":
            ds = load_dataset("parquet", data_files=loc, **kwargs)
        elif kind == "text":
            ds = load_dataset("text", data_files=loc, **kwargs)
        else:
            ds = load_dataset(loc, **kwargs)

        def to_text(ex):
            if self.input_field and self.target_field:
                return {"text": f"{ex[self.input_field]}\n{ex[self.target_field]}"}
            if self.text_field:
                return {"text": ex[self.text_field]}
            for k, v in ex.items():
                if isinstance(v, str):
                    return {"text": v}
            raise ValueError("No text column found; specify text_field or input+target_field.")

        ds = ds.map(to_text, batched=False)
        if not self.streaming and self.max_samples:
            ds = ds.select(range(min(len(ds), self.max_samples)))
        return ds

    def map_tokenize(self, ds: Dataset, tokenizer, max_length=2048, batched=True) -> Dataset:
        def tok(batch):
            enc = tokenizer(batch["text"], truncation=True, padding="max_length",
                            max_length=max_length)
            enc["labels"] = enc["input_ids"].copy()
            return enc
        return ds.map(tok, batched=batched, remove_columns=[c for c in ds.column_names if c != "text"])
