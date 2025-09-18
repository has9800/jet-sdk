from datasets import load_dataset

class DatasetBuilder:
    def __init__(
        self, source: str, split: str = "train",
        text_field: str | None = None,
        input_field: str | None = None,
        target_field: str | None = None,
        streaming: bool = False,
        max_samples: int | None = None,
    ):
        self.source, self.split = source, split
        self.text_field, self.input_field, self.target_field = text_field, input_field, target_field
        self.streaming, self.max_samples = streaming, max_samples

    def _detect(self, path: str):
        import os
        if ":" in path and os.path.sep in path:
            kind, loc = path.split(":", 1)
            return kind, loc
        return "auto", path

    def load(self):
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
            # fallback: first string field
            for k, v in ex.items():
                if isinstance(v, str):
                    return {"text": v}
            raise ValueError("Specify text_field or input+target_field")

        ds = ds.map(to_text, batched=False)
        if not self.streaming and self.max_samples:
            ds = ds.select(range(min(len(ds), self.max_samples)))
        return ds
