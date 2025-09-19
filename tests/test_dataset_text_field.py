# tests/test_dataset_text_field.py
from jet.dataset import DatasetBuilder

def test_builder_falls_back_on_invalid_text_field(monkeypatch):
    # Fake datasets.load_dataset -> dataset with text+label
    class FakeDS(list):
        def map(self, fn, batched=False): return list(map(fn, self))
        def select(self, idxs): return self
    def fake_load_dataset(kind, path, split=None, streaming=False): 
        return FakeDS([{"text":"hello","label":0}])
    monkeypatch.setattr("jet.dataset.load_dataset", fake_load_dataset)

    ds = DatasetBuilder("hf:ag_news", split="train", text_field="not_a_col").load()
    assert isinstance(ds[0]["text"], str) and ds[0]["text"] == "hello"  # fallback string field [web:1083]
