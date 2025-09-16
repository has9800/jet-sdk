# tests/test_dataset.py
import types
from easyllm.dataset import DatasetBuilder

def test_dataset_builder_offline(monkeypatch):
    # Fake HF datasets.load_dataset returning a minimal Dataset-like object
    fake_ds = types.SimpleNamespace(
        column_names=["text"],
        map=lambda fn, batched=False, remove_columns=None: fake_ds,
        select=lambda rng: fake_ds,
        sort=lambda col: fake_ds,
        __len__=lambda self=0: 10,
        __getitem__=lambda i: {"text": f"row {i}"}
    )
    def fake_load_dataset(*args, **kwargs):
        return fake_ds
    monkeypatch.setattr("easyllm.dataset.load_dataset", fake_load_dataset)

    ds = DatasetBuilder("csv:/any/path.csv", split="train", text_field="text").load()
    assert hasattr(ds, "map") and hasattr(ds, "select")  # contract: Datasets-like methods exist
