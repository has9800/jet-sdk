from jet.dataset import DatasetBuilder

def test_load_local_text(tmp_path):
    p = tmp_path / "sample.txt"
    p.write_text("Hello\nWorld\n")
    dsb = DatasetBuilder(f"text:{p}", split="train", text_field="text")
    ds = dsb.load()
    assert len(ds) > 0 and "text" in ds[0]

def test_load_hf_dataset_smoke():
    dsb = DatasetBuilder("hf:sshleifer/tiny-shakespeare", split="train", text_field="text")
    ds = dsb.load()
    assert len(ds) > 0 and "text" in ds[0]
