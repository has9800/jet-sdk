# tests/test_merge.py
from unittest.mock import MagicMock
from jet.merge import merge_lora

def test_merge_lora_invokes_merge_and_save(monkeypatch, tmp_path):
    class Base(MagicMock):
        @staticmethod
        def from_pretrained(*a, **k): return Base()
    class Peft(MagicMock):
        @staticmethod
        def from_pretrained(base, adapter_dir): 
            m = MagicMock()
            merged = MagicMock()
            merged.save_pretrained = lambda out: open(out + "/config.json", "w").write("{}")
            m.merge_and_unload.return_value = merged
            return m

    monkeypatch.setattr("jet.merge.AutoModelForCausalLM", Base)
    monkeypatch.setattr("jet.merge.PeftModel", Peft)
    out_dir = str(tmp_path / "merged")
    merge_lora("base-id", "adapter-dir", out_dir)
    assert (tmp_path / "merged" / "config.json").exists()  # merged model saved [web:1504]
