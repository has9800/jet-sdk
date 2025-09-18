import os
from jet.options import TrainOptions
from jet.dataset import DatasetBuilder
from jet.train import train_with_options

def test_cpu_smoke(tmp_path):
    p = tmp_path / "sample.txt"
    p.write_text("hello world\nsecond sample\n")
    ds = DatasetBuilder(f"text:{p}", split="train").load()
    out = tmp_path / "model"
    opts = TrainOptions(model="sshleifer/tiny-gpt2", engine="hf", epochs=1, max_seq=128, output_dir=str(out))
    job = train_with_options(opts, ds)
    assert os.path.isdir(job.model_dir)
