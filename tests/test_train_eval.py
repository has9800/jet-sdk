from jet.options import TrainOptions
from jet.dataset import DatasetBuilder
from jet.train import train_with_options

def test_train_smoke_cpu(tmp_path):
    dsb = DatasetBuilder("hf:karpathy/tiny_shakespeare", split="train", text_field="text")
    ds = dsb.load()
    opts = TrainOptions(
        model="sshleifer/tiny-gpt2",
        engine="hf",
        epochs=1,
        max_seq=128,
        output_dir=str(tmp_path),
        dataset_source="hf:sshleifer/tiny-shakespeare",
        text_field="text",
    )
    job = train_with_options(opts, ds)
    assert job.model_dir == str(tmp_path)

def test_eval_smoke(tmp_path):
    # This assumes artifacts from a previous training run or a minimal placeholder in tmp_path
    from jet.eval import Evaluator
    evaluator = Evaluator(str(tmp_path), bf16=False)
    rep = evaluator.evaluate(["Hello"], ["Hi"])
    assert rep is not None
