# src/jet/train.py
from jet.merge import merge_lora  # [web:1504]

def has_supported_gpu() -> bool:
    try:
        import torch
        return (hasattr(torch, "cuda") and torch.cuda.is_available()) or (hasattr(torch, "xpu") and torch.xpu.is_available())
    except Exception:
        return False

def train_with_options(opts, train_ds, eval_ds=None):
    engine = opts.engine if getattr(opts, "engine", "auto") != "auto" else ("unsloth" if has_supported_gpu() else "hf")
    if engine == "unsloth":
        from .engine_unsloth import train as engine_train
    else:
        from .engine_hf import train as engine_train

    job = engine_train(opts, train_ds, eval_ds)  # [web:1406]

    merged_dir = None
    if getattr(opts, "merge_weights", False):
        try:
            merged_dir = f"{opts.output_dir}-merged"
            merge_lora(opts.model, opts.output_dir, merged_dir)  # [web:1504]
        except Exception as e:
            return type("Job", (), {"model_dir": opts.output_dir, "merged_dir": None, "merge_error": str(e)})
    return type("Job", (), {"model_dir": opts.output_dir, "merged_dir": merged_dir})
