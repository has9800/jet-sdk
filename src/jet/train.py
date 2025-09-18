def has_supported_gpu() -> bool:
    try:
        import torch
        return (hasattr(torch, "cuda") and torch.cuda.is_available()) or (hasattr(torch, "xpu") and torch.xpu.is_available())
    except Exception:
        return False

def train_with_options(opts, train_ds, eval_ds=None):
    engine = opts.engine if opts.engine != "auto" else ("unsloth" if has_supported_gpu() else "hf")
    if engine == "unsloth":
        from .engine_unsloth import train as engine_train
    else:
        from .engine_hf import train as engine_train
    return engine_train(opts, train_ds, eval_ds)
