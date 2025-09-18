def has_supported_gpu() -> bool:
    try:
        import torch
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            return True
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return True
    except Exception:
        pass
    return False

def train_with_options(opts, train_ds, eval_ds=None):
    engine = opts.engine
    if engine == "auto":
        engine = "unsloth" if has_supported_gpu() else "hf"
    if engine == "unsloth":
        from .engine_unsloth import train as engine_train  # unsloth is imported inside that module first
    else:
        from .engine_hf import train as engine_train
    return engine_train(opts, train_ds, eval_ds)
