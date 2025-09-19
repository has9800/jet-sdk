import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed  # core HF [web:1361]
from trl import SFTTrainer, SFTConfig  # TRL SFT [web:347]
from jet.decoding import make_generation_config  # shared decoding presets [web:1361]

def _normalize_precision(opts):
    cuda = torch.cuda.is_available()
    xpu = hasattr(torch, "xpu") and torch.xpu.is_available()
    if not (cuda or xpu):
        return torch.float32, False, False  # CPU fp32 [web:1361]
    if cuda and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        return torch.bfloat16, True, False  # bf16 where supported [web:1361]
    return torch.float16, False, True  # else fp16 [web:1361]

def _build_sft_config(opts, use_bf16, use_fp16, include_field=True):
    base = dict(
        output_dir=opts.output_dir,
        per_device_train_batch_size=opts.per_device_batch,
        gradient_accumulation_steps=opts.grad_accum,
        learning_rate=opts.lr,
        num_train_epochs=opts.epochs,
        bf16=use_bf16,
        fp16=use_fp16,
        logging_steps=50,
        save_strategy="epoch",
        report_to=[],
        packing=False,
    )
    if include_field:
        base["dataset_text_field"] = opts.text_field or "text"
    try:
        return SFTConfig(**base), True  # field accepted here [web:347]
    except TypeError:
        if "dataset_text_field" in base:
            base.pop("dataset_text_field", None)
            return SFTConfig(**base), False  # pass field to trainer instead [web:347]
        raise

def _make_trainer(model, tok, train_ds, eval_ds, sft_cfg, dataset_field_name):
    kwargs = dict(model=model, train_dataset=train_ds, eval_dataset=eval_ds, args=sft_cfg)
    try:
        return SFTTrainer(tokenizer=tok, **kwargs)  # older TRL param [web:347]
    except TypeError:
        pass
    if dataset_field_name:
        try:
            return SFTTrainer(processing_class=tok, dataset_text_field=dataset_field_name, **kwargs)  # newer TRL [web:347]
        except TypeError:
            pass
    return SFTTrainer(processing_class=tok, **kwargs)  # final fallback [web:347]

def train(opts, train_ds, eval_ds=None):
    set_seed(opts.seed)  # reproducibility [web:347]
    dtype, use_bf16, use_fp16 = _normalize_precision(opts)  # auto precision [web:1361]

    tok = AutoTokenizer.from_pretrained(opts.model, use_fast=True)  # tokenizer [web:1361]
    if getattr(tok, "pad_token", None) is None:
        tok.pad_token = getattr(tok, "eos_token", None)  # set pad if missing [web:1361]
    tok.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(opts.model, torch_dtype=dtype)  # model [web:1361]

    # Attach and persist decoding defaults once
    gen_cfg = make_generation_config(tok, opts)  # [web:1361]
    try:
        model.generation_config = gen_cfg  # [web:1361]
    except Exception:
        pass

    sft_cfg, cfg_has_field = _build_sft_config(opts, use_bf16, use_fp16, include_field=True)  # TRL config [web:347]
    dataset_field_name = opts.text_field or "text"
    trainer = _make_trainer(model, tok, train_ds, eval_ds, sft_cfg, None if cfg_has_field else dataset_field_name)  # tolerant [web:347]

    trainer.train()  # run SFT [web:347]
    trainer.save_model(opts.output_dir)  # persist model [web:1406]
    tok.save_pretrained(opts.output_dir)  # persist tokenizer [web:1406]
    try:
        gen_cfg.save_pretrained(opts.output_dir)  # [web:1361]
    except Exception:
        pass

    return type("Job", (), {"model_dir": opts.output_dir})  # handle [web:1406]