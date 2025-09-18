# src/jet/engine_hf.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, GenerationConfig  # HF core + generation [web:1361]
from trl import SFTTrainer, SFTConfig  # TRL supervised fine-tuning [web:347]

def _normalize_precision(opts):
    cuda = torch.cuda.is_available()  # GPU availability [web:1262]
    xpu = hasattr(torch, "xpu") and torch.xpu.is_available()  # Intel XPU [web:1262]
    if not (cuda or xpu):
        return torch.float32, False, False  # dtype, use_bf16, use_fp16 (CPU) [web:1260]
    if cuda and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        return torch.bfloat16, True, False  # prefer bf16 on supported GPUs [web:1268]
    return torch.float16, False, True  # otherwise fp16 [web:1260]

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
    )  # minimal for cross-version TRL compatibility [web:1216]
    if include_field:
        base["dataset_text_field"] = opts.text_field or "text"
    try:
        return SFTConfig(**base), True  # field accepted here [web:1216]
    except TypeError:
        if "dataset_text_field" in base:
            base.pop("dataset_text_field", None)
            return SFTConfig(**base), False  # fallback: pass field to trainer [web:1216]
        raise

def _make_trainer(model, tok, train_ds, eval_ds, sft_cfg, dataset_field_name):
    kwargs = dict(model=model, train_dataset=train_ds, eval_dataset=eval_ds, args=sft_cfg)
    try:
        return SFTTrainer(tokenizer=tok, **kwargs)  # older/minor TRL accepts tokenizer [web:1216]
    except TypeError:
        pass
    if dataset_field_name:
        try:
            return SFTTrainer(processing_class=tok, dataset_text_field=dataset_field_name, **kwargs)  # newer TRL [web:1229]
        except TypeError:
            pass
    return SFTTrainer(processing_class=tok, **kwargs)  # final fallback [web:1229]

def _safe_int_id(v):
    return v if isinstance(v, int) else None  # only ints or None for GenerationConfig validation [web:1378]

def train(opts, train_ds, eval_ds=None):
    set_seed(opts.seed)  # reproducibility [web:1216]
    dtype, use_bf16, use_fp16 = _normalize_precision(opts)  # auto by hardware [web:1260]

    tok = AutoTokenizer.from_pretrained(opts.model, use_fast=True)  # fast tokenizer [web:1361]
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token  # ensure padding token [web:1361]
    tok.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(opts.model, torch_dtype=dtype)  # respect dtype [web:1361]

    # Persist default decoding settings to curb repetition and enable sampling [web:1361]
    pad_id = _safe_int_id(getattr(tok, "pad_token_id", None)) or _safe_int_id(getattr(tok, "eos_token_id", None))
    eos_id = _safe_int_id(getattr(tok, "eos_token_id", None))
    gen_cfg = GenerationConfig(
        do_sample=getattr(opts, "do_sample", True),
        temperature=getattr(opts, "temperature", 0.7),
        top_p=getattr(opts, "top_p", 0.9),
        top_k=getattr(opts, "top_k", 50),
        repetition_penalty=getattr(opts, "repetition_penalty", 1.1),
        no_repeat_ngram_size=getattr(opts, "no_repeat_ngram_size", 2),
        max_new_tokens=getattr(opts, "max_new_tokens", 64),
        pad_token_id=pad_id,
        eos_token_id=eos_id,
    )  # supports save_pretrained / from_pretrained [web:1361]
    try:
        model.generation_config = gen_cfg  # attach defaults [web:1361]
    except Exception:
        pass

    sft_cfg, cfg_has_field = _build_sft_config(opts, use_bf16, use_fp16, include_field=True)  # TRL config [web:1216]
    dataset_field_name = opts.text_field or "text"
    trainer = _make_trainer(model, tok, train_ds, eval_ds, sft_cfg, None if cfg_has_field else dataset_field_name)  # tolerant [web:1229]

    trainer.train()  # run SFT [web:1216]
    trainer.save_model(opts.output_dir)  # model + config.json [web:1406]
    tok.save_pretrained(opts.output_dir)  # tokenizer [web:1406]
    try:
        gen_cfg.save_pretrained(opts.output_dir)  # generation_config.json [web:1361]
    except Exception:
        pass

    return type("Job", (), {"model_dir": opts.output_dir})  # simple handle [web:1216]