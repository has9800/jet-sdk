# src/jet/engine_unsloth.py
import torch
from transformers import set_seed, GenerationConfig  # reproducibility + generation [web:1361]
from trl import SFTTrainer, SFTConfig  # TRL SFT [web:347]

# Avoid importing Unsloth at module import (GPU checks); allow DI in tests. [web:1263]
class FastLanguageModel:
    _placeholder = True
    @staticmethod
    def from_pretrained(*args, **kwargs):
        raise RuntimeError("UNSLOTH_PLACEHOLDER")
    @staticmethod
    def get_peft_model(model, **kwargs):
        raise RuntimeError("UNSLOTH_PLACEHOLDER")

def _normalize_precision(opts):
    cuda = torch.cuda.is_available()  # NVIDIA [web:1262]
    xpu = hasattr(torch, "xpu") and torch.xpu.is_available()  # Intel [web:1262]
    if not (cuda or xpu):
        return torch.float32, False, False, False  # dtype, bf16, fp16, use_4bit on CPU [web:1260][web:1263]
    if cuda and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        return torch.bfloat16, True, False, getattr(opts, "use_4bit", True)  # bf16 preferred [web:1268]
    return torch.float16, False, True, getattr(opts, "use_4bit", True)  # fp16 otherwise [web:1260]

def _import_unsloth():
    from unsloth import FastLanguageModel as _RealFastLanguageModel  # may raise on CPU [web:1263]
    return _RealFastLanguageModel

def _ensure_unsloth_loaded():
    global FastLanguageModel
    if getattr(FastLanguageModel, "_placeholder", False):
        FastLanguageModel = _import_unsloth()  # swap in real impl [web:1263]

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
    )  # minimal TRL config [web:1216]
    if include_field:
        base["dataset_text_field"] = opts.text_field or "text"
    try:
        return SFTConfig(**base), True
    except TypeError:
        if "dataset_text_field" in base:
            base.pop("dataset_text_field", None)
            return SFTConfig(**base), False
        raise

def _make_trainer(model, tok, train_ds, eval_ds, sft_cfg, dataset_field_name):
    kwargs = dict(model=model, train_dataset=train_ds, eval_dataset=eval_ds, args=sft_cfg)
    try:
        return SFTTrainer(tokenizer=tok, **kwargs)  # older/minor TRL [web:1216]
    except TypeError:
        pass
    if dataset_field_name:
        try:
            return SFTTrainer(processing_class=tok, dataset_text_field=dataset_field_name, **kwargs)  # newer TRL [web:1229]
        except TypeError:
            pass
    return SFTTrainer(processing_class=tok, **kwargs)

def _safe_int_id(v):
    return v if isinstance(v, int) else None  # ints or None only for GenerationConfig [web:1378]

def train(opts, train_ds, eval_ds=None):
    set_seed(opts.seed)  # reproducibility [web:1216]
    dtype, use_bf16, use_fp16, use_4bit = _normalize_precision(opts)  # auto precision + 4-bit policy [web:1260][web:1263]

    # Try DI binding first; if placeholder, lazily import Unsloth [web:1263]
    try:
        model, tok = FastLanguageModel.from_pretrained(
            model_name=opts.model,
            max_seq_length=opts.max_seq,
            load_in_4bit=use_4bit,
            dtype=dtype,
            device_map="auto",
        )
    except RuntimeError as e:
        if "UNSLOTH_PLACEHOLDER" not in str(e):
            raise
        try:
            _ensure_unsloth_loaded()
            model, tok = FastLanguageModel.from_pretrained(
                model_name=opts.model,
                max_seq_length=opts.max_seq,
                load_in_4bit=use_4bit,
                dtype=dtype,
                device_map="auto",
            )
        except NotImplementedError as ne:
            raise RuntimeError("Unsloth requires NVIDIA CUDA or Intel XPU; use engine='hf' on CPU.") from ne  # clear guidance [web:1263]
        except Exception as ie:
            raise RuntimeError(f"Unsloth load failed: {ie}")

    # Tokenizer safety for stable collation [web:1361]
    if getattr(tok, "pad_token", None) is None:
        tok.pad_token = getattr(tok, "eos_token", None) or "<|pad|>"
    tok.padding_side = "right"

    # LoRA via Unsloth helper [web:1263]
    try:
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
            bias="none",
            use_gradient_checkpointing="unsloth",
        )
    except RuntimeError as e:
        if "UNSLOTH_PLACEHOLDER" in str(e):
            _ensure_unsloth_loaded()
            model = FastLanguageModel.get_peft_model(
                model,
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
                bias="none",
                use_gradient_checkpointing="unsloth",
            )
        else:
            raise

    # Prefer flash attention when available [web:1216]
    if hasattr(model.config, "attn_implementation"):
        model.config.attn_implementation = "flash_attention_2"

    # Persist default decoding settings (sampling, no-repeat n-grams) [web:1361]
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
    )  # validated and saveable [web:1361]
    try:
        model.generation_config = gen_cfg
    except Exception:
        pass

    sft_cfg, cfg_has_field = _build_sft_config(opts, use_bf16, use_fp16, include_field=True)  # TRL config [web:1216]
    dataset_field_name = opts.text_field or "text"
    trainer = _make_trainer(model, tok, train_ds, eval_ds, sft_cfg, None if cfg_has_field else dataset_field_name)  # tolerant [web:1229]

    trainer.train()
    trainer.save_model(opts.output_dir)  # persist model/adapter [web:1406]
    tok.save_pretrained(opts.output_dir)  # tokenizer [web:1406]
    try:
        gen_cfg.save_pretrained(opts.output_dir)  # generation_config.json [web:1361]
    except Exception:
        pass

    return type("Job", (), {"model_dir": opts.output_dir})  # handle for tests/CLI [web:1216]
