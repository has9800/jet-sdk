import unsloth  # must be first for patching [web:764]
from unsloth import FastLanguageModel
import torch
from transformers import set_seed
from trl import SFTTrainer, SFTConfig

def train(opts, train_ds, eval_ds=None):
    set_seed(opts.seed)
    model, tok = FastLanguageModel.from_pretrained(
        model_name=opts.model,
        max_seq_length=opts.max_seq,
        load_in_4bit=opts.use_4bit,
        dtype=torch.bfloat16 if opts.bf16 else torch.float16,
        device_map="auto",
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    model = FastLanguageModel.get_peft_model(
        model,
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
    )
    if opts.flash_attn2 and hasattr(model.config, "attn_implementation"):
        model.config.attn_implementation = "flash_attention_2"

    sft = SFTConfig(
        output_dir=opts.output_dir,
        per_device_train_batch_size=opts.per_device_batch,
        gradient_accumulation_steps=opts.grad_accum,
        learning_rate=opts.lr,
        num_train_epochs=opts.epochs,
        bf16=opts.bf16, fp16=not opts.bf16,
        logging_steps=50,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_ds is not None else "no",
        report_to=[],
        dataset_text_field=opts.text_field or "text",
        packing=False,
        max_seq_length=opts.max_seq,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tok,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=sft,
    )
    trainer.train()
    trainer.save_model(opts.output_dir)
    tok.save_pretrained(opts.output_dir)
    return type("Job", (), {"model_dir": opts.output_dir})