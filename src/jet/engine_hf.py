from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTTrainer, SFTConfig

def train(opts, train_ds, eval_ds=None):
    set_seed(opts.seed)
    tok = AutoTokenizer.from_pretrained(opts.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(opts.model)

    sft = SFTConfig(
        output_dir=opts.output_dir,
        per_device_train_batch_size=opts.per_device_batch,
        gradient_accumulation_steps=opts.grad_accum,
        learning_rate=opts.lr,
        num_train_epochs=opts.epochs,
        bf16=False, fp16=False,
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
