# easyllm/train.py
import os, mlflow, torch
from typing import Optional
from transformers import set_seed, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer
from unsloth import FastLanguageModel
from .options import TrainOptions
from .curriculum import add_length_difficulty, add_loss_difficulty, make_buckets, one_pass, baby_steps

class _CurriculumCB:
    def __init__(self, builder): self.builder = builder
    def on_epoch_begin(self, args, state, control, **kw):
        trainer = kw["trainer"]; epoch = int(state.epoch) if state.epoch is not None else 0
        total = int(args.num_train_epochs); trainer.train_dataset = self.builder(epoch, total); return control

class FineTuner:
    def __init__(self, model_id: str, opts: Optional[TrainOptions] = None):
        self.model_id = model_id
        self.opts = opts or TrainOptions()

    def _build_model(self):
        m, tok = FastLanguageModel.from_pretrained(
            model_name=self.model_id, max_seq_length=self.opts.max_seq,
            load_in_4bit=self.opts.use_4bit, dtype=torch.bfloat16 if self.opts.bf16 else torch.float16,
            device_map="auto"
        )
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        tok.padding_side = "right"
        m = FastLanguageModel.get_peft_model(
            m, r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
            bias="none", use_gradient_checkpointing="unsloth"
        )
        if self.opts.flash_attn2 and hasattr(m.config, "attn_implementation"):
            m.config.attn_implementation = "flash_attention_2"
        return m, tok

    def train(self, train_ds, eval_ds=None):
        set_seed(self.opts.seed)
        mlflow.set_experiment(self.opts.mlflow_experiment)
        with mlflow.start_run():
            mlflow.log_params(self.opts.model_dump())
            model, tok = self._build_model()

            callback = None
            if self.opts.curriculum != "off":
                if self.opts.curriculum.startswith("length"):
                    train_ds = add_length_difficulty(train_ds)
                else:
                    base = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=torch.bfloat16 if self.opts.bf16 else torch.float16).to("cuda").eval()
                    base_tok = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
                    train_ds = add_loss_difficulty(train_ds, base, base_tok, max_len=min(512, self.opts.max_seq), batch_size=4)
                buckets = make_buckets(train_ds, buckets=self.opts.buckets)
                builder = (lambda e,T: one_pass(buckets,e,T)) if "onepass" in self.opts.curriculum else (lambda e,T: baby_steps(buckets,e,T))
                callback = _CurriculumCB(builder)

            args = TrainingArguments(
                output_dir=self.opts.output_dir,
                per_device_train_batch_size=self.opts.per_device_batch,
                gradient_accumulation_steps=self.opts.grad_accum,
                learning_rate=self.opts.lr,
                num_train_epochs=self.opts.epochs,
                bf16=self.opts.bf16,
                fp16=not self.opts.bf16,
                logging_steps=50,
                save_strategy="epoch",
                evaluation_strategy="epoch" if eval_ds is not None else "no",
                report_to=[],
            )
            trainer = SFTTrainer(
                model=model, tokenizer=tok,
                train_dataset=train_ds, eval_dataset=eval_ds,
                args=args, dataset_text_field="text", packing=False, max_seq_length=self.opts.max_seq
            )
            if callback: trainer.add_callback(callback)
            trainer.train()
            os.makedirs(self.opts.output_dir, exist_ok=True)
            trainer.save_model(self.opts.output_dir); tok.save_pretrained(self.opts.output_dir)
            mlflow.log_artifacts(self.opts.output_dir, artifact_path="model")
            return type("Job", (), {"model_dir": self.opts.output_dir})
