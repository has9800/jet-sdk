# easyllm/eval.py
import mlflow, torch
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import evaluate as hf_evaluate

class Evaluator:
    def __init__(self, model_dir: str, bf16: bool = True, seed: int = 42):
        self.model_dir = model_dir; self.bf16 = bf16; self.seed = seed

    def evaluate(self, prompts: List[str], references: List[str], max_new_tokens: int = 256, do_sample: bool = False) -> Dict:
        set_seed(self.seed)
        tok = AutoTokenizer.from_pretrained(self.model_dir, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_dir, torch_dtype=torch.bfloat16 if self.bf16 else torch.float16
        ).to("cuda").eval()

        preds=[]
        for p in prompts:
            inp = tok(p, return_tensors="pt").to("cuda")
            gen = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=1.0, top_p=1.0)
            txt = tok.decode(gen, skip_special_tokens=True)
            preds.append(txt[len(p):].strip() if txt.startswith(p) else txt.strip())

        rouge = hf_evaluate.load("rouge")
        rouge_scores = rouge.compute(predictions=preds, references=references)

        @torch.no_grad()
        def ppl(prompt, ref):
            ep = tok(prompt, return_tensors="pt")
            ef = tok(prompt + ref, return_tensors="pt")
            ids = ef["input_ids"].to("cuda"); attn=ef["attention_mask"].to("cuda")
            labels = ids.clone(); labels[:, :ep["input_ids"].shape[32]] = -100
            out = model(input_ids=ids, attention_mask=attn, labels=labels)
            return torch.exp(out.loss).item()

        ppls = [ppl(p, r) for p, r in zip(prompts, references)]
        res = {"rouge": rouge_scores, "per_sample_ppl": ppls, "mean_ppl": sum(ppls)/len(ppls)}
        mlflow.log_dict(res, "mini_eval.json")
        return res
