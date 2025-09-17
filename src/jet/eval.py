# src/jet/eval.py
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import torch, evaluate

class Evaluator:
    def __init__(self, model_dir: str, bf16: bool = True, seed: int = 42):
        self.model_dir, self.bf16, self.seed = model_dir, bf16, seed
    def evaluate(self, prompts, references, max_new_tokens=128, do_sample=False):
        set_seed(self.seed)
        tok = AutoTokenizer.from_pretrained(self.model_dir, use_fast=True)
        dtype = torch.bfloat16 if self.bf16 and torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(self.model_dir, torch_dtype=dtype)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device).eval()
        preds=[]
        for p in prompts:
            inp = tok(p, return_tensors="pt").to(device)
            gen = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=do_sample)
            txt = tok.decode(gen[0], skip_special_tokens=True)
            preds.append(txt[len(p):].strip() if txt.startswith(p) else txt.strip())
        rouge = evaluate.load("rouge")
        rouge_scores = rouge.compute(predictions=preds, references=references)
        with torch.no_grad():
            def ppl(prompt, ref):
                ep = tok(prompt, return_tensors="pt")
                ef = tok(prompt + ref, return_tensors="pt")
                ids = ef["input_ids"].to(device); attn = ef["attention_mask"].to(device)
                labels = ids.clone(); labels[:, :ep["input_ids"].shape[1]] = -100
                out = model(input_ids=ids, attention_mask=attn, labels=labels)
                return torch.exp(out.loss).item()
            ppls = [ppl(p, r) for p, r in zip(prompts, references)]
        return {"rouge": rouge_scores, "per_sample_ppl": ppls, "mean_ppl": sum(ppls)/len(ppls)}