# src/jet/metrics.py
import evaluate  # pip install evaluate
import math, torch

def compute_rouge(preds, refs):
    rouge = evaluate.load("rouge")  # returns ROUGE-1/2/L/Lsum scores [web:1497]
    return rouge.compute(predictions=preds, references=refs)  # dict of scores [web:1489]

def compute_perplexity(texts, model, tokenizer, stride=512, max_len=None):
    device = next(model.parameters()).device  # infer device [web:1503]
    enc = tokenizer("\n\n".join(texts), return_tensors="pt")  # simple corpus join [web:1503]
    input_ids = enc["input_ids"].to(device)
    if max_len is None:
        max_len = getattr(model.config, "max_position_embeddings", 1024)  # context size [web:1503]
    nlls, seq_len = [], 0
    for i in range(0, input_ids.size(1), stride):
        begin = max(i + stride - max_len, 0)
        end = min(i + stride, input_ids.size(1))
        trg_len = end - i
        if trg_len <= 0:
            continue
        input_ids_slice = input_ids[:, begin:end]
        target_ids = input_ids_slice.clone()
        target_ids[:, :-trg_len] = -100  # ignore context positions [web:1503]
        with torch.no_grad():
            loss = model(input_ids=input_ids_slice, labels=target_ids).loss  # NLL over target region [web:1503]
        nlls.append(loss.float() * trg_len)
        seq_len += trg_len
    ppl = math.exp(torch.stack(nlls).sum().item() / seq_len)
    return {"Perplexity Score (lower = better)": ppl}
