# easyllm/curriculum.py
import torch
from datasets import Dataset, concatenate_datasets

def add_length_difficulty(ds: Dataset) -> Dataset:
    return ds.map(lambda ex: {"difficulty": len(ex["text"]) if isinstance(ex["text"], str) else 0})

@torch.no_grad()
def add_loss_difficulty(ds: Dataset, model, tok, max_len=512, batch_size=4) -> Dataset:
    texts = ds["text"]; scores=[]
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tok(batch, truncation=True, max_length=max_len, padding=True, return_tensors="pt").to("cuda")
        out = model(**enc, labels=enc["input_ids"])
        scores.extend([out.loss.item()]*len(batch))
    return ds.add_column("difficulty", scores)

def make_buckets(ds: Dataset, buckets=3):
    ds_sorted = ds.sort("difficulty")
    n = len(ds_sorted); base=n//buckets; rem=n%buckets; parts=[]; s=0
    for b in range(buckets):
        size = base + (1 if b<rem else 0)
        parts.append(ds_sorted.select(range(s, s+size))); s += size
    return parts

def one_pass(buckets, epoch, total): 
    return buckets[min(len(buckets)-1, epoch*len(buckets)//max(1,total))]

def baby_steps(buckets, epoch, total, mix=[1.0,0.5,0.2]):
    pools=[]
    for i,b in enumerate(buckets):
        weight = mix[i]*(epoch+1)/max(1,total); take=max(1, int(len(b)*min(1.0, weight)))
        pools.append(b.select(range(min(len(b), take))))
    return concatenate_datasets(pools)
