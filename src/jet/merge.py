# src/jet/merge.py
import os
from transformers import AutoModelForCausalLM  # load/save [web:1406]
from peft import PeftModel  # adapter attach [web:1504]

def merge_lora(base_model_id: str, adapter_dir: str, out_dir: str, torch_dtype=None):
    base = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch_dtype)  # base weights [web:1406]
    peft = PeftModel.from_pretrained(base, adapter_dir)  # attach adapter [web:1504]
    merged = peft.merge_and_unload()  # merge into base [web:1504]
    os.makedirs(out_dir, exist_ok=True)  # create target path [web:1546]
    merged.save_pretrained(out_dir)  # write merged model [web:1406]
    return out_dir
