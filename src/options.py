# easyllm/options.py
from pydantic import BaseModel
from typing import Optional

class TrainOptions(BaseModel):
    epochs: int = 1
    max_seq: int = 2048
    lr: float = 2e-4
    per_device_batch: int = 1
    grad_accum: int = 16
    seed: int = 42
    use_4bit: bool = True
    bf16: bool = True
    flash_attn2: bool = True
    curriculum: str = "off"          # "off" | "length-baby" | "loss-onepass"
    buckets: int = 3
    output_dir: str = "outputs/model"
    mlflow_experiment: str = "easyllm-finetune"
