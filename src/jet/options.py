from typing import Literal
from pydantic import BaseModel

class TrainOptions(BaseModel):
    engine: Literal["auto","unsloth","hf"] = "auto"
    model: str
    epochs: int = 1
    max_seq: int = 2048
    output_dir: str = "outputs/model"
    per_device_batch: int = 1
    grad_accum: int = 16
    lr: float = 2e-4
    seed: int = 42
    use_4bit: bool = True
    bf16: bool = True
    flash_attn2: bool = True