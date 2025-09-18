from typing import Optional, Literal
from pydantic import BaseModel

class TrainOptions(BaseModel):
    engine: Literal["auto","unsloth","hf"] = "auto"
    model: str
    dataset_source: Optional[str] = None
    text_field: Optional[str] = "text"
    input_field: Optional[str] = None
    target_field: Optional[str] = None
    epochs: int = 1
    max_seq: int = 2048
    output_dir: str = "outputs/model"
    per_device_batch: int = 1
    grad_accum: int = 16
    lr: float = 2e-4
    seed: int = 42
    use_4bit: bool = True
    bf16: bool = True
