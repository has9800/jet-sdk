from typing import Literal, Optional
from pydantic import BaseModel

class TrainOptions(BaseModel):
    engine: Literal["auto","unsloth","hf"] = "auto"
    model: str
    # data mapping
    dataset_source: Optional[str] = None         # e.g., "text:./file.txt", "csv:./data.csv", "hf:org/name", "https://..."
    text_field: Optional[str] = "text"
    input_field: Optional[str] = None
    target_field: Optional[str] = None

    # training
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
