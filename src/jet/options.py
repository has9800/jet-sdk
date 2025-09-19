# src/jet/options.py (snippet)
from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainOptions:
    engine: str
    model: str
    output_dir: str = "outputs/model"
    epochs: int = 1
    max_seq: int = 1024
    per_device_batch: int = 1
    grad_accum: int = 16
    lr: float = 2e-4
    seed: int = 42
    use_4bit: bool = True
    bf16: bool = True
    text_field: Optional[str] = "text"
    input_field: Optional[str] = None
    target_field: Optional[str] = None
    dataset_source: Optional[str] = None  # keep for tests [web:1361]
    decoding_preset: Optional[str] = None
    do_sample: bool = True
    temperature: float = 0.7
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 50
    repetition_penalty: float = 1.1
    no_repeat_ngram_size: int = 2
    max_new_tokens: int = 64
    merge_weights: bool = True  # deployment-first default [web:1504]
