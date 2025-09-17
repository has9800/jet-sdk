# src/jet/__init__.py
__version__ = "0.3.1"
from .options import TrainOptions
from .dataset import DatasetBuilder
from .train import train_with_options
from .eval import Evaluator