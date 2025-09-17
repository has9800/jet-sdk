# src/jet/cli.py
import argparse
from .options import TrainOptions
from .dataset import DatasetBuilder
from .train import train_with_options

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--engine", default="auto", choices=["auto","unsloth","hf"])
    args = ap.parse_args()

    ds = DatasetBuilder(args.data).load()
    opts = TrainOptions(model=args.model, engine=args.engine)
    train_with_options(opts, ds)

def app():
    main()