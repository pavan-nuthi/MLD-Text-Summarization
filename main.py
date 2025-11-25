import os
# Set MPS High Watermark Ratio to 0.0 to avoid premature OOM
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import argparse
import sys
from train import train
from evaluation import evaluate

def main():
    parser = argparse.ArgumentParser(description="Text Summarization Project")
    subparsers = parser.add_subparsers(dest="mode", help="Mode: train or evaluate")
    
    # Train Parser
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--model_type", type=str, required=True, choices=['bertsum', 'bart'])
    train_parser.add_argument("--data_path", type=str, default=None)
    train_parser.add_argument("--epochs", type=int, default=3)
    train_parser.add_argument("--batch_size", type=int, default=4)
    train_parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    
    # Evaluate Parser
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a model")
    eval_parser.add_argument("--model_type", type=str, required=True, choices=['bertsum', 'bart'])
    eval_parser.add_argument("--model_path", type=str, required=True)
    eval_parser.add_argument("--data_path", type=str, default=None)
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train(args)
    elif args.mode == "evaluate":
        evaluate(args)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
