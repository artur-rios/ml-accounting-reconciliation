import argparse
from pathlib import Path

from reconciliacao.comparison import run
from reconciliacao.utils.config import load_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare RF, SVM and Logistic Regression on the leak-free task."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--seeds", type=int, default=None,
                        help="override comparison.n_seeds (use 2 for a dry run)")
    parser.add_argument("--output", default="data/results/comparison")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.seeds is not None and args.seeds <= 0:
        parser.error(f"--seeds must be a positive integer, got {args.seeds}")

    if "comparison" not in config:
        parser.error(
            f"{args.config} has no 'comparison' block; add one before running "
            "the algorithm comparison."
        )

    total = args.seeds if args.seeds is not None else config["comparison"]["n_seeds"]
    run(config, total, Path(args.output))
