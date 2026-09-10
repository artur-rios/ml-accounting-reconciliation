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
    total = args.seeds or config["comparison"]["n_seeds"]
    run(config, total, Path(args.output))
