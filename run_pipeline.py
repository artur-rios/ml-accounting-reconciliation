import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from reconciliacao.etl.cleaner import clean_nfse, clean_pagamentos
from reconciliacao.etl.labeler import label_records
from reconciliacao.etl.loader import load_nfse, load_pagamentos
from reconciliacao.models.evaluator import evaluate_all
from reconciliacao.models.features import build_features
from reconciliacao.models.trainer import train_all
from reconciliacao.simulation.excel_generator import (
    DEFAULT_REFERENCE_DATE,
    generate_payment_records,
    write_excel,
)
from reconciliacao.simulation.xml_generator import generate_nfse, write_xml
from reconciliacao.utils.config import load_config


def _reference_date(sim: dict) -> date:
    """Anchor for the simulated date window, from config.

    PyYAML already parses an unquoted ISO date into a `date`, but the key may
    also arrive as a string, and its absence must not silently fall back to
    the wall clock -- that fallback was the defect.
    """
    valor = sim.get("reference_date", DEFAULT_REFERENCE_DATE)
    return valor if isinstance(valor, date) else date.fromisoformat(str(valor))


def run(scenario: str, cfg: dict) -> None:
    scfg = cfg["scenarios"][scenario]
    sim = cfg["simulation"]

    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    results_dir = Path(f"data/results/{scenario}")

    print(f"[1/5] Simulating {sim['n_records']} records...")
    reference_date = _reference_date(sim)
    df_pag = generate_payment_records(sim["n_records"], sim["random_seed"], reference_date)
    write_excel(df_pag, raw_dir / "pagamentos.xlsx")
    df_nfse_sim = generate_nfse(
        df_pag, sim["conciliation_rate"], sim["random_seed"], reference_date
    )
    write_xml(df_nfse_sim, raw_dir / "nfse.xml")

    print("[2/5] Running ETL...")
    df_pag_raw = load_pagamentos(raw_dir / "pagamentos.xlsx")
    df_nfse_raw = load_nfse(raw_dir / "nfse.xml")
    df_pag_clean = clean_pagamentos(df_pag_raw)
    df_nfse_clean = clean_nfse(df_nfse_raw)
    df_reconciled = label_records(
        df_pag_clean,
        df_nfse_clean,
        date_tolerance_days=scfg["date_tolerance_days"],
        value_tolerance_pct=scfg["value_tolerance_pct"],
        n_classes=scfg["n_classes"],
    )
    df_reconciled["scenario"] = scenario
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_path = processed_dir / f"{scenario}_reconciliado.csv"
    df_reconciled.to_csv(processed_path, index=False)
    print(f"    Label distribution:\n{df_reconciled['label'].value_counts().to_string()}")

    print("[3/5] Building features...")
    # Features are built from the persisted CSV, not from the frame still in
    # memory, so that this pipeline and everything that consumes its output --
    # run_ablation.py and the notebooks -- compute on byte-identical input.
    #
    # The two are not equivalent, and the difference is instructive. A CSV
    # round-trip perturbs delta_valor_pct by about 1e-14 on 94 of 7,518 rows,
    # which is pure float repr noise. That noise survives StandardScaler and
    # reaches lbfgs near its convergence tolerance, and the cross-validated
    # F1-macro of the logistic regression in the fuzzy scenario moves from
    # 0.8678 to 0.8674 -- a 3e-4 shift from a 1e-14 input difference, landing
    # exactly in the fourth decimal the write-up reports. Test-set metrics and
    # the selected hyperparameter are unaffected.
    #
    # Reading back is the honest direction: the CSV is the artifact a reader
    # can inspect, so it should be the artifact the model is fitted on.
    X, y = build_features(pd.read_csv(processed_path))

    print("[4/5] Training models (this may take a few minutes)...")
    training_results = train_all(X, y, cfg, results_dir / "models")

    print("[5/5] Evaluating models...")
    df_metrics = evaluate_all(training_results, scfg["n_classes"], results_dir)
    print("\n" + df_metrics.to_string())
    print(f"\nArtifacts saved to {results_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the ML reconciliation pipeline.")
    parser.add_argument("--scenario", choices=["exact", "fuzzy"], required=True)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run(args.scenario, load_config(args.config))
