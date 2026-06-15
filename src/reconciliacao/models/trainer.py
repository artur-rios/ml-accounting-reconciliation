from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def get_pipelines(cfg: dict) -> dict[str, tuple]:
    seed = cfg["simulation"]["random_seed"]
    mcfg = cfg["models"]

    return {
        "random_forest": (
            Pipeline([
                ("clf", RandomForestClassifier(
                    n_estimators=mcfg["random_forest"]["n_estimators"],
                    class_weight=mcfg["random_forest"]["class_weight"],
                    random_state=seed,
                )),
            ]),
            {"clf__n_estimators": [100, 200], "clf__max_depth": [None, 10]},
        ),
        "svm": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(
                    kernel=mcfg["svm"]["kernel"],
                    class_weight=mcfg["svm"]["class_weight"],
                    probability=True,
                    random_state=seed,
                )),
            ]),
            {"clf__C": [0.1, 1.0, 10.0]},
        ),
        "logistic_regression": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    penalty=mcfg["logistic_regression"]["penalty"],
                    class_weight=mcfg["logistic_regression"]["class_weight"],
                    max_iter=1000,
                    random_state=seed,
                )),
            ]),
            {"clf__C": [0.01, 0.1, 1.0, 10.0]},
        ),
    }


def train_all(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: dict,
    output_dir: Path,
) -> dict:
    seed = cfg["simulation"]["random_seed"]
    mcfg = cfg["models"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=mcfg["test_size"],
        stratify=y,
        random_state=seed,
    )

    cv = StratifiedKFold(n_splits=mcfg["cv_folds"], shuffle=True, random_state=seed)
    results = {}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, (estimator, param_grid) in get_pipelines(cfg).items():
        gs = GridSearchCV(
            estimator,
            param_grid,
            cv=cv,
            scoring=mcfg["scoring"],
            n_jobs=-1,
            refit=True,
        )
        gs.fit(X_train, y_train)
        joblib.dump(gs.best_estimator_, output_dir / f"{name}.joblib")
        best_idx = gs.best_index_
        results[name] = {
            "estimator": gs.best_estimator_,
            "X_test": X_test,
            "y_test": y_test,
            "cv_score_mean": float(gs.cv_results_["mean_test_score"][best_idx]),
            "cv_score_std": float(gs.cv_results_["std_test_score"][best_idx]),
        }

    return results
