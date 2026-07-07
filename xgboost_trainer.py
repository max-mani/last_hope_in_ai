"""
UYIR XGBoost Trainer — shared dataset logging + (re)training logic.

Used by two paths:
  1. app.py's existing manual endpoints (/log-feature, /train-model) —
     those keep their own xgb_clf loading/refinement logic untouched;
     this module just centralizes the CSV path + training routine so a
     second path (below) can drive it too without duplicating the
     sklearn/xgboost code a second time.
  2. The automatic LLM-verification pipeline (app.py's
     _auto_verify_and_train, called after a confirmed accident) — logs a
     labeled row from the external LLM's verdict and retrains once enough
     data has accumulated, exactly like clicking "Train Model" yourself.

CSV_FILE / XGB_MODEL_PATH intentionally match the constants already used
in app.py so both paths read and write the same files.
"""

import csv
import logging
import os

logger = logging.getLogger("XGBoostTrainer")

CSV_FILE = "accident_features.csv"
XGB_MODEL_PATH = "model_output/accident_xgboost.json"

_FEATURE_COLUMNS = [
    "proximity", "trajectory", "anomaly", "cnn",
    "occlusion", "merge", "kinetic", "density",
    "avg_speed", "stopped_ratio",
]


def init_csv_file():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(_FEATURE_COLUMNS + ["label"])


def log_feature_row(features: dict, label: int):
    """
    Append one labeled row. `features` keys should match _FEATURE_COLUMNS;
    anything missing defaults to 0.0 rather than failing the whole log.
    """
    init_csv_file()
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([features.get(col, 0.0) for col in _FEATURE_COLUMNS] + [label])
    logger.info(f"Logged auto-labeled feature row (label={label}).")


def dataset_counts():
    """Returns (total_rows, class_0_rows, class_1_rows)."""
    if not os.path.exists(CSV_FILE):
        return 0, 0, 0
    import pandas as pd
    df = pd.read_csv(CSV_FILE)
    total = len(df)
    if "label" not in df.columns:
        return total, 0, 0
    counts = df["label"].value_counts()
    return total, int(counts.get(0, 0)), int(counts.get(1, 0))


def retrain(min_rows_per_class=50):
    """
    Trains XGBoost from the current CSV and saves it to XGB_MODEL_PATH if
    there's enough labeled data. Returns a result dict; never raises.
    Caller is responsible for reloading any already-loaded model object
    afterward (app.py's load_xgboost_model()) — this module only writes
    the file.
    """
    try:
        import pandas as pd
        from xgboost import XGBClassifier
        XGBClassifier._estimator_type = "classifier"
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score

        if not os.path.exists(CSV_FILE):
            return {"trained": False, "reason": "Dataset not found."}

        df = pd.read_csv(CSV_FILE)
        if len(df) < 5 or "label" not in df.columns or df["label"].nunique() < 2:
            return {"trained": False, "reason": "Not enough data or only one class so far."}

        counts = df["label"].value_counts()
        n0, n1 = int(counts.get(0, 0)), int(counts.get(1, 0))
        if n0 < min_rows_per_class or n1 < min_rows_per_class:
            return {
                "trained": False,
                "reason": f"Need >= {min_rows_per_class} rows per class (have {n0}/{n1}).",
            }

        X = df.drop("label", axis=1)
        y = df["label"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        clf = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, eval_metric="logloss")
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))

        os.makedirs(os.path.dirname(XGB_MODEL_PATH), exist_ok=True)
        clf.save_model(XGB_MODEL_PATH)
        logger.info(f"XGBoost retrained — accuracy={acc:.3f}, rows={len(df)}")
        return {"trained": True, "accuracy": float(acc), "total_rows": len(df)}
    except Exception as e:
        logger.exception("XGBoost retrain failed")
        return {"trained": False, "reason": str(e)}
