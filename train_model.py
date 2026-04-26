from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import train_test_split

from src.data import load_data_bundle, load_uploaded_data_bundle
from src.ml_model import (
    CATEGORICAL_FEATURES,
    prepare_model_matrix,
    prepare_training_frame,
    save_model_artifacts,
    train_prediction_models,
)

DEFAULT_EXTERNAL_FILES = [
    r"c:\Users\kshan\Downloads\dataset_weather.xlsx",
    r"c:\Users\kshan\Downloads\dataset_energy.xlsx",
    r"c:\Users\kshan\Downloads\dataset_iot.xlsx",
    r"c:\Users\kshan\Downloads\dataset_hr.xlsx",
    r"c:\Users\kshan\Downloads\dataset_movies.xlsx",
    r"c:\Users\kshan\Downloads\dataset_traffic.xlsx",
    r"c:\Users\kshan\Downloads\dataset_sales.xlsx",
    r"c:\Users\kshan\Downloads\dataset_banking.xlsx",
    r"c:\Users\kshan\Downloads\dataset_health.xlsx",
    r"c:\Users\kshan\Downloads\dataset_students.xlsx",
    r"c:\Users\kshan\Downloads\dataset_ecommerce.xlsx",
    r"c:\Users\kshan\Downloads\dataset_logistics.xlsx",
]


def load_compatible_external_training_data(project_root: Path, extra_files: list[str] | None = None) -> tuple[pd.DataFrame, list[str], list[str]]:
    loaded_frames: list[pd.DataFrame] = []
    used_files: list[str] = []
    skipped_files: list[str] = []

    candidate_files = extra_files or DEFAULT_EXTERNAL_FILES
    for file_path in candidate_files:
        path = Path(file_path)
        if not path.exists():
            continue
        try:
            bundle = load_uploaded_data_bundle(project_root, path.read_bytes(), path.name)
            frame = bundle.orders.copy()
            if frame.empty or "lead_time_days" not in frame.columns:
                skipped_files.append(f"{path.name}: no supervised lead-time rows after import")
                continue
            frame["order_id"] = frame["order_id"].astype(str).map(lambda value: f"AUX-{path.stem}-{value}")
            loaded_frames.append(frame)
            used_files.append(path.name)
        except Exception as exc:
            skipped_files.append(f"{path.name}: {exc}")
    if loaded_frames:
        return pd.concat(loaded_frames, ignore_index=True, sort=False), used_files, skipped_files
    return pd.DataFrame(), used_files, skipped_files


def run_optuna_tuning(orders, trials: int = 20) -> dict[str, float]:
    import optuna

    frame, _ = prepare_training_frame(orders)
    X = prepare_model_matrix(frame)
    y_reg = frame["lead_time_days"]
    y_clf = frame["delay_flag"]
    cat_feature_indices = [X.columns.get_loc(column) for column in CATEGORICAL_FEATURES]
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X,
        y_reg,
        y_clf,
        test_size=0.2,
        random_state=42,
        stratify=y_clf,
    )

    def objective(trial: optuna.Trial) -> float:
        depth = trial.suggest_int("depth", 6, 10)
        learning_rate = trial.suggest_float("learning_rate", 0.02, 0.12, log=True)
        iterations = trial.suggest_int("iterations", 300, 900)
        reg = CatBoostRegressor(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            loss_function="RMSE",
            verbose=False,
            allow_writing_files=False,
            random_seed=42,
        )
        clf = CatBoostClassifier(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            loss_function="Logloss",
            eval_metric="AUC",
            verbose=False,
            allow_writing_files=False,
            auto_class_weights="Balanced",
            random_seed=42,
        )
        reg.fit(X_train, y_reg_train, cat_features=cat_feature_indices)
        clf.fit(X_train, y_clf_train, cat_features=cat_feature_indices)
        reg_pred = reg.predict(X_test)
        clf_prob = clf.predict_proba(X_test)[:, 1]
        mae = mean_absolute_error(y_reg_test, reg_pred)
        auc = roc_auc_score(y_clf_test, clf_prob)
        return mae - (auc * 25)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=trials)
    return study.best_params


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true", help="Run Optuna tuning before final training.")
    parser.add_argument("--trials", type=int, default=20, help="Optuna trials when --tune is used.")
    parser.add_argument("--extra", nargs="*", default=None, help="Optional external dataset file paths to try including in training.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    bundle = load_data_bundle(project_root)
    extra_frame, used_files, skipped_files = load_compatible_external_training_data(project_root, args.extra)

    training_orders = bundle.orders.copy()
    if not extra_frame.empty:
        training_orders = pd.concat([training_orders, extra_frame], ignore_index=True, sort=False)

    if args.tune:
        best = run_optuna_tuning(training_orders, trials=args.trials)
        print("Optuna best parameters:")
        for key, value in best.items():
            print(f"  {key}: {value}")

    models = train_prediction_models(training_orders, perform_cross_validation=True)
    output_dir = project_root / "models"
    save_model_artifacts(models, output_dir)

    print("Training complete.")
    print(f"Artifacts saved to: {output_dir}")
    print(f"Base rows: {len(bundle.orders):,}")
    print(f"External rows used: {len(extra_frame):,}")
    if used_files:
        print("External datasets used:")
        for item in used_files:
            print(f"  - {item}")
    if skipped_files:
        print("External datasets skipped:")
        for item in skipped_files:
            print(f"  - {item}")
    print("Metrics:")
    for key, value in models.metrics.items():
        if key in {"mae", "cv_mae"}:
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
