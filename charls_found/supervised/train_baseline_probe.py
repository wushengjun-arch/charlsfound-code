#!/usr/bin/env python3
"""Train leakage-filtered rolling next-wave and discrete hazard baselines."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sksurv.metrics import concordance_index_censored
from xgboost import XGBClassifier

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_CONFIG = SCRIPT_PATH.parents[1] / "configs" / "supervised.yaml"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def feature_key(feature: str) -> str:
    prefix = "feature__"
    return feature[len(prefix) :] if feature.startswith(prefix) else feature


def make_preprocessor(
    features: list[str], feature_type_by_variable: dict[str, str | None], scale_numeric: bool
) -> ColumnTransformer:
    numeric_features = []
    categorical_features = []
    for feature in features:
        variable = feature_key(feature)
        if feature_type_by_variable.get(variable) == "continuous":
            numeric_features.append(feature)
        else:
            categorical_features.append(feature)

    transformers = []
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    if numeric_features:
        transformers.append(("num", Pipeline(numeric_steps), numeric_features))
    if categorical_features:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        (
                            "to_object",
                            FunctionTransformer(lambda array: array.astype(object), validate=False),
                        ),
                        ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
                        (
                            "to_string",
                            FunctionTransformer(lambda array: array.astype(str), validate=False),
                        ),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
                    ]
                ),
                categorical_features,
            )
        )
    return ColumnTransformer(transformers, sparse_threshold=0.3)


def make_model(
    name: str,
    config: dict[str, Any],
    y_train: pd.Series,
    features: list[str],
    feature_type_by_variable: dict[str, str | None],
) -> Pipeline:
    random_state = int(config["models"]["random_state"])
    n_jobs = int(config["models"].get("n_jobs", 4))
    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    if name == "logistic_regression":
        classifier = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            C=1e12,
            random_state=random_state,
        )
        scale_numeric = True
    elif name == "lightgbm":
        classifier = LGBMClassifier(
            n_estimators=int(config["models"].get("lightgbm_estimators", 250)),
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.85,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=n_jobs,
            verbosity=-1,
        )
        scale_numeric = False
    elif name == "xgboost":
        xgboost_device = os.environ.get("CHARLS_XGBOOST_DEVICE") or config["models"].get(
            "xgboost_device"
        )
        xgboost_kwargs: dict[str, Any] = {}
        if xgboost_device:
            xgboost_kwargs["device"] = str(xgboost_device)
        classifier = XGBClassifier(
            n_estimators=int(config["models"].get("xgboost_estimators", 250)),
            learning_rate=0.03,
            max_depth=3,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            n_jobs=n_jobs,
            **xgboost_kwargs,
        )
        scale_numeric = False
    else:
        raise ValueError(f"Unknown model: {name}")

    return Pipeline(
        [
            (
                "preprocess",
                make_preprocessor(features, feature_type_by_variable, scale_numeric=scale_numeric),
            ),
            ("classifier", classifier),
        ]
    )


def predict_probability(model: Pipeline, x_frame: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_frame)[:, 1]
    decision = model.decision_function(x_frame)
    return 1 / (1 + np.exp(-decision))


def calibration_slope_intercept(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    if len(np.unique(y_true)) < 2:
        return math.nan, math.nan
    clipped = np.clip(y_prob, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    try:
        model = LogisticRegression(C=1e12, solver="lbfgs", max_iter=1000)
        model.fit(logit, y_true)
    except Exception:
        return math.nan, math.nan
    return float(model.intercept_[0]), float(model.coef_[0][0])


def evaluate_predictions(y_true: pd.Series, y_prob: np.ndarray) -> dict[str, float]:
    y_array = y_true.to_numpy(dtype=int)
    if len(np.unique(y_array)) < 2:
        roc_auc = math.nan
        pr_auc = math.nan
    else:
        roc_auc = float(roc_auc_score(y_array, y_prob))
        pr_auc = float(average_precision_score(y_array, y_prob))
    brier = float(brier_score_loss(y_array, y_prob))
    calibration_intercept, calibration_slope = calibration_slope_intercept(y_array, y_prob)
    event_rate = float(y_array.mean())
    mean_predicted_risk = float(np.mean(y_prob))
    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier": brier,
        "event_rate": event_rate,
        "mean_predicted_risk": mean_predicted_risk,
        "calibration_mean_error": mean_predicted_risk - event_rate,
        "calibration_intercept": calibration_intercept,
        "calibration_slope": calibration_slope,
    }


def discrete_hazard_c_index(
    frame: pd.DataFrame, probabilities: np.ndarray, id_column: str
) -> dict[str, float]:
    risk_frame = frame[
        [id_column, "origin_wave", "next_wave", "origin_year", "next_year", "event_next_wave"]
    ].copy()
    risk_frame["probability"] = np.clip(probabilities, 0, 1)
    rows: list[dict[str, Any]] = []
    for id_value, group in risk_frame.sort_values([id_column, "origin_wave"]).groupby(
        id_column, sort=False
    ):
        first = group.iloc[0]
        event_rows = group[group["event_next_wave"].astype(int).eq(1)]
        event_observed = not event_rows.empty
        terminal = event_rows.iloc[0] if event_observed else group.iloc[-1]
        followup_time = float(terminal["next_year"] - first["origin_year"])
        if followup_time <= 0:
            continue
        rows.append(
            {
                id_column: id_value,
                "event_observed": event_observed,
                "time_to_event_years": followup_time,
                "risk_score": float(first["probability"]),
            }
        )
    survival = pd.DataFrame(rows)
    if survival.empty or survival["event_observed"].sum() == 0:
        return {
            "c_index": math.nan,
            "ids": int(len(survival)),
            "events": int(survival.get("event_observed", pd.Series(dtype=bool)).sum()),
        }
    try:
        c_index = float(
            concordance_index_censored(
                survival["event_observed"].to_numpy(dtype=bool),
                survival["time_to_event_years"].to_numpy(dtype=float),
                survival["risk_score"].to_numpy(dtype=float),
            )[0]
        )
    except Exception:
        c_index = math.nan
    return {
        "c_index": c_index,
        "ids": int(len(survival)),
        "events": int(survival["event_observed"].sum()),
    }


def verify_group_splits(matrix: pd.DataFrame, id_column: str) -> None:
    split_counts = matrix.groupby(id_column)["split"].nunique()
    leaked = split_counts[split_counts.gt(1)]
    if not leaked.empty:
        raise RuntimeError(f"ID-level split leakage detected for {len(leaked)} IDs")


def metric_row(
    outcome_id: str,
    model_name: str,
    split: str,
    metric_scope: str,
    origin_wave: str | int,
    n: int,
    ids: int,
    events: int,
    features: int,
    values: dict[str, float],
) -> dict[str, Any]:
    return {
        "outcome_id": outcome_id,
        "model": model_name,
        "split": split,
        "metric_scope": metric_scope,
        "origin_wave": origin_wave,
        "n": int(n),
        "ids": int(ids),
        "events": int(events),
        "features": int(features),
        "roc_auc": values.get("roc_auc", math.nan),
        "pr_auc": values.get("pr_auc", math.nan),
        "brier": values.get("brier", math.nan),
        "c_index": values.get("c_index", math.nan),
        "event_rate": values.get("event_rate", math.nan),
        "mean_predicted_risk": values.get("mean_predicted_risk", math.nan),
        "calibration_mean_error": values.get("calibration_mean_error", math.nan),
        "calibration_intercept": values.get("calibration_intercept", math.nan),
        "calibration_slope": values.get("calibration_slope", math.nan),
    }


def fmt(value: Any, digits: int = 4) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def write_report(
    path: Path, metrics: pd.DataFrame, config: dict[str, Any], manifest: dict[str, Any]
) -> None:
    test_pooled = metrics[metrics["split"].eq("test") & metrics["metric_scope"].eq("pooled")].copy()
    best_by_outcome = (
        test_pooled.sort_values(["outcome_id", "roc_auc", "pr_auc"], ascending=[True, False, False])
        .groupby("outcome_id", as_index=False)
        .head(1)
    )
    best_model_by_outcome = dict(zip(best_by_outcome["outcome_id"], best_by_outcome["model"]))
    test_origin = metrics[metrics["split"].eq("test") & metrics["metric_scope"].eq("origin")].copy()
    test_origin = test_origin[
        test_origin.apply(
            lambda row: best_model_by_outcome.get(row["outcome_id"]) == row["model"], axis=1
        )
    ]
    test_survival = metrics[
        metrics["split"].eq("test") & metrics["metric_scope"].eq("survival")
    ].copy()

    lines = [
        "# CHARLS Rolling Baseline Probe Report",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
        f"- Outcomes evaluated: {metrics['outcome_id'].nunique()}",
        f"- Models: {', '.join(metrics['model'].drop_duplicates())}",
        f"- Matrix type: `{manifest['metadata'].get('matrix_type', 'unknown')}`",
        "- Split: respondent-level `GroupShuffleSplit`; no ID crosses train/validation/test.",
        "- Metrics: pooled next-wave ROC-AUC/PR-AUC/Brier, per-origin ROC-AUC, and first-origin landmark C-index.",
        "",
        "## Best Test Pooled Model Per Outcome",
        "",
        "| outcome_id | model | rows | ids | events | event_rate | roc_auc | pr_auc | brier | calibration_mean_error | calibration_slope |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in best_by_outcome.iterrows():
        lines.append(
            f"| {row['outcome_id']} | {row['model']} | {int(row['n'])} | {int(row['ids'])} | {int(row['events'])} | {fmt(row['event_rate'])} | {fmt(row['roc_auc'])} | {fmt(row['pr_auc'])} | {fmt(row['brier'])} | {fmt(row['calibration_mean_error'])} | {fmt(row['calibration_slope'])} |"
        )

    lines.extend(
        [
            "",
            "## Test Per-Origin ROC-AUC For Best Pooled Model",
            "",
            "| outcome_id | model | origin_wave | rows | events | roc_auc | pr_auc | brier |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in test_origin.sort_values(["outcome_id", "origin_wave"]).iterrows():
        lines.append(
            f"| {row['outcome_id']} | {row['model']} | {row['origin_wave']} | {int(row['n'])} | {int(row['events'])} | {fmt(row['roc_auc'])} | {fmt(row['pr_auc'])} | {fmt(row['brier'])} |"
        )

    lines.extend(
        [
            "",
            "## Test First-Origin Landmark C-Index",
            "",
            "| outcome_id | model | ids | events | c_index |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for _, row in test_survival.sort_values(["outcome_id", "model"]).iterrows():
        lines.append(
            f"| {row['outcome_id']} | {row['model']} | {int(row['ids'])} | {int(row['events'])} | {fmt(row['c_index'])} |"
        )

    lines.extend(
        [
            "",
            "The C-index uses each respondent's first eligible origin-wave hazard as the ID-level risk score, with later event/censor time from the person-period path. This avoids inflating risk simply because a censored respondent contributed more observed periods.",
            "",
            "## Feature Guardrails",
            "",
            "| outcome_id | safe_predictors | time_features |",
            "|---|---:|---|",
        ]
    )
    for outcome_id in manifest["selected_outcomes"]:
        time_features = ", ".join(manifest.get("time_features", []))
        lines.append(
            f"| {outcome_id} | {len(manifest['per_outcome_predictors'][outcome_id])} | {time_features} |"
        )

    lines.extend(
        [
            "",
            "Per-outcome feature lists come from the leakage matrix after excluding direct targets, disease diagnosis years, disease-specific diagnostic/treatment constructs, and multimorbidity disease-count constructs. The matrix stores a union of feature columns for efficiency; each model still uses only its own per-outcome safe predictor list plus `origin_wave` and `interval_years`.",
            "",
            "## Inputs",
            "",
            f"- Matrix: `{config['outputs']['supervised_outcome_matrix']}`",
            f"- Time-to-event: `{config['outputs']['supervised_time_to_event_matrix']}`",
            f"- Manifest: `{config['outputs']['supervised_matrix_manifest']}`",
            f"- Feature union count: {len(manifest['feature_columns'])}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--models", nargs="+", default=["logistic_regression", "lightgbm", "xgboost"]
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    matrix = pd.read_parquet(resolve_path(config["outputs"]["supervised_outcome_matrix"]))
    manifest = load_json(resolve_path(config["outputs"]["supervised_matrix_manifest"]))
    feature_type_by_variable = manifest["feature_type_by_variable"]
    id_column = manifest["metadata"]["id_column"]
    verify_group_splits(matrix, id_column)

    rows: list[dict[str, Any]] = []
    warnings.filterwarnings("ignore", category=UserWarning)
    target_column = "event_next_wave"
    time_features = manifest.get("time_features", ["origin_wave", "interval_years"])
    for outcome_id in manifest["selected_outcomes"]:
        predictor_variables = manifest["per_outcome_predictors"][outcome_id]
        features = [f"feature__{variable}" for variable in predictor_variables] + time_features
        missing_features = sorted(set(features) - set(matrix.columns))
        if missing_features:
            raise RuntimeError(f"{outcome_id} requested missing features: {missing_features[:20]}")
        outcome_columns = list(
            dict.fromkeys(
                [
                    id_column,
                    "split",
                    "origin_wave",
                    "next_wave",
                    "origin_year",
                    "next_year",
                    target_column,
                ]
                + features
            )
        )
        outcome_frame = matrix.loc[matrix["outcome_id"].eq(outcome_id), outcome_columns].copy()
        outcome_frame[target_column] = outcome_frame[target_column].astype(int)

        train_frame = outcome_frame[outcome_frame["split"].eq("train")]
        if train_frame[target_column].nunique() < 2:
            print(f"Skipping {outcome_id}: train split has one class")
            continue
        x_train = train_frame[features]
        y_train = train_frame[target_column]

        for model_name in args.models:
            print(
                f"Training {model_name} for {outcome_id} on {len(train_frame)} train person-period rows"
            )
            model = make_model(model_name, config, y_train, features, feature_type_by_variable)
            model.fit(x_train, y_train)
            for split in ["validation", "test"]:
                split_frame = outcome_frame[outcome_frame["split"].eq(split)]
                if split_frame.empty:
                    continue
                y_true = split_frame[target_column].astype(int)
                y_prob = predict_probability(model, split_frame[features])
                pooled_metrics = evaluate_predictions(y_true, y_prob)
                rows.append(
                    metric_row(
                        outcome_id=outcome_id,
                        model_name=model_name,
                        split=split,
                        metric_scope="pooled",
                        origin_wave="pooled",
                        n=len(split_frame),
                        ids=split_frame[id_column].nunique(),
                        events=int(y_true.sum()),
                        features=len(features),
                        values=pooled_metrics,
                    )
                )

                split_with_prob = split_frame.copy()
                split_with_prob["probability"] = y_prob
                for origin_wave, origin_frame in split_with_prob.groupby("origin_wave", sort=True):
                    origin_y = origin_frame[target_column].astype(int)
                    origin_metrics = evaluate_predictions(
                        origin_y, origin_frame["probability"].to_numpy(dtype=float)
                    )
                    rows.append(
                        metric_row(
                            outcome_id=outcome_id,
                            model_name=model_name,
                            split=split,
                            metric_scope="origin",
                            origin_wave=int(origin_wave),
                            n=len(origin_frame),
                            ids=origin_frame[id_column].nunique(),
                            events=int(origin_y.sum()),
                            features=len(features),
                            values=origin_metrics,
                        )
                    )

                survival_metrics = discrete_hazard_c_index(split_frame, y_prob, id_column)
                rows.append(
                    metric_row(
                        outcome_id=outcome_id,
                        model_name=model_name,
                        split=split,
                        metric_scope="survival",
                        origin_wave="all",
                        n=survival_metrics["ids"],
                        ids=survival_metrics["ids"],
                        events=survival_metrics["events"],
                        features=len(features),
                        values=survival_metrics,
                    )
                )

    metrics_df = pd.DataFrame(rows)
    if metrics_df.empty:
        raise RuntimeError("No baseline probe metrics were produced")
    metrics_path = resolve_path(config["outputs"]["baseline_probe_metrics"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(metrics_path, index=False)

    report_path = resolve_path(config["outputs"]["baseline_probe_report"])
    write_report(report_path, metrics_df, config, manifest)

    test_pooled = metrics_df[
        metrics_df["split"].eq("test") & metrics_df["metric_scope"].eq("pooled")
    ]
    test_survival = metrics_df[
        metrics_df["split"].eq("test") & metrics_df["metric_scope"].eq("survival")
    ]
    summary = {
        "outcomes": int(metrics_df["outcome_id"].nunique()),
        "models": sorted(metrics_df["model"].unique().tolist()),
        "test_pooled_rows": int(test_pooled["n"].sum()),
        "best_test_roc_auc_min": float(test_pooled.groupby("outcome_id")["roc_auc"].max().min()),
        "best_test_roc_auc_max": float(test_pooled.groupby("outcome_id")["roc_auc"].max().max()),
        "best_test_c_index_min": float(test_survival.groupby("outcome_id")["c_index"].max().min()),
        "best_test_c_index_max": float(test_survival.groupby("outcome_id")["c_index"].max().max()),
        "metrics": str(metrics_path.relative_to(PROJECT_ROOT)),
        "report": str(report_path.relative_to(PROJECT_ROOT)),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
