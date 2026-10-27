#!/usr/bin/env python3
"""Train HRS rolling next-wave baselines and report honest per-origin AUCs."""

from __future__ import annotations

import argparse
import json
import math
import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sksurv.metrics import concordance_index_censored
from xgboost import XGBClassifier

from hrs_utils import (
    ID_COLUMN,
    OUTPUT_DIR,
    PROJECT_ROOT,
    SELECTED_OUTCOMES,
    fmt,
    generated_timestamp,
    verify_group_splits,
    write_text,
)

CHARLS_REFERENCE_AUC = {
    "low_cognition_onset_data_driven": 0.799,
    "iadl_new_difficulty": 0.771,
    "adl_new_difficulty": 0.745,
    "high_depressive_symptoms_onset": 0.702,
    "incident_diabetes": 0.682,
    "multimorbidity_onset": 0.669,
    "incident_hypertension": 0.620,
    "hospitalization_onset": 0.636,
}


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
    y_train: pd.Series,
    features: list[str],
    feature_type_by_variable: dict[str, str | None],
    random_state: int,
    n_jobs: int,
    estimators: int,
) -> Pipeline:
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
            n_estimators=estimators,
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
        xgboost_device = os.environ.get("CHARLS_XGBOOST_DEVICE")
        xgboost_kwargs: dict[str, Any] = {}
        if xgboost_device:
            xgboost_kwargs["device"] = str(xgboost_device)
        classifier = XGBClassifier(
            n_estimators=estimators,
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


def discrete_hazard_c_index(frame: pd.DataFrame, probabilities: np.ndarray) -> dict[str, float]:
    risk_frame = frame[
        [ID_COLUMN, "origin_wave", "next_wave", "origin_year", "next_year", "event_next_wave"]
    ].copy()
    risk_frame["probability"] = np.clip(probabilities, 0, 1)
    rows: list[dict[str, Any]] = []
    for id_value, group in risk_frame.sort_values([ID_COLUMN, "origin_wave"]).groupby(
        ID_COLUMN, sort=False
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
                ID_COLUMN: id_value,
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


def origin_mean_row(
    origin_rows: list[dict[str, Any]], outcome_id: str, model_name: str, split: str, features: int
) -> dict[str, Any]:
    origin_df = pd.DataFrame(origin_rows)
    values = {
        "roc_auc": float(origin_df["roc_auc"].mean(skipna=True)),
        "pr_auc": float(origin_df["pr_auc"].mean(skipna=True)),
        "brier": float(origin_df["brier"].mean(skipna=True)),
        "event_rate": (
            float(origin_df["events"].sum() / origin_df["n"].sum())
            if origin_df["n"].sum()
            else math.nan
        ),
        "mean_predicted_risk": float(origin_df["mean_predicted_risk"].mean(skipna=True)),
        "calibration_mean_error": float(origin_df["calibration_mean_error"].mean(skipna=True)),
        "calibration_intercept": float(origin_df["calibration_intercept"].mean(skipna=True)),
        "calibration_slope": float(origin_df["calibration_slope"].mean(skipna=True)),
    }
    return metric_row(
        outcome_id=outcome_id,
        model_name=model_name,
        split=split,
        metric_scope="origin_mean",
        origin_wave="mean",
        n=int(origin_df["n"].sum()),
        ids=int(origin_df["ids"].max()),
        events=int(origin_df["events"].sum()),
        features=features,
        values=values,
    )


def best_model_by_validation(metrics: pd.DataFrame) -> dict[str, str]:
    validation = metrics[
        metrics["split"].eq("validation") & metrics["metric_scope"].eq("origin_mean")
    ].copy()
    if validation.empty:
        raise RuntimeError("No validation origin-mean rows available for model selection")
    best = (
        validation.sort_values(["outcome_id", "roc_auc", "pr_auc"], ascending=[True, False, False])
        .groupby("outcome_id", as_index=False)
        .head(1)
    )
    return dict(zip(best["outcome_id"], best["model"]))


def write_report(metrics: pd.DataFrame, manifest: dict[str, Any]) -> None:
    best_map = best_model_by_validation(metrics)
    test_mean = metrics[
        metrics["split"].eq("test") & metrics["metric_scope"].eq("origin_mean")
    ].copy()
    test_mean = test_mean[
        test_mean.apply(lambda row: best_map.get(row["outcome_id"]) == row["model"], axis=1)
    ]
    test_pooled = metrics[metrics["split"].eq("test") & metrics["metric_scope"].eq("pooled")].copy()
    test_pooled = test_pooled[
        test_pooled.apply(lambda row: best_map.get(row["outcome_id"]) == row["model"], axis=1)
    ]
    test_origin = metrics[metrics["split"].eq("test") & metrics["metric_scope"].eq("origin")].copy()
    test_origin = test_origin[
        test_origin.apply(lambda row: best_map.get(row["outcome_id"]) == row["model"], axis=1)
    ]
    test_survival = metrics[
        metrics["split"].eq("test") & metrics["metric_scope"].eq("survival")
    ].copy()
    test_survival = test_survival[
        test_survival.apply(lambda row: best_map.get(row["outcome_id"]) == row["model"], axis=1)
    ]

    lines = [
        "# HRS Rolling Baseline Probe Report",
        "",
        f"Generated: {generated_timestamp()}",
        "",
        "## Summary",
        "",
        f"- Outcomes evaluated: {metrics['outcome_id'].nunique()}",
        f"- Models: {', '.join(metrics['model'].drop_duplicates())}",
        f"- Matrix type: `{manifest['metadata'].get('matrix_type', 'unknown')}`",
        "- Model selection: highest validation per-origin mean ROC-AUC, with PR-AUC as tie-breaker.",
        "- Main test metric: test per-origin mean ROC-AUC. Pooled ROC-AUC is context only.",
        "- Split: respondent-level `GroupShuffleSplit`; no ID crosses train/validation/test.",
        "",
        "## Main Test Readout",
        "",
        "| outcome_id | selected_model | HRS test per-origin mean AUC | CHARLS reference AUC | delta HRS-CHARLS | pooled test AUC | rows | events |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    pooled_by_outcome = test_pooled.set_index("outcome_id")
    for _, row in test_mean.sort_values("outcome_id").iterrows():
        outcome_id = row["outcome_id"]
        pooled_auc = (
            pooled_by_outcome.loc[outcome_id, "roc_auc"]
            if outcome_id in pooled_by_outcome.index
            else math.nan
        )
        charls_auc = CHARLS_REFERENCE_AUC.get(outcome_id, math.nan)
        delta = (
            float(row["roc_auc"] - charls_auc)
            if not pd.isna(charls_auc) and not pd.isna(row["roc_auc"])
            else math.nan
        )
        lines.append(
            f"| {outcome_id} | {row['model']} | {fmt(row['roc_auc'])} | {fmt(charls_auc)} | {fmt(delta)} | {fmt(pooled_auc)} | {int(row['n'])} | {int(row['events'])} |"
        )

    lines.extend(
        [
            "",
            "## Test Per-Origin Detail",
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
            "This report completes the HRS native rolling baseline validation. The foundation-model embedding comparison is not adjudicated here; it requires the HRS encoder/export/probe scripts in the next validation step.",
            "",
        ]
    )
    write_text(OUTPUT_DIR / "hrs_baseline_probe_report.md", lines)


def write_external_validation_status(metrics: pd.DataFrame) -> None:
    best_map = best_model_by_validation(metrics)
    test_mean = metrics[
        metrics["split"].eq("test") & metrics["metric_scope"].eq("origin_mean")
    ].copy()
    test_mean = test_mean[
        test_mean.apply(lambda row: best_map.get(row["outcome_id"]) == row["model"], axis=1)
    ]
    lines = [
        "# HRS External Validation Report",
        "",
        f"Generated: {generated_timestamp()}",
        "",
        "## Anchor Questions",
        "",
        "| question | current answer | evidence |",
        "|---|---|---|",
        "| Does the rigorous rolling per-origin benchmark carry to HRS? | Baseline stage complete; see per-origin HRS AUCs below. | `hrs_baseline_metrics.csv` and `hrs_baseline_probe_report.md` |",
        "| Does the CHARLS multimorbidity FM embedding gain reproduce in HRS? | Not yet adjudicated. | HRS encoder/export/embedding-vs-tree scripts remain pending. |",
        "",
        "## Baseline AUC Comparison",
        "",
        "| outcome_id | selected_model | HRS test per-origin mean AUC | CHARLS reference AUC | delta HRS-CHARLS |",
        "|---|---|---:|---:|---:|",
    ]
    for _, row in test_mean.sort_values("outcome_id").iterrows():
        outcome_id = row["outcome_id"]
        charls_auc = CHARLS_REFERENCE_AUC.get(outcome_id, math.nan)
        delta = (
            float(row["roc_auc"] - charls_auc)
            if not pd.isna(charls_auc) and not pd.isna(row["roc_auc"])
            else math.nan
        )
        lines.append(
            f"| {outcome_id} | {row['model']} | {fmt(row['roc_auc'])} | {fmt(charls_auc)} | {fmt(delta)} |"
        )
    lines.extend(
        [
            "",
            "## Completed Artifacts",
            "",
            "- `hrs_discovery_report.md`",
            "- `external_variable_crosswalk.csv`",
            "- `external_outcome_crosswalk.csv`",
            "- `hrs_person_wave.parquet`",
            "- `hrs_per_outcome_leakage.csv`",
            "- `hrs_supervised_outcome_matrix.parquet`",
            "- `hrs_baseline_metrics.csv`",
            "",
            "## Pending For Full Stage D",
            "",
            "- Train HRS-native CHARLS-found-lite encoder.",
            "- Export unmasked and diagnostic-masked HRS embeddings.",
            "- Run HRS embedding-vs-tree comparison and multimorbidity stability bootstrap.",
            "",
        ]
    )
    write_text(OUTPUT_DIR / "09_external_validation_report.md", lines)


def train_and_evaluate(args: argparse.Namespace) -> pd.DataFrame:
    matrix_path = OUTPUT_DIR / "hrs_supervised_outcome_matrix.parquet"
    manifest_path = OUTPUT_DIR / "hrs_supervised_matrix_manifest.json"
    if not matrix_path.exists() or not manifest_path.exists():
        raise RuntimeError("Run hrs_build_supervised_matrix.py before training HRS baselines")
    matrix = pd.read_parquet(matrix_path)
    manifest = load_json(manifest_path)
    feature_type_by_variable = manifest["feature_type_by_variable"]
    verify_group_splits(matrix, ID_COLUMN)

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
                    ID_COLUMN,
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
            model = make_model(
                model_name,
                y_train,
                features,
                feature_type_by_variable,
                random_state=args.random_state,
                n_jobs=args.n_jobs,
                estimators=args.estimators,
            )
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
                        ids=split_frame[ID_COLUMN].nunique(),
                        events=int(y_true.sum()),
                        features=len(features),
                        values=pooled_metrics,
                    )
                )

                split_with_prob = split_frame.copy()
                split_with_prob["probability"] = y_prob
                origin_rows = []
                for origin_wave, origin_frame in split_with_prob.groupby("origin_wave", sort=True):
                    origin_y = origin_frame[target_column].astype(int)
                    origin_metrics = evaluate_predictions(
                        origin_y, origin_frame["probability"].to_numpy(dtype=float)
                    )
                    row = metric_row(
                        outcome_id=outcome_id,
                        model_name=model_name,
                        split=split,
                        metric_scope="origin",
                        origin_wave=int(origin_wave),
                        n=len(origin_frame),
                        ids=origin_frame[ID_COLUMN].nunique(),
                        events=int(origin_y.sum()),
                        features=len(features),
                        values=origin_metrics,
                    )
                    rows.append(row)
                    origin_rows.append(row)
                if origin_rows:
                    rows.append(
                        origin_mean_row(origin_rows, outcome_id, model_name, split, len(features))
                    )

                survival_metrics = discrete_hazard_c_index(split_frame, y_prob)
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
        raise RuntimeError("No HRS baseline metrics were produced")
    metrics_df.to_csv(OUTPUT_DIR / "hrs_baseline_metrics.csv", index=False)
    write_report(metrics_df, manifest)
    write_external_validation_status(metrics_df)
    return metrics_df


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+", default=["logistic_regression", "lightgbm", "xgboost"]
    )
    parser.add_argument("--random-state", type=int, default=20260620)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--estimators", type=int, default=250)
    args = parser.parse_args()

    metrics = train_and_evaluate(args)
    test_mean = metrics[metrics["split"].eq("test") & metrics["metric_scope"].eq("origin_mean")]
    summary = {
        "outcomes": int(metrics["outcome_id"].nunique()),
        "models": sorted(metrics["model"].unique().tolist()),
        "test_origin_mean_rows": int(len(test_mean)),
        "metrics": str((OUTPUT_DIR / "hrs_baseline_metrics.csv").relative_to(PROJECT_ROOT)),
        "report": str((OUTPUT_DIR / "hrs_baseline_probe_report.md").relative_to(PROJECT_ROOT)),
        "external_validation_status": str(
            (OUTPUT_DIR / "09_external_validation_report.md").relative_to(PROJECT_ROOT)
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
