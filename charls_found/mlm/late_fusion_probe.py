#!/usr/bin/env python3
"""Evaluate probability-level late fusion of tree and embedding probes."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_MLM_CONFIG = SCRIPT_PATH.parents[1] / "configs" / "mlm.yaml"
DEFAULT_SUPERVISED_CONFIG = SCRIPT_PATH.parents[1] / "configs" / "supervised.yaml"


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BASELINE = load_module(
    "charls_late_fusion_baseline",
    PROJECT_ROOT / "charls_found" / "supervised" / "train_baseline_probe.py",
)
COMPARE = load_module(
    "charls_late_fusion_compare",
    PROJECT_ROOT / "charls_found" / "mlm" / "compare_embedding_probe.py",
)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def fmt(value: Any, digits: int = 4) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def tree_features(manifest: dict[str, Any], outcome_id: str) -> list[str]:
    predictor_variables = manifest["per_outcome_predictors"][outcome_id]
    return [f"feature__{variable}" for variable in predictor_variables] + manifest.get(
        "time_features", ["origin_wave", "interval_years"]
    )


def origin_mean_auc(frame: pd.DataFrame, probability_column: str) -> float:
    aucs: list[float] = []
    for _, origin_frame in frame.groupby("origin_wave", sort=True):
        values = COMPARE.evaluate_predictions(
            origin_frame["event_next_wave"].astype(int),
            origin_frame[probability_column].to_numpy(dtype=float),
        )
        if not pd.isna(values["roc_auc"]):
            aucs.append(float(values["roc_auc"]))
    return float(np.mean(aucs)) if aucs else math.nan


def fit_tree_model(
    frame: pd.DataFrame,
    features: list[str],
    model_name: str,
    supervised_config: dict[str, Any],
    manifest: dict[str, Any],
) -> Any:
    train = frame[frame["split"].eq("train")]
    if train["event_next_wave"].nunique() < 2:
        raise RuntimeError(f"{model_name} train split has one class")
    model = BASELINE.make_model(
        model_name,
        supervised_config,
        train["event_next_wave"].astype(int),
        features,
        manifest["feature_type_by_variable"],
    )
    model.fit(train[features], train["event_next_wave"].astype(int))
    return model


def predict_tree(
    frame: pd.DataFrame, features: list[str], model: Any, split: str, probability_column: str
) -> pd.DataFrame:
    split_frame = frame[frame["split"].eq(split)].copy()
    split_frame[probability_column] = BASELINE.predict_probability(model, split_frame[features])
    return split_frame


def fit_embedding_model(
    frame: pd.DataFrame,
    model_name: str,
    pooling: str,
    include_time_features: bool,
    comparison_config: dict[str, Any],
) -> Any:
    train = frame[frame["split"].eq("train")]
    if train["event_next_wave"].nunique() < 2:
        raise RuntimeError(f"{model_name}/{pooling} train split has one class")
    model = COMPARE.make_embedding_probe(model_name, comparison_config)
    COMPARE.fit_embedding_probe(
        model,
        model_name,
        COMPARE.embedding_matrix(train, pooling, include_time_features),
        train["event_next_wave"].astype(int),
    )
    return model


def predict_embedding(
    frame: pd.DataFrame,
    model: Any,
    pooling: str,
    include_time_features: bool,
    split: str,
    probability_column: str,
) -> pd.DataFrame:
    split_frame = frame[frame["split"].eq(split)].copy()
    split_frame[probability_column] = model.predict_proba(
        COMPARE.embedding_matrix(split_frame, pooling, include_time_features)
    )[:, 1]
    return split_frame


def select_tree_candidate(
    frame: pd.DataFrame,
    features: list[str],
    tree_models: list[str],
    supervised_config: dict[str, Any],
    manifest: dict[str, Any],
) -> tuple[str, Any, float]:
    candidates: list[tuple[float, str, Any]] = []
    for model_name in tree_models:
        model = fit_tree_model(frame, features, model_name, supervised_config, manifest)
        validation = predict_tree(frame, features, model, "validation", "tree_probability")
        candidates.append((origin_mean_auc(validation, "tree_probability"), model_name, model))
    candidates = [candidate for candidate in candidates if not pd.isna(candidate[0])]
    if not candidates:
        raise RuntimeError("No valid tree candidate")
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1], candidates[0][2], float(candidates[0][0])


def select_embedding_candidate(
    frame: pd.DataFrame,
    embedding_models: list[str],
    poolings: list[str],
    include_time_features: bool,
    comparison_config: dict[str, Any],
) -> tuple[str, str, Any, float]:
    candidates: list[tuple[float, str, str, Any]] = []
    for model_name in embedding_models:
        for pooling in poolings:
            model = fit_embedding_model(
                frame, model_name, pooling, include_time_features, comparison_config
            )
            validation = predict_embedding(
                frame, model, pooling, include_time_features, "validation", "embedding_probability"
            )
            candidates.append(
                (origin_mean_auc(validation, "embedding_probability"), model_name, pooling, model)
            )
    candidates = [candidate for candidate in candidates if not pd.isna(candidate[0])]
    if not candidates:
        raise RuntimeError("No valid embedding candidate")
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1], candidates[0][2], candidates[0][3], float(candidates[0][0])


def paired_predictions(
    tree_frame: pd.DataFrame, embedding_frame: pd.DataFrame, id_column: str
) -> pd.DataFrame:
    columns = [
        "row_id",
        id_column,
        "origin_wave",
        "next_wave",
        "origin_year",
        "next_year",
        "interval_years",
        "event_next_wave",
        "tree_probability",
    ]
    paired = tree_frame[columns].merge(
        embedding_frame[["row_id", "embedding_probability"]],
        on="row_id",
        how="inner",
        validate="one_to_one",
    )
    if len(paired) != len(tree_frame):
        raise RuntimeError("Tree and embedding prediction rows are not aligned")
    return paired


def fusion_matrix(frame: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    return frame[feature_names].to_numpy(dtype=np.float32)


def make_fusion_model() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000, class_weight="balanced", solver="lbfgs", random_state=20260621
                ),
            ),
        ]
    )


def fit_fusion_model(validation_frame: pd.DataFrame, feature_names: list[str]) -> Pipeline:
    if validation_frame["event_next_wave"].nunique() < 2:
        raise RuntimeError("Fusion validation split has one class")
    model = make_fusion_model()
    model.fit(
        fusion_matrix(validation_frame, feature_names),
        validation_frame["event_next_wave"].astype(int),
    )
    return model


def predict_fusion(frame: pd.DataFrame, model: Pipeline, feature_names: list[str]) -> np.ndarray:
    return model.predict_proba(fusion_matrix(frame, feature_names))[:, 1]


def metric_rows(
    outcome_id: str,
    method: str,
    model_name: str,
    split: str,
    frame: pd.DataFrame,
    probability_column: str,
    id_column: str,
    mask_mode: str,
    tree_model: str,
    embedding_model: str,
    embedding_pooling: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    y_true = frame["event_next_wave"].astype(int)
    pooled = COMPARE.evaluate_predictions(y_true, frame[probability_column].to_numpy(dtype=float))
    rows.append(
        {
            "outcome_id": outcome_id,
            "method": method,
            "model": model_name,
            "split": split,
            "metric_scope": "pooled",
            "origin_wave": "pooled",
            "n": int(len(frame)),
            "ids": int(frame[id_column].nunique()),
            "events": int(y_true.sum()),
            "roc_auc": pooled["roc_auc"],
            "pr_auc": pooled["pr_auc"],
            "brier": pooled["brier"],
            "c_index": math.nan,
            "event_rate": pooled["event_rate"],
            "mean_predicted_risk": pooled["mean_predicted_risk"],
            "mask_mode": mask_mode,
            "selected_tree_model": tree_model,
            "selected_embedding_model": embedding_model,
            "selected_embedding_pooling": embedding_pooling,
        }
    )
    for origin_wave, origin_frame in frame.groupby("origin_wave", sort=True):
        origin_y = origin_frame["event_next_wave"].astype(int)
        values = COMPARE.evaluate_predictions(
            origin_y, origin_frame[probability_column].to_numpy(dtype=float)
        )
        rows.append(
            {
                "outcome_id": outcome_id,
                "method": method,
                "model": model_name,
                "split": split,
                "metric_scope": "origin",
                "origin_wave": int(origin_wave),
                "n": int(len(origin_frame)),
                "ids": int(origin_frame[id_column].nunique()),
                "events": int(origin_y.sum()),
                "roc_auc": values["roc_auc"],
                "pr_auc": values["pr_auc"],
                "brier": values["brier"],
                "c_index": math.nan,
                "event_rate": values["event_rate"],
                "mean_predicted_risk": values["mean_predicted_risk"],
                "mask_mode": mask_mode,
                "selected_tree_model": tree_model,
                "selected_embedding_model": embedding_model,
                "selected_embedding_pooling": embedding_pooling,
            }
        )
    survival = COMPARE.first_origin_c_index(
        frame, frame[probability_column].to_numpy(dtype=float), id_column
    )
    rows.append(
        {
            "outcome_id": outcome_id,
            "method": method,
            "model": model_name,
            "split": split,
            "metric_scope": "survival",
            "origin_wave": "all",
            "n": int(survival["ids"]),
            "ids": int(survival["ids"]),
            "events": int(survival["events"]),
            "roc_auc": math.nan,
            "pr_auc": math.nan,
            "brier": math.nan,
            "c_index": survival["c_index"],
            "event_rate": math.nan,
            "mean_predicted_risk": math.nan,
            "mask_mode": mask_mode,
            "selected_tree_model": tree_model,
            "selected_embedding_model": embedding_model,
            "selected_embedding_pooling": embedding_pooling,
        }
    )
    return rows


def per_origin_summary(metrics: pd.DataFrame, outcome_id: str, method: str) -> dict[str, float]:
    subset = metrics[
        metrics["outcome_id"].eq(outcome_id)
        & metrics["method"].eq(method)
        & metrics["split"].eq("test")
        & metrics["metric_scope"].eq("origin")
    ]
    auc_by_wave = {int(row["origin_wave"]): float(row["roc_auc"]) for _, row in subset.iterrows()}
    values = [value for value in auc_by_wave.values() if not pd.isna(value)]
    return {
        "mean_auc": float(np.mean(values)) if values else math.nan,
        "origin_1_auc": auc_by_wave.get(1, math.nan),
        "origin_2_auc": auc_by_wave.get(2, math.nan),
        "origin_3_auc": auc_by_wave.get(3, math.nan),
        "origin_4_auc": auc_by_wave.get(4, math.nan),
    }


def c_index(metrics: pd.DataFrame, outcome_id: str, method: str) -> float:
    subset = metrics[
        metrics["outcome_id"].eq(outcome_id)
        & metrics["method"].eq(method)
        & metrics["split"].eq("test")
        & metrics["metric_scope"].eq("survival")
    ]
    return float(subset["c_index"].iloc[0]) if not subset.empty else math.nan


def build_decision_table(
    metrics: pd.DataFrame,
    outcomes: list[str],
    metadata: dict[str, dict[str, Any]],
    threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for outcome_id in outcomes:
        tree = per_origin_summary(metrics, outcome_id, "tree")
        embedding = per_origin_summary(metrics, outcome_id, "embedding")
        fusion = per_origin_summary(metrics, outcome_id, "tree+embedding")
        tree_c = c_index(metrics, outcome_id, "tree")
        embedding_c = c_index(metrics, outcome_id, "embedding")
        fusion_c = c_index(metrics, outcome_id, "tree+embedding")
        best_single_auc = max(tree["mean_auc"], embedding["mean_auc"])
        best_single_method = "tree" if tree["mean_auc"] >= embedding["mean_auc"] else "embedding"
        fusion_minus_tree = fusion["mean_auc"] - tree["mean_auc"]
        fusion_minus_embedding = fusion["mean_auc"] - embedding["mean_auc"]
        fusion_minus_best = fusion["mean_auc"] - best_single_auc
        if fusion_minus_best >= threshold:
            verdict = "fusion improves"
        elif fusion_minus_tree >= threshold:
            verdict = "adds beyond tree"
        elif fusion_minus_tree <= -threshold:
            verdict = "worse than tree"
        else:
            verdict = "tie vs tree"
        info = metadata[outcome_id]
        rows.append(
            {
                "outcome_id": outcome_id,
                "mask_mode": info["mask_mode"],
                "tree_model": info["tree_model"],
                "embedding_model": info["embedding_model"],
                "embedding_pooling": info["embedding_pooling"],
                "tree_validation_origin_mean_auc": info["tree_validation_origin_mean_auc"],
                "embedding_validation_origin_mean_auc": info[
                    "embedding_validation_origin_mean_auc"
                ],
                "tree_test_per_origin_mean_auc": tree["mean_auc"],
                "embedding_test_per_origin_mean_auc": embedding["mean_auc"],
                "tree_embedding_test_per_origin_mean_auc": fusion["mean_auc"],
                "delta_embedding_minus_tree": embedding["mean_auc"] - tree["mean_auc"],
                "delta_fusion_minus_tree": fusion_minus_tree,
                "delta_fusion_minus_embedding": fusion_minus_embedding,
                "delta_fusion_minus_best_single": fusion_minus_best,
                "best_single_method": best_single_method,
                "tree_test_c_index": tree_c,
                "embedding_test_c_index": embedding_c,
                "tree_embedding_test_c_index": fusion_c,
                "delta_c_fusion_minus_tree": fusion_c - tree_c,
                "delta_c_fusion_minus_embedding": fusion_c - embedding_c,
                "late_fusion_verdict": verdict,
                "tree_origin_1_auc": tree["origin_1_auc"],
                "tree_origin_2_auc": tree["origin_2_auc"],
                "tree_origin_3_auc": tree["origin_3_auc"],
                "tree_origin_4_auc": tree["origin_4_auc"],
                "embedding_origin_1_auc": embedding["origin_1_auc"],
                "embedding_origin_2_auc": embedding["origin_2_auc"],
                "embedding_origin_3_auc": embedding["origin_3_auc"],
                "embedding_origin_4_auc": embedding["origin_4_auc"],
                "fusion_origin_1_auc": fusion["origin_1_auc"],
                "fusion_origin_2_auc": fusion["origin_2_auc"],
                "fusion_origin_3_auc": fusion["origin_3_auc"],
                "fusion_origin_4_auc": fusion["origin_4_auc"],
            }
        )
    return pd.DataFrame(rows)


def write_report(path: Path, decision: pd.DataFrame, config: dict[str, Any]) -> None:
    lines = [
        "# CHARLS Late-Fusion Probe Report",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "- Main metric: test per-origin mean ROC-AUC.",
        "- Late fusion is probability-level stacking: selected tree probability + selected embedding probability.",
        "- Tree and embedding candidates are selected on validation per-origin mean ROC-AUC.",
        "- Fusion logistic regression is fit on the validation split and evaluated once on test.",
        "- Disease and multimorbidity outcomes use diagnostic-masked embeddings; clean outcomes use unmasked embeddings.",
        "",
        "## Decision Table",
        "",
        "| outcome_id | tree | embedding | tree+embedding | fusion-tree | fusion-embedding | tree_c | embedding_c | fusion_c | verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in decision.iterrows():
        lines.append(
            f"| {row['outcome_id']} | {fmt(row['tree_test_per_origin_mean_auc'])} | "
            f"{fmt(row['embedding_test_per_origin_mean_auc'])} | {fmt(row['tree_embedding_test_per_origin_mean_auc'])} | "
            f"{fmt(row['delta_fusion_minus_tree'])} | {fmt(row['delta_fusion_minus_embedding'])} | "
            f"{fmt(row['tree_test_c_index'])} | {fmt(row['embedding_test_c_index'])} | {fmt(row['tree_embedding_test_c_index'])} | "
            f"{row['late_fusion_verdict']} |"
        )
    lines.extend(
        [
            "",
            "## Inputs",
            "",
            f"- Supervised matrix: `{config['inputs']['supervised_outcome_matrix']}`",
            f"- Embedding run: `outputs/07_mlm/runs/{config['comparison']['embedding_run_name']}`",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlm-config", type=Path, default=DEFAULT_MLM_CONFIG)
    parser.add_argument("--supervised-config", type=Path, default=DEFAULT_SUPERVISED_CONFIG)
    args = parser.parse_args()

    mlm_config = load_yaml(args.mlm_config)
    supervised_config = load_yaml(args.supervised_config)
    comparison_config = mlm_config["comparison"]
    late_fusion_config = mlm_config.get("late_fusion", {})
    fusion_features = list(
        late_fusion_config.get("fusion_features", ["tree_probability", "embedding_probability"])
    )
    manifest = load_json(resolve_path(mlm_config["inputs"]["supervised_matrix_manifest"]))
    id_column = manifest["metadata"]["id_column"]
    selected_outcomes = list(manifest["selected_outcomes"])
    include_time_features = bool(comparison_config.get("include_time_features", True))

    registry = pd.read_csv(
        resolve_path("outputs/02_outcome_registry/outcome_registry_validated.csv")
    )
    domain_by_outcome = registry.set_index("outcome_id")["domain"].to_dict()
    disease_domains = set(comparison_config["disease_domains"])
    clean_mask_mode = comparison_config["clean_mask_mode"]
    disease_mask_mode = comparison_config["disease_mask_mode"]

    matrix = pd.read_parquet(resolve_path(mlm_config["inputs"]["supervised_outcome_matrix"]))
    embedding_cache: dict[Path, pd.DataFrame] = {}
    rows: list[dict[str, Any]] = []
    metadata: dict[str, dict[str, Any]] = {}

    for outcome_id in selected_outcomes:
        print(f"Late-fusion evaluation for {outcome_id}")
        outcome_matrix = matrix[matrix["outcome_id"].eq(outcome_id)].copy().reset_index(drop=True)
        outcome_matrix["row_id"] = np.arange(len(outcome_matrix))
        features = tree_features(manifest, outcome_id)
        missing_features = sorted(set(features) - set(outcome_matrix.columns))
        if missing_features:
            raise RuntimeError(
                f"{outcome_id} requested missing tree features: {missing_features[:20]}"
            )
        tree_columns = list(
            dict.fromkeys(
                [
                    "row_id",
                    id_column,
                    "split",
                    "origin_wave",
                    "next_wave",
                    "origin_year",
                    "next_year",
                    "interval_years",
                    "event_next_wave",
                ]
                + features
            )
        )
        tree_frame = outcome_matrix[tree_columns].copy()
        tree_model_name, tree_model, tree_validation_auc = select_tree_candidate(
            tree_frame,
            features,
            list(comparison_config["baseline_tree_models"]),
            supervised_config,
            manifest,
        )

        mask_mode = (
            disease_mask_mode
            if domain_by_outcome.get(outcome_id) in disease_domains
            else clean_mask_mode
        )
        emb_path = COMPARE.embedding_path(
            comparison_config["embedding_run_name"], outcome_id, mask_mode
        )
        if emb_path not in embedding_cache:
            embedding_cache[emb_path] = COMPARE.load_embedding_frame(emb_path, id_column)
        embedding_frame = outcome_matrix[
            [
                "row_id",
                id_column,
                "split",
                "origin_wave",
                "next_wave",
                "origin_year",
                "next_year",
                "interval_years",
                "event_next_wave",
            ]
        ].merge(
            embedding_cache[emb_path],
            on=[id_column, "origin_wave"],
            how="left",
            suffixes=("", "_embedding"),
            validate="many_to_one",
        )
        if embedding_frame["mean_embedding"].isna().any():
            missing = int(embedding_frame["mean_embedding"].isna().sum())
            raise RuntimeError(f"{outcome_id} has {missing} rows without embeddings")
        embedding_model_name, embedding_pooling, embedding_model, embedding_validation_auc = (
            select_embedding_candidate(
                embedding_frame,
                list(comparison_config.get("embedding_models", ["logistic_regression"])),
                list(comparison_config["embedding_poolings"]),
                include_time_features,
                comparison_config,
            )
        )

        validation_tree = predict_tree(
            tree_frame, features, tree_model, "validation", "tree_probability"
        )
        validation_embedding = predict_embedding(
            embedding_frame,
            embedding_model,
            embedding_pooling,
            include_time_features,
            "validation",
            "embedding_probability",
        )
        validation_paired = paired_predictions(validation_tree, validation_embedding, id_column)
        fusion_model = fit_fusion_model(validation_paired, fusion_features)

        test_tree = predict_tree(tree_frame, features, tree_model, "test", "tree_probability")
        test_embedding = predict_embedding(
            embedding_frame,
            embedding_model,
            embedding_pooling,
            include_time_features,
            "test",
            "embedding_probability",
        )
        test_paired = paired_predictions(test_tree, test_embedding, id_column)
        test_paired["fusion_probability"] = predict_fusion(
            test_paired, fusion_model, fusion_features
        )

        metadata[outcome_id] = {
            "mask_mode": mask_mode,
            "tree_model": tree_model_name,
            "embedding_model": embedding_model_name,
            "embedding_pooling": embedding_pooling,
            "tree_validation_origin_mean_auc": tree_validation_auc,
            "embedding_validation_origin_mean_auc": embedding_validation_auc,
        }
        rows.extend(
            metric_rows(
                outcome_id,
                "tree",
                tree_model_name,
                "test",
                test_paired,
                "tree_probability",
                id_column,
                mask_mode,
                tree_model_name,
                embedding_model_name,
                embedding_pooling,
            )
        )
        rows.extend(
            metric_rows(
                outcome_id,
                "embedding",
                embedding_model_name,
                "test",
                test_paired,
                "embedding_probability",
                id_column,
                mask_mode,
                tree_model_name,
                embedding_model_name,
                embedding_pooling,
            )
        )
        rows.extend(
            metric_rows(
                outcome_id,
                "tree+embedding",
                str(late_fusion_config.get("fusion_model", "stacked_logistic_regression")),
                "test",
                test_paired,
                "fusion_probability",
                id_column,
                mask_mode,
                tree_model_name,
                embedding_model_name,
                embedding_pooling,
            )
        )

    metrics = pd.DataFrame(rows)
    decision = build_decision_table(
        metrics,
        selected_outcomes,
        metadata,
        float(comparison_config.get("fm_delta_threshold", 0.01)),
    )
    metrics_path = resolve_path(mlm_config["outputs"]["late_fusion_metrics"])
    decision_path = resolve_path(mlm_config["outputs"]["late_fusion_decision_table"])
    report_path = resolve_path(mlm_config["outputs"]["late_fusion_report"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(metrics_path, index=False)
    decision.to_csv(decision_path, index=False)
    write_report(report_path, decision, mlm_config)
    print(
        json.dumps(
            {
                "metrics": str(metrics_path.relative_to(PROJECT_ROOT)),
                "decision_table": str(decision_path.relative_to(PROJECT_ROOT)),
                "report": str(report_path.relative_to(PROJECT_ROOT)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    raise SystemExit(main())
