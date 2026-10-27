#!/usr/bin/env python3
"""Run source-native late fusion for external validation cohorts."""

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


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BASELINE = load_module(
    "external_late_fusion_baseline",
    PROJECT_ROOT / "charls_found" / "supervised" / "train_baseline_probe.py",
)
COMPARE = load_module(
    "external_late_fusion_compare",
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
    return COMPARE.fmt(value, digits)


def normalize_id(series: pd.Series) -> pd.Series:
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        return series.astype(str)
    return pd.to_numeric(series, errors="raise").round().astype("int64")


def embedding_path(config: dict[str, Any], run_name: str, outcome_id: str, mask_mode: str) -> Path:
    run_dir = resolve_path(config["outputs"]["mlm_directory"]) / "runs" / run_name
    if mask_mode == "none":
        return run_dir / "person_wave_embeddings_none.parquet"
    return run_dir / f"person_wave_embeddings_{mask_mode}_{outcome_id}.parquet"


def load_embedding_frame(path: Path, id_column: str, id_reference: pd.Series) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing embedding file: {path}")
    columns = [
        "ID",
        "wave",
        "split",
        "cls_embedding",
        "mean_embedding",
        "embedding_mask_mode",
        "embedding_mask_outcome_id",
    ]
    frame = pd.read_parquet(path, columns=columns)
    frame = frame.rename(columns={"ID": id_column, "wave": "origin_wave"})
    if pd.api.types.is_object_dtype(id_reference) or pd.api.types.is_string_dtype(id_reference):
        frame[id_column] = frame[id_column].astype(str)
    else:
        frame[id_column] = pd.to_numeric(frame[id_column], errors="raise").round().astype("int64")
    return frame


def make_embedding_matrix(origin_waves: list[int]):
    def embedding_matrix(
        frame: pd.DataFrame, pooling: str, include_time_features: bool
    ) -> np.ndarray:
        embedding = np.vstack(
            frame[pooling].map(lambda value: np.asarray(value, dtype=np.float32)).to_numpy()
        )
        if not include_time_features:
            return embedding
        origin = pd.get_dummies(frame["origin_wave"].astype(int), prefix="origin", dtype=float)
        for wave in origin_waves:
            column = f"origin_{wave}"
            if column not in origin:
                origin[column] = 0.0
        origin = origin[[f"origin_{wave}" for wave in origin_waves]]
        interval = frame[["interval_years"]].to_numpy(dtype=np.float32)
        return np.hstack([embedding, origin.to_numpy(dtype=np.float32), interval])

    return embedding_matrix


def tree_features(manifest: dict[str, Any], outcome_id: str) -> list[str]:
    predictor_variables = manifest["per_outcome_predictors"][outcome_id]
    return [f"feature__{variable}" for variable in predictor_variables] + manifest.get(
        "time_features", ["origin_wave", "interval_years"]
    )


def fit_tree(
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


def predict_tree(frame: pd.DataFrame, model: Any, features: list[str], split: str) -> pd.DataFrame:
    out = frame[frame["split"].eq(split)].copy()
    out["tree_probability"] = BASELINE.predict_probability(model, out[features])
    return out


def fit_embedding(
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
    frame: pd.DataFrame, model: Any, pooling: str, include_time_features: bool, split: str
) -> pd.DataFrame:
    out = frame[frame["split"].eq(split)].copy()
    out["embedding_probability"] = model.predict_proba(
        COMPARE.embedding_matrix(out, pooling, include_time_features)
    )[:, 1]
    return out


def pair_predictions(
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
        raise RuntimeError("Tree and embedding predictions are not aligned")
    return paired


def make_fusion_model() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000, class_weight="balanced", solver="lbfgs", random_state=20260625
                ),
            ),
        ]
    )


def predict_fusion(model: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(
        frame[["tree_probability", "embedding_probability"]].to_numpy(dtype=np.float32)
    )[:, 1]


def origin_summary(frame: pd.DataFrame, probability_column: str) -> dict[str, Any]:
    auc_by_origin: dict[int, float] = {}
    pr_by_origin: dict[int, float] = {}
    brier_by_origin: dict[int, float] = {}
    for origin_wave, origin_frame in frame.groupby("origin_wave", sort=True):
        values = COMPARE.evaluate_predictions(
            origin_frame["event_next_wave"].astype(int),
            origin_frame[probability_column].to_numpy(dtype=float),
        )
        auc_by_origin[int(origin_wave)] = values["roc_auc"]
        pr_by_origin[int(origin_wave)] = values["pr_auc"]
        brier_by_origin[int(origin_wave)] = values["brier"]
    aucs = [value for value in auc_by_origin.values() if not pd.isna(value)]
    prs = [value for value in pr_by_origin.values() if not pd.isna(value)]
    briers = [value for value in brier_by_origin.values() if not pd.isna(value)]
    return {
        "origin_mean_auc": float(np.mean(aucs)) if aucs else math.nan,
        "origin_mean_pr_auc": float(np.mean(prs)) if prs else math.nan,
        "origin_mean_brier": float(np.mean(briers)) if briers else math.nan,
        "auc_by_origin": auc_by_origin,
    }


def metric_rows(
    outcome_id: str,
    method: str,
    model_name: str,
    frame: pd.DataFrame,
    probability_column: str,
    id_column: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    y_true = frame["event_next_wave"].astype(int)
    pooled = COMPARE.evaluate_predictions(y_true, frame[probability_column].to_numpy(dtype=float))
    rows.append(
        {
            "outcome_id": outcome_id,
            "method": method,
            "model": model_name,
            "split": "test",
            "metric_scope": "pooled",
            "origin_wave": "pooled",
            "n": int(len(frame)),
            "ids": int(frame[id_column].nunique()),
            "events": int(y_true.sum()),
            "roc_auc": pooled["roc_auc"],
            "pr_auc": pooled["pr_auc"],
            "brier": pooled["brier"],
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
                "split": "test",
                "metric_scope": "origin",
                "origin_wave": int(origin_wave),
                "n": int(len(origin_frame)),
                "ids": int(origin_frame[id_column].nunique()),
                "events": int(origin_y.sum()),
                "roc_auc": values["roc_auc"],
                "pr_auc": values["pr_auc"],
                "brier": values["brier"],
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
            "split": "test",
            "metric_scope": "survival",
            "origin_wave": "all",
            "n": int(survival["ids"]),
            "ids": int(survival["ids"]),
            "events": int(survival["events"]),
            "roc_auc": math.nan,
            "pr_auc": math.nan,
            "brier": math.nan,
            "c_index": survival["c_index"],
        }
    )
    return rows


def report_title(cohort_label: str) -> str:
    return f"{cohort_label} Selected Late Fusion"


def write_report(
    path: Path, decision: pd.DataFrame, cohort_label: str, config: dict[str, Any]
) -> None:
    lines = [
        f"# {report_title(cohort_label)}",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "- Fusion uses the tree and embedding candidates already selected by validation per-origin mean ROC-AUC.",
        "- Fusion is probability-level stacking: selected tree probability + selected embedding probability.",
        "- The fusion logistic regression is fit on validation predictions and evaluated once on test.",
        "",
        "| outcome_id | tree | embedding | fusion | fusion-tree | fusion-embedding | fusion-best | verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in decision.iterrows():
        lines.append(
            f"| {row['outcome_id']} | {fmt(row['tree_test_per_origin_mean_auc'])} | "
            f"{fmt(row['embedding_test_per_origin_mean_auc'])} | {fmt(row['fusion_test_per_origin_mean_auc'])} | "
            f"{fmt(row['delta_fusion_minus_tree'])} | {fmt(row['delta_fusion_minus_embedding'])} | "
            f"{fmt(row['delta_fusion_minus_best_single'])} | {row['late_fusion_verdict']} |"
        )
    lines.extend(
        [
            "",
            "## Inputs",
            "",
            f"- Supervised matrix: `{config['inputs']['supervised_outcome_matrix']}`",
            f"- Embedding run: `{config['outputs']['mlm_directory']}/runs/{config['comparison']['embedding_run_name']}`",
            f"- Embedding-vs-tree decision table: `{config['outputs']['embedding_comparison_decision_table']}`",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_decision_row(
    selected: pd.Series,
    tree_summary: dict[str, Any],
    embedding_summary: dict[str, Any],
    fusion_summary: dict[str, Any],
    tree_c: float,
    embedding_c: float,
    fusion_c: float,
    threshold: float,
) -> dict[str, Any]:
    tree_auc = tree_summary["origin_mean_auc"]
    embedding_auc = embedding_summary["origin_mean_auc"]
    fusion_auc = fusion_summary["origin_mean_auc"]
    best_single_auc = max(tree_auc, embedding_auc)
    best_single_method = "tree" if tree_auc >= embedding_auc else "embedding"
    delta_fusion_minus_tree = fusion_auc - tree_auc
    delta_fusion_minus_embedding = fusion_auc - embedding_auc
    delta_fusion_minus_best = fusion_auc - best_single_auc
    if delta_fusion_minus_best >= threshold:
        verdict = "fusion improves"
    elif delta_fusion_minus_tree >= threshold:
        verdict = "adds beyond tree"
    elif delta_fusion_minus_tree < 0:
        verdict = "below tree"
    else:
        verdict = "tie vs tree"
    row = {
        "outcome_id": selected["outcome_id"],
        "tree_model": selected["tree_model"],
        "embedding_model": selected["embedding_model"],
        "embedding_pooling": selected["embedding_pooling_selected_by_validation"],
        "embedding_mask_mode": selected["embedding_mask_mode"],
        "tree_test_per_origin_mean_auc": tree_auc,
        "embedding_test_per_origin_mean_auc": embedding_auc,
        "fusion_test_per_origin_mean_auc": fusion_auc,
        "delta_embedding_minus_tree": embedding_auc - tree_auc,
        "delta_fusion_minus_tree": delta_fusion_minus_tree,
        "delta_fusion_minus_embedding": delta_fusion_minus_embedding,
        "delta_fusion_minus_best_single": delta_fusion_minus_best,
        "best_single_method": best_single_method,
        "tree_test_c_index": tree_c,
        "embedding_test_c_index": embedding_c,
        "fusion_test_c_index": fusion_c,
        "delta_c_fusion_minus_tree": fusion_c - tree_c,
        "delta_c_fusion_minus_embedding": fusion_c - embedding_c,
        "late_fusion_verdict": verdict,
        "tree_auc_by_origin_json": json.dumps(tree_summary["auc_by_origin"], sort_keys=True),
        "embedding_auc_by_origin_json": json.dumps(
            embedding_summary["auc_by_origin"], sort_keys=True
        ),
        "fusion_auc_by_origin_json": json.dumps(fusion_summary["auc_by_origin"], sort_keys=True),
    }
    for wave, value in tree_summary["auc_by_origin"].items():
        row[f"tree_origin_{wave}_auc"] = value
    for wave, value in embedding_summary["auc_by_origin"].items():
        row[f"embedding_origin_{wave}_auc"] = value
    for wave, value in fusion_summary["auc_by_origin"].items():
        row[f"fusion_origin_{wave}_auc"] = value
    return row


def output_paths(config: dict[str, Any], cohort: str) -> tuple[Path, Path, Path]:
    out_dir = resolve_path(config["outputs"]["embedding_comparison_directory"])
    return (
        out_dir / f"{cohort}_late_fusion_metrics.csv",
        out_dir / f"{cohort}_late_fusion_decision_table.csv",
        out_dir / f"{cohort}_late_fusion_report.md",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--cohort-label")
    args = parser.parse_args()

    config = load_yaml(args.config)
    comparison_config = config["comparison"]
    supervised_config = {"models": {"random_state": int(config["training"]["seed"]), "n_jobs": 4}}
    supervised_config["models"].update(config.get("models", {}))
    manifest = load_json(resolve_path(config["inputs"]["supervised_matrix_manifest"]))
    id_column = manifest["metadata"]["id_column"]
    origin_waves = [int(wave) for wave in manifest["metadata"]["origin_waves"]]
    COMPARE.embedding_matrix = make_embedding_matrix(origin_waves)

    matrix = pd.read_parquet(resolve_path(config["inputs"]["supervised_outcome_matrix"]))
    matrix[id_column] = normalize_id(matrix[id_column])
    decision_in = pd.read_csv(
        resolve_path(config["outputs"]["embedding_comparison_decision_table"])
    )
    threshold = float(comparison_config.get("fm_delta_threshold", 0.01))
    include_time_features = bool(comparison_config.get("include_time_features", True))
    run_name = comparison_config["embedding_run_name"]

    rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    embedding_cache: dict[Path, pd.DataFrame] = {}
    id_reference = matrix[id_column]

    for _, selected in decision_in.iterrows():
        outcome_id = str(selected["outcome_id"])
        print(f"{args.cohort}: selected late fusion for {outcome_id}", flush=True)
        outcome_matrix = matrix[matrix["outcome_id"].eq(outcome_id)].copy().reset_index(drop=True)
        outcome_matrix["row_id"] = np.arange(len(outcome_matrix))
        features = tree_features(manifest, outcome_id)
        missing = sorted(set(features) - set(outcome_matrix.columns))
        if missing:
            raise RuntimeError(f"{outcome_id} missing tree features: {missing[:20]}")

        tree_frame = outcome_matrix[
            list(
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
                        *features,
                    ]
                )
            )
        ].copy()
        tree_model_name = str(selected["tree_model"])
        tree_model = fit_tree(tree_frame, features, tree_model_name, supervised_config, manifest)

        mask_mode = str(selected["embedding_mask_mode"])
        pooling = str(selected["embedding_pooling_selected_by_validation"])
        embedding_model_name = str(selected["embedding_model"])
        emb_path = embedding_path(config, run_name, outcome_id, mask_mode)
        if emb_path not in embedding_cache:
            embedding_cache[emb_path] = load_embedding_frame(emb_path, id_column, id_reference)
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
        if embedding_frame[pooling].isna().any():
            missing_embeddings = int(embedding_frame[pooling].isna().sum())
            raise RuntimeError(
                f"{outcome_id} has {missing_embeddings} rows without {pooling} embeddings"
            )
        embedding_model = fit_embedding(
            embedding_frame, embedding_model_name, pooling, include_time_features, comparison_config
        )

        validation_tree = predict_tree(tree_frame, tree_model, features, "validation")
        validation_embedding = predict_embedding(
            embedding_frame, embedding_model, pooling, include_time_features, "validation"
        )
        validation_paired = pair_predictions(validation_tree, validation_embedding, id_column)
        fusion_model = make_fusion_model()
        fusion_model.fit(
            validation_paired[["tree_probability", "embedding_probability"]].to_numpy(
                dtype=np.float32
            ),
            validation_paired["event_next_wave"].astype(int),
        )

        test_tree = predict_tree(tree_frame, tree_model, features, "test")
        test_embedding = predict_embedding(
            embedding_frame, embedding_model, pooling, include_time_features, "test"
        )
        test_paired = pair_predictions(test_tree, test_embedding, id_column)
        test_paired["fusion_probability"] = predict_fusion(fusion_model, test_paired)

        rows.extend(
            metric_rows(
                outcome_id, "tree", tree_model_name, test_paired, "tree_probability", id_column
            )
        )
        rows.extend(
            metric_rows(
                outcome_id,
                "embedding",
                embedding_model_name,
                test_paired,
                "embedding_probability",
                id_column,
            )
        )
        rows.extend(
            metric_rows(
                outcome_id,
                "tree+embedding",
                "selected_stacked_logistic_regression",
                test_paired,
                "fusion_probability",
                id_column,
            )
        )

        tree_summary = origin_summary(test_paired, "tree_probability")
        embedding_summary = origin_summary(test_paired, "embedding_probability")
        fusion_summary = origin_summary(test_paired, "fusion_probability")
        tree_c = COMPARE.first_origin_c_index(
            test_paired, test_paired["tree_probability"].to_numpy(dtype=float), id_column
        )["c_index"]
        embedding_c = COMPARE.first_origin_c_index(
            test_paired, test_paired["embedding_probability"].to_numpy(dtype=float), id_column
        )["c_index"]
        fusion_c = COMPARE.first_origin_c_index(
            test_paired, test_paired["fusion_probability"].to_numpy(dtype=float), id_column
        )["c_index"]
        decision_rows.append(
            build_decision_row(
                selected,
                tree_summary,
                embedding_summary,
                fusion_summary,
                tree_c,
                embedding_c,
                fusion_c,
                threshold,
            )
        )

    metrics = pd.DataFrame(rows)
    decision = pd.DataFrame(decision_rows)
    metrics_path, decision_path, report_path = output_paths(config, args.cohort)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(metrics_path, index=False)
    decision.to_csv(decision_path, index=False)
    write_report(report_path, decision, args.cohort_label or args.cohort.upper(), config)
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
