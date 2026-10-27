#!/usr/bin/env python3
"""Stability checks for the HRS multimorbidity embedding-vs-tree delta."""

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
from sklearn.model_selection import GroupShuffleSplit

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_CONFIG = SCRIPT_PATH.parents[1] / "configs" / "hrs_mlm.yaml"


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BASELINE = load_module(
    "hrs_stability_baseline",
    PROJECT_ROOT / "charls_found" / "external_validation" / "hrs_train_baselines.py",
)
HRS_COMPARE = load_module(
    "hrs_stability_compare",
    PROJECT_ROOT / "charls_found" / "external_validation" / "hrs_embedding_vs_tree.py",
)
SHARED_COMPARE = HRS_COMPARE.COMPARE


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


def percentile_interval(
    values: np.ndarray, low: float = 2.5, high: float = 97.5
) -> tuple[float, float]:
    clean = values[~np.isnan(values)]
    if clean.size == 0:
        return math.nan, math.nan
    return float(np.percentile(clean, low)), float(np.percentile(clean, high))


def summarize(values: np.ndarray) -> dict[str, float]:
    clean = values[~np.isnan(values)]
    low, high = percentile_interval(clean)
    if clean.size == 0:
        return {"mean": math.nan, "median": math.nan, "ci_low": math.nan, "ci_high": math.nan}
    return {
        "mean": float(np.mean(clean)),
        "median": float(np.median(clean)),
        "ci_low": low,
        "ci_high": high,
    }


def origin_mean_auc(frame: pd.DataFrame, probability_column: str) -> float:
    aucs: list[float] = []
    for _, origin_frame in frame.groupby("origin_wave", sort=True):
        values = SHARED_COMPARE.evaluate_predictions(
            origin_frame["event_next_wave"].astype(int),
            origin_frame[probability_column].to_numpy(dtype=float),
        )
        if not pd.isna(values["roc_auc"]):
            aucs.append(float(values["roc_auc"]))
    return float(np.mean(aucs)) if aucs else math.nan


def assign_group_split(
    frame: pd.DataFrame,
    id_column: str,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
) -> pd.Series:
    if not math.isclose(train_fraction + validation_fraction + test_fraction, 1.0):
        raise ValueError("train/validation/test fractions must sum to 1")
    ids = pd.Series(frame[id_column].drop_duplicates().to_numpy())
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    train_validation_index, test_index = next(splitter.split(ids.to_frame(), groups=ids))
    train_validation_ids = ids.iloc[train_validation_index].reset_index(drop=True)
    test_ids = set(ids.iloc[test_index].tolist())
    validation_relative = validation_fraction / (train_fraction + validation_fraction)
    validation_splitter = GroupShuffleSplit(
        n_splits=1, test_size=validation_relative, random_state=seed + 100_000
    )
    train_index, validation_index = next(
        validation_splitter.split(train_validation_ids.to_frame(), groups=train_validation_ids)
    )
    train_ids = set(train_validation_ids.iloc[train_index].tolist())
    validation_ids = set(train_validation_ids.iloc[validation_index].tolist())
    split = np.full(len(frame), "train", dtype=object)
    id_values = frame[id_column].to_numpy()
    split[np.isin(id_values, list(validation_ids))] = "validation"
    split[np.isin(id_values, list(test_ids))] = "test"
    split[np.isin(id_values, list(train_ids))] = "train"
    return pd.Series(split, index=frame.index, name="split")


def tree_features(manifest: dict[str, Any], outcome_id: str) -> list[str]:
    predictor_variables = manifest["per_outcome_predictors"][outcome_id]
    return [f"feature__{variable}" for variable in predictor_variables] + manifest.get(
        "time_features", ["origin_wave", "interval_years"]
    )


def fit_tree_model(
    frame: pd.DataFrame,
    features: list[str],
    model_name: str,
    manifest: dict[str, Any],
    random_state: int,
) -> Any:
    train = frame[frame["split"].eq("train")]
    if train["event_next_wave"].nunique() < 2:
        raise RuntimeError(f"{model_name} train split has one class")
    model = BASELINE.make_model(
        model_name,
        train["event_next_wave"].astype(int),
        features,
        manifest["feature_type_by_variable"],
        random_state=random_state,
        n_jobs=4,
        estimators=250,
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
    model = SHARED_COMPARE.make_embedding_probe(model_name, comparison_config)
    SHARED_COMPARE.fit_embedding_probe(
        model,
        model_name,
        SHARED_COMPARE.embedding_matrix(train, pooling, include_time_features),
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
        SHARED_COMPARE.embedding_matrix(split_frame, pooling, include_time_features)
    )[:, 1]
    return split_frame


def select_tree_candidate(
    frame: pd.DataFrame,
    features: list[str],
    tree_models: list[str],
    manifest: dict[str, Any],
    random_state: int,
) -> tuple[str, Any, float]:
    candidates: list[tuple[float, str, Any]] = []
    for model_name in tree_models:
        model = fit_tree_model(frame, features, model_name, manifest, random_state)
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


def paired_prediction_frame(
    tree_frame: pd.DataFrame, embedding_frame: pd.DataFrame, id_column: str
) -> pd.DataFrame:
    columns = [
        id_column,
        "row_id",
        "origin_wave",
        "next_wave",
        "origin_year",
        "next_year",
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


def paired_delta(
    paired: pd.DataFrame, id_column: str, include_c_index: bool = True
) -> dict[str, float]:
    tree_auc = origin_mean_auc(paired, "tree_probability")
    embedding_auc = origin_mean_auc(paired, "embedding_probability")
    values = {
        "tree_auc": tree_auc,
        "embedding_auc": embedding_auc,
        "delta_auc": embedding_auc - tree_auc,
    }
    if include_c_index:
        tree_c = SHARED_COMPARE.first_origin_c_index(
            paired, paired["tree_probability"].to_numpy(dtype=float), id_column
        )["c_index"]
        embedding_c = SHARED_COMPARE.first_origin_c_index(
            paired, paired["embedding_probability"].to_numpy(dtype=float), id_column
        )["c_index"]
        values.update(
            {
                "tree_c_index": tree_c,
                "embedding_c_index": embedding_c,
                "delta_c_index": embedding_c - tree_c,
            }
        )
    return values


def id_cluster_bootstrap(
    paired: pd.DataFrame, id_column: str, iterations: int, seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ids = paired[id_column].drop_duplicates().to_numpy()
    indices_by_id = {
        id_value: group.index.to_numpy()
        for id_value, group in paired.groupby(id_column, sort=False)
    }
    rows: list[dict[str, Any]] = []
    for iteration in range(iterations):
        sampled_ids = rng.choice(ids, size=len(ids), replace=True)
        sampled_indices = np.concatenate([indices_by_id[id_value] for id_value in sampled_ids])
        sample = paired.loc[sampled_indices].copy()
        values = paired_delta(sample, id_column, include_c_index=False)
        values.update(
            {
                "iteration": iteration,
                "sampled_ids": int(len(sampled_ids)),
                "sampled_rows": int(len(sample)),
            }
        )
        rows.append(values)
    return pd.DataFrame(rows)


def repeated_split_evaluation(
    tree_base: pd.DataFrame,
    embedding_base: pd.DataFrame,
    features: list[str],
    manifest: dict[str, Any],
    comparison_config: dict[str, Any],
    stability_config: dict[str, Any],
    id_column: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    seed = int(stability_config["seed"])
    tree_models = list(comparison_config["baseline_tree_models"])
    embedding_models = list(comparison_config.get("embedding_models", ["logistic_regression"]))
    poolings = list(comparison_config["embedding_poolings"])
    include_time_features = bool(comparison_config.get("include_time_features", True))
    for repeat in range(int(stability_config["repeated_splits"])):
        split_seed = seed + repeat
        split = assign_group_split(
            tree_base,
            id_column,
            split_seed,
            float(stability_config["train_fraction"]),
            float(stability_config["validation_fraction"]),
            float(stability_config["test_fraction"]),
        )
        tree_frame = tree_base.copy()
        embedding_frame = embedding_base.copy()
        tree_frame["split"] = split
        embedding_frame["split"] = split
        if tree_frame.groupby("split")["event_next_wave"].nunique().min() < 2:
            print(f"Skipping repeat {repeat}: one split has one class", flush=True)
            continue
        tree_model_name, tree_model, tree_validation_auc = select_tree_candidate(
            tree_frame,
            features,
            tree_models,
            manifest,
            split_seed,
        )
        embedding_model_name, pooling, embedding_model, embedding_validation_auc = (
            select_embedding_candidate(
                embedding_frame,
                embedding_models,
                poolings,
                include_time_features,
                comparison_config,
            )
        )
        tree_test = predict_tree(tree_frame, features, tree_model, "test", "tree_probability")
        embedding_test = predict_embedding(
            embedding_frame,
            embedding_model,
            pooling,
            include_time_features,
            "test",
            "embedding_probability",
        )
        paired = paired_prediction_frame(tree_test, embedding_test, id_column)
        values = paired_delta(paired, id_column)
        values.update(
            {
                "repeat": repeat,
                "seed": split_seed,
                "selected_tree_model": tree_model_name,
                "tree_validation_origin_mean_auc": tree_validation_auc,
                "selected_embedding_model": embedding_model_name,
                "selected_embedding_pooling": pooling,
                "embedding_validation_origin_mean_auc": embedding_validation_auc,
                "test_rows": int(len(paired)),
                "test_ids": int(paired[id_column].nunique()),
                "test_events": int(paired["event_next_wave"].sum()),
            }
        )
        rows.append(values)
        print(
            f"repeat {repeat}: tree={tree_model_name} embedding={embedding_model_name}/{pooling} "
            f"delta_auc={values['delta_auc']:.4f}",
            flush=True,
        )
    return pd.DataFrame(rows)


def write_report(
    path: Path,
    original: dict[str, Any],
    bootstrap: pd.DataFrame,
    repeated: pd.DataFrame,
    threshold: float,
) -> None:
    bootstrap_delta = bootstrap["delta_auc"].to_numpy(dtype=float)
    repeated_delta = (
        repeated["delta_auc"].to_numpy(dtype=float)
        if not repeated.empty
        else np.asarray([], dtype=float)
    )
    bootstrap_summary = summarize(bootstrap_delta)
    repeated_summary = summarize(repeated_delta)
    lines = [
        "# HRS Multimorbidity Stability Check",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Original Split",
        "",
        f"- Tree: `{original['tree_model']}`",
        f"- Embedding: `{original['embedding_model']}` / `{original['embedding_pooling']}`",
        f"- Test per-origin AUC delta: {original['delta_auc']:.4f}",
        f"- Test C-index delta: {original['delta_c_index']:.4f}",
        "",
        "## ID-Cluster Bootstrap On Original Test Split",
        "",
        f"- Iterations: {len(bootstrap)}",
        f"- Mean delta AUC: {bootstrap_summary['mean']:.4f}",
        f"- Median delta AUC: {bootstrap_summary['median']:.4f}",
        f"- 95% percentile interval: [{bootstrap_summary['ci_low']:.4f}, {bootstrap_summary['ci_high']:.4f}]",
        f"- P(delta > 0): {float(np.mean(bootstrap_delta > 0)):.3f}",
        f"- P(delta > {threshold:.3f}): {float(np.mean(bootstrap_delta > threshold)):.3f}",
        "",
        "## Repeated ID-Level Splits",
        "",
        f"- Completed repeats: {len(repeated)}",
        f"- Mean delta AUC: {repeated_summary['mean']:.4f}",
        f"- Median delta AUC: {repeated_summary['median']:.4f}",
        f"- 95% empirical interval: [{repeated_summary['ci_low']:.4f}, {repeated_summary['ci_high']:.4f}]",
        f"- Share delta > 0: {float(np.mean(repeated_delta > 0)) if repeated_delta.size else math.nan:.3f}",
        f"- Share delta > {threshold:.3f}: {float(np.mean(repeated_delta > threshold)) if repeated_delta.size else math.nan:.3f}",
        "",
    ]
    if not repeated.empty:
        lines.extend(
            [
                "## Selected Models Across Repeats",
                "",
                "| selected_tree_model | count |",
                "|---|---:|",
            ]
        )
        for model_name, count in (
            repeated["selected_tree_model"].value_counts().sort_index().items()
        ):
            lines.append(f"| {model_name} | {int(count)} |")
        lines.extend(["", "| selected_embedding | count |", "|---|---:|"])
        selected_embedding = (
            repeated["selected_embedding_model"] + "/" + repeated["selected_embedding_pooling"]
        )
        for model_name, count in selected_embedding.value_counts().sort_index().items():
            lines.append(f"| {model_name} | {int(count)} |")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--bootstrap-iterations", type=int)
    parser.add_argument("--repeated-splits", type=int)
    args = parser.parse_args()

    config = load_yaml(args.config)
    stability_config = dict(config["stability"])
    if args.bootstrap_iterations is not None:
        stability_config["bootstrap_iterations"] = args.bootstrap_iterations
    if args.repeated_splits is not None:
        stability_config["repeated_splits"] = args.repeated_splits
    comparison_config = config["comparison"]
    outcome_id = stability_config["outcome_id"]
    manifest = load_json(resolve_path(config["inputs"]["supervised_matrix_manifest"]))
    id_column = manifest["metadata"]["id_column"]
    origin_waves = [int(wave) for wave in manifest["metadata"]["origin_waves"]]
    SHARED_COMPARE.embedding_matrix = HRS_COMPARE.make_embedding_matrix(origin_waves)

    matrix = pd.read_parquet(resolve_path(config["inputs"]["supervised_outcome_matrix"]))
    matrix[id_column] = pd.to_numeric(matrix[id_column], errors="raise").round().astype("int64")
    outcome_matrix = matrix[matrix["outcome_id"].eq(outcome_id)].copy().reset_index(drop=True)
    outcome_matrix["row_id"] = np.arange(len(outcome_matrix))
    features = tree_features(manifest, outcome_id)
    missing_features = sorted(set(features) - set(outcome_matrix.columns))
    if missing_features:
        raise RuntimeError(f"Missing tree features: {missing_features[:20]}")
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
    tree_base = outcome_matrix[tree_columns].copy()

    run_name = comparison_config["embedding_run_name"]
    embeddings = HRS_COMPARE.load_embedding_frame(
        HRS_COMPARE.embedding_path(
            config, run_name, outcome_id, comparison_config["disease_mask_mode"]
        ),
        id_column,
    )
    embedding_base = outcome_matrix[
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
        embeddings,
        on=[id_column, "origin_wave"],
        how="left",
        suffixes=("", "_embedding"),
        validate="many_to_one",
    )
    if embedding_base["mean_embedding"].isna().any():
        raise RuntimeError("Missing diagnostic embeddings for HRS multimorbidity")

    decision = pd.read_csv(resolve_path(config["outputs"]["embedding_comparison_decision_table"]))
    selected = decision[decision["outcome_id"].eq(outcome_id)].iloc[0]
    tree_model_name = str(selected["tree_model"])
    embedding_model_name = str(selected["embedding_model"])
    embedding_pooling = str(selected["embedding_pooling_selected_by_validation"])
    include_time_features = bool(comparison_config.get("include_time_features", True))

    tree_model = fit_tree_model(
        tree_base, features, tree_model_name, manifest, int(stability_config["seed"])
    )
    embedding_model = fit_embedding_model(
        embedding_base,
        embedding_model_name,
        embedding_pooling,
        include_time_features,
        comparison_config,
    )
    tree_test = predict_tree(tree_base, features, tree_model, "test", "tree_probability")
    embedding_test = predict_embedding(
        embedding_base,
        embedding_model,
        embedding_pooling,
        include_time_features,
        "test",
        "embedding_probability",
    )
    paired = paired_prediction_frame(tree_test, embedding_test, id_column)
    original = paired_delta(paired, id_column)
    original.update(
        {
            "tree_model": tree_model_name,
            "embedding_model": embedding_model_name,
            "embedding_pooling": embedding_pooling,
        }
    )

    bootstrap = id_cluster_bootstrap(
        paired,
        id_column,
        int(stability_config["bootstrap_iterations"]),
        int(stability_config["seed"]),
    )
    repeated = repeated_split_evaluation(
        tree_base,
        embedding_base,
        features,
        manifest,
        comparison_config,
        stability_config,
        id_column,
    )

    bootstrap_path = resolve_path(config["outputs"]["multimorbidity_stability_bootstrap"])
    repeated_path = resolve_path(config["outputs"]["multimorbidity_stability_repeated_splits"])
    report_path = resolve_path(config["outputs"]["multimorbidity_stability_report"])
    bootstrap_path.parent.mkdir(parents=True, exist_ok=True)
    bootstrap.to_csv(bootstrap_path, index=False)
    repeated.to_csv(repeated_path, index=False)
    write_report(
        report_path,
        original,
        bootstrap,
        repeated,
        float(comparison_config.get("fm_delta_threshold", 0.01)),
    )

    summary = {
        "original_delta_auc": float(original["delta_auc"]),
        "bootstrap_delta_auc_mean": float(bootstrap["delta_auc"].mean()),
        "bootstrap_delta_auc_ci": list(
            percentile_interval(bootstrap["delta_auc"].to_numpy(dtype=float))
        ),
        "repeated_delta_auc_mean": (
            float(repeated["delta_auc"].mean()) if not repeated.empty else math.nan
        ),
        "repeated_delta_auc_ci": (
            list(percentile_interval(repeated["delta_auc"].to_numpy(dtype=float)))
            if not repeated.empty
            else [math.nan, math.nan]
        ),
        "bootstrap": str(bootstrap_path.relative_to(PROJECT_ROOT)),
        "repeated_splits": str(repeated_path.relative_to(PROJECT_ROOT)),
        "report": str(report_path.relative_to(PROJECT_ROOT)),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    raise SystemExit(main())
