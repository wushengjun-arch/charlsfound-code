#!/usr/bin/env python3
"""Unified ID-cluster bootstrap CIs for core embedding-vs-tree deltas."""

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

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]

CORE_OUTCOMES = [
    "adl_new_difficulty",
    "iadl_new_difficulty",
    "incident_diabetes",
    "incident_hypertension",
    "multimorbidity_onset",
    "high_depressive_symptoms_onset",
    "low_cognition_onset_data_driven",
]


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


LATE = load_module(
    "phase4_external_late_fusion_helpers",
    PROJECT_ROOT / "charls_found" / "external_validation" / "late_fusion.py",
)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def percentile(values: np.ndarray, q: float) -> float:
    clean = values[~np.isnan(values)]
    if clean.size == 0:
        return math.nan
    return float(np.percentile(clean, q))


def finite_std(values: np.ndarray) -> float:
    clean = values[~np.isnan(values)]
    if clean.size < 2:
        return math.nan
    return float(np.std(clean, ddof=1))


def stability_label(delta: float, ci_low: float, ci_high: float, threshold: float) -> str:
    if pd.isna(delta) or pd.isna(ci_low) or pd.isna(ci_high):
        return "uncertain/tie"
    if delta >= threshold and ci_low > 0:
        return "stable FM win"
    if delta <= -threshold and ci_high < 0:
        return "stable tree stronger"
    return "uncertain/tie"


def origin_metrics(frame: pd.DataFrame, probability_column: str, id_column: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for origin_wave, origin_frame in frame.groupby("origin_wave", sort=True):
        y_true = origin_frame["event_next_wave"].astype(int)
        values = LATE.COMPARE.evaluate_predictions(
            y_true,
            origin_frame[probability_column].to_numpy(dtype=float),
        )
        rows.append(
            {
                "origin_wave": int(origin_wave),
                "n": int(len(origin_frame)),
                "ids": int(origin_frame[id_column].nunique()),
                "events": int(y_true.sum()),
                **values,
            }
        )
    origin_df = pd.DataFrame(rows)
    return {
        "origin_mean_auc": float(origin_df["roc_auc"].mean(skipna=True)),
        "origin_mean_pr_auc": float(origin_df["pr_auc"].mean(skipna=True)),
        "origin_mean_brier": float(origin_df["brier"].mean(skipna=True)),
        "origin_json": json.dumps(
            {
                int(row["origin_wave"]): float(row["roc_auc"])
                for _, row in origin_df.iterrows()
                if not pd.isna(row["roc_auc"])
            },
            sort_keys=True,
        ),
        "origin_detail": origin_df,
    }


def paired_scores(frame: pd.DataFrame, id_column: str) -> dict[str, float]:
    tree = origin_metrics(frame, "tree_probability", id_column)
    embedding = origin_metrics(frame, "embedding_probability", id_column)
    return {
        "tree_auc": tree["origin_mean_auc"],
        "embedding_auc": embedding["origin_mean_auc"],
        "delta_auc": embedding["origin_mean_auc"] - tree["origin_mean_auc"],
        "tree_pr_auc": tree["origin_mean_pr_auc"],
        "embedding_pr_auc": embedding["origin_mean_pr_auc"],
        "delta_pr_auc": embedding["origin_mean_pr_auc"] - tree["origin_mean_pr_auc"],
        "tree_brier": tree["origin_mean_brier"],
        "embedding_brier": embedding["origin_mean_brier"],
        "delta_brier": embedding["origin_mean_brier"] - tree["origin_mean_brier"],
    }


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
        sample = paired.loc[sampled_indices]
        values = paired_scores(sample, id_column)
        values.update(
            {
                "iteration": iteration,
                "sampled_ids": int(len(sampled_ids)),
                "sampled_rows": int(len(sample)),
            }
        )
        rows.append(values)
    return pd.DataFrame(rows)


def make_supervised_config(config: dict[str, Any]) -> dict[str, Any]:
    supervised_config = {"models": {"random_state": int(config["training"]["seed"]), "n_jobs": 4}}
    supervised_config["models"].update(config.get("models", {}))
    return supervised_config


def build_prediction_frame(
    config: dict[str, Any],
    manifest: dict[str, Any],
    matrix: pd.DataFrame,
    id_column: str,
    id_reference: pd.Series,
    selected: pd.Series,
    embedding_cache: dict[Path, pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    comparison_config = config["comparison"]
    include_time_features = bool(comparison_config.get("include_time_features", True))
    supervised_config = make_supervised_config(config)
    outcome_id = str(selected["outcome_id"])
    run_name = str(comparison_config["embedding_run_name"])

    outcome_matrix = matrix[matrix["outcome_id"].eq(outcome_id)].copy().reset_index(drop=True)
    outcome_matrix["row_id"] = np.arange(len(outcome_matrix), dtype=np.int64)
    features = LATE.tree_features(manifest, outcome_id)
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
    tree_model = LATE.fit_tree(tree_frame, features, tree_model_name, supervised_config, manifest)
    test_tree = LATE.predict_tree(tree_frame, tree_model, features, "test")

    mask_mode = str(selected["embedding_mask_mode"])
    pooling = str(selected["embedding_pooling_selected_by_validation"])
    embedding_model_name = str(selected["embedding_model"])
    emb_path = LATE.embedding_path(config, run_name, outcome_id, mask_mode)
    if emb_path not in embedding_cache:
        embedding_cache[emb_path] = LATE.load_embedding_frame(emb_path, id_column, id_reference)
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
    embedding_model = LATE.fit_embedding(
        embedding_frame,
        embedding_model_name,
        pooling,
        include_time_features,
        comparison_config,
    )
    test_embedding = LATE.predict_embedding(
        embedding_frame, embedding_model, pooling, include_time_features, "test"
    )
    paired = LATE.pair_predictions(test_tree, test_embedding, id_column)
    return paired, {
        "tree_model": tree_model_name,
        "embedding_model": embedding_model_name,
        "embedding_pooling": pooling,
        "embedding_mask_mode": mask_mode,
    }


def output_base(config: dict[str, Any], cohort: str) -> Path:
    out_dir = resolve_path(config["outputs"]["embedding_comparison_directory"])
    return out_dir / f"{cohort}_core_delta_bootstrap_ci.csv"


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def run(args: argparse.Namespace) -> int:
    config = LATE.load_yaml(args.config)
    manifest = LATE.load_json(resolve_path(config["inputs"]["supervised_matrix_manifest"]))
    id_column = str(manifest["metadata"]["id_column"])
    origin_waves = [int(wave) for wave in manifest["metadata"]["origin_waves"]]
    LATE.COMPARE.embedding_matrix = LATE.make_embedding_matrix(origin_waves)

    matrix = pd.read_parquet(resolve_path(config["inputs"]["supervised_outcome_matrix"]))
    matrix[id_column] = LATE.normalize_id(matrix[id_column])
    id_reference = matrix[id_column]

    decision = pd.read_csv(resolve_path(config["outputs"]["embedding_comparison_decision_table"]))
    requested_outcomes = args.outcomes if args.outcomes else CORE_OUTCOMES
    decision = decision[decision["outcome_id"].isin(requested_outcomes)].copy()
    missing_outcomes = sorted(set(requested_outcomes) - set(decision["outcome_id"]))
    if missing_outcomes:
        print(
            f"[{args.cohort}] skipping unavailable outcomes: {', '.join(missing_outcomes)}",
            flush=True,
        )
    if decision.empty:
        raise RuntimeError(
            f"No requested outcomes are available in {config['outputs']['embedding_comparison_decision_table']}"
        )

    threshold = float(config["comparison"].get("fm_delta_threshold", 0.01))
    output = args.output or output_base(config, args.cohort)
    origin_output = args.origin_output
    predictions_dir = args.predictions_dir
    if predictions_dir:
        predictions_dir.mkdir(parents=True, exist_ok=True)

    embedding_cache: dict[Path, pd.DataFrame] = {}
    rows: list[dict[str, Any]] = []
    origin_rows: list[dict[str, Any]] = []

    for outcome_index, selected in enumerate(
        decision.sort_values("outcome_id").itertuples(index=False), start=1
    ):
        selected_series = pd.Series(selected._asdict())
        outcome_id = str(selected_series["outcome_id"])
        print(
            f"[{args.cohort}] {outcome_index}/{len(decision)} bootstrap {outcome_id}: "
            f"{selected_series['tree_model']} vs {selected_series['embedding_model']}/"
            f"{selected_series['embedding_pooling_selected_by_validation']} "
            f"({selected_series['embedding_mask_mode']})",
            flush=True,
        )
        paired, model_info = build_prediction_frame(
            config=config,
            manifest=manifest,
            matrix=matrix,
            id_column=id_column,
            id_reference=id_reference,
            selected=selected_series,
            embedding_cache=embedding_cache,
        )
        if predictions_dir:
            paired.to_parquet(
                predictions_dir / f"{args.cohort}_{outcome_id}_selected_test_predictions.parquet",
                index=False,
            )

        tree_origin = origin_metrics(paired, "tree_probability", id_column)
        embedding_origin = origin_metrics(paired, "embedding_probability", id_column)
        origin_detail = tree_origin["origin_detail"].merge(
            embedding_origin["origin_detail"],
            on=["origin_wave", "n", "ids", "events"],
            suffixes=("_tree", "_embedding"),
        )
        origin_detail["delta_auc"] = (
            origin_detail["roc_auc_embedding"] - origin_detail["roc_auc_tree"]
        )
        origin_detail["delta_pr_auc"] = (
            origin_detail["pr_auc_embedding"] - origin_detail["pr_auc_tree"]
        )
        origin_detail["delta_brier"] = (
            origin_detail["brier_embedding"] - origin_detail["brier_tree"]
        )
        origin_detail.insert(0, "outcome_id", outcome_id)
        origin_detail.insert(0, "cohort", args.cohort)
        origin_rows.extend(origin_detail.to_dict(orient="records"))

        observed = paired_scores(paired, id_column)
        bootstrap = id_cluster_bootstrap(
            paired,
            id_column=id_column,
            iterations=int(args.bootstrap_iterations),
            seed=int(args.seed) + outcome_index * 1009,
        )
        delta_values = bootstrap["delta_auc"].to_numpy(dtype=float)
        pr_values = bootstrap["delta_pr_auc"].to_numpy(dtype=float)
        brier_values = bootstrap["delta_brier"].to_numpy(dtype=float)
        origin_win_count = int((origin_detail["delta_auc"] > 0).sum())
        origin_count = int(len(origin_detail))
        row = {
            "cohort": args.cohort,
            "outcome_id": outcome_id,
            **model_info,
            "test_rows": int(len(paired)),
            "test_ids": int(paired[id_column].nunique()),
            "test_events": int(paired["event_next_wave"].astype(int).sum()),
            "origin_count": origin_count,
            "embedding_origin_win_count": origin_win_count,
            "embedding_origin_win_share": (
                origin_win_count / origin_count if origin_count else math.nan
            ),
            "tree_test_per_origin_mean_auc": observed["tree_auc"],
            "embedding_test_per_origin_mean_auc": observed["embedding_auc"],
            "delta_embedding_minus_tree": observed["delta_auc"],
            "delta_auc_bootstrap_mean": float(np.nanmean(delta_values)),
            "delta_auc_bootstrap_median": float(np.nanmedian(delta_values)),
            "delta_auc_bootstrap_sd": finite_std(delta_values),
            "delta_auc_ci_low": percentile(delta_values, 2.5),
            "delta_auc_ci_high": percentile(delta_values, 97.5),
            "p_delta_gt_0": float(np.nanmean(delta_values > 0)),
            "p_delta_gt_threshold": float(np.nanmean(delta_values > threshold)),
            "p_delta_lt_0": float(np.nanmean(delta_values < 0)),
            "p_delta_lt_minus_threshold": float(np.nanmean(delta_values < -threshold)),
            "delta_pr_auc": observed["delta_pr_auc"],
            "delta_pr_auc_bootstrap_sd": finite_std(pr_values),
            "delta_pr_auc_ci_low": percentile(pr_values, 2.5),
            "delta_pr_auc_ci_high": percentile(pr_values, 97.5),
            "delta_brier": observed["delta_brier"],
            "delta_brier_bootstrap_sd": finite_std(brier_values),
            "delta_brier_ci_low": percentile(brier_values, 2.5),
            "delta_brier_ci_high": percentile(brier_values, 97.5),
            "tree_auc_by_origin_json": tree_origin["origin_json"],
            "embedding_auc_by_origin_json": embedding_origin["origin_json"],
        }
        row["bootstrap_stability_label"] = stability_label(
            row["delta_embedding_minus_tree"],
            row["delta_auc_ci_low"],
            row["delta_auc_ci_high"],
            threshold,
        )
        rows.append(row)

    result = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    if origin_output:
        origin_output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(origin_rows).to_csv(origin_output, index=False)
    print(
        json.dumps(
            {
                "cohort": args.cohort,
                "outcomes": int(len(result)),
                "bootstrap_iterations": int(args.bootstrap_iterations),
                "output": display_path(output),
                "stable_fm": int(result["bootstrap_stability_label"].eq("stable FM win").sum()),
                "stable_tree": int(
                    result["bootstrap_stability_label"].eq("stable tree stronger").sum()
                ),
                "elapsed_seconds": round(time.perf_counter() - args.started_at, 1),
            },
            indent=2,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--origin-output", type=Path)
    parser.add_argument("--predictions-dir", type=Path)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--outcomes", nargs="+")
    args = parser.parse_args()
    args.started_at = time.perf_counter()
    return run(args)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    raise SystemExit(main())
