#!/usr/bin/env python3
"""Stability checks for the Elsa multimorbidity embedding-vs-tree delta."""

from __future__ import annotations

import importlib.util
import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_CONFIG = SCRIPT_PATH.parents[1] / "configs" / "elsa_mlm.yaml"


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


STABILITY = load_module(
    "elsa_stability_shared",
    PROJECT_ROOT / "charls_found" / "external_validation" / "hrs_multimorbidity_stability.py",
)
BASELINE = load_module(
    "elsa_stability_baseline",
    PROJECT_ROOT / "charls_found" / "external_validation" / "elsa_train_baselines.py",
)
ELSA_COMPARE = load_module(
    "elsa_stability_compare",
    PROJECT_ROOT / "charls_found" / "external_validation" / "elsa_embedding_vs_tree.py",
)


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
    bootstrap_summary = STABILITY.summarize(bootstrap_delta)
    repeated_summary = STABILITY.summarize(repeated_delta)
    lines = [
        "# Elsa Multimorbidity Stability Check",
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
    STABILITY.DEFAULT_CONFIG = DEFAULT_CONFIG
    STABILITY.BASELINE = BASELINE
    STABILITY.HRS_COMPARE = ELSA_COMPARE
    STABILITY.SHARED_COMPARE = ELSA_COMPARE.COMPARE
    STABILITY.write_report = write_report

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--bootstrap-iterations", type=int)
    parser.add_argument("--repeated-splits", type=int)
    args = parser.parse_args()

    config = STABILITY.load_yaml(args.config)
    stability_config = dict(config["stability"])
    if args.bootstrap_iterations is not None:
        stability_config["bootstrap_iterations"] = args.bootstrap_iterations
    if args.repeated_splits is not None:
        stability_config["repeated_splits"] = args.repeated_splits
    comparison_config = config["comparison"]
    outcome_id = stability_config["outcome_id"]
    manifest = STABILITY.load_json(
        STABILITY.resolve_path(config["inputs"]["supervised_matrix_manifest"])
    )
    id_column = manifest["metadata"]["id_column"]
    origin_waves = [int(wave) for wave in manifest["metadata"]["origin_waves"]]
    STABILITY.SHARED_COMPARE.embedding_matrix = ELSA_COMPARE.make_embedding_matrix(origin_waves)

    matrix = pd.read_parquet(STABILITY.resolve_path(config["inputs"]["supervised_outcome_matrix"]))
    matrix[id_column] = matrix[id_column].astype(str)
    outcome_matrix = matrix[matrix["outcome_id"].eq(outcome_id)].copy().reset_index(drop=True)
    outcome_matrix["row_id"] = np.arange(len(outcome_matrix))
    features = STABILITY.tree_features(manifest, outcome_id)
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
    embeddings = ELSA_COMPARE.load_embedding_frame(
        ELSA_COMPARE.embedding_path(
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
        raise RuntimeError("Missing diagnostic embeddings for ELSA multimorbidity")

    decision = pd.read_csv(
        STABILITY.resolve_path(config["outputs"]["embedding_comparison_decision_table"])
    )
    selected = decision[decision["outcome_id"].eq(outcome_id)].iloc[0]
    tree_model_name = str(selected["tree_model"])
    embedding_model_name = str(selected["embedding_model"])
    embedding_pooling = str(selected["embedding_pooling_selected_by_validation"])
    include_time_features = bool(comparison_config.get("include_time_features", True))

    tree_model = STABILITY.fit_tree_model(
        tree_base, features, tree_model_name, manifest, int(stability_config["seed"])
    )
    embedding_model = STABILITY.fit_embedding_model(
        embedding_base,
        embedding_model_name,
        embedding_pooling,
        include_time_features,
        comparison_config,
    )
    tree_test = STABILITY.predict_tree(tree_base, features, tree_model, "test", "tree_probability")
    embedding_test = STABILITY.predict_embedding(
        embedding_base,
        embedding_model,
        embedding_pooling,
        include_time_features,
        "test",
        "embedding_probability",
    )
    paired = STABILITY.paired_prediction_frame(tree_test, embedding_test, id_column)
    original = STABILITY.paired_delta(paired, id_column)
    original.update(
        {
            "tree_model": tree_model_name,
            "embedding_model": embedding_model_name,
            "embedding_pooling": embedding_pooling,
        }
    )

    bootstrap = STABILITY.id_cluster_bootstrap(
        paired,
        id_column,
        int(stability_config["bootstrap_iterations"]),
        int(stability_config["seed"]),
    )
    repeated = STABILITY.repeated_split_evaluation(
        tree_base,
        embedding_base,
        features,
        manifest,
        comparison_config,
        stability_config,
        id_column,
    )

    bootstrap_path = STABILITY.resolve_path(config["outputs"]["multimorbidity_stability_bootstrap"])
    repeated_path = STABILITY.resolve_path(
        config["outputs"]["multimorbidity_stability_repeated_splits"]
    )
    report_path = STABILITY.resolve_path(config["outputs"]["multimorbidity_stability_report"])
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
            STABILITY.percentile_interval(bootstrap["delta_auc"].to_numpy(dtype=float))
        ),
        "repeated_delta_auc_mean": (
            float(repeated["delta_auc"].mean()) if not repeated.empty else math.nan
        ),
        "repeated_delta_auc_ci": (
            list(STABILITY.percentile_interval(repeated["delta_auc"].to_numpy(dtype=float)))
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
    raise SystemExit(main())
