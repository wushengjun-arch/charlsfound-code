#!/usr/bin/env python3
"""Compare ELSA-native encoder embeddings against ELSA rolling tree baselines."""

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


COMPARE = load_module(
    "elsa_compare_shared", PROJECT_ROOT / "charls_found" / "mlm" / "compare_embedding_probe.py"
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


def embedding_path(config: dict[str, Any], run_name: str, outcome_id: str, mask_mode: str) -> Path:
    run_dir = resolve_path(config["outputs"]["mlm_directory"]) / "runs" / run_name
    if mask_mode == "none":
        return run_dir / "person_wave_embeddings_none.parquet"
    return run_dir / f"person_wave_embeddings_{mask_mode}_{outcome_id}.parquet"


def load_embedding_frame(path: Path, id_column: str) -> pd.DataFrame:
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
    frame[id_column] = frame[id_column].astype(str)
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


def origin_auc_summary(
    metrics: pd.DataFrame,
    outcome_id: str,
    method: str,
    model: str,
    split: str,
    pooling: str = "",
) -> dict[str, Any]:
    subset = metrics[
        metrics["outcome_id"].eq(outcome_id)
        & metrics["method"].eq(method)
        & metrics["model"].eq(model)
        & metrics["split"].eq(split)
        & metrics["metric_scope"].eq("origin")
    ].copy()
    if pooling:
        subset = subset[subset["embedding_pooling"].eq(pooling)]
    auc_by_wave = {int(row["origin_wave"]): float(row["roc_auc"]) for _, row in subset.iterrows()}
    values = [value for value in auc_by_wave.values() if not pd.isna(value)]
    return {
        "origin_mean_auc": float(np.mean(values)) if values else math.nan,
        "auc_by_origin": auc_by_wave,
    }


def baseline_tree_rows(
    baseline_metrics: pd.DataFrame, selected_outcomes: list[str], tree_models: list[str]
) -> tuple[pd.DataFrame, dict[str, str]]:
    rows: list[pd.DataFrame] = []
    best_model_by_outcome: dict[str, str] = {}
    for outcome_id in selected_outcomes:
        validation = baseline_metrics[
            baseline_metrics["outcome_id"].eq(outcome_id)
            & baseline_metrics["split"].eq("validation")
            & baseline_metrics["model"].isin(tree_models)
        ].copy()
        if validation.empty:
            raise RuntimeError(f"No Elsa tree baseline metrics found for {outcome_id}")
        origin_mean = validation[validation["metric_scope"].eq("origin_mean")]
        if origin_mean.empty:
            origin_mean = (
                validation[validation["metric_scope"].eq("origin")]
                .groupby(["outcome_id", "model"], as_index=False)
                .agg(roc_auc=("roc_auc", "mean"), pr_auc=("pr_auc", "mean"))
            )
        model = str(
            origin_mean.sort_values(["roc_auc", "pr_auc"], ascending=False).iloc[0]["model"]
        )
        best_model_by_outcome[outcome_id] = model
        rows.append(
            baseline_metrics[
                baseline_metrics["outcome_id"].eq(outcome_id)
                & baseline_metrics["model"].eq(model)
                & baseline_metrics["split"].isin(["validation", "test"])
            ].copy()
        )
    baseline = pd.concat(rows, ignore_index=True)
    baseline["method"] = "tree"
    baseline["mask_mode"] = ""
    baseline["embedding_pooling"] = ""
    return baseline, best_model_by_outcome


def choose_embedding_probe(
    metrics: pd.DataFrame, outcome_id: str, models: list[str], poolings: list[str]
) -> tuple[str, str]:
    candidates = []
    for model_name in models:
        for pooling in poolings:
            summary = origin_auc_summary(
                metrics, outcome_id, "embedding", model_name, "validation", pooling
            )
            candidates.append((summary["origin_mean_auc"], model_name, pooling))
    candidates = [candidate for candidate in candidates if not pd.isna(candidate[0])]
    if not candidates:
        raise RuntimeError(f"No validation per-origin AUC for {outcome_id}")
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1], candidates[0][2]


def build_decision_table(
    all_metrics: pd.DataFrame,
    selected_outcomes: list[str],
    tree_model_by_outcome: dict[str, str],
    selected_embedding: dict[str, tuple[str, str]],
    outcome_mask_mode: dict[str, str],
    delta_threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for outcome_id in selected_outcomes:
        tree_model = tree_model_by_outcome[outcome_id]
        embedding_model, pooling = selected_embedding[outcome_id]
        tree_origin = origin_auc_summary(all_metrics, outcome_id, "tree", tree_model, "test")
        emb_origin = origin_auc_summary(
            all_metrics, outcome_id, "embedding", embedding_model, "test", pooling
        )
        tree_c = all_metrics[
            all_metrics["outcome_id"].eq(outcome_id)
            & all_metrics["method"].eq("tree")
            & all_metrics["model"].eq(tree_model)
            & all_metrics["split"].eq("test")
            & all_metrics["metric_scope"].eq("survival")
        ]["c_index"].iloc[0]
        emb_c = all_metrics[
            all_metrics["outcome_id"].eq(outcome_id)
            & all_metrics["method"].eq("embedding")
            & all_metrics["model"].eq(embedding_model)
            & all_metrics["embedding_pooling"].eq(pooling)
            & all_metrics["split"].eq("test")
            & all_metrics["metric_scope"].eq("survival")
        ]["c_index"].iloc[0]
        tree_pooled = all_metrics[
            all_metrics["outcome_id"].eq(outcome_id)
            & all_metrics["method"].eq("tree")
            & all_metrics["model"].eq(tree_model)
            & all_metrics["split"].eq("test")
            & all_metrics["metric_scope"].eq("pooled")
        ]["roc_auc"].iloc[0]
        emb_pooled = all_metrics[
            all_metrics["outcome_id"].eq(outcome_id)
            & all_metrics["method"].eq("embedding")
            & all_metrics["model"].eq(embedding_model)
            & all_metrics["embedding_pooling"].eq(pooling)
            & all_metrics["split"].eq("test")
            & all_metrics["metric_scope"].eq("pooled")
        ]["roc_auc"].iloc[0]
        delta = emb_origin["origin_mean_auc"] - tree_origin["origin_mean_auc"]
        if delta >= delta_threshold:
            verdict = "FM improves"
        elif delta <= -delta_threshold:
            verdict = "tree stronger"
        else:
            verdict = "tie"
        row = {
            "outcome_id": outcome_id,
            "tree_model": tree_model,
            "embedding_model": embedding_model,
            "embedding_pooling_selected_by_validation": pooling,
            "embedding_mask_mode": outcome_mask_mode[outcome_id],
            "tree_test_per_origin_mean_auc": tree_origin["origin_mean_auc"],
            "embedding_test_per_origin_mean_auc": emb_origin["origin_mean_auc"],
            "delta_embedding_minus_tree": delta,
            "tree_test_c_index": tree_c,
            "embedding_test_c_index": emb_c,
            "delta_c_index": emb_c - tree_c,
            "tree_test_pooled_auc_context_only": tree_pooled,
            "embedding_test_pooled_auc_context_only": emb_pooled,
            "fm_adds_value": verdict,
            "tree_auc_by_origin_json": json.dumps(tree_origin["auc_by_origin"], sort_keys=True),
            "embedding_auc_by_origin_json": json.dumps(emb_origin["auc_by_origin"], sort_keys=True),
        }
        for wave, value in tree_origin["auc_by_origin"].items():
            row[f"tree_origin_{wave}_auc"] = value
        for wave, value in emb_origin["auc_by_origin"].items():
            row[f"embedding_origin_{wave}_auc"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(path: Path, decision: pd.DataFrame, config: dict[str, Any], run_name: str) -> None:
    lines = [
        "# Elsa Embedding vs Tree Comparison",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"- Embedding run: `{run_name}`",
        "- Main decision metric: test per-origin mean ROC-AUC.",
        "- Pooled AUC is context only because wave/base-rate differences can inflate it.",
        "- Tree model, embedding head, and pooling are selected on validation per-origin mean ROC-AUC, then evaluated once on test.",
        "- Chronic disease and multimorbidity outcomes use Elsa diagnostic-masked embeddings; other outcomes use unmasked embeddings.",
        "",
        "## Decision Table",
        "",
        "| outcome_id | mask | head | pooling | tree_auc | embedding_auc | delta | tree_c | embedding_c | verdict |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in decision.iterrows():
        lines.append(
            f"| {row['outcome_id']} | {row['embedding_mask_mode']} | {row['embedding_model']} | "
            f"{row['embedding_pooling_selected_by_validation']} | {fmt(row['tree_test_per_origin_mean_auc'])} | "
            f"{fmt(row['embedding_test_per_origin_mean_auc'])} | {fmt(row['delta_embedding_minus_tree'])} | "
            f"{fmt(row['tree_test_c_index'])} | {fmt(row['embedding_test_c_index'])} | {row['fm_adds_value']} |"
        )
    lines.extend(
        [
            "",
            "## Inputs",
            "",
            f"- Baseline metrics: `{config['inputs']['baseline_probe_metrics']}`",
            f"- Supervised matrix: `{config['inputs']['supervised_outcome_matrix']}`",
            f"- Elsa MLM directory: `{config['outputs']['mlm_directory']}/runs/{run_name}`",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-name")
    parser.add_argument("--poolings", nargs="+")
    args = parser.parse_args()

    config = load_yaml(args.config)
    comparison_config = config["comparison"]
    run_name = args.run_name or comparison_config["embedding_run_name"]
    embedding_models = list(comparison_config.get("embedding_models", ["logistic_regression"]))
    poolings = args.poolings or list(comparison_config["embedding_poolings"])
    include_time_features = bool(comparison_config.get("include_time_features", True))
    disease_domains = set(comparison_config["disease_domains"])
    clean_mask_mode = comparison_config["clean_mask_mode"]
    disease_mask_mode = comparison_config["disease_mask_mode"]

    supervised_manifest = load_json(resolve_path(config["inputs"]["supervised_matrix_manifest"]))
    id_column = supervised_manifest["metadata"]["id_column"]
    origin_waves = [int(wave) for wave in supervised_manifest["metadata"]["origin_waves"]]
    COMPARE.embedding_matrix = make_embedding_matrix(origin_waves)

    selected_outcomes = list(supervised_manifest["selected_outcomes"])
    matrix_columns = [
        id_column,
        "split",
        "outcome_id",
        "origin_wave",
        "next_wave",
        "origin_year",
        "next_year",
        "interval_years",
        "event_next_wave",
    ]
    matrix = pd.read_parquet(
        resolve_path(config["inputs"]["supervised_outcome_matrix"]), columns=matrix_columns
    )
    matrix[id_column] = matrix[id_column].astype(str)

    registry = pd.read_csv(resolve_path(config["inputs"]["outcome_registry"]))
    domain_by_outcome = registry.set_index("outcome_id")["domain"].to_dict()

    baseline_metrics = pd.read_csv(resolve_path(config["inputs"]["baseline_probe_metrics"]))
    baseline_tree, tree_model_by_outcome = baseline_tree_rows(
        baseline_metrics,
        selected_outcomes,
        list(comparison_config["baseline_tree_models"]),
    )

    embedding_rows: list[dict[str, Any]] = []
    outcome_mask_mode: dict[str, str] = {}
    embedding_cache: dict[Path, pd.DataFrame] = {}
    for outcome_id in selected_outcomes:
        mask_mode = (
            disease_mask_mode
            if domain_by_outcome.get(outcome_id) in disease_domains
            else clean_mask_mode
        )
        outcome_mask_mode[outcome_id] = mask_mode
        path = embedding_path(config, run_name, outcome_id, mask_mode)
        if path not in embedding_cache:
            embedding_cache[path] = load_embedding_frame(path, id_column)
        embeddings = embedding_cache[path]
        print(
            f"Training Elsa embedding probes for {outcome_id} using {mask_mode} embeddings from {path}",
            flush=True,
        )
        for model_name in embedding_models:
            for pooling in poolings:
                print(f"  - {model_name} / {pooling}", flush=True)
                embedding_rows.extend(
                    COMPARE.evaluate_embedding_probe(
                        matrix=matrix,
                        embeddings=embeddings,
                        outcome_id=outcome_id,
                        model_name=model_name,
                        pooling=pooling,
                        include_time_features=include_time_features,
                        id_column=id_column,
                        mask_mode=mask_mode,
                        comparison_config=comparison_config,
                    )
                )

    embedding_metrics = pd.DataFrame(embedding_rows)
    combined = pd.concat([baseline_tree, embedding_metrics], ignore_index=True, sort=False)
    selected_embedding = {
        outcome_id: choose_embedding_probe(
            embedding_metrics, outcome_id, embedding_models, poolings
        )
        for outcome_id in selected_outcomes
    }
    decision = build_decision_table(
        combined,
        selected_outcomes,
        tree_model_by_outcome,
        selected_embedding,
        outcome_mask_mode,
        float(comparison_config.get("fm_delta_threshold", 0.01)),
    )

    metrics_path = resolve_path(config["outputs"]["embedding_comparison_metrics"])
    decision_path = resolve_path(config["outputs"]["embedding_comparison_decision_table"])
    report_path = resolve_path(config["outputs"]["embedding_comparison_report"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(metrics_path, index=False)
    decision.to_csv(decision_path, index=False)
    write_report(report_path, decision, config, run_name)

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
