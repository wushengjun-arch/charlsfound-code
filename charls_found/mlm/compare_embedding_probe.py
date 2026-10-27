#!/usr/bin/env python3
"""Compare rolling tree baselines against CHARLS-found-lite embedding probes."""

from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

try:
    from sksurv.metrics import concordance_index_censored as sksurv_concordance_index_censored
except Exception:  # pragma: no cover - optional dependency fallback for GPU workers.
    sksurv_concordance_index_censored = None


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_CONFIG = SCRIPT_PATH.parents[1] / "configs" / "mlm.yaml"


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


def embedding_path(run_name: str, outcome_id: str, mask_mode: str) -> Path:
    run_dir = PROJECT_ROOT / "outputs" / "07_mlm" / "runs" / run_name
    if mask_mode == "none":
        return run_dir / "person_wave_embeddings_none.parquet"
    return run_dir / f"person_wave_embeddings_{mask_mode}_{outcome_id}.parquet"


def fmt(value: Any, digits: int = 4) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def evaluate_predictions(y_true: pd.Series, y_prob: np.ndarray) -> dict[str, float]:
    y_array = y_true.to_numpy(dtype=int)
    if len(np.unique(y_array)) < 2:
        roc_auc = math.nan
        pr_auc = math.nan
    else:
        roc_auc = float(roc_auc_score(y_array, y_prob))
        pr_auc = float(average_precision_score(y_array, y_prob))
    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier": float(brier_score_loss(y_array, y_prob)),
        "event_rate": float(y_array.mean()),
        "mean_predicted_risk": float(np.mean(y_prob)),
    }


def harrell_c_index(
    event_observed: np.ndarray, followup_time: np.ndarray, risk_score: np.ndarray
) -> float:
    comparable = 0.0
    concordant = 0.0
    event_indices = np.flatnonzero(event_observed)
    for index in event_indices:
        later = followup_time[index] < followup_time
        later_scores = risk_score[later]
        comparable += float(later_scores.size)
        if later_scores.size == 0:
            continue
        concordant += float(np.sum(risk_score[index] > later_scores))
        concordant += float(np.sum(risk_score[index] == later_scores)) * 0.5
    return float(concordant / comparable) if comparable else math.nan


def concordance_index(
    event_observed: np.ndarray, followup_time: np.ndarray, risk_score: np.ndarray
) -> float:
    if sksurv_concordance_index_censored is not None:
        return float(
            sksurv_concordance_index_censored(event_observed, followup_time, risk_score)[0]
        )
    return harrell_c_index(event_observed, followup_time, risk_score)


def first_origin_c_index(
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
        return {"c_index": math.nan, "ids": int(len(survival)), "events": 0}
    c_index = concordance_index(
        survival["event_observed"].to_numpy(dtype=bool),
        survival["time_to_event_years"].to_numpy(dtype=float),
        survival["risk_score"].to_numpy(dtype=float),
    )
    return {
        "c_index": c_index,
        "ids": int(len(survival)),
        "events": int(survival["event_observed"].sum()),
    }


def metric_row(
    outcome_id: str,
    method: str,
    model: str,
    split: str,
    metric_scope: str,
    origin_wave: str | int,
    n: int,
    ids: int,
    events: int,
    values: dict[str, float],
    mask_mode: str = "",
    embedding_pooling: str = "",
) -> dict[str, Any]:
    return {
        "outcome_id": outcome_id,
        "method": method,
        "model": model,
        "split": split,
        "metric_scope": metric_scope,
        "origin_wave": origin_wave,
        "n": int(n),
        "ids": int(ids),
        "events": int(events),
        "roc_auc": values.get("roc_auc", math.nan),
        "pr_auc": values.get("pr_auc", math.nan),
        "brier": values.get("brier", math.nan),
        "c_index": values.get("c_index", math.nan),
        "event_rate": values.get("event_rate", math.nan),
        "mean_predicted_risk": values.get("mean_predicted_risk", math.nan),
        "mask_mode": mask_mode,
        "embedding_pooling": embedding_pooling,
    }


def embedding_matrix(frame: pd.DataFrame, pooling: str, include_time_features: bool) -> np.ndarray:
    embedding = np.vstack(
        frame[pooling].map(lambda value: np.asarray(value, dtype=np.float32)).to_numpy()
    )
    if not include_time_features:
        return embedding
    origin = pd.get_dummies(frame["origin_wave"].astype(int), prefix="origin", dtype=float)
    for wave in [1, 2, 3, 4]:
        column = f"origin_{wave}"
        if column not in origin:
            origin[column] = 0.0
    origin = origin[[f"origin_{wave}" for wave in [1, 2, 3, 4]]]
    interval = frame[["interval_years"]].to_numpy(dtype=np.float32)
    return np.hstack([embedding, origin.to_numpy(dtype=np.float32), interval])


def make_embedding_probe(model_name: str, comparison_config: dict[str, Any]) -> Pipeline:
    if model_name == "logistic_regression":
        classifier = LogisticRegression(
            max_iter=2000, class_weight="balanced", solver="lbfgs", random_state=20260621
        )
    elif model_name == "mlp":
        classifier = MLPClassifier(
            hidden_layer_sizes=tuple(
                int(value) for value in comparison_config.get("mlp_hidden_layer_sizes", [64])
            ),
            activation="relu",
            solver="adam",
            alpha=float(comparison_config.get("mlp_alpha", 1e-4)),
            batch_size=int(comparison_config.get("mlp_batch_size", 256)),
            learning_rate_init=float(comparison_config.get("mlp_learning_rate_init", 1e-3)),
            max_iter=int(comparison_config.get("mlp_max_iter", 120)),
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=int(comparison_config.get("mlp_n_iter_no_change", 12)),
            random_state=20260621,
        )
    else:
        raise ValueError(f"Unknown embedding probe model: {model_name}")
    return Pipeline([("scaler", StandardScaler()), ("classifier", classifier)])


def fit_embedding_probe(
    model: Pipeline, model_name: str, x_train: np.ndarray, y_train: pd.Series
) -> Pipeline:
    if model_name == "mlp":
        sample_weight = compute_sample_weight(
            class_weight="balanced", y=y_train.to_numpy(dtype=int)
        )
        try:
            model.fit(x_train, y_train, classifier__sample_weight=sample_weight)
        except TypeError as exc:
            if "sample_weight" not in str(exc):
                raise
            model.fit(x_train, y_train)
    else:
        model.fit(x_train, y_train)
    return model


def load_embedding_frame(path: Path, id_column: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing embedding file: {path}. Export it with "
            "`charls_found/mlm/export_person_wave_embeddings.py --run-name <run> --outcome-id <outcome> --mask-mode diagnostic`."
        )
    columns = [
        id_column,
        "wave",
        "split",
        "cls_embedding",
        "mean_embedding",
        "embedding_mask_mode",
        "embedding_mask_outcome_id",
    ]
    frame = pd.read_parquet(path, columns=columns)
    return frame.rename(columns={"wave": "origin_wave"})


def evaluate_embedding_probe(
    matrix: pd.DataFrame,
    embeddings: pd.DataFrame,
    outcome_id: str,
    model_name: str,
    pooling: str,
    include_time_features: bool,
    id_column: str,
    mask_mode: str,
    comparison_config: dict[str, Any],
) -> list[dict[str, Any]]:
    outcome_frame = matrix[matrix["outcome_id"].eq(outcome_id)].merge(
        embeddings,
        on=[id_column, "origin_wave"],
        how="left",
        suffixes=("", "_embedding"),
        validate="many_to_one",
    )
    if outcome_frame[pooling].isna().any():
        missing = int(outcome_frame[pooling].isna().sum())
        raise RuntimeError(f"{outcome_id} has {missing} rows without {pooling} embeddings")
    train = outcome_frame[outcome_frame["split"].eq("train")].copy()
    if train["event_next_wave"].nunique() < 2:
        raise RuntimeError(f"{outcome_id} train split has one class")

    model = make_embedding_probe(model_name, comparison_config)
    fit_embedding_probe(
        model,
        model_name,
        embedding_matrix(train, pooling, include_time_features),
        train["event_next_wave"].astype(int),
    )
    rows: list[dict[str, Any]] = []
    for split in ["validation", "test"]:
        split_frame = outcome_frame[outcome_frame["split"].eq(split)].copy()
        y_true = split_frame["event_next_wave"].astype(int)
        y_prob = model.predict_proba(embedding_matrix(split_frame, pooling, include_time_features))[
            :, 1
        ]
        pooled = evaluate_predictions(y_true, y_prob)
        rows.append(
            metric_row(
                outcome_id,
                method="embedding",
                model=model_name,
                split=split,
                metric_scope="pooled",
                origin_wave="pooled",
                n=len(split_frame),
                ids=split_frame[id_column].nunique(),
                events=int(y_true.sum()),
                values=pooled,
                mask_mode=mask_mode,
                embedding_pooling=pooling,
            )
        )

        with_prob = split_frame.copy()
        with_prob["probability"] = y_prob
        for origin_wave, origin_frame in with_prob.groupby("origin_wave", sort=True):
            origin_y = origin_frame["event_next_wave"].astype(int)
            values = evaluate_predictions(
                origin_y, origin_frame["probability"].to_numpy(dtype=float)
            )
            rows.append(
                metric_row(
                    outcome_id,
                    method="embedding",
                    model=model_name,
                    split=split,
                    metric_scope="origin",
                    origin_wave=int(origin_wave),
                    n=len(origin_frame),
                    ids=origin_frame[id_column].nunique(),
                    events=int(origin_y.sum()),
                    values=values,
                    mask_mode=mask_mode,
                    embedding_pooling=pooling,
                )
            )
        survival = first_origin_c_index(split_frame, y_prob, id_column)
        rows.append(
            metric_row(
                outcome_id,
                method="embedding",
                model=model_name,
                split=split,
                metric_scope="survival",
                origin_wave="all",
                n=survival["ids"],
                ids=survival["ids"],
                events=survival["events"],
                values=survival,
                mask_mode=mask_mode,
                embedding_pooling=pooling,
            )
        )
    return rows


def per_origin_summary(
    metrics: pd.DataFrame, outcome_id: str, method: str, model: str, split: str, pooling: str = ""
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
        "origin_1_auc": auc_by_wave.get(1, math.nan),
        "origin_2_auc": auc_by_wave.get(2, math.nan),
        "origin_3_auc": auc_by_wave.get(3, math.nan),
        "origin_4_auc": auc_by_wave.get(4, math.nan),
    }


def baseline_tree_rows(
    baseline_metrics: pd.DataFrame, selected_outcomes: list[str], tree_models: list[str]
) -> tuple[pd.DataFrame, dict[str, str]]:
    rows: list[pd.DataFrame] = []
    best_model_by_outcome: dict[str, str] = {}
    for outcome_id in selected_outcomes:
        origin_validation = baseline_metrics[
            baseline_metrics["outcome_id"].eq(outcome_id)
            & baseline_metrics["split"].eq("validation")
            & baseline_metrics["metric_scope"].eq("origin")
            & baseline_metrics["model"].isin(tree_models)
        ]
        if origin_validation.empty:
            raise RuntimeError(f"No tree baseline metrics found for {outcome_id}")
        model_scores = (
            origin_validation.groupby("model", as_index=False)
            .agg(origin_mean_auc=("roc_auc", "mean"), origin_mean_pr_auc=("pr_auc", "mean"))
            .sort_values(["origin_mean_auc", "origin_mean_pr_auc"], ascending=False)
        )
        model = str(model_scores.iloc[0]["model"])
        best_model_by_outcome[outcome_id] = model
        rows.append(
            baseline_metrics[
                baseline_metrics["outcome_id"].eq(outcome_id)
                & baseline_metrics["model"].eq(model)
                & baseline_metrics["split"].isin(["validation", "test"])
            ]
        )
    baseline = pd.concat(rows, ignore_index=True)
    baseline = baseline.assign(method="tree", mask_mode="", embedding_pooling="")
    return baseline, best_model_by_outcome


def choose_embedding_probe(
    metrics: pd.DataFrame, outcome_id: str, models: list[str], poolings: list[str]
) -> tuple[str, str]:
    candidates = []
    for model_name in models:
        for pooling in poolings:
            summary = per_origin_summary(
                metrics, outcome_id, "embedding", model_name, "validation", pooling
            )
            candidates.append((summary["origin_mean_auc"], model_name, pooling))
    candidates = [
        (score, model_name, pooling)
        for score, model_name, pooling in candidates
        if not pd.isna(score)
    ]
    if not candidates:
        raise RuntimeError(f"No validation per-origin AUC for {outcome_id}")
    candidates.sort(reverse=True)
    return candidates[0][1], candidates[0][2]


def build_decision_table(
    all_metrics: pd.DataFrame,
    selected_outcomes: list[str],
    tree_model_by_outcome: dict[str, str],
    selected_embedding: dict[str, tuple[str, str]],
    outcome_mask_mode: dict[str, str],
    delta_threshold: float,
) -> pd.DataFrame:
    rows = []
    for outcome_id in selected_outcomes:
        tree_model = tree_model_by_outcome[outcome_id]
        embedding_model, pooling = selected_embedding[outcome_id]
        tree_origin = per_origin_summary(all_metrics, outcome_id, "tree", tree_model, "test")
        emb_origin = per_origin_summary(
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
        rows.append(
            {
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
                "tree_origin_1_auc": tree_origin["origin_1_auc"],
                "tree_origin_2_auc": tree_origin["origin_2_auc"],
                "tree_origin_3_auc": tree_origin["origin_3_auc"],
                "tree_origin_4_auc": tree_origin["origin_4_auc"],
                "embedding_origin_1_auc": emb_origin["origin_1_auc"],
                "embedding_origin_2_auc": emb_origin["origin_2_auc"],
                "embedding_origin_3_auc": emb_origin["origin_3_auc"],
                "embedding_origin_4_auc": emb_origin["origin_4_auc"],
            }
        )
    return pd.DataFrame(rows)


def write_report(path: Path, decision: pd.DataFrame, config: dict[str, Any], run_name: str) -> None:
    lines = [
        "# CHARLS Embedding vs Tree Comparison",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"- Embedding run: `{run_name}`",
        "- Main decision metric: test per-origin mean ROC-AUC.",
        "- Pooled AUC is included only as context because wave/base-rate differences can inflate it.",
        "- Tree model family, embedding head, and embedding pooling are selected on validation per-origin mean ROC-AUC, then evaluated once on test.",
        "- Clean outcomes use unmasked embeddings; chronic disease and multimorbidity outcomes use diagnostic-masked embeddings.",
        "",
        "## Decision Table",
        "",
        "| outcome_id | mask | head | pooling | tree_auc | embedding_auc | delta | tree_c | embedding_c | verdict |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in decision.iterrows():
        lines.append(
            f"| {row['outcome_id']} | {row['embedding_mask_mode']} | {row['embedding_model']} | {row['embedding_pooling_selected_by_validation']} | "
            f"{fmt(row['tree_test_per_origin_mean_auc'])} | {fmt(row['embedding_test_per_origin_mean_auc'])} | "
            f"{fmt(row['delta_embedding_minus_tree'])} | {fmt(row['tree_test_c_index'])} | {fmt(row['embedding_test_c_index'])} | {row['fm_adds_value']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A positive delta means the selected CHARLS-found-lite embedding probe beat the selected tree baseline on the honest per-origin test AUC. `tie` means the absolute difference is smaller than the configured threshold.",
            "",
            "## Inputs",
            "",
            f"- Baseline metrics: `{config['inputs'].get('baseline_probe_metrics', 'outputs/06_supervised/baseline_probe_metrics.csv')}`",
            f"- Supervised matrix: `{config['inputs']['supervised_outcome_matrix']}`",
            f"- Embedding run directory: `outputs/07_mlm/runs/{run_name}`",
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
    registry = pd.read_csv(
        resolve_path("outputs/02_outcome_registry/outcome_registry_validated.csv")
    )
    domain_by_outcome = registry.set_index("outcome_id")["domain"].to_dict()

    baseline_metrics_path = resolve_path(
        config["inputs"].get(
            "baseline_probe_metrics", "outputs/06_supervised/baseline_probe_metrics.csv"
        )
    )
    baseline_metrics = pd.read_csv(baseline_metrics_path)
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
        path = embedding_path(run_name, outcome_id, mask_mode)
        if path not in embedding_cache:
            embedding_cache[path] = load_embedding_frame(path, id_column)
        embeddings = embedding_cache[path]
        print(
            f"Training embedding probes for {outcome_id} using {mask_mode} embeddings from {path}"
        )
        for model_name in embedding_models:
            for pooling in poolings:
                print(f"  - {model_name} / {pooling}")
                embedding_rows.extend(
                    evaluate_embedding_probe(
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
                "decision_table": str(decision_path.relative_to(PROJECT_ROOT)),
                "report": str(report_path.relative_to(PROJECT_ROOT)),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    raise SystemExit(main())
