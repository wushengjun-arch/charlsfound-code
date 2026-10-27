#!/usr/bin/env python3
"""Fusion add-on uncertainty and calibration for core cohort-outcome rows."""

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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
WORKSPACE_ROOT = PROJECT_ROOT

CORE_OUTCOMES = [
    "adl_new_difficulty",
    "iadl_new_difficulty",
    "incident_diabetes",
    "incident_hypertension",
    "multimorbidity_onset",
    "high_depressive_symptoms_onset",
    "low_cognition_onset_data_driven",
]

OUTCOME_LABELS = {
    "adl_new_difficulty": "ADL",
    "iadl_new_difficulty": "IADL",
    "incident_diabetes": "Diabetes",
    "incident_hypertension": "Hypertension",
    "multimorbidity_onset": "Multimorbidity",
    "high_depressive_symptoms_onset": "Depression",
    "low_cognition_onset_data_driven": "Low cognition",
}

CONFIGS = {
    "charls": PROJECT_ROOT / "charls_found" / "configs" / "mlm.yaml",
    "hrs": PROJECT_ROOT / "charls_found" / "configs" / "hrs_mlm.yaml",
    "klosa": PROJECT_ROOT / "charls_found" / "configs" / "klosa_mlm.yaml",
    "elsa": PROJECT_ROOT / "charls_found" / "configs" / "elsa_mlm.yaml",
    "share": PROJECT_ROOT / "charls_found" / "configs" / "share_mlm.yaml",
}

COHORT_LABELS = {
    "charls": "CHARLS",
    "hrs": "HRS",
    "klosa": "KLoSA",
    "elsa": "ELSA",
    "share": "SHARE",
}

COHORT_COLORS = {
    "charls": "#D85A30",
    "hrs": "#378ADD",
    "klosa": "#1D9E75",
    "elsa": "#7F77DD",
    "share": "#BA7517",
}

COHORT_MARKERS = {
    "charls": "o",
    "hrs": "^",
    "klosa": "s",
    "elsa": "D",
    "share": "*",
}


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


LATE = load_module(
    "fusion_uncertainty_late_helpers",
    PROJECT_ROOT / "charls_found" / "external_validation" / "late_fusion.py",
)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def safe_mean(values: list[float]) -> float:
    clean = [value for value in values if not pd.isna(value)]
    return float(np.mean(clean)) if clean else math.nan


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

    validation_tree = LATE.predict_tree(tree_frame, tree_model, features, "validation")
    validation_embedding = LATE.predict_embedding(
        embedding_frame, embedding_model, pooling, include_time_features, "validation"
    )
    validation_paired = LATE.pair_predictions(validation_tree, validation_embedding, id_column)
    fusion_model = LATE.make_fusion_model()
    fusion_model.fit(
        validation_paired[["tree_probability", "embedding_probability"]].to_numpy(dtype=np.float32),
        validation_paired["event_next_wave"].astype(int),
    )

    test_tree = LATE.predict_tree(tree_frame, tree_model, features, "test")
    test_embedding = LATE.predict_embedding(
        embedding_frame, embedding_model, pooling, include_time_features, "test"
    )
    paired = LATE.pair_predictions(test_tree, test_embedding, id_column)
    paired["fusion_probability"] = LATE.predict_fusion(fusion_model, paired)
    paired["participant_id"] = paired[id_column].astype(str)
    paired["outcome_id"] = outcome_id
    return paired, {
        "tree_model": tree_model_name,
        "embedding_model": embedding_model_name,
        "embedding_pooling": pooling,
        "embedding_mask_mode": mask_mode,
    }


def origin_metric_values(frame: pd.DataFrame, probability_column: str) -> dict[str, float]:
    aucs: list[float] = []
    prs: list[float] = []
    briers: list[float] = []
    for _, origin_frame in frame.groupby("origin_wave", sort=True):
        values = LATE.COMPARE.evaluate_predictions(
            origin_frame["event_next_wave"].astype(int),
            origin_frame[probability_column].to_numpy(dtype=float),
        )
        aucs.append(values["roc_auc"])
        prs.append(values["pr_auc"])
        briers.append(values["brier"])
    return {
        "auc": safe_mean(aucs),
        "pr_auc": safe_mean(prs),
        "brier": safe_mean(briers),
    }


def score_frame(frame: pd.DataFrame) -> dict[str, float]:
    scores: dict[str, float] = {}
    for method, column in [
        ("tree", "tree_probability"),
        ("embedding", "embedding_probability"),
        ("fusion", "fusion_probability"),
    ]:
        values = origin_metric_values(frame, column)
        for metric, value in values.items():
            scores[f"{method}_{metric}"] = value
    scores["delta_embedding_minus_tree_auc"] = scores["embedding_auc"] - scores["tree_auc"]
    scores["delta_fusion_minus_tree_auc"] = scores["fusion_auc"] - scores["tree_auc"]
    scores["delta_fusion_minus_embedding_auc"] = scores["fusion_auc"] - scores["embedding_auc"]
    scores["delta_fusion_minus_best_single_auc"] = scores["fusion_auc"] - max(
        scores["tree_auc"], scores["embedding_auc"]
    )
    scores["delta_fusion_minus_tree_pr_auc"] = scores["fusion_pr_auc"] - scores["tree_pr_auc"]
    scores["delta_fusion_minus_tree_brier"] = scores["fusion_brier"] - scores["tree_brier"]
    return scores


def id_cluster_bootstrap(paired: pd.DataFrame, iterations: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ids = paired["participant_id"].drop_duplicates().to_numpy()
    indices_by_id = {
        id_value: group.index.to_numpy()
        for id_value, group in paired.groupby("participant_id", sort=False)
    }
    rows: list[dict[str, Any]] = []
    for iteration in range(iterations):
        sampled_ids = rng.choice(ids, size=len(ids), replace=True)
        sampled_indices = np.concatenate([indices_by_id[id_value] for id_value in sampled_ids])
        sample = paired.loc[sampled_indices]
        row = score_frame(sample)
        row.update(
            {
                "iteration": iteration,
                "sampled_ids": int(len(sampled_ids)),
                "sampled_rows": int(len(sample)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_bootstrap(
    observed: dict[str, float], bootstrap: pd.DataFrame, threshold: float
) -> dict[str, Any]:
    row: dict[str, Any] = dict(observed)
    for column in [
        "delta_embedding_minus_tree_auc",
        "delta_fusion_minus_tree_auc",
        "delta_fusion_minus_embedding_auc",
        "delta_fusion_minus_best_single_auc",
        "delta_fusion_minus_tree_pr_auc",
        "delta_fusion_minus_tree_brier",
    ]:
        values = bootstrap[column].to_numpy(dtype=float)
        row[f"{column}_bootstrap_mean"] = float(np.nanmean(values))
        row[f"{column}_bootstrap_median"] = float(np.nanmedian(values))
        row[f"{column}_bootstrap_sd"] = finite_std(values)
        row[f"{column}_ci_low"] = percentile(values, 2.5)
        row[f"{column}_ci_high"] = percentile(values, 97.5)
        row[f"p_{column}_gt_0"] = float(np.nanmean(values > 0))
        row[f"p_{column}_gt_threshold"] = float(np.nanmean(values > threshold))
        row[f"p_{column}_lt_0"] = float(np.nanmean(values < 0))
    delta = row["delta_fusion_minus_tree_auc"]
    low = row["delta_fusion_minus_tree_auc_ci_low"]
    high = row["delta_fusion_minus_tree_auc_ci_high"]
    if pd.isna(delta) or pd.isna(low) or pd.isna(high):
        label = "uncertain"
    elif delta >= threshold and low > 0:
        label = "stable fusion add-on"
    elif delta > 0:
        label = "directional fusion add-on"
    elif high < 0:
        label = "fusion below tree"
    else:
        label = "fusion tie/uncertain"
    row["fusion_uncertainty_label"] = label
    return row


def logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values.astype(float), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def calibration_intercept_slope(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    if len(np.unique(y_true)) < 2 or len(np.unique(y_prob)) < 2:
        return math.nan, math.nan
    x = logit(y_prob)
    x_design = sm.add_constant(x)
    try:
        model = sm.Logit(y_true, x_design).fit(disp=False, maxiter=100)
    except Exception:
        return math.nan, math.nan
    return float(model.params[0]), float(model.params[1])


def calibration_bins(
    frame: pd.DataFrame,
    probability_column: str,
    method: str,
    cohort: str,
    outcome_id: str,
    bins: int,
) -> pd.DataFrame:
    tmp = frame[["participant_id", "event_next_wave", probability_column]].copy()
    tmp = tmp.rename(columns={probability_column: "probability"})
    q = max(1, min(bins, len(tmp)))
    ranks = tmp["probability"].rank(method="first")
    tmp["calibration_bin"] = pd.qcut(ranks, q=q, labels=False, duplicates="drop")
    rows = []
    for bin_id, group in tmp.groupby("calibration_bin", sort=True):
        rows.append(
            {
                "cohort": cohort,
                "cohort_label": COHORT_LABELS[cohort],
                "outcome_id": outcome_id,
                "outcome_label": OUTCOME_LABELS[outcome_id],
                "method": method,
                "bin": int(bin_id),
                "n": int(len(group)),
                "ids": int(group["participant_id"].nunique()),
                "events": int(group["event_next_wave"].astype(int).sum()),
                "mean_predicted": float(group["probability"].mean()),
                "observed_rate": float(group["event_next_wave"].astype(int).mean()),
            }
        )
    return pd.DataFrame(rows)


def calibration_metrics(
    frame: pd.DataFrame,
    probability_column: str,
    method: str,
    cohort: str,
    outcome_id: str,
    bins: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    y_true = frame["event_next_wave"].astype(int).to_numpy()
    y_prob = np.clip(frame[probability_column].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
    intercept, slope = calibration_intercept_slope(y_true, y_prob)
    bin_table = calibration_bins(frame, probability_column, method, cohort, outcome_id, bins)
    weights = bin_table["n"].to_numpy(dtype=float) / float(len(frame))
    abs_errors = (
        (bin_table["observed_rate"] - bin_table["mean_predicted"]).abs().to_numpy(dtype=float)
    )
    metrics = {
        "cohort": cohort,
        "cohort_label": COHORT_LABELS[cohort],
        "outcome_id": outcome_id,
        "outcome_label": OUTCOME_LABELS[outcome_id],
        "method": method,
        "n": int(len(frame)),
        "ids": int(frame["participant_id"].nunique()),
        "events": int(y_true.sum()),
        "event_rate": float(y_true.mean()),
        "mean_predicted": float(y_prob.mean()),
        "calibration_in_the_large": float(y_prob.mean() - y_true.mean()),
        "brier": float(np.mean((y_prob - y_true) ** 2)),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "ece_quantile": float(np.sum(weights * abs_errors)),
        "max_bin_abs_error": float(np.max(abs_errors)) if len(abs_errors) else math.nan,
        "bins": int(bin_table["bin"].nunique()) if not bin_table.empty else 0,
    }
    return metrics, bin_table


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(WORKSPACE_ROOT))
    except ValueError:
        return str(path)


def plot_fusion_uncertainty(summary: pd.DataFrame, output_dir: Path, experiment_dir: Path) -> None:
    df = summary.sort_values("delta_fusion_minus_tree_auc").reset_index(drop=True)
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(7.2, 8.6), dpi=180)
    ax.axvspan(-0.005, 0.005, color="#F4F4F4", zorder=0)
    ax.axvline(0, color="#6B7280", linestyle="--", linewidth=1.0)
    for i, row in df.iterrows():
        cohort = row["cohort"]
        ax.plot(
            [row["delta_fusion_minus_tree_auc_ci_low"], row["delta_fusion_minus_tree_auc_ci_high"]],
            [i, i],
            color=COHORT_COLORS[cohort],
            alpha=0.65,
            linewidth=1.1,
        )
        ax.scatter(
            row["delta_fusion_minus_tree_auc"],
            i,
            color=COHORT_COLORS[cohort],
            marker=COHORT_MARKERS[cohort],
            s=26,
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{row.cohort_label} - {OUTCOME_LABELS[row.outcome_id]}" for row in df.itertuples()],
        fontsize=6,
    )
    ax.set_xlabel("Fusion - tree AUC (ID-cluster bootstrap 95% CI)")
    ax.set_title(
        "Fusion add-on uncertainty across core cohort-outcome rows",
        loc="left",
        fontsize=10,
        fontweight="bold",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    for base in [output_dir / "figures", experiment_dir]:
        base.mkdir(parents=True, exist_ok=True)
        fig.savefig(base / "fusion_uncertainty_forest.png", dpi=300, facecolor="white")
        fig.savefig(base / "fusion_uncertainty_forest.pdf", facecolor="white")
    plt.close(fig)


def plot_calibration_summary(summary: pd.DataFrame, output_dir: Path, experiment_dir: Path) -> None:
    df = summary.copy()
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=180)
    ax.axvline(0, color="#6B7280", linestyle="--", linewidth=0.9)
    ax.axhline(0, color="#6B7280", linestyle="--", linewidth=0.9)
    ax.axvspan(0, df["delta_fusion_minus_tree_auc"].max() + 0.005, color="#F7FBF8", zorder=0)
    ax.axhspan(df["delta_fusion_minus_tree_brier"].min() - 0.005, 0, color="#F7FBF8", zorder=0)
    for cohort, group in df.groupby("cohort", sort=False):
        ax.scatter(
            group["delta_fusion_minus_tree_auc"],
            group["delta_fusion_minus_tree_brier"],
            color=COHORT_COLORS[cohort],
            marker=COHORT_MARKERS[cohort],
            s=42,
            edgecolor="white",
            linewidth=0.4,
            label=COHORT_LABELS[cohort],
        )
    ax.set_xlabel("Fusion - tree AUC")
    ax.set_ylabel("Fusion - tree Brier score")
    ax.set_title(
        "Discrimination gain versus calibration error", loc="left", fontsize=10, fontweight="bold"
    )
    ax.text(
        0.98,
        0.05,
        "Lower Brier is better",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        color="#6B7280",
    )
    ax.legend(frameon=False, ncol=3, fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    for base in [output_dir / "figures", experiment_dir]:
        base.mkdir(parents=True, exist_ok=True)
        fig.savefig(base / "fusion_calibration_tradeoff.png", dpi=300, facecolor="white")
        fig.savefig(base / "fusion_calibration_tradeoff.pdf", facecolor="white")
    plt.close(fig)


def write_report(path: Path, summary: pd.DataFrame, iterations: int, elapsed: float) -> None:
    threshold_stable = int(summary["fusion_uncertainty_label"].eq("stable fusion add-on").sum())
    ci_positive = int((summary["delta_fusion_minus_tree_auc_ci_low"] > 0).sum())
    ci_negative = int((summary["delta_fusion_minus_tree_auc_ci_high"] < 0).sum())
    directional_ci_crossing = int(
        (
            (summary["delta_fusion_minus_tree_auc"] > 0)
            & (summary["delta_fusion_minus_tree_auc_ci_low"] <= 0)
            & (summary["delta_fusion_minus_tree_auc_ci_high"] >= 0)
        ).sum()
    )
    ge_tree = int((summary["delta_fusion_minus_tree_auc"] >= 0).sum())
    gt_best = int((summary["delta_fusion_minus_best_single_auc"] > 0).sum())
    brier_not_worse = int((summary["delta_fusion_minus_tree_brier"] <= 0).sum())
    external = summary[summary["cohort"].ne("charls")]
    external_threshold_stable = int(
        external["fusion_uncertainty_label"].eq("stable fusion add-on").sum()
    )
    external_ci_positive = int((external["delta_fusion_minus_tree_auc_ci_low"] > 0).sum())
    external_ci_negative = int((external["delta_fusion_minus_tree_auc_ci_high"] < 0).sum())
    external_ge = int((external["delta_fusion_minus_tree_auc"] >= 0).sum())
    external_best = int((external["delta_fusion_minus_best_single_auc"] > 0).sum())
    external_brier_not_worse = int((external["delta_fusion_minus_tree_brier"] <= 0).sum())
    lines = [
        "# Fusion Uncertainty And Calibration Analysis",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"- Bootstrap iterations per cohort-outcome row: {iterations}.",
        "- Bootstrap unit: participant ID cluster; all person-period rows for sampled IDs are retained.",
        "- Fusion model: validation-fit stacked logistic regression using selected tree and embedding probabilities.",
        "",
        "## Headline",
        "",
        f"- Core rows analyzed: {len(summary)}.",
        f"- Fusion point estimate >= tree: {ge_tree}/{len(summary)}.",
        f"- Fusion point estimate > best single: {gt_best}/{len(summary)}.",
        f"- Fusion AUC CI entirely > 0 versus tree: {ci_positive}/{len(summary)}.",
        f"- Fusion AUC CI entirely < 0 versus tree: {ci_negative}/{len(summary)}.",
        f"- Positive fusion point estimate with CI crossing zero: {directional_ci_crossing}/{len(summary)}.",
        f"- Fusion AUC gain >= +0.01 with CI entirely > 0: {threshold_stable}/{len(summary)}.",
        f"- Fusion origin-wave-averaged Brier score not worse than tree: {brier_not_worse}/{len(summary)}.",
        f"- External rows: fusion >= tree {external_ge}/{len(external)}, fusion > best single {external_best}/{len(external)}, CI > 0 {external_ci_positive}/{len(external)}, CI < 0 {external_ci_negative}/{len(external)}, >= +0.01 and CI > 0 {external_threshold_stable}/{len(external)}, origin-wave-averaged Brier not worse {external_brier_not_worse}/{len(external)}.",
        "",
        "## Interpretation Boundary",
        "",
        "Fusion remains a useful discrimination add-on signal, but manuscript language should distinguish point-estimate add-on frequency, bootstrap intervals excluding zero, and the stricter prespecified +0.01 AUC threshold. Calibration and Brier metrics are test-set diagnostics; current fusion probabilities should not be described as finalized absolute-risk estimates without a separate recalibration layer.",
        "",
        "## Outputs",
        "",
        "- `outputs/10_evidence_map/fusion_uncertainty/fusion_uncertainty_summary.csv`",
        "- `outputs/10_evidence_map/fusion_uncertainty/fusion_uncertainty_bootstrap_iterations.csv`",
        "- `outputs/10_evidence_map/fusion_uncertainty/fusion_calibration_metrics.csv`",
        "- `outputs/10_evidence_map/fusion_uncertainty/fusion_calibration_bins.csv`",
        "- `outputs/10_evidence_map/fusion_uncertainty/predictions/`",
        "- `outputs/figures/fusion_uncertainty_forest.png` and `.pdf`",
        "- `outputs/figures/fusion_calibration_tradeoff.png` and `.pdf`",
        "",
        f"Elapsed seconds: {elapsed:.1f}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    output_dir = PROJECT_ROOT / "outputs" / "10_evidence_map" / "fusion_uncertainty"
    prediction_dir = output_dir / "predictions"
    experiment_fig_dir = PROJECT_ROOT / "outputs" / "figures"
    experiment_analysis_dir = PROJECT_ROOT / "outputs" / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[pd.DataFrame] = []
    calibration_rows: list[dict[str, Any]] = []
    calibration_bin_frames: list[pd.DataFrame] = []
    all_predictions: list[pd.DataFrame] = []

    for cohort in args.cohorts:
        config_path = CONFIGS[cohort]
        config = LATE.load_yaml(config_path)
        manifest = LATE.load_json(resolve_path(config["inputs"]["supervised_matrix_manifest"]))
        id_column = str(manifest["metadata"]["id_column"])
        origin_waves = [int(wave) for wave in manifest["metadata"]["origin_waves"]]
        LATE.COMPARE.embedding_matrix = LATE.make_embedding_matrix(origin_waves)

        matrix = pd.read_parquet(resolve_path(config["inputs"]["supervised_outcome_matrix"]))
        matrix[id_column] = LATE.normalize_id(matrix[id_column])
        decision = pd.read_csv(
            resolve_path(config["outputs"]["embedding_comparison_decision_table"])
        )
        decision = decision[decision["outcome_id"].isin(CORE_OUTCOMES)].copy()
        decision["outcome_order"] = decision["outcome_id"].map(
            {outcome: i for i, outcome in enumerate(CORE_OUTCOMES)}
        )
        decision = decision.sort_values("outcome_order")
        threshold = float(config["comparison"].get("fm_delta_threshold", 0.01))
        embedding_cache: dict[Path, pd.DataFrame] = {}

        for outcome_index, selected in enumerate(decision.itertuples(index=False), start=1):
            selected_series = pd.Series(selected._asdict())
            outcome_id = str(selected_series["outcome_id"])
            print(
                f"[{cohort}] {outcome_index}/{len(decision)} fusion uncertainty: {outcome_id}",
                flush=True,
            )
            paired, model_info = build_prediction_frame(
                config=config,
                manifest=manifest,
                matrix=matrix,
                id_column=id_column,
                id_reference=matrix[id_column],
                selected=selected_series,
                embedding_cache=embedding_cache,
            )
            paired.insert(0, "cohort", cohort)
            paired.insert(1, "cohort_label", COHORT_LABELS[cohort])
            keep_cols = [
                "cohort",
                "cohort_label",
                "outcome_id",
                "participant_id",
                "origin_wave",
                "next_wave",
                "origin_year",
                "next_year",
                "interval_years",
                "event_next_wave",
                "tree_probability",
                "embedding_probability",
                "fusion_probability",
            ]
            pred_out = paired[keep_cols].copy()
            pred_out.to_parquet(
                prediction_dir / f"{cohort}_{outcome_id}_fusion_test_predictions.parquet",
                index=False,
            )
            all_predictions.append(pred_out)

            observed = score_frame(paired)
            bootstrap = id_cluster_bootstrap(
                paired,
                iterations=int(args.bootstrap_iterations),
                seed=int(args.seed) + 100_003 * list(CONFIGS).index(cohort) + outcome_index * 1009,
            )
            bootstrap.insert(0, "outcome_id", outcome_id)
            bootstrap.insert(0, "outcome_label", OUTCOME_LABELS[outcome_id])
            bootstrap.insert(0, "cohort", cohort)
            bootstrap.insert(1, "cohort_label", COHORT_LABELS[cohort])
            bootstrap_rows.append(bootstrap)

            summary = summarize_bootstrap(observed, bootstrap, threshold)
            summary.update(
                {
                    "cohort": cohort,
                    "cohort_label": COHORT_LABELS[cohort],
                    "outcome_id": outcome_id,
                    "outcome_label": OUTCOME_LABELS[outcome_id],
                    "test_rows": int(len(paired)),
                    "test_ids": int(paired["participant_id"].nunique()),
                    "test_events": int(paired["event_next_wave"].astype(int).sum()),
                    **model_info,
                }
            )
            summary_rows.append(summary)

            for method, column in [
                ("tree", "tree_probability"),
                ("embedding", "embedding_probability"),
                ("fusion", "fusion_probability"),
            ]:
                metrics, bins = calibration_metrics(
                    paired, column, method, cohort, outcome_id, int(args.calibration_bins)
                )
                calibration_rows.append(metrics)
                calibration_bin_frames.append(bins)

    summary = pd.DataFrame(summary_rows)
    front_cols = [
        "cohort",
        "cohort_label",
        "outcome_id",
        "outcome_label",
        "test_rows",
        "test_ids",
        "test_events",
        "tree_model",
        "embedding_model",
        "embedding_pooling",
        "embedding_mask_mode",
        "fusion_uncertainty_label",
    ]
    summary = summary[
        front_cols + [column for column in summary.columns if column not in front_cols]
    ]
    bootstrap_all = pd.concat(bootstrap_rows, ignore_index=True)
    calibration = pd.DataFrame(calibration_rows)
    calibration_bins_all = pd.concat(calibration_bin_frames, ignore_index=True)
    predictions_all = pd.concat(all_predictions, ignore_index=True)

    summary.to_csv(output_dir / "fusion_uncertainty_summary.csv", index=False)
    bootstrap_all.to_csv(output_dir / "fusion_uncertainty_bootstrap_iterations.csv", index=False)
    calibration.to_csv(output_dir / "fusion_calibration_metrics.csv", index=False)
    calibration_bins_all.to_csv(output_dir / "fusion_calibration_bins.csv", index=False)
    predictions_all.to_parquet(output_dir / "fusion_test_predictions_all.parquet", index=False)

    plot_fusion_uncertainty(summary, output_dir, experiment_fig_dir)
    plot_calibration_summary(summary, output_dir, experiment_fig_dir)
    elapsed = time.perf_counter() - started
    write_report(
        output_dir / "fusion_uncertainty_summary.md",
        summary,
        int(args.bootstrap_iterations),
        elapsed,
    )
    write_report(
        experiment_analysis_dir / "fusion_calibration_analysis.md",
        summary,
        int(args.bootstrap_iterations),
        elapsed,
    )

    print(
        json.dumps(
            {
                "summary": display_path(output_dir / "fusion_uncertainty_summary.csv"),
                "bootstrap_iterations": int(args.bootstrap_iterations),
                "rows": int(len(summary)),
                "stable_fusion_add_on": int(
                    summary["fusion_uncertainty_label"].eq("stable fusion add-on").sum()
                ),
                "report": display_path(experiment_analysis_dir / "fusion_calibration_analysis.md"),
                "elapsed_seconds": round(elapsed, 1),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--cohorts", nargs="+", choices=list(CONFIGS), default=list(CONFIGS))
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    raise SystemExit(main())
