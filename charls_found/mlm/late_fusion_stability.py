#!/usr/bin/env python3
"""Bootstrap and repeated-split stability for late-fusion gains."""

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


LATE_FUSION = load_module(
    "charls_late_fusion_stability_probe",
    PROJECT_ROOT / "charls_found" / "mlm" / "late_fusion_probe.py",
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


def percentile_interval(
    values: np.ndarray, low: float = 2.5, high: float = 97.5
) -> tuple[float, float]:
    clean = values[~np.isnan(values)]
    if clean.size == 0:
        return math.nan, math.nan
    return float(np.percentile(clean, low)), float(np.percentile(clean, high))


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


def prepare_outcome_frames(
    outcome_id: str,
    matrix: pd.DataFrame,
    manifest: dict[str, Any],
    domain_by_outcome: dict[str, str],
    mlm_config: dict[str, Any],
    embedding_cache: dict[Path, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], str]:
    id_column = manifest["metadata"]["id_column"]
    comparison_config = mlm_config["comparison"]
    disease_domains = set(comparison_config["disease_domains"])
    mask_mode = (
        comparison_config["disease_mask_mode"]
        if domain_by_outcome.get(outcome_id) in disease_domains
        else comparison_config["clean_mask_mode"]
    )
    outcome_matrix = matrix[matrix["outcome_id"].eq(outcome_id)].copy().reset_index(drop=True)
    outcome_matrix["row_id"] = np.arange(len(outcome_matrix))
    features = LATE_FUSION.tree_features(manifest, outcome_id)
    missing_features = sorted(set(features) - set(outcome_matrix.columns))
    if missing_features:
        raise RuntimeError(f"{outcome_id} requested missing tree features: {missing_features[:20]}")
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
    emb_path = LATE_FUSION.COMPARE.embedding_path(
        comparison_config["embedding_run_name"], outcome_id, mask_mode
    )
    if emb_path not in embedding_cache:
        embedding_cache[emb_path] = LATE_FUSION.COMPARE.load_embedding_frame(emb_path, id_column)
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
    return tree_frame, embedding_frame, features, mask_mode


def train_and_predict(
    outcome_id: str,
    tree_frame: pd.DataFrame,
    embedding_frame: pd.DataFrame,
    features: list[str],
    supervised_config: dict[str, Any],
    manifest: dict[str, Any],
    mlm_config: dict[str, Any],
    mask_mode: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    id_column = manifest["metadata"]["id_column"]
    comparison_config = mlm_config["comparison"]
    late_fusion_config = mlm_config.get("late_fusion", {})
    include_time_features = bool(comparison_config.get("include_time_features", True))
    fusion_features = list(
        late_fusion_config.get("fusion_features", ["tree_probability", "embedding_probability"])
    )
    tree_model_name, tree_model, tree_validation_auc = LATE_FUSION.select_tree_candidate(
        tree_frame,
        features,
        list(comparison_config["baseline_tree_models"]),
        supervised_config,
        manifest,
    )
    embedding_model_name, embedding_pooling, embedding_model, embedding_validation_auc = (
        LATE_FUSION.select_embedding_candidate(
            embedding_frame,
            list(comparison_config.get("embedding_models", ["logistic_regression"])),
            list(comparison_config["embedding_poolings"]),
            include_time_features,
            comparison_config,
        )
    )
    validation_tree = LATE_FUSION.predict_tree(
        tree_frame, features, tree_model, "validation", "tree_probability"
    )
    validation_embedding = LATE_FUSION.predict_embedding(
        embedding_frame,
        embedding_model,
        embedding_pooling,
        include_time_features,
        "validation",
        "embedding_probability",
    )
    validation_paired = LATE_FUSION.paired_predictions(
        validation_tree, validation_embedding, id_column
    )
    fusion_model = LATE_FUSION.fit_fusion_model(validation_paired, fusion_features)
    test_tree = LATE_FUSION.predict_tree(
        tree_frame, features, tree_model, "test", "tree_probability"
    )
    test_embedding = LATE_FUSION.predict_embedding(
        embedding_frame,
        embedding_model,
        embedding_pooling,
        include_time_features,
        "test",
        "embedding_probability",
    )
    test_paired = LATE_FUSION.paired_predictions(test_tree, test_embedding, id_column)
    test_paired["fusion_probability"] = LATE_FUSION.predict_fusion(
        test_paired, fusion_model, fusion_features
    )
    metadata = {
        "outcome_id": outcome_id,
        "mask_mode": mask_mode,
        "selected_tree_model": tree_model_name,
        "selected_embedding_model": embedding_model_name,
        "selected_embedding_pooling": embedding_pooling,
        "tree_validation_origin_mean_auc": tree_validation_auc,
        "embedding_validation_origin_mean_auc": embedding_validation_auc,
    }
    return test_paired, metadata


def c_index(frame: pd.DataFrame, probability_column: str, id_column: str) -> float:
    return float(
        LATE_FUSION.COMPARE.first_origin_c_index(
            frame, frame[probability_column].to_numpy(dtype=float), id_column
        )["c_index"]
    )


def score_paired(frame: pd.DataFrame, id_column: str, include_c_index: bool) -> dict[str, float]:
    values = {
        "tree_auc": LATE_FUSION.origin_mean_auc(frame, "tree_probability"),
        "embedding_auc": LATE_FUSION.origin_mean_auc(frame, "embedding_probability"),
        "fusion_auc": LATE_FUSION.origin_mean_auc(frame, "fusion_probability"),
    }
    values["delta_embedding_minus_tree"] = values["embedding_auc"] - values["tree_auc"]
    values["delta_fusion_minus_tree"] = values["fusion_auc"] - values["tree_auc"]
    values["delta_fusion_minus_embedding"] = values["fusion_auc"] - values["embedding_auc"]
    values["delta_fusion_minus_best_single"] = values["fusion_auc"] - max(
        values["tree_auc"], values["embedding_auc"]
    )
    if include_c_index:
        values["tree_c_index"] = c_index(frame, "tree_probability", id_column)
        values["embedding_c_index"] = c_index(frame, "embedding_probability", id_column)
        values["fusion_c_index"] = c_index(frame, "fusion_probability", id_column)
        values["delta_c_embedding_minus_tree"] = (
            values["embedding_c_index"] - values["tree_c_index"]
        )
        values["delta_c_fusion_minus_tree"] = values["fusion_c_index"] - values["tree_c_index"]
        values["delta_c_fusion_minus_embedding"] = (
            values["fusion_c_index"] - values["embedding_c_index"]
        )
    return values


def id_cluster_bootstrap(
    outcome_id: str, paired: pd.DataFrame, id_column: str, iterations: int, seed: int
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
        row = score_paired(sample, id_column, include_c_index=False)
        row.update(
            {
                "outcome_id": outcome_id,
                "iteration": iteration,
                "sampled_ids": int(len(sampled_ids)),
                "sampled_rows": int(len(sample)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def repeated_split_stability(
    outcome_id: str,
    tree_base: pd.DataFrame,
    embedding_base: pd.DataFrame,
    features: list[str],
    mask_mode: str,
    supervised_config: dict[str, Any],
    manifest: dict[str, Any],
    mlm_config: dict[str, Any],
    stability_config: dict[str, Any],
) -> pd.DataFrame:
    id_column = manifest["metadata"]["id_column"]
    rows: list[dict[str, Any]] = []
    for repeat in range(int(stability_config["repeated_splits"])):
        split_seed = int(stability_config["seed"]) + repeat
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
            print(f"Skipping {outcome_id} repeat {repeat}: one split has one class")
            continue
        paired, metadata = train_and_predict(
            outcome_id,
            tree_frame,
            embedding_frame,
            features,
            supervised_config,
            manifest,
            mlm_config,
            mask_mode,
        )
        row = score_paired(paired, id_column, include_c_index=True)
        row.update(metadata)
        row.update(
            {
                "repeat": repeat,
                "seed": split_seed,
                "test_rows": int(len(paired)),
                "test_ids": int(paired[id_column].nunique()),
                "test_events": int(paired["event_next_wave"].sum()),
            }
        )
        rows.append(row)
        print(
            f"{outcome_id} repeat {repeat}: tree={metadata['selected_tree_model']} "
            f"embedding={metadata['selected_embedding_model']}/{metadata['selected_embedding_pooling']} "
            f"fusion-tree={row['delta_fusion_minus_tree']:.4f} fusion-embedding={row['delta_fusion_minus_embedding']:.4f}"
        )
    return pd.DataFrame(rows)


def summary_row(
    values: pd.Series, measure: str, outcome_id: str, source: str, threshold: float
) -> dict[str, Any]:
    array = values.to_numpy(dtype=float)
    clean = array[~np.isnan(array)]
    low, high = percentile_interval(clean)
    is_delta = measure.startswith("delta")
    return {
        "outcome_id": outcome_id,
        "source": source,
        "measure": measure,
        "n": int(clean.size),
        "mean": float(np.mean(clean)) if clean.size else math.nan,
        "median": float(np.median(clean)) if clean.size else math.nan,
        "ci_low": low,
        "ci_high": high,
        "p_gt_0": float(np.mean(clean > 0)) if is_delta and clean.size else math.nan,
        "p_gt_threshold": (
            float(np.mean(clean > threshold)) if is_delta and clean.size else math.nan
        ),
    }


def build_summary(
    original: pd.DataFrame, bootstrap: pd.DataFrame, repeated: pd.DataFrame, threshold: float
) -> pd.DataFrame:
    measures = [
        "tree_auc",
        "embedding_auc",
        "fusion_auc",
        "delta_embedding_minus_tree",
        "delta_fusion_minus_tree",
        "delta_fusion_minus_embedding",
        "delta_fusion_minus_best_single",
        "tree_c_index",
        "embedding_c_index",
        "fusion_c_index",
        "delta_c_embedding_minus_tree",
        "delta_c_fusion_minus_tree",
        "delta_c_fusion_minus_embedding",
    ]
    rows: list[dict[str, Any]] = []
    for outcome_id in sorted(original["outcome_id"].unique()):
        original_row = original[original["outcome_id"].eq(outcome_id)].iloc[0]
        for measure in measures:
            if measure in original_row.index:
                rows.append(
                    {
                        "outcome_id": outcome_id,
                        "source": "original",
                        "measure": measure,
                        "n": 1,
                        "mean": float(original_row[measure]),
                        "median": float(original_row[measure]),
                        "ci_low": math.nan,
                        "ci_high": math.nan,
                        "p_gt_0": math.nan,
                        "p_gt_threshold": math.nan,
                    }
                )
        boot_subset = bootstrap[bootstrap["outcome_id"].eq(outcome_id)]
        rep_subset = repeated[repeated["outcome_id"].eq(outcome_id)]
        for measure in measures:
            if measure in boot_subset:
                rows.append(
                    summary_row(boot_subset[measure], measure, outcome_id, "bootstrap", threshold)
                )
            if measure in rep_subset:
                rows.append(
                    summary_row(
                        rep_subset[measure], measure, outcome_id, "repeated_split", threshold
                    )
                )
    return pd.DataFrame(rows)


def summary_lookup(summary: pd.DataFrame, outcome_id: str, source: str, measure: str) -> pd.Series:
    subset = summary[
        summary["outcome_id"].eq(outcome_id)
        & summary["source"].eq(source)
        & summary["measure"].eq(measure)
    ]
    if subset.empty:
        return pd.Series(dtype=object)
    return subset.iloc[0]


def write_report(
    path: Path,
    original: pd.DataFrame,
    summary: pd.DataFrame,
    repeated: pd.DataFrame,
    threshold: float,
) -> None:
    lines = [
        "# CHARLS Late-Fusion Stability Report",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "- Original split uses the locked test IDs from the rolling person-period matrix.",
        "- Bootstrap is ID-cluster resampling of the original test split and reports per-origin AUC intervals.",
        "- Repeated split redoes ID-level train/validation/test splits, model selection, fusion fitting, and test evaluation.",
        "",
        "## Original Test Split",
        "",
        "| outcome_id | tree | embedding | tree+embedding | emb-tree | fusion-tree | fusion-embedding | tree_c | embedding_c | fusion_c |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in original.iterrows():
        lines.append(
            f"| {row['outcome_id']} | {fmt(row['tree_auc'])} | {fmt(row['embedding_auc'])} | {fmt(row['fusion_auc'])} | "
            f"{fmt(row['delta_embedding_minus_tree'])} | {fmt(row['delta_fusion_minus_tree'])} | {fmt(row['delta_fusion_minus_embedding'])} | "
            f"{fmt(row['tree_c_index'])} | {fmt(row['embedding_c_index'])} | {fmt(row['fusion_c_index'])} |"
        )
    lines.extend(
        [
            "",
            "## Stability Of Key Deltas",
            "",
            f"`P_gt_threshold` uses threshold = {threshold:.3f}.",
            "",
            "| outcome_id | delta | bootstrap_mean | bootstrap_95ci | bootstrap_P>0 | bootstrap_P>thr | repeated_mean | repeated_95ci | repeated_P>0 | repeated_P>thr |",
            "|---|---|---:|---|---:|---:|---:|---|---:|---:|",
        ]
    )
    key_measures = [
        "delta_embedding_minus_tree",
        "delta_fusion_minus_tree",
        "delta_fusion_minus_embedding",
        "delta_fusion_minus_best_single",
    ]
    for outcome_id in original["outcome_id"]:
        for measure in key_measures:
            boot = summary_lookup(summary, outcome_id, "bootstrap", measure)
            rep = summary_lookup(summary, outcome_id, "repeated_split", measure)
            lines.append(
                f"| {outcome_id} | {measure} | {fmt(boot.get('mean', math.nan))} | "
                f"[{fmt(boot.get('ci_low', math.nan))}, {fmt(boot.get('ci_high', math.nan))}] | "
                f"{fmt(boot.get('p_gt_0', math.nan), 3)} | {fmt(boot.get('p_gt_threshold', math.nan), 3)} | "
                f"{fmt(rep.get('mean', math.nan))} | [{fmt(rep.get('ci_low', math.nan))}, {fmt(rep.get('ci_high', math.nan))}] | "
                f"{fmt(rep.get('p_gt_0', math.nan), 3)} | {fmt(rep.get('p_gt_threshold', math.nan), 3)} |"
            )
    lines.extend(
        [
            "",
            "## Repeated Split Model Selection",
            "",
            "| outcome_id | selected_tree_model | count |",
            "|---|---|---:|",
        ]
    )
    for (outcome_id, model_name), count in (
        repeated.groupby(["outcome_id", "selected_tree_model"]).size().sort_index().items()
    ):
        lines.append(f"| {outcome_id} | {model_name} | {int(count)} |")
    lines.extend(["", "| outcome_id | selected_embedding | count |", "|---|---|---:|"])
    repeated = repeated.copy()
    repeated["selected_embedding"] = (
        repeated["selected_embedding_model"] + "/" + repeated["selected_embedding_pooling"]
    )
    for (outcome_id, model_name), count in (
        repeated.groupby(["outcome_id", "selected_embedding"]).size().sort_index().items()
    ):
        lines.append(f"| {outcome_id} | {model_name} | {int(count)} |")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlm-config", type=Path, default=DEFAULT_MLM_CONFIG)
    parser.add_argument("--supervised-config", type=Path, default=DEFAULT_SUPERVISED_CONFIG)
    parser.add_argument("--outcomes", nargs="+")
    parser.add_argument("--bootstrap-iterations", type=int)
    parser.add_argument("--repeated-splits", type=int)
    args = parser.parse_args()

    mlm_config = load_yaml(args.mlm_config)
    supervised_config = load_yaml(args.supervised_config)
    stability_config = dict(mlm_config["late_fusion_stability"])
    if args.outcomes:
        stability_config["outcomes"] = args.outcomes
    if args.bootstrap_iterations is not None:
        stability_config["bootstrap_iterations"] = args.bootstrap_iterations
    if args.repeated_splits is not None:
        stability_config["repeated_splits"] = args.repeated_splits

    manifest = load_json(resolve_path(mlm_config["inputs"]["supervised_matrix_manifest"]))
    id_column = manifest["metadata"]["id_column"]
    registry = pd.read_csv(
        resolve_path("outputs/02_outcome_registry/outcome_registry_validated.csv")
    )
    domain_by_outcome = registry.set_index("outcome_id")["domain"].to_dict()
    matrix = pd.read_parquet(resolve_path(mlm_config["inputs"]["supervised_outcome_matrix"]))
    embedding_cache: dict[Path, pd.DataFrame] = {}
    original_rows: list[dict[str, Any]] = []
    bootstrap_frames: list[pd.DataFrame] = []
    repeated_frames: list[pd.DataFrame] = []

    for outcome_offset, outcome_id in enumerate(stability_config["outcomes"]):
        print(f"Late-fusion stability for {outcome_id}")
        tree_base, embedding_base, features, mask_mode = prepare_outcome_frames(
            outcome_id,
            matrix,
            manifest,
            domain_by_outcome,
            mlm_config,
            embedding_cache,
        )
        paired, metadata = train_and_predict(
            outcome_id,
            tree_base,
            embedding_base,
            features,
            supervised_config,
            manifest,
            mlm_config,
            mask_mode,
        )
        original = score_paired(paired, id_column, include_c_index=True)
        original.update(metadata)
        original.update(
            {
                "test_rows": int(len(paired)),
                "test_ids": int(paired[id_column].nunique()),
                "test_events": int(paired["event_next_wave"].sum()),
            }
        )
        original_rows.append(original)
        bootstrap_frames.append(
            id_cluster_bootstrap(
                outcome_id,
                paired,
                id_column,
                int(stability_config["bootstrap_iterations"]),
                int(stability_config["seed"]) + outcome_offset * 10_000,
            )
        )
        repeated_frames.append(
            repeated_split_stability(
                outcome_id,
                tree_base,
                embedding_base,
                features,
                mask_mode,
                supervised_config,
                manifest,
                mlm_config,
                stability_config,
            )
        )

    original_df = pd.DataFrame(original_rows)
    bootstrap_df = pd.concat(bootstrap_frames, ignore_index=True)
    repeated_df = pd.concat(repeated_frames, ignore_index=True)
    threshold = float(mlm_config["comparison"].get("fm_delta_threshold", 0.01))
    summary_df = build_summary(original_df, bootstrap_df, repeated_df, threshold)

    bootstrap_path = resolve_path(mlm_config["outputs"]["late_fusion_stability_bootstrap"])
    repeated_path = resolve_path(mlm_config["outputs"]["late_fusion_stability_repeated_splits"])
    summary_path = resolve_path(mlm_config["outputs"]["late_fusion_stability_summary"])
    report_path = resolve_path(mlm_config["outputs"]["late_fusion_stability_report"])
    bootstrap_path.parent.mkdir(parents=True, exist_ok=True)
    bootstrap_df.to_csv(bootstrap_path, index=False)
    repeated_df.to_csv(repeated_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    write_report(report_path, original_df, summary_df, repeated_df, threshold)
    print(
        json.dumps(
            {
                "outcomes": list(stability_config["outcomes"]),
                "bootstrap": str(bootstrap_path.relative_to(PROJECT_ROOT)),
                "repeated_splits": str(repeated_path.relative_to(PROJECT_ROOT)),
                "summary": str(summary_path.relative_to(PROJECT_ROOT)),
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
