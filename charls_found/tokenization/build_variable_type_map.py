#!/usr/bin/env python3
"""Build the first CHARLS tokenization variable-role map."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_CONFIG = SCRIPT_PATH.parents[1] / "configs" / "tokenization.yaml"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def split_vars(value: Any) -> list[str]:
    if pd.isna(value) or str(value).strip() == "":
        return []
    return [part.strip() for part in str(value).split("|") if part.strip()]


def classify_tokenization_type(
    row: pd.Series,
    has_value_labels: bool,
    high_missing: bool,
    high_cardinality_allowlist: set[str],
) -> tuple[str, str]:
    variable = str(row["variable"])
    module = str(row["module"])
    type_model = str(row["type_model"])
    leakage = str(row["leakage_risk_level"])
    unique_count = int(row.get("unique_count", 0) or 0)

    if bool(row.get("is_identifier", False)) or leakage == "identifier":
        return "exclude_identifier", "identifier variables are not tokenized"
    if bool(row.get("is_time_variable", False)) or module == "wave/time":
        if variable == "wave":
            return "wave_context", "wave is encoded as context only"
        return "exclude_time", "time variables are not ordinary trait tokens"
    if module == "mortality/death timing":
        return "exclude_timing_review", "mortality timing semantics are not validated"
    if high_missing:
        return "exclude_high_missing", "missingness is above v1 threshold"
    if has_value_labels:
        return "categorical_value_label", "categorical values use verified value-label dictionary"
    if type_model in {"binary_or_boolean", "categorical_numeric", "categorical_text"}:
        return "categorical_raw_code", "categorical values use observed raw codes"
    if variable in high_cardinality_allowlist:
        return (
            "categorical_high_cardinality_context",
            "allowlisted high-cardinality context variable",
        )
    if type_model == "continuous":
        return "continuous_quantile_bin", "continuous values require global quantile bins"
    if unique_count <= 20:
        return "categorical_raw_code", "low-cardinality variable treated as categorical"
    return "review_needed", "manual review needed before tokenization"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    config = load_yaml(args.config)
    schema = pd.read_csv(resolve_path(config["inputs"]["schema"]))
    value_labels = pd.read_csv(resolve_path(config["inputs"]["value_labels"]))
    registry = pd.read_csv(resolve_path(config["inputs"]["outcome_registry"]))
    leakage = pd.read_csv(resolve_path(config["inputs"]["leakage_matrix"]))

    usable_registry = registry[registry["usable"].astype(bool)].copy()
    usable_outcome_source: dict[str, list[str]] = {}
    all_outcome_source: dict[str, list[str]] = {}
    for _, row in registry.iterrows():
        for variable in split_vars(row["source_variables"]):
            all_outcome_source.setdefault(variable, []).append(str(row["outcome_id"]))
            if bool(row.get("usable", False)):
                usable_outcome_source.setdefault(variable, []).append(str(row["outcome_id"]))

    label_counts = value_labels.groupby("variable").size().rename("full_value_label_count")
    schema = schema.merge(label_counts, on="variable", how="left")
    schema["full_value_label_count"] = schema["full_value_label_count"].fillna(0).astype(int)

    usable_leakage = leakage[leakage["outcome_usable"].astype(bool)].copy()
    decision_counts = (
        usable_leakage.groupby(["variable", "decision"]).size().unstack(fill_value=0).reset_index()
    )
    for column in ["include", "review", "exclude"]:
        if column not in decision_counts:
            decision_counts[column] = 0
    decision_counts = decision_counts.rename(
        columns={
            "include": "usable_outcome_include_count",
            "review": "usable_outcome_review_count",
            "exclude": "usable_outcome_exclude_count",
        }
    )
    schema = schema.merge(decision_counts, on="variable", how="left")
    for column in [
        "usable_outcome_include_count",
        "usable_outcome_review_count",
        "usable_outcome_exclude_count",
    ]:
        schema[column] = schema[column].fillna(0).astype(int)

    missing_threshold = float(config["thresholds"]["v1_exclude_missing_rate_ge"])
    rows = []
    high_cardinality_allowlist = set(
        config.get("rules", {}).get("high_cardinality_categorical_allowlist", [])
    )
    for _, row in schema.iterrows():
        variable = str(row["variable"])
        missing_rate = float(row.get("missing_rate_overall", 0) or 0)
        high_missing = missing_rate >= missing_threshold
        has_value_labels = int(row["full_value_label_count"]) > 0
        tokenization_type, note = classify_tokenization_type(
            row, has_value_labels, high_missing, high_cardinality_allowlist
        )
        include_in_tokenization_v1 = (
            not tokenization_type.startswith("exclude_") and tokenization_type != "wave_context"
        )
        include_as_wave_context = tokenization_type == "wave_context"
        include_in_pretraining_v1 = include_in_tokenization_v1
        baseline_predictor_general = (
            int(row["usable_outcome_include_count"]) > 0 and include_in_tokenization_v1
        )
        direct_usable_outcome_source = usable_outcome_source.get(variable, [])
        direct_any_outcome_source = all_outcome_source.get(variable, [])
        direct_target_source = len(direct_usable_outcome_source) > 0
        requires_per_outcome_filter = (
            int(row["usable_outcome_review_count"]) > 0
            or int(row["usable_outcome_exclude_count"]) > 0
            or direct_target_source
        )
        if tokenization_type == "wave_context":
            review_status = "context_only"
        elif tokenization_type.startswith("exclude_"):
            review_status = "exclude"
        elif tokenization_type == "review_needed":
            review_status = "review"
        elif requires_per_outcome_filter:
            review_status = "tokenize_with_modeling_filter"
        else:
            review_status = "tokenize"

        rows.append(
            {
                "variable": variable,
                "label_zh": row.get("label_zh"),
                "module": row.get("module"),
                "type_raw": row.get("type_raw"),
                "type_index": row.get("type_index"),
                "type_model": row.get("type_model"),
                "missing_rate_overall": missing_rate,
                "unique_count": int(row.get("unique_count", 0) or 0),
                "value_label_count_preview": row.get("value_label_count"),
                "full_value_label_count": int(row["full_value_label_count"]),
                "tokenization_type": tokenization_type,
                "tokenization_note": note,
                "include_in_tokenization_v1": include_in_tokenization_v1,
                "include_in_pretraining_v1": include_in_pretraining_v1,
                "include_as_wave_context": include_as_wave_context,
                "baseline_predictor_general": baseline_predictor_general,
                "requires_per_outcome_filter": requires_per_outcome_filter,
                "direct_usable_outcome_source": direct_target_source,
                "usable_outcome_source_count": len(direct_usable_outcome_source),
                "usable_outcome_source_ids": "|".join(sorted(set(direct_usable_outcome_source))),
                "any_outcome_source_ids": "|".join(sorted(set(direct_any_outcome_source))),
                "usable_outcome_include_count": int(row["usable_outcome_include_count"]),
                "usable_outcome_review_count": int(row["usable_outcome_review_count"]),
                "usable_outcome_exclude_count": int(row["usable_outcome_exclude_count"]),
                "review_status": review_status,
            }
        )

    variable_map = pd.DataFrame(rows)
    out_path = resolve_path(config["outputs"]["variable_type_map"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    variable_map.to_csv(out_path, index=False)

    report_path = resolve_path(config["outputs"]["variable_type_map_report"])
    type_counts = variable_map["tokenization_type"].value_counts().reset_index()
    status_counts = variable_map["review_status"].value_counts().reset_index()
    module_counts = (
        variable_map[variable_map["include_in_tokenization_v1"]]
        .groupby("module")
        .size()
        .sort_values(ascending=False)
        .reset_index(name="tokenized_variable_count")
    )
    high_missing = variable_map[variable_map["tokenization_type"] == "exclude_high_missing"]
    lines = [
        "# CHARLS Variable Type Map Report",
        "",
        "## Inputs",
        "",
        f"- Schema: `{config['inputs']['schema']}`",
        f"- Value labels: `{config['inputs']['value_labels']}`",
        f"- Outcome registry: `{config['inputs']['outcome_registry']}`",
        f"- Leakage matrix: `{config['inputs']['leakage_matrix']}`",
        "",
        "## Summary",
        "",
        f"- Total variables: {len(variable_map)}",
        f"- Included in v1 tokenization: {int(variable_map['include_in_tokenization_v1'].sum())}",
        f"- Included as wave context only: {int(variable_map['include_as_wave_context'].sum())}",
        f"- Included in v1 self-supervised pretraining: {int(variable_map['include_in_pretraining_v1'].sum())}",
        f"- General baseline predictor candidates: {int(variable_map['baseline_predictor_general'].sum())}",
        f"- Direct usable outcome source variables: {int(variable_map['direct_usable_outcome_source'].sum())}",
        f"- Variables requiring per-outcome modeling filter: {int(variable_map['requires_per_outcome_filter'].sum())}",
        "",
        "## Tokenization Type Counts",
        "",
        "| tokenization_type | variable_count |",
        "|---|---:|",
    ]
    for _, count_row in type_counts.iterrows():
        lines.append(f"| {count_row['tokenization_type']} | {count_row['count']} |")
    lines.extend(
        ["", "## Review Status Counts", "", "| review_status | variable_count |", "|---|---:|"]
    )
    for _, count_row in status_counts.iterrows():
        lines.append(f"| {count_row['review_status']} | {count_row['count']} |")
    lines.extend(
        [
            "",
            "## Included Variables by Module",
            "",
            "| module | tokenized_variable_count |",
            "|---|---:|",
        ]
    )
    for _, module_row in module_counts.iterrows():
        lines.append(f"| {module_row['module']} | {module_row['tokenized_variable_count']} |")
    lines.extend(["", "## High-Missing Exclusions", ""])
    if high_missing.empty:
        lines.append("No variables exceeded the v1 missingness threshold.")
    else:
        lines.extend(["| variable | label | missing_rate |", "|---|---|---:|"])
        for _, hm_row in (
            high_missing.sort_values("missing_rate_overall", ascending=False).head(30).iterrows()
        ):
            label = str(hm_row.get("label_zh", "")).replace("|", "/")
            lines.append(
                f"| {hm_row['variable']} | {label} | {hm_row['missing_rate_overall']:.3f} |"
            )
    lines.extend(
        [
            "",
            "## Gate Before Next Script",
            "",
            "- Review variables with `review_status=review` or `review_status=tokenize_with_modeling_filter`.",
            "- Confirm that direct outcome source variables remain tokenizable for self-supervised pretraining but are filtered per outcome during supervised modeling.",
            "- Confirm that high-missing rare variables should be excluded from v1 tokenization rather than handled in a sensitivity run.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
