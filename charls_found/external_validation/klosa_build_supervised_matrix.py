#!/usr/bin/env python3
"""Build KLoSA leakage-filtered rolling person-period supervised matrices."""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import pandas as pd

from klosa_utils import (
    API_BASE,
    ID_COLUMN,
    MAIN_FILE,
    ORIGIN_WAVES,
    OUTPUT_DIR,
    PROJECT_ROOT,
    SELECTED_OUTCOMES,
    SOURCE_ID,
    WAVE_COLUMN,
    WAVE_YEARS,
    build_person_period_rows,
    build_time_to_event_matrix,
    extract_columns,
    fmt,
    generated_timestamp,
    make_group_split,
    transition_audit,
    validation_window,
    verify_group_splits,
    write_text,
)


def safe_predictors(leakage: pd.DataFrame) -> dict[str, list[str]]:
    safe_rows = leakage[
        leakage["outcome_id"].isin(SELECTED_OUTCOMES)
        & leakage["outcome_usable"].astype(bool)
        & leakage["decision"].eq("include")
        & leakage["reason"].eq("safe_baseline_predictor")
        & leakage["usable_as_baseline_predictor_after_review"].astype(bool)
    ].copy()
    if safe_rows.empty:
        raise RuntimeError("No KLoSA safe predictors found")
    predictors = {
        outcome_id: safe_rows.loc[safe_rows["outcome_id"].eq(outcome_id), "variable"]
        .drop_duplicates()
        .sort_values()
        .tolist()
        for outcome_id in SELECTED_OUTCOMES
    }
    empty = [outcome_id for outcome_id, variables in predictors.items() if not variables]
    if empty:
        raise RuntimeError(f"KLoSA outcomes without safe predictors: {empty}")
    return predictors


def write_transition_audit_report(audit: pd.DataFrame) -> None:
    lines = [
        "# KLoSA Person-Period Transition Audit",
        "",
        f"Generated: {generated_timestamp()}",
        "",
        "| outcome_id | origin_wave | next_wave | rows | events | event_rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in audit.iterrows():
        lines.append(
            f"| {row['outcome_id']} | {int(row['origin_wave'])} | {int(row['next_wave'])} | {int(row['rows'])} | {int(row['events'])} | {fmt(row['event_rate'])} |"
        )
    lines.extend(
        [
            "",
            "Rows are adjacent KLoSA transitions where the respondent is observed and still at risk at the origin wave, and the next-wave label is observed.",
            "An observed event at `t+1` removes the respondent from later origin waves for the same outcome.",
            "",
        ]
    )
    write_text(OUTPUT_DIR / "klosa_transition_audit_report.md", lines)


def write_matrix_report(
    matrix: pd.DataFrame,
    time_to_event: pd.DataFrame,
    audit: pd.DataFrame,
    per_outcome_predictors: dict[str, list[str]],
) -> None:
    split_rows = (
        matrix.groupby("split")
        .agg(row_count=(ID_COLUMN, "size"), ids=(ID_COLUMN, "nunique"))
        .reset_index()
    )
    outcome_rows = []
    for outcome_id in SELECTED_OUTCOMES:
        subset = matrix[matrix["outcome_id"].eq(outcome_id)]
        survival = time_to_event[time_to_event["outcome_id"].eq(outcome_id)]
        outcome_rows.append(
            {
                "outcome_id": outcome_id,
                "rows": int(len(subset)),
                "ids": int(subset[ID_COLUMN].nunique()),
                "events": int(subset["event_next_wave"].sum()),
                "event_rate": float(subset["event_next_wave"].mean()),
                "survival_ids": int(survival[ID_COLUMN].nunique()),
                "safe_predictors": len(per_outcome_predictors[outcome_id]),
            }
        )
    outcome_df = pd.DataFrame(outcome_rows)

    lines = [
        "# KLoSA Person-Period Supervised Matrix Report",
        "",
        f"Generated: {generated_timestamp()}",
        "",
        "## Summary",
        "",
        f"- Person-period rows: {len(matrix)}",
        f"- Unique respondent IDs: {matrix[ID_COLUMN].nunique()}",
        f"- Selected outcomes: {len(SELECTED_OUTCOMES)}",
        f"- Feature columns: {len([col for col in matrix.columns if col.startswith('feature__')])}",
        "- Matrix unit: one at-risk adjacent transition per `(pid, outcome_id, origin_wave)`.",
        "- Split: `GroupShuffleSplit` by KLoSA person ID; all origins and outcomes for an ID share one split.",
        "- Features come from wave t only; labels come from adjacent wave t+1.",
        "",
        "## Split Counts",
        "",
        "| split | row_count | ids |",
        "|---|---:|---:|",
    ]
    for _, row in split_rows.iterrows():
        lines.append(f"| {row['split']} | {int(row['row_count'])} | {int(row['ids'])} |")

    lines.extend(
        [
            "",
            "## Outcome Targets",
            "",
            "| outcome_id | rows | ids | events | event_rate | survival_ids | safe_predictors |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in outcome_df.iterrows():
        lines.append(
            f"| {row['outcome_id']} | {row['rows']} | {row['ids']} | {row['events']} | {fmt(row['event_rate'])} | {row['survival_ids']} | {row['safe_predictors']} |"
        )

    lines.extend(
        [
            "",
            "## Transition Counts",
            "",
            "| outcome_id | origin_wave | next_wave | rows | events | event_rate |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in audit.iterrows():
        lines.append(
            f"| {row['outcome_id']} | {int(row['origin_wave'])} | {int(row['next_wave'])} | {int(row['rows'])} | {int(row['events'])} | {fmt(row['event_rate'])} |"
        )

    lines.extend(
        [
            "",
            "Guardrails:",
            "",
            "- Respondents exit an outcome risk set after the first observed event.",
            "- Adjacent transitions with missing origin status or missing next-wave label are not treated as negatives.",
            "- KLoSA diagnostic/treatment leakage exclusions are applied per outcome before features are chosen.",
            "- Cross-split `pid` overlap is verified before writing outputs.",
            "",
        ]
    )
    write_text(OUTPUT_DIR / "klosa_supervised_matrix_report.md", lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", default=API_BASE)
    parser.add_argument("--row-limit", type=int, default=1000000)
    parser.add_argument("--chunk-size", type=int, default=70)
    parser.add_argument("--seed", type=int, default=20260625)
    args = parser.parse_args()

    person_wave_path = OUTPUT_DIR / "klosa_person_wave.parquet"
    leakage_path = OUTPUT_DIR / "klosa_per_outcome_leakage.csv"
    variable_map_path = OUTPUT_DIR / "klosa_variable_type_map.csv"
    if not person_wave_path.exists() or not leakage_path.exists() or not variable_map_path.exists():
        raise RuntimeError("Run KLoSA scripts 02 and 03 before building the supervised matrix")

    person_wave = pd.read_parquet(person_wave_path)
    leakage = pd.read_csv(leakage_path)
    variable_map = pd.read_csv(variable_map_path)
    per_outcome_predictors = safe_predictors(leakage)
    predictor_variables = sorted(
        set().union(*[set(values) for values in per_outcome_predictors.values()])
    )

    source_df = extract_columns(
        predictor_variables,
        row_limit=args.row_limit,
        api_base=args.api_base,
        chunk_size=args.chunk_size,
    )
    source_df = validation_window(source_df)
    matrix = build_person_period_rows(
        person_wave, selected_outcomes=SELECTED_OUTCOMES, origin_waves=ORIGIN_WAVES
    )

    feature_frame = source_df[[ID_COLUMN, WAVE_COLUMN] + predictor_variables].copy()
    feature_frame = feature_frame.rename(
        columns={
            WAVE_COLUMN: "origin_wave",
            **{variable: f"feature__{variable}" for variable in predictor_variables},
        }
    )
    matrix = matrix.merge(
        feature_frame, on=[ID_COLUMN, "origin_wave"], how="left", validate="many_to_one"
    )

    split_map = make_group_split(
        matrix[ID_COLUMN], seed=args.seed, train_ratio=0.8, validation_ratio=0.1
    )
    matrix.insert(2, "split", matrix[ID_COLUMN].map(split_map))
    if matrix["split"].isna().any():
        raise RuntimeError("Some KLoSA person-period rows did not receive a split")
    verify_group_splits(matrix, ID_COLUMN)

    time_to_event = build_time_to_event_matrix(matrix)
    audit = transition_audit(matrix)

    matrix_path = OUTPUT_DIR / "klosa_supervised_outcome_matrix.parquet"
    time_to_event_path = OUTPUT_DIR / "klosa_time_to_event_matrix.parquet"
    audit_path = OUTPUT_DIR / "klosa_transition_audit.csv"
    manifest_path = OUTPUT_DIR / "klosa_supervised_matrix_manifest.json"
    matrix.to_parquet(matrix_path, index=False)
    time_to_event.to_parquet(time_to_event_path, index=False)
    audit.to_csv(audit_path, index=False)
    write_transition_audit_report(audit)

    type_by_variable = variable_map.set_index("variable")["type_model"].to_dict()
    tokenization_by_variable = variable_map.set_index("variable")["tokenization_type"].to_dict()
    feature_type_by_variable = {
        variable: type_by_variable.get(variable, "categorical_numeric")
        for variable in predictor_variables
    }
    feature_type_by_variable.update({"origin_wave": "categorical", "interval_years": "continuous"})
    split_counts = {
        split: {
            "rows": int(subset.shape[0]),
            "ids": int(subset[ID_COLUMN].nunique()),
        }
        for split, subset in matrix.groupby("split")
    }
    manifest: dict[str, Any] = {
        "metadata": {
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "matrix_type": "klosa_person_period_discrete_time_hazard",
            "source_id": SOURCE_ID,
            "main_file": MAIN_FILE,
            "api_base": args.api_base,
            "row_limit": args.row_limit,
            "id_column": ID_COLUMN,
            "wave_column": WAVE_COLUMN,
            "origin_waves": ORIGIN_WAVES,
            "wave_years": WAVE_YEARS,
            "split_method": "GroupShuffleSplit by pid",
            "feature_selection": "KLoSA per-outcome leakage table, decision=include, reason=safe_baseline_predictor",
        },
        "selected_outcomes": SELECTED_OUTCOMES,
        "feature_columns": [f"feature__{variable}" for variable in predictor_variables],
        "feature_variables": predictor_variables,
        "time_features": ["origin_wave", "interval_years"],
        "feature_type_by_variable": feature_type_by_variable,
        "tokenization_type_by_variable": {
            variable: tokenization_by_variable.get(variable, "") for variable in predictor_variables
        },
        "per_outcome_predictors": per_outcome_predictors,
        "target_columns": {
            outcome_id: {
                "row_filter": f"outcome_id == '{outcome_id}'",
                "event": "event_next_wave",
                "time": "interval_years",
            }
            for outcome_id in SELECTED_OUTCOMES
        },
        "split_counts": split_counts,
        "outputs": {
            "person_period_matrix": str(matrix_path.relative_to(PROJECT_ROOT)),
            "time_to_event_matrix": str(time_to_event_path.relative_to(PROJECT_ROOT)),
            "transition_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_matrix_report(matrix, time_to_event, audit, per_outcome_predictors)

    print(
        {
            "rows": int(len(matrix)),
            "ids": int(matrix[ID_COLUMN].nunique()),
            "features": len(predictor_variables),
            "matrix": str(matrix_path.relative_to(PROJECT_ROOT)),
            "manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
