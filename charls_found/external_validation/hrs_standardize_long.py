#!/usr/bin/env python3
"""Standardize HRS person-wave data and recompute validation outcomes."""

from __future__ import annotations

import argparse
import math
from typing import Any

import pandas as pd

from hrs_utils import (
    API_BASE,
    HRS_COMMON_CHRONIC_USED,
    HRS_OUTCOME_SOURCES,
    ID_COLUMN,
    INTERVIEW_YEAR_COLUMN,
    OUTPUT_DIR,
    OUTCOME_DOMAINS,
    OUTCOME_NAMES,
    PROJECT_ROOT,
    SELECTED_OUTCOMES,
    VALIDATION_WAVES,
    WAVE_COLUMN,
    add_outcome_status_columns,
    extract_columns,
    fmt,
    generated_timestamp,
    infer_depression_threshold,
    infer_low_cognition_threshold,
    to_numeric,
    validation_window,
    write_text,
)


def split_vars(value: Any) -> list[str]:
    if pd.isna(value) or str(value).strip() == "":
        return []
    return [part.strip() for part in str(value).split("|") if part.strip()]


def build_outcome_registry(
    person_wave: pd.DataFrame, depression_threshold: int, low_cog_threshold: float
) -> pd.DataFrame:
    rows = []
    followup_waves = [wave for wave in VALIDATION_WAVES if wave != 10]
    for outcome_id in SELECTED_OUTCOMES:
        status_col = f"status__{outcome_id}"
        wide = person_wave.pivot(index=ID_COLUMN, columns=WAVE_COLUMN, values=status_col)
        baseline = wide.get(10, pd.Series(dtype="float64"))
        follow = wide[[wave for wave in followup_waves if wave in wide.columns]]
        follow_observed = follow.notna().any(axis=1)
        follow_positive = follow.eq(1).any(axis=1)
        baseline_observed = baseline.notna()
        eligible = baseline_observed & baseline.eq(0) & follow_observed
        events = eligible & follow_positive
        n_events = int(events.sum())
        denominator = int(eligible.sum())
        event_rate = float(n_events / denominator) if denominator else math.nan

        event_definition = (
            "Baseline-negative at HRS wave 10 and any follow-up positive in waves 11-15."
        )
        if outcome_id == "high_depressive_symptoms_onset":
            event_definition = f"Baseline CESD < {depression_threshold} and any follow-up CESD >= {depression_threshold} or depressive=1."
        elif outcome_id == "low_cognition_onset_data_driven":
            event_definition = f"Baseline cog27 above wave-10 Q25 ({low_cog_threshold:g}) and any follow-up cog27 <= threshold."
        elif outcome_id == "multimorbidity_onset":
            event_definition = f"Baseline common-disease count <2 and any follow-up count >=2, using {'|'.join(HRS_COMMON_CHRONIC_USED)}."

        rows.append(
            {
                "outcome_id": outcome_id,
                "outcome_name": OUTCOME_NAMES[outcome_id],
                "domain": OUTCOME_DOMAINS[outcome_id],
                "source_variables": "|".join(HRS_OUTCOME_SOURCES[outcome_id]),
                "definition_type": "binary incident outcome",
                "baseline_wave": 10,
                "followup_waves": "|".join(str(wave) for wave in followup_waves),
                "baseline_eligibility": "baseline wave observed and baseline-negative",
                "event_definition": event_definition,
                "n_baseline": int(baseline_observed.sum()),
                "n_followup": denominator,
                "n_events": n_events,
                "event_rate": event_rate,
                "minimum_event_count_for_usable": 100,
                "usable": n_events >= 100,
                "notes": "HRS-native external validation definition; no CHARLS numeric thresholds copied.",
            }
        )
    return pd.DataFrame(rows)


def status_summary(person_wave: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for outcome_id in SELECTED_OUTCOMES:
        status_col = f"status__{outcome_id}"
        for wave, group in person_wave.groupby(WAVE_COLUMN, sort=True):
            status = to_numeric(group[status_col])
            rows.append(
                {
                    "outcome_id": outcome_id,
                    "wave": int(wave),
                    "rows": int(len(group)),
                    "observed": int(status.notna().sum()),
                    "positive": int(status.eq(1).sum()),
                    "positive_rate_observed": (
                        float(status.eq(1).sum() / status.notna().sum())
                        if status.notna().sum()
                        else math.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def write_standardization_report(
    person_wave: pd.DataFrame,
    registry: pd.DataFrame,
    status_audit: pd.DataFrame,
    depression_threshold: int,
    low_cog_threshold: float,
) -> None:
    lines = [
        "# HRS Person-Wave Standardization Report",
        "",
        f"Generated: {generated_timestamp()}",
        "",
        "## Summary",
        "",
        f"- Rows in validation window: {len(person_wave)}",
        f"- Unique HRS IDs: {person_wave[ID_COLUMN].nunique()}",
        f"- Waves: {', '.join(str(wave) for wave in sorted(person_wave[WAVE_COLUMN].dropna().unique()))}",
        f"- Depression threshold: HRS CESD >= {depression_threshold}",
        f"- Low-cognition threshold: HRS wave-10 `cog27` Q25 = {low_cog_threshold:g}",
        f"- Multimorbidity disease set: {'|'.join(HRS_COMMON_CHRONIC_USED)}",
        "",
        "## Outcome Registry",
        "",
        "| outcome_id | source_variables | n_baseline | n_followup | n_events | event_rate | usable |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for _, row in registry.iterrows():
        lines.append(
            f"| {row['outcome_id']} | {row['source_variables']} | {int(row['n_baseline'])} | {int(row['n_followup'])} | {int(row['n_events'])} | {fmt(row['event_rate'])} | {row['usable']} |"
        )

    lines.extend(
        [
            "",
            "## Status Counts By Wave",
            "",
            "| outcome_id | wave | observed | positive | positive_rate_observed |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in status_audit.iterrows():
        lines.append(
            f"| {row['outcome_id']} | {int(row['wave'])} | {int(row['observed'])} | {int(row['positive'])} | {fmt(row['positive_rate_observed'])} |"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "",
            "- `cogtotp` is not used because it is unavailable in the wave-10 baseline window.",
            "- HRS wave 10-15 is used for nominal 2010-2020 comparability with CHARLS 2011-2020.",
            "- HRS special missing values are already exposed as missing in the API-backed parquet for these harmonized variables.",
            "",
        ]
    )
    write_text(OUTPUT_DIR / "hrs_standardization_report.md", lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", default=API_BASE)
    parser.add_argument("--row-limit", type=int, default=300000)
    parser.add_argument("--chunk-size", type=int, default=70)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outcome_variables = sorted(
        {variable for variables in HRS_OUTCOME_SOURCES.values() for variable in variables}
    )
    base_columns = [ID_COLUMN, WAVE_COLUMN, INTERVIEW_YEAR_COLUMN, "ragey_e", "ragender"]
    frame = extract_columns(
        outcome_variables,
        row_limit=args.row_limit,
        api_base=args.api_base,
        chunk_size=args.chunk_size,
        base_columns=base_columns,
    )
    person_wave = validation_window(frame)
    depression_threshold = infer_depression_threshold(person_wave)
    low_cog_threshold = infer_low_cognition_threshold(person_wave, variable="cog27", quantile=0.25)
    person_wave = add_outcome_status_columns(person_wave, depression_threshold, low_cog_threshold)

    registry = build_outcome_registry(person_wave, depression_threshold, low_cog_threshold)
    status_audit = status_summary(person_wave)

    person_wave.to_parquet(OUTPUT_DIR / "hrs_person_wave.parquet", index=False)
    registry.to_csv(OUTPUT_DIR / "hrs_outcome_registry.csv", index=False)
    status_audit.to_csv(OUTPUT_DIR / "hrs_status_by_wave_audit.csv", index=False)
    write_standardization_report(
        person_wave, registry, status_audit, depression_threshold, low_cog_threshold
    )

    print(
        {
            "person_wave": str((OUTPUT_DIR / "hrs_person_wave.parquet").relative_to(PROJECT_ROOT)),
            "outcome_registry": str(
                (OUTPUT_DIR / "hrs_outcome_registry.csv").relative_to(PROJECT_ROOT)
            ),
            "report": str((OUTPUT_DIR / "hrs_standardization_report.md").relative_to(PROJECT_ROOT)),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
