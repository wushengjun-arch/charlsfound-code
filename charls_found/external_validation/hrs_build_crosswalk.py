#!/usr/bin/env python3
"""Discover HRS validation variables and build CHARLS-to-HRS crosswalks."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from hrs_utils import (
    API_BASE,
    CHARLS_CHRONIC_14,
    HRS_COMMON_CHRONIC_USED,
    HRS_OUTCOME_SOURCES,
    ID_COLUMN,
    INTERVIEW_YEAR_COLUMN,
    MAIN_FILE,
    OUTPUT_DIR,
    PROJECT_ROOT,
    SELECTED_OUTCOMES,
    SOURCE_ID,
    VALIDATION_WAVES,
    WAVE_COLUMN,
    WAVE_YEARS,
    extract_columns,
    fetch_schema,
    fmt,
    generated_timestamp,
    infer_depression_threshold,
    infer_low_cognition_threshold,
    post_json,
    resolve_main_file,
    resolve_project_path,
    validation_window,
    write_text,
)

CHARLS_VARIABLE_MAP = PROJECT_ROOT / "outputs" / "05_tokenization" / "charls_variable_type_map.csv"

MANUAL_VARIABLE_MAP = {
    "ID": ID_COLUMN,
    "age": "ragey_e",
    "hospital": "hosp",
    "hospital_time": "hsptim",
    "doctor_time": "doctim",
    "hrural": "urbrur",
    "rural2": "rural",
    "raeduc_c": "raeducl",
    "sleep_night": "sleepe",
    "drinkl": "drink",
    "smokef": "smoken",
}


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame[columns].iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return lines


def top_search_codes(api_base: str, queries: list[str]) -> dict[str, str]:
    payload = post_json(
        api_base,
        "/search/batch",
        {"q": queries, "source": SOURCE_ID, "kind": "auto", "limit": 8},
        timeout=300,
    )
    out: dict[str, str] = {}
    for result in payload.get("results", []):
        query = result.get("query", "")
        codes = []
        for hit in result.get("results", []):
            code = hit.get("code") or hit.get("physical_column")
            if code and code not in codes:
                codes.append(str(code))
        out[str(query)] = ", ".join(codes[:8]) if codes else "no hits"
    return out


def build_variable_crosswalk(schema: set[str]) -> pd.DataFrame:
    charls_map = pd.read_csv(CHARLS_VARIABLE_MAP)
    rows: list[dict[str, Any]] = []
    for _, row in charls_map.iterrows():
        variable = str(row["variable"])
        is_safe = as_bool(row.get("baseline_predictor_general", False))
        include_pretraining = as_bool(row.get("include_in_pretraining_v1", False))
        if not is_safe and not include_pretraining:
            continue

        if variable in schema:
            hrs_variable = variable
            status = "mapped"
            mapping_type = "same_name"
            reason = "HRS schema contains the same harmonized variable name"
        elif variable in MANUAL_VARIABLE_MAP and MANUAL_VARIABLE_MAP[variable] in schema:
            hrs_variable = MANUAL_VARIABLE_MAP[variable]
            status = "mapped"
            mapping_type = "manual"
            reason = f"Manual harmonized alias {variable}->{hrs_variable}"
        else:
            hrs_variable = ""
            status = "unmappable"
            mapping_type = "none"
            reason = "No HRS schema variable found for this CHARLS predictor"

        rows.append(
            {
                "charls_variable": variable,
                "hrs_variable": hrs_variable,
                "status": status,
                "mapping_type": mapping_type,
                "reason": reason,
                "charls_module": row.get("module", ""),
                "charls_type_model": row.get("type_model", ""),
                "charls_tokenization_type": row.get("tokenization_type", ""),
                "charls_baseline_predictor_general": is_safe,
                "api_schema_verified": bool(hrs_variable and hrs_variable in schema),
                "api_file": MAIN_FILE if hrs_variable else "",
            }
        )
    return pd.DataFrame(rows).sort_values(["status", "charls_variable"], kind="stable")


def build_outcome_crosswalk(
    schema: set[str], depression_threshold: int, low_cog_threshold: float
) -> pd.DataFrame:
    charls_sources = {
        "adl_new_difficulty": "dressa|batha|eata|beda|toilta|urina",
        "iadl_new_difficulty": "housewka|mealsa|shopa|phonea|medsa|moneya",
        "incident_diabetes": "diabe",
        "incident_hypertension": "hibpe",
        "multimorbidity_onset": "|".join(CHARLS_CHRONIC_14),
        "high_depressive_symptoms_onset": "cesd10",
        "hospitalization_onset": "hospital",
        "low_cognition_onset_data_driven": "total_cognition",
    }
    rows: list[dict[str, Any]] = []
    for outcome_id in SELECTED_OUTCOMES:
        hrs_variables = HRS_OUTCOME_SOURCES[outcome_id]
        missing = sorted(set(hrs_variables) - schema)
        status = "mapped" if not missing else "unmappable"
        note = "HRS-native definition in 2010-2020 validation waves"
        if outcome_id == "adl_new_difficulty":
            note = "HRS six-item ADL uses walksa instead of CHARLS urina"
        elif outcome_id == "iadl_new_difficulty":
            note = "HRS five-item IADL omits CHARLS housework item; iadl5a exists for audit"
        elif outcome_id == "multimorbidity_onset":
            note = "Uses observed 2010-2020 common disease flags by harmonized name; memrye exists but has no 2010-2020 coverage"
        elif outcome_id == "high_depressive_symptoms_onset":
            note = f"HRS CESD is 0-8; derived depressive flag implies cutpoint >= {depression_threshold}"
        elif outcome_id == "low_cognition_onset_data_driven":
            note = (
                f"cogtotp is unavailable at wave 10; uses cog27 wave-10 Q25={low_cog_threshold:g}"
            )
        rows.append(
            {
                "outcome_id": outcome_id,
                "charls_source_variables": charls_sources[outcome_id],
                "hrs_source_variables": "|".join(hrs_variables),
                "status": status,
                "missing_hrs_variables": "|".join(missing),
                "definition_note": note,
                "api_schema_verified": not missing,
                "api_file": MAIN_FILE,
            }
        )
    return pd.DataFrame(rows)


def write_discovery_report(
    path: Path,
    resolve_payload: dict[str, Any],
    search_codes: dict[str, str],
    wave_summary: pd.DataFrame,
    chronic_summary: pd.DataFrame,
    depression_threshold: int,
    low_cog_threshold: float,
    variable_crosswalk: pd.DataFrame,
    outcome_crosswalk: pd.DataFrame,
) -> None:
    mapped_safe = variable_crosswalk[
        variable_crosswalk["charls_baseline_predictor_general"].astype(bool)
        & variable_crosswalk["status"].eq("mapped")
    ]
    safe_total = int(variable_crosswalk["charls_baseline_predictor_general"].astype(bool).sum())
    safe_mapped = int(len(mapped_safe))
    lines = [
        "# HRS Discovery Report",
        "",
        f"Generated: {generated_timestamp()}",
        "",
        "## API Evidence",
        "",
        f"- Source: `{SOURCE_ID}`",
        f"- File key: `{MAIN_FILE}`",
        f"- Resolve exists: `{resolve_payload.get('exists')}`",
        f"- API base: `{API_BASE}`",
        f"- Validation waves: {', '.join(str(wave) for wave in VALIDATION_WAVES)}",
        "",
        "## Search Checks",
        "",
        "| query | top API hits |",
        "|---|---|",
    ]
    for query, codes in search_codes.items():
        lines.append(f"| {query} | {codes} |")

    lines.extend(["", "## Wave-To-Year Check", ""])
    lines.extend(
        markdown_table(
            wave_summary, ["wave", "nominal_year", "iwendy_min", "iwendy_max", "rows", "ids"]
        )
    )

    lines.extend(
        [
            "",
            "## Required Discovery Decisions",
            "",
            f"- IADL is available: `moneya|phonea|medsa|mealsa|shopa` and derived `iadl5a`; first-round HRS IADL keeps the five-item HRS scale and records CHARLS `housewka` as unmappable.",
            f"- HRS CESD is 0-8. The HRS derived `depressive` flag implies high symptoms at `cesd >= {depression_threshold}`, so the CHARLS CESD10 >=10 cutpoint is not reused.",
            f"- Low cognition uses HRS `cog27`, because `cogtotp` has no wave-10 observations. The wave-10 Q25 threshold is `{low_cog_threshold:g}`.",
            "- HRS validation window is wave 10-15, nominally 2010-2020; interview-year ranges are shown above.",
            "- Multimorbidity uses the common observed HRS/CHARLS harmonized-name disease set below, with threshold `<2 -> >=2`.",
            "",
            "## Multimorbidity Disease Set",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            chronic_summary,
            ["variable", "in_hrs_schema", "validation_nonmissing", "used_in_hrs_multimorbidity"],
        )
    )

    lines.extend(
        [
            "",
            "## Crosswalk Summary",
            "",
            f"- CHARLS safe baseline predictors considered: {safe_total}",
            f"- Safe predictors mapped to HRS: {safe_mapped}",
            f"- Safe predictors unmappable: {safe_total - safe_mapped}",
            f"- Outcome definitions mapped: {int(outcome_crosswalk['status'].eq('mapped').sum())}/{len(outcome_crosswalk)}",
            "",
            "Outputs:",
            "",
            "- `external_variable_crosswalk.csv`",
            "- `external_outcome_crosswalk.csv`",
            "",
            "## Outcome Crosswalk",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            outcome_crosswalk, ["outcome_id", "status", "hrs_source_variables", "definition_note"]
        )
    )
    write_text(path, lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", default=API_BASE)
    parser.add_argument("--row-limit", type=int, default=300000)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    schema = set(fetch_schema(args.api_base))
    resolve_payload = resolve_main_file(args.api_base)
    search_codes = top_search_codes(
        args.api_base, ["iadl", "cesd", "cognition", "blood pressure", "diabetes medication"]
    )

    discovery_columns = sorted(
        set(
            [
                ID_COLUMN,
                WAVE_COLUMN,
                INTERVIEW_YEAR_COLUMN,
                "cesd",
                "depressive",
                "cog27",
                "cogtotp",
                "iadl5a",
                "systo",
                "diasto",
                "rxhibp",
                "rxdiab",
                "rxdiabo",
                "rxdiabi",
                *CHARLS_CHRONIC_14,
                *HRS_COMMON_CHRONIC_USED,
            ]
        )
        & schema
    )
    discovery = extract_columns(discovery_columns, row_limit=args.row_limit, api_base=args.api_base)
    window = validation_window(discovery)
    depression_threshold = infer_depression_threshold(window)
    low_cog_threshold = infer_low_cognition_threshold(window, variable="cog27", quantile=0.25)

    wave_summary = (
        window.groupby(WAVE_COLUMN, as_index=False)
        .agg(
            iwendy_min=(INTERVIEW_YEAR_COLUMN, "min"),
            iwendy_max=(INTERVIEW_YEAR_COLUMN, "max"),
            rows=(ID_COLUMN, "size"),
            ids=(ID_COLUMN, "nunique"),
        )
        .rename(columns={WAVE_COLUMN: "wave"})
    )
    wave_summary["nominal_year"] = wave_summary["wave"].map(WAVE_YEARS)
    wave_summary = wave_summary[["wave", "nominal_year", "iwendy_min", "iwendy_max", "rows", "ids"]]

    chronic_rows = []
    for variable in CHARLS_CHRONIC_14:
        in_schema = variable in schema
        nonmissing = (
            int(window[variable].notna().sum()) if in_schema and variable in window.columns else 0
        )
        chronic_rows.append(
            {
                "variable": variable,
                "in_hrs_schema": in_schema,
                "validation_nonmissing": nonmissing,
                "used_in_hrs_multimorbidity": variable in HRS_COMMON_CHRONIC_USED
                and nonmissing > 0,
            }
        )
    chronic_summary = pd.DataFrame(chronic_rows)

    variable_crosswalk = build_variable_crosswalk(schema)
    outcome_crosswalk = build_outcome_crosswalk(schema, depression_threshold, low_cog_threshold)

    variable_crosswalk.to_csv(OUTPUT_DIR / "external_variable_crosswalk.csv", index=False)
    outcome_crosswalk.to_csv(OUTPUT_DIR / "external_outcome_crosswalk.csv", index=False)
    wave_summary.to_csv(OUTPUT_DIR / "hrs_wave_year_audit.csv", index=False)
    chronic_summary.to_csv(OUTPUT_DIR / "hrs_multimorbidity_common_disease_audit.csv", index=False)
    write_discovery_report(
        OUTPUT_DIR / "hrs_discovery_report.md",
        resolve_payload=resolve_payload,
        search_codes=search_codes,
        wave_summary=wave_summary,
        chronic_summary=chronic_summary,
        depression_threshold=depression_threshold,
        low_cog_threshold=low_cog_threshold,
        variable_crosswalk=variable_crosswalk,
        outcome_crosswalk=outcome_crosswalk,
    )

    print(
        {
            "variable_crosswalk": str(
                (OUTPUT_DIR / "external_variable_crosswalk.csv").relative_to(PROJECT_ROOT)
            ),
            "outcome_crosswalk": str(
                (OUTPUT_DIR / "external_outcome_crosswalk.csv").relative_to(PROJECT_ROOT)
            ),
            "discovery_report": str(
                (OUTPUT_DIR / "hrs_discovery_report.md").relative_to(PROJECT_ROOT)
            ),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
