#!/usr/bin/env python3
"""Build ELSA per-outcome leakage exclusions and predictor type map."""

from __future__ import annotations

import argparse
import math
from typing import Any

import pandas as pd

from elsa_utils import (
    API_BASE,
    ID_COLUMN,
    ELSA_COMMON_CHRONIC_USED,
    ELSA_OUTCOME_SOURCES,
    OUTPUT_DIR,
    PROJECT_ROOT,
    SELECTED_OUTCOMES,
    WAVE_COLUMN,
    extract_columns,
    fetch_schema,
    fmt,
    generated_timestamp,
    to_numeric,
    validation_window,
    write_text,
)

PHASE0_VARIABLE_CROSSWALK = OUTPUT_DIR / "elsa_variable_crosswalk.csv"

NATIVE_SAFE_CANDIDATES = [
    "agey",
    "ragender",
    "raeducl",
    "raeduc_k",
    "rural",
    "metrop_k",
    "region_k",
    "relig_k",
    "mstath",
    "shlt",
    "bmi",
    "height",
    "weight",
    "bmicat",
    "obese",
    "sighta",
    "hearinga",
    "fall",
    "fallnum",
    "fallinj",
    "painfr",
    "painhlv",
    "paina",
    "vigact",
    "vigactf_k",
    "drinkev",
    "drink",
    "drinkx",
    "drinkn_k",
    "smokev",
    "smoken",
    "smokef",
    "doctor",
    "doctim",
    "trdmed2y",
    "homcar",
    "hctim",
    "dentst",
    "dentim",
    "drugs",
    "oopdoc",
    "oopdrug",
    "oopmd",
    "higovt_k",
    "hipriv",
    "hiltca_k",
    "hiltc_k",
    "atotf",
    "atotb",
    "itearn",
    "itot",
    "itothhinc",
    "ctot",
    "cperc",
    "hres",
    "child",
    "momliv",
    "dadliv",
    "ramomeducl",
    "radadeducl",
    "coresd",
    "socwk",
    "socrelg_k",
    "relgwk",
    "tcany",
    "tcamt",
    "fcany",
    "fcamt",
    "tpany",
    "tpamt",
    "fpany",
    "fpamt",
    "work",
    "slfemp",
    "lbrf_k",
    "lgrip",
    "rgrip",
    "gripsum",
    "gripcomp",
    "wealth",
]

GLOBAL_EXCLUDE = {
    ID_COLUMN,
    WAVE_COLUMN,
    "hhidc",
    "iwstat",
    "iwm",
    "iwy",
    "iwindy",
    "rabyear",
    "rabmonth",
    "radyear",
    "radmonth",
}

FUNCTION_VARS = {
    "dressa",
    "batha",
    "eata",
    "toilta",
    "beda",
    "walkra",
    "housewka",
    "mealsa",
    "shopa",
    "phonea",
    "medsa",
    "moneya",
    "dressb",
    "bathb",
    "eatb",
    "toiltb",
    "bedb_k",
    "brushb",
    "urinb",
    "mealsb",
    "shopb",
    "medsb",
    "moneyb",
    "phoneb",
    "transb",
    "gooutb",
    "laundryb",
    "housewkb",
    "groomb",
    "dementia",
}

UTILIZATION_VARS = {
    "doctor",
    "doctim",
    "hcfcl2y",
    "hcftim2y",
    "hcfnit2y",
    "trdmed2y",
    "trdmdtim2y",
    "homcar",
    "hctim",
    "dentst",
    "dentim",
    "drugs",
    "oopdoc",
    "oopdrug",
    "oopmd",
}

HYPERTENSION_LEAKS = {"hibpe", "radiaghibp", "rxhibp"}
DIABETES_LEAKS = {"diabe", "radiagdiab", "rxdiab"}
CHRONIC_DISEASE_AND_TREATMENT = {
    *ELSA_COMMON_CHRONIC_USED,
    "digeste",
    "radiaghibp",
    "radiagdiab",
    "radiagcancr",
    "radiaglung",
    "radiagheart",
    "radiagstrok",
    "radiagpsych",
    "radiagarthr",
    "radiagliver",
    "radiagdigest",
    "rxhibp",
    "rxdiab",
    "rxcancr",
    "rxlung",
    "rxheart",
    "rxstrok",
    "rxpsych",
    "rxarthr",
    "rxliver",
    "rxdigest",
}
DEPRESSION_LEAKS = {
    "cesd",
    "depressive",
    "cesd10a",
    "cesd10b",
    *[f"freq_{index}" for index in range(1, 8)],
}
COGNITION_LEAK_PREFIXES = (
    "cog",
    "tcog",
    "imrc",
    "dlrc",
    "ser7",
    "tr6",
    "object",
    "rpsnt",
    "execu",
    "task",
    "write",
    "draw",
)
COGNITION_LEAKS = {"tcog_z_z", "cog_total", "dementia", "dw", "dat", "ssn", "place", "cgdaddr"}

CONTINUOUS_HINTS = {
    "agey",
    "bmi",
    "height",
    "weight",
    "fallnum",
    "doctim",
    "hctim",
    "dentim",
    "oopdoc",
    "oopdrug",
    "oopmd",
    "atotf",
    "atotb",
    "itearn",
    "itot",
    "itothhinc",
    "ctot",
    "cperc",
    "hres",
    "child",
    "tcamt",
    "fcamt",
    "tpamt",
    "fpamt",
    "lgrip",
    "rgrip",
    "gripsum",
    "wealth",
    "smokef",
}


def load_candidate_variables(schema: set[str]) -> list[str]:
    candidates = set(NATIVE_SAFE_CANDIDATES)
    if PHASE0_VARIABLE_CROSSWALK.exists():
        crosswalk = pd.read_csv(PHASE0_VARIABLE_CROSSWALK)
        mapped = crosswalk[
            crosswalk["charls_baseline_predictor_general"].astype(bool)
            & crosswalk["status"].eq("mapped")
            & crosswalk["cohort_variable"].notna()
        ]["cohort_variable"].astype(str)
        candidates.update(mapped.tolist())
    return sorted(
        variable for variable in candidates if variable in schema and variable not in GLOBAL_EXCLUDE
    )


def infer_type(series: pd.Series, variable: str) -> tuple[str, str, int, float]:
    numeric = to_numeric(series)
    observed = numeric.dropna()
    missing_rate = float(series.isna().mean())
    unique_count = int(series.dropna().nunique())
    if variable in CONTINUOUS_HINTS:
        return "continuous", "continuous_quantile_bin", unique_count, missing_rate
    observed_values = set(observed.dropna().unique().tolist())
    if observed_values.issubset({0, 1}) or unique_count <= 12:
        return (
            "binary_or_boolean" if observed_values.issubset({0, 1}) else "categorical_numeric",
            "categorical_value_label",
            unique_count,
            missing_rate,
        )
    return "continuous", "continuous_quantile_bin", unique_count, missing_rate


def build_variable_type_map(frame: pd.DataFrame, candidate_variables: list[str]) -> pd.DataFrame:
    rows = []
    for variable in candidate_variables:
        type_model, tokenization_type, unique_count, missing_rate = infer_type(
            frame[variable], variable
        )
        rows.append(
            {
                "variable": variable,
                "type_model": type_model,
                "tokenization_type": tokenization_type,
                "missing_rate_overall": missing_rate,
                "unique_count": unique_count,
                "include_in_tokenization_v1": missing_rate < 0.8,
                "include_in_pretraining_v1": missing_rate < 0.8,
                "usable_as_baseline_predictor_after_review": missing_rate < 0.8,
            }
        )
    return pd.DataFrame(rows).sort_values("variable", kind="stable")


def leakage_decision(outcome_id: str, variable: str) -> tuple[str, str]:
    if variable in GLOBAL_EXCLUDE:
        return "exclude", "identifier_or_time_index"
    if variable in set(ELSA_OUTCOME_SOURCES[outcome_id]):
        return "exclude", "target_source_variable"
    if outcome_id in {"adl_new_difficulty", "iadl_new_difficulty"} and variable in FUNCTION_VARS:
        return "exclude", "function_construct_leakage"
    if outcome_id == "incident_hypertension" and variable in HYPERTENSION_LEAKS:
        return "exclude", "hypertension_diagnostic_or_treatment"
    if outcome_id == "incident_diabetes" and variable in DIABETES_LEAKS:
        return "exclude", "diabetes_diagnostic_or_treatment"
    if outcome_id == "multimorbidity_onset" and variable in CHRONIC_DISEASE_AND_TREATMENT:
        return "exclude", "chronic_disease_or_treatment_construct"
    if outcome_id == "hospitalization_onset" and variable in UTILIZATION_VARS:
        return "exclude", "medical_utilization_construct"
    if outcome_id == "high_depressive_symptoms_onset" and variable in DEPRESSION_LEAKS:
        return "exclude", "depression_scale_construct"
    if outcome_id == "low_cognition_onset_data_driven" and (
        variable in COGNITION_LEAKS or variable.startswith(COGNITION_LEAK_PREFIXES)
    ):
        return "exclude", "cognition_construct"
    return "include", "safe_baseline_predictor"


def build_leakage_table(variable_map: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    usable_by_outcome = dict(zip(registry["outcome_id"], registry["usable"].astype(bool)))
    for outcome_id in SELECTED_OUTCOMES:
        for _, variable_row in variable_map.iterrows():
            variable = str(variable_row["variable"])
            decision, reason = leakage_decision(outcome_id, variable)
            rows.append(
                {
                    "outcome_id": outcome_id,
                    "variable": variable,
                    "decision": decision,
                    "reason": reason,
                    "outcome_usable": bool(usable_by_outcome.get(outcome_id, False)),
                    "usable_as_baseline_predictor_after_review": bool(
                        variable_row["usable_as_baseline_predictor_after_review"]
                    ),
                    "type_model": variable_row["type_model"],
                    "tokenization_type": variable_row["tokenization_type"],
                    "missing_rate_overall": float(variable_row["missing_rate_overall"]),
                }
            )
    return pd.DataFrame(rows)


def residual_audit(leakage: pd.DataFrame) -> pd.DataFrame:
    checks = {
        "incident_hypertension": HYPERTENSION_LEAKS,
        "incident_diabetes": DIABETES_LEAKS,
        "multimorbidity_onset": CHRONIC_DISEASE_AND_TREATMENT,
        "hospitalization_onset": UTILIZATION_VARS,
        "high_depressive_symptoms_onset": DEPRESSION_LEAKS,
        "low_cognition_onset_data_driven": set(COGNITION_LEAKS),
    }
    rows = []
    for outcome_id, forbidden in checks.items():
        included = set(
            leakage.loc[
                leakage["outcome_id"].eq(outcome_id)
                & leakage["decision"].eq("include")
                & leakage["usable_as_baseline_predictor_after_review"].astype(bool),
                "variable",
            ]
        )
        residual = sorted(included & forbidden)
        rows.append(
            {
                "outcome_id": outcome_id,
                "residual_leak_count": len(residual),
                "residual_variables": "|".join(residual),
            }
        )
    return pd.DataFrame(rows)


def write_report(leakage: pd.DataFrame, variable_map: pd.DataFrame, audit: pd.DataFrame) -> None:
    summary = (
        leakage.groupby(["outcome_id", "decision", "reason"], as_index=False)
        .agg(variables=("variable", "nunique"))
        .sort_values(["outcome_id", "decision", "reason"], kind="stable")
    )
    safe_counts = (
        leakage[
            leakage["decision"].eq("include")
            & leakage["reason"].eq("safe_baseline_predictor")
            & leakage["usable_as_baseline_predictor_after_review"].astype(bool)
        ]
        .groupby("outcome_id", as_index=False)
        .agg(safe_predictors=("variable", "nunique"))
    )
    lines = [
        "# ELSA Leakage Exclusion Report",
        "",
        f"Generated: {generated_timestamp()}",
        "",
        f"- Candidate predictor variables: {len(variable_map)}",
        "- Disease diagnostic/treatment constructs are excluded per ELSA variable names.",
        "- ELSA function, depression, and cognition constructs are excluded from their corresponding target outcomes.",
        "",
        "## Safe Predictor Counts",
        "",
        "| outcome_id | safe_predictors |",
        "|---|---:|",
    ]
    for _, row in safe_counts.iterrows():
        lines.append(f"| {row['outcome_id']} | {int(row['safe_predictors'])} |")
    lines.extend(
        [
            "",
            "## Residual Leak Audit",
            "",
            "| outcome_id | residual_leak_count | residual_variables |",
            "|---|---:|---|",
        ]
    )
    for _, row in audit.iterrows():
        lines.append(
            f"| {row['outcome_id']} | {int(row['residual_leak_count'])} | {row['residual_variables']} |"
        )
    lines.extend(
        [
            "",
            "## Decision Summary",
            "",
            "| outcome_id | decision | reason | variables |",
            "|---|---|---|---:|",
        ]
    )
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['outcome_id']} | {row['decision']} | {row['reason']} | {int(row['variables'])} |"
        )
    write_text(OUTPUT_DIR / "elsa_leakage_exclusion_report.md", lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", default=API_BASE)
    parser.add_argument("--row-limit", type=int, default=1000000)
    parser.add_argument("--chunk-size", type=int, default=70)
    args = parser.parse_args()

    person_wave_path = OUTPUT_DIR / "elsa_person_wave.parquet"
    registry_path = OUTPUT_DIR / "elsa_outcome_registry.csv"
    if not person_wave_path.exists() or not registry_path.exists():
        raise RuntimeError("Run elsa_standardize_long.py before leakage-table construction")

    schema = set(fetch_schema(args.api_base))
    candidate_variables = load_candidate_variables(schema)
    frame = extract_columns(
        candidate_variables,
        row_limit=args.row_limit,
        api_base=args.api_base,
        chunk_size=args.chunk_size,
    )
    frame = validation_window(frame)
    variable_map = build_variable_type_map(frame, candidate_variables)
    registry = pd.read_csv(registry_path)
    leakage = build_leakage_table(variable_map, registry)
    audit = residual_audit(leakage)
    if audit["residual_leak_count"].sum() != 0:
        raise RuntimeError(f"ELSA residual leakage audit failed:\n{audit.to_string(index=False)}")

    variable_map.to_csv(OUTPUT_DIR / "elsa_variable_type_map.csv", index=False)
    leakage.to_csv(OUTPUT_DIR / "elsa_per_outcome_leakage.csv", index=False)
    audit.to_csv(OUTPUT_DIR / "elsa_leakage_residual_audit.csv", index=False)
    write_report(leakage, variable_map, audit)
    print(
        {
            "variables": int(len(variable_map)),
            "leakage": str((OUTPUT_DIR / "elsa_per_outcome_leakage.csv").relative_to(PROJECT_ROOT)),
            "report": str(
                (OUTPUT_DIR / "elsa_leakage_exclusion_report.md").relative_to(PROJECT_ROOT)
            ),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
