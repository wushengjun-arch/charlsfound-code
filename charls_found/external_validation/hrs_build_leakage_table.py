#!/usr/bin/env python3
"""Build HRS per-outcome leakage exclusions and predictor type map."""

from __future__ import annotations

import argparse
from typing import Any

import pandas as pd

from hrs_utils import (
    API_BASE,
    HRS_COMMON_CHRONIC_USED,
    HRS_OUTCOME_SOURCES,
    ID_COLUMN,
    OUTPUT_DIR,
    OUTCOME_DOMAINS,
    OUTCOME_NAMES,
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

NATIVE_SAFE_CANDIDATES = [
    "ragey_e",
    "ragender",
    "rahispan",
    "raracem",
    "race",
    "cenreg",
    "hacohort",
    "raedyrs",
    "raeducl",
    "urbrur",
    "rural",
    "shlt",
    "bmi",
    "height",
    "weight",
    "bmicat",
    "obese",
    "smokev",
    "smoken",
    "smokef",
    "drink",
    "drinkd",
    "drinkb",
    "binged",
    "sleepe",
    "vgactx",
    "mdactx",
    "ltactx",
    "work",
    "slfemp",
    "unemp",
    "sayret",
    "atotn",
    "atotb",
    "atotw",
    "iearn",
    "itot",
    "covr",
    "covs",
    "higov",
    "hiothp",
    "hiltc",
    "lifein",
    "peninc",
    "hhres",
    "dadliv",
    "momliv",
    "lvwith",
    "relgwk",
    "socrelg_h",
    "socwk",
    "socmn",
    "walk",
    "care_adult",
    "with_grand",
    "volunteer",
    "charity",
    "education",
    "club",
    "nonreligious",
    "pray",
    "read",
    "watch_tel",
    "word_game",
    "play_card",
    "writing",
    "use_computer",
    "gardening",
    "bake",
    "sew",
    "do_hobby",
    "exercize",
    "art",
    "cntc",
    "cntr",
    "cntf",
    "hobby",
    "vol",
    "hour_vol",
    "away_child",
    "satlife_h",
    "satlifez",
    "cantril",
    "dtrmnd",
    "enthstc",
    "active",
    "proud",
    "intrstd",
    "fhappy",
    "attntv",
    "content",
    "insprd",
    "hopeful",
    "alert",
    "calm",
    "exctd",
]

GLOBAL_EXCLUDE = {
    ID_COLUMN,
    WAVE_COLUMN,
    "hhhid",
    "iwstat",
    "radmonth",
    "radyear",
    "radtimtdth",
    "iwendm",
    "iwendy",
    "iwendy",
    "rabmonth",
    "rabyear",
    "wthh",
    "wtresp",
    "wtr_nh",
    "wtcrnh",
}

FUNCTION_VARS = {
    "walksa",
    "joga",
    "walk1a",
    "walkra",
    "sita",
    "chaira",
    "climsa",
    "clim1a",
    "stoopa",
    "lifta",
    "dimea",
    "armsa",
    "pusha",
    "dressa",
    "batha",
    "eata",
    "beda",
    "toilta",
    "moneya",
    "phonea",
    "medsa",
    "mealsa",
    "shopa",
    "adl6a",
    "iadl5a",
    "dependency",
    "frailty",
}

UTILIZATION_VARS = {
    "hosp",
    "hsptim",
    "hspnit",
    "nrshom",
    "nrstim",
    "nrsnit",
    "doctor",
    "doctim",
    "homcar",
    "oopmd",
}

HYPERTENSION_LEAKS = {
    "hibpe",
    "rxhibp",
    "systo",
    "systo1",
    "systo2",
    "systo3",
    "diasto",
    "diasto1",
    "diasto2",
    "diasto3",
    "bpsys",
    "bpdia",
    "bpcomp",
}

DIABETES_LEAKS = {"diabe", "rxdiab", "rxdiabo", "rxdiabi", "radiagdiab"}

CHRONIC_DISEASE_AND_TREATMENT = {
    *HRS_COMMON_CHRONIC_USED,
    "memrye",
    "alzhe",
    "alzhee",
    "demen",
    "demene",
    "hchole",
    "osteoe",
    "hrtatt",
    "angin",
    "conhrtf",
    "hrtrhm",
    "hrtatte",
    "angine",
    "conhrtfe",
    "hrtrhme",
    "rxhibp",
    "rxdiab",
    "rxdiabo",
    "rxdiabi",
    "rxstrok",
    "rxangina",
    "rxchf",
    "rxlung",
    "trpsych",
    "cncrchem",
    "cncrsurg",
    "cncrradn",
    "cncrothr",
    "cncrmeds",
    "rxhrtat",
    "rxheart",
    "rxmemry",
    "radiagdiab",
    "reccancr",
    "rechrtatt",
    "recstrok",
    "rafrhrtatt",
    "radiagchf",
    "radiaghrtr",
    "radiagangin",
    "hrtsrg",
    "jointr",
    "rxhchol",
    "rxbreath",
    "rxstom",
    "rxdepres",
}

DEPRESSION_LEAKS = {
    "cesd",
    "depressive",
    "cididep",
    "cidianh",
    "cidisymp",
    "cidimde3",
    "cidimde5",
    "lstsf",
    "pfsad",
    "dstres",
    "rxdepres",
}

COGNITION_LEAK_PREFIXES = ("cog", "imrc", "dlrc", "ser7", "bwc", "tr20", "mstot", "vocab", "orient")
COGNITION_LEAKS = {
    "memory_z",
    "executive_z",
    "tcog_z_z",
    "slfmem",
    "dementia",
    "demen",
    "demene",
    "alzhe",
    "alzhee",
    "memrye",
    "rxmemry",
}


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def load_candidate_variables(schema: set[str]) -> list[str]:
    crosswalk_path = OUTPUT_DIR / "external_variable_crosswalk.csv"
    if not crosswalk_path.exists():
        raise RuntimeError("Run hrs_build_crosswalk.py before building HRS leakage table")
    crosswalk = pd.read_csv(crosswalk_path)
    mapped_safe = crosswalk[
        crosswalk["status"].eq("mapped")
        & crosswalk["charls_baseline_predictor_general"].map(as_bool)
        & crosswalk["hrs_variable"].notna()
        & crosswalk["hrs_variable"].astype(str).ne("")
    ]["hrs_variable"].astype(str)
    candidates = set(mapped_safe.tolist()) | set(NATIVE_SAFE_CANDIDATES)
    candidates = {variable for variable in candidates if variable in schema}
    candidates -= GLOBAL_EXCLUDE
    return sorted(candidates)


def variable_stats(frame: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    rows = []
    for variable in variables:
        series = frame[variable]
        numeric = to_numeric(series)
        observed = int(series.notna().sum())
        missing_rate = float(1 - observed / len(frame)) if len(frame) else 1.0
        unique_count = int(series.dropna().nunique())
        if unique_count <= 2:
            type_model = "binary_or_boolean"
        elif unique_count <= 20:
            type_model = "categorical_numeric"
        else:
            type_model = (
                "continuous" if numeric.notna().sum() >= observed * 0.8 else "categorical_text"
            )
        rows.append(
            {
                "variable": variable,
                "missing_rate_overall": missing_rate,
                "unique_count": unique_count,
                "type_model": type_model,
                "tokenization_type": (
                    "continuous_quantile_bin"
                    if type_model == "continuous"
                    else "categorical_raw_code"
                ),
                "include_in_tokenization_v1": missing_rate < 0.8,
                "include_in_pretraining_v1": missing_rate < 0.8,
            }
        )
    return pd.DataFrame(rows)


def cognition_leak(variable: str) -> bool:
    return variable in COGNITION_LEAKS or variable.startswith(COGNITION_LEAK_PREFIXES)


def decide_variable(
    outcome_id: str, variable: str, missing_rate: float
) -> tuple[str, str, str, bool]:
    if missing_rate >= 0.8:
        return "exclude", "high_missingness_ge_80_percent", "quality", False
    if variable in HRS_OUTCOME_SOURCES[outcome_id]:
        return "exclude", "target_source_variable", "critical", False
    if variable in GLOBAL_EXCLUDE:
        return "exclude", "identifier_time_or_future_variable", "critical", False

    if outcome_id in {"adl_new_difficulty", "iadl_new_difficulty"} and variable in FUNCTION_VARS:
        return "exclude", "function_construct_for_function_outcome", "critical", False
    if outcome_id == "incident_hypertension" and variable in HYPERTENSION_LEAKS:
        return "exclude", "hrs_hypertension_diagnostic_or_treatment_construct", "critical", False
    if outcome_id == "incident_diabetes" and variable in DIABETES_LEAKS:
        return "exclude", "hrs_diabetes_diagnostic_or_treatment_construct", "critical", False
    if outcome_id == "multimorbidity_onset" and variable in CHRONIC_DISEASE_AND_TREATMENT:
        return "exclude", "hrs_multimorbidity_component_or_treatment_construct", "critical", False
    if outcome_id == "high_depressive_symptoms_onset" and variable in DEPRESSION_LEAKS:
        return "exclude", "hrs_depression_scale_or_treatment_construct", "critical", False
    if outcome_id == "hospitalization_onset" and variable in UTILIZATION_VARS:
        return "exclude", "hrs_medical_utilization_construct", "critical", False
    if outcome_id == "low_cognition_onset_data_driven" and cognition_leak(variable):
        return "exclude", "hrs_cognition_construct", "critical", False

    if (
        outcome_id in {"incident_diabetes", "incident_hypertension"}
        and variable in CHRONIC_DISEASE_AND_TREATMENT
    ):
        return "review", "same_domain_chronic_predictor_review", "moderate", True
    return "include", "safe_baseline_predictor", "low", True


def build_leakage(stats: pd.DataFrame) -> pd.DataFrame:
    rows = []
    stats_by_var = stats.set_index("variable")
    for outcome_id in SELECTED_OUTCOMES:
        for variable, row in stats_by_var.iterrows():
            decision, reason, severity, usable = decide_variable(
                outcome_id, variable, float(row["missing_rate_overall"])
            )
            rows.append(
                {
                    "outcome_id": outcome_id,
                    "outcome_name": OUTCOME_NAMES[outcome_id],
                    "outcome_domain": OUTCOME_DOMAINS[outcome_id],
                    "outcome_usable": True,
                    "variable": variable,
                    "label_zh": "",
                    "variable_module": "",
                    "variable_type_model": row["type_model"],
                    "missing_rate_overall": row["missing_rate_overall"],
                    "decision": decision,
                    "reason": reason,
                    "severity": severity,
                    "usable_as_baseline_predictor_after_review": usable,
                }
            )
    return pd.DataFrame(rows)


def residual_audit(leakage: pd.DataFrame) -> pd.DataFrame:
    checks = {
        "incident_hypertension": HYPERTENSION_LEAKS,
        "incident_diabetes": DIABETES_LEAKS,
        "multimorbidity_onset": CHRONIC_DISEASE_AND_TREATMENT,
    }
    rows = []
    for outcome_id, variables in checks.items():
        included = (
            leakage[
                leakage["outcome_id"].eq(outcome_id)
                & leakage["decision"].eq("include")
                & leakage["variable"].isin(variables)
            ]["variable"]
            .sort_values()
            .tolist()
        )
        rows.append(
            {
                "outcome_id": outcome_id,
                "leak_construct_set_size": len(variables),
                "included_residual_count": len(included),
                "included_residual_variables": "|".join(included),
                "pass": len(included) == 0,
            }
        )
    return pd.DataFrame(rows)


def write_report(leakage: pd.DataFrame, stats: pd.DataFrame, audit: pd.DataFrame) -> None:
    include_counts = (
        leakage[leakage["decision"].eq("include")]
        .groupby("outcome_id")
        .size()
        .reset_index(name="included_predictors")
    )
    decision_counts = (
        leakage.groupby(["outcome_id", "decision"]).size().unstack(fill_value=0).reset_index()
    )
    lines = [
        "# HRS Leakage Exclusion Report",
        "",
        f"Generated: {generated_timestamp()}",
        "",
        "## Summary",
        "",
        f"- Candidate HRS predictors: {len(stats)}",
        "- Predictor candidates are mapped CHARLS safe predictors plus HRS-native low-risk covariates, all verified against the HRS API schema.",
        "- Model feature lists use only rows with `decision=include` and `reason=safe_baseline_predictor`.",
        "",
        "## Included Predictor Counts",
        "",
        "| outcome_id | included_predictors |",
        "|---|---:|",
    ]
    for _, row in include_counts.iterrows():
        lines.append(f"| {row['outcome_id']} | {int(row['included_predictors'])} |")

    lines.extend(
        [
            "",
            "## Decision Counts",
            "",
            "| outcome_id | include | review | exclude |",
            "|---|---:|---:|---:|",
        ]
    )
    for _, row in decision_counts.iterrows():
        lines.append(
            f"| {row['outcome_id']} | {int(row.get('include', 0))} | {int(row.get('review', 0))} | {int(row.get('exclude', 0))} |"
        )

    lines.extend(
        [
            "",
            "## Residual Leak Audit",
            "",
            "| outcome_id | included_residual_count | pass | included_residual_variables |",
            "|---|---:|---|---|",
        ]
    )
    for _, row in audit.iterrows():
        lines.append(
            f"| {row['outcome_id']} | {int(row['included_residual_count'])} | {row['pass']} | {row['included_residual_variables'] or ''} |"
        )
    lines.extend(
        [
            "",
            "Guardrails:",
            "",
            "- Hypertension excludes diagnosis, blood-pressure measurement, and antihypertensive treatment variables.",
            "- Diabetes excludes diagnosis and antidiabetic treatment variables.",
            "- Multimorbidity excludes every disease component and disease-treatment construct considered in HRS.",
            "- Cognition, function, depression, and utilization outcomes exclude their direct construct families.",
            "",
        ]
    )
    write_text(OUTPUT_DIR / "hrs_leakage_exclusion_report.md", lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", default=API_BASE)
    parser.add_argument("--row-limit", type=int, default=300000)
    parser.add_argument("--chunk-size", type=int, default=70)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    schema = set(fetch_schema(args.api_base))
    candidates = load_candidate_variables(schema)
    frame = extract_columns(
        candidates, row_limit=args.row_limit, api_base=args.api_base, chunk_size=args.chunk_size
    )
    frame = validation_window(frame)
    stats = variable_stats(frame, candidates)
    leakage = build_leakage(stats)
    audit = residual_audit(leakage)
    if not audit["pass"].all():
        raise RuntimeError(f"HRS leakage residual audit failed: {audit.to_dict(orient='records')}")

    stats.to_csv(OUTPUT_DIR / "hrs_variable_type_map.csv", index=False)
    leakage.to_csv(OUTPUT_DIR / "hrs_per_outcome_leakage.csv", index=False)
    audit.to_csv(OUTPUT_DIR / "hrs_leakage_residual_audit.csv", index=False)
    write_report(leakage, stats, audit)
    print(
        {
            "leakage": str((OUTPUT_DIR / "hrs_per_outcome_leakage.csv").relative_to(PROJECT_ROOT)),
            "variable_type_map": str(
                (OUTPUT_DIR / "hrs_variable_type_map.csv").relative_to(PROJECT_ROOT)
            ),
            "audit": str((OUTPUT_DIR / "hrs_leakage_residual_audit.csv").relative_to(PROJECT_ROOT)),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
