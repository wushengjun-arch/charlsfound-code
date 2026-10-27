#!/usr/bin/env python3
"""Build per-outcome predictor leakage exclusion and review tables."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_CONFIG = SCRIPT_PATH.parents[1] / "configs" / "data.yaml"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def split_vars(value: Any) -> list[str]:
    if pd.isna(value) or str(value).strip() == "":
        return []
    return [part.strip() for part in str(value).split("|") if part.strip()]


def canonical_domain(domain: str) -> str:
    mapping = {
        "chronic_disease": "chronic diseases",
        "function": "function",
        "cognition": "cognition",
        "mental_health": "mental health",
        "medical_utilization": "medical utilization",
        "physical_measures": "physical measures",
        "multimorbidity": "chronic diseases",
        "mortality": "mortality/death timing",
    }
    return mapping.get(domain, domain)


DISEASE_DIAGNOSTIC_EXCLUSION_VARIABLES = {
    "incident_arthritis": {
        "rxarthr",
        "rxarthr_c",
    },
    "incident_diabetes": {
        "bl_glu",
        "bl_hbalc",
        "rxdiab",
        "rxdiab_c",
        "rxdiabi",
    },
    "incident_digestive_disease": {
        "rxdigest",
        "rxdigest_c",
    },
    "incident_dyslipidemia": {
        "bl_cho",
        "bl_hdl",
        "bl_ldl",
        "bl_tg",
        "bl_top_coding_tg",
        "rxdyslip",
        "rxdyslip_c",
    },
    "incident_heart_disease": {
        "rxheart",
        "rxheart_c",
    },
    "incident_hypertension": {
        "bpact30",
        "bpcomp",
        "diasto",
        "diasto1",
        "diasto2",
        "diasto3",
        "pulse",
        "pulse1",
        "pulse2",
        "pulse3",
        "rxhibp",
        "rxhibp_c",
        "systo",
        "systo1",
        "systo2",
        "systo3",
    },
    "incident_kidney_disease": {
        "bl_bun",
        "bl_crea",
        "bl_cysc",
        "rxkidney",
        "rxkidney_c",
    },
    "incident_liver_disease": {
        "rxliver",
        "rxliver_c",
    },
    "incident_lung_disease": {
        "rxlung",
        "rxlung_c",
    },
    "incident_memory_disease": {
        "rxmemry",
        "rxmemry_c",
    },
    "incident_psychiatric_disease": {
        "rxpsych",
    },
    "incident_stroke": {
        "rxstrok",
        "rxstrok_c",
    },
}

MULTIMORBIDITY_CONSTRUCT_EXCLUSION_VARIABLES = {
    "bl_bun",
    "bl_cho",
    "bl_crea",
    "bl_cysc",
    "bl_glu",
    "bl_hbalc",
    "bl_hdl",
    "bl_ldl",
    "bl_tg",
    "bl_top_coding_tg",
    "bpact30",
    "bpcomp",
    "chronic_num",
    "diasto",
    "diasto1",
    "diasto2",
    "diasto3",
    "pulse",
    "pulse1",
    "pulse2",
    "pulse3",
    "rxarthr",
    "rxarthr_c",
    "rxdiab",
    "rxdiab_c",
    "rxdiabi",
    "rxdigest",
    "rxdigest_c",
    "rxdyslip",
    "rxdyslip_c",
    "rxheart",
    "rxheart_c",
    "rxhibp",
    "rxhibp_c",
    "rxkidney",
    "rxkidney_c",
    "rxliver",
    "rxliver_c",
    "rxlung",
    "rxlung_c",
    "rxmemry",
    "rxmemry_c",
    "rxpsych",
    "rxstrok",
    "rxstrok_c",
    "systo",
    "systo1",
    "systo2",
    "systo3",
}


def diagnostic_construct_component(variable: str, outcome_id: str, domain: str) -> tuple[bool, str]:
    if domain in {"chronic_disease", "multimorbidity"} and variable.startswith("da009_1_"):
        return True, "disease_diagnosis_year_construct"
    if variable in DISEASE_DIAGNOSTIC_EXCLUSION_VARIABLES.get(outcome_id, set()):
        return True, "disease_diagnostic_or_treatment_construct"
    if domain == "multimorbidity" and variable in MULTIMORBIDITY_CONSTRUCT_EXCLUSION_VARIABLES:
        return True, "multimorbidity_diagnostic_or_count_construct"
    return False, ""


def related_component(
    variable: str, module: str, outcome_id: str, source_vars: set[str], domain: str
) -> tuple[bool, str]:
    if variable in source_vars:
        return True, "target_source_variable"
    if outcome_id == "adl_new_difficulty" and (
        module == "ADL" or variable in {"adlab_c", "frailtyb"}
    ):
        return True, "adl_component_or_derived_score"
    if outcome_id == "iadl_new_difficulty" and (
        module == "IADL" or variable in {"iadl", "frailtyb"}
    ):
        return True, "iadl_component_or_derived_score"
    if "cognition" in outcome_id and (module == "cognition" or variable in {"cog_status"}):
        return True, "cognition_component_or_derived_score"
    if "depressive" in outcome_id or outcome_id == "cesd10_change":
        if module == "mental health" or variable == "cesd10":
            return True, "mental_health_component"
    if outcome_id == "hospitalization_onset" and module == "medical utilization":
        return True, "medical_utilization_same_window"
    if outcome_id.startswith("incident_cancer") and (
        variable.startswith("da017") or variable == "cancre"
    ):
        return True, "cancer_site_or_target_component"
    if outcome_id.startswith("incident_hypertension") and variable in {"hibpe", "hibpe_self"}:
        return True, "hypertension_target_or_awareness"
    if outcome_id.startswith("incident_lung") and variable in {"lunge", "lunge_self"}:
        return True, "lung_target_or_awareness"
    if outcome_id.startswith("incident_psychiatric") and variable in {"psyche", "psyche_self"}:
        return True, "psychiatric_target_or_awareness"
    if domain == "mortality" and module == "mortality/death timing":
        return True, "mortality_timing_field"
    return False, ""


def leakage_decision(outcome: pd.Series, variable: pd.Series) -> tuple[str, str, str, bool]:
    outcome_id = outcome["outcome_id"]
    domain = outcome["domain"]
    source_vars = set(split_vars(outcome["source_variables"]))
    var = variable["variable"]
    module = variable["module"]
    label = variable.get("label_zh", "")
    missing_rate = float(variable.get("missing_rate_overall", 0) or 0)
    usable = bool(outcome.get("usable", False))

    if not usable:
        return "exclude", "outcome_not_usable", "critical", False
    if module == "identifiers" or bool(variable.get("is_identifier", False)):
        return "exclude", "identifier", "critical", False
    if module == "wave/time" or bool(variable.get("is_time_variable", False)):
        return "exclude", "time_or_wave_variable", "critical", False
    if module == "mortality/death timing" or var in {"radyear", "radmonth", "iwstat"}:
        return "exclude", "future_or_death_timing_field", "critical", False
    related, reason = related_component(var, module, outcome_id, source_vars, domain)
    if related:
        return "exclude", reason, "critical", False
    diagnostic_related, diagnostic_reason = diagnostic_construct_component(var, outcome_id, domain)
    if diagnostic_related:
        return "exclude", diagnostic_reason, "critical", False
    if missing_rate >= 0.8:
        return "exclude", "high_missingness_ge_80_percent", "quality", False
    if module == "derived scores":
        return "review", "derived_score_review", "moderate", True
    if domain == "multimorbidity" and module == "chronic diseases":
        return "exclude", "multimorbidity_component_chronic_disease", "critical", False
    if canonical_domain(domain) == module:
        return "review", "same_domain_predictor_review", "moderate", True
    if "outcome_same_domain" == str(variable.get("leakage_risk_level", "")):
        return "review", "generic_outcome_domain_variable", "moderate", True
    if str(label).startswith("频率/") and domain in {"function", "cognition", "mental_health"}:
        return "include", "safe_baseline_predictor", "low", True
    return "include", "safe_baseline_predictor", "low", True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    config = load_yaml(args.config)
    output_dir = args.output_dir or PROJECT_ROOT / config["outputs"].get(
        "leakage_directory", config["outputs"]["directory"]
    )
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    registry_dir = PROJECT_ROOT / config["outputs"].get(
        "outcome_registry_directory", config["outputs"]["directory"]
    )
    if not registry_dir.is_absolute():
        registry_dir = PROJECT_ROOT / registry_dir
    audit_dir = PROJECT_ROOT / config["outputs"].get(
        "data_audit_directory", config["outputs"]["directory"]
    )
    if not audit_dir.is_absolute():
        audit_dir = PROJECT_ROOT / audit_dir
    registry_path = registry_dir / "outcome_registry_validated.csv"
    if not registry_path.exists():
        registry_path = registry_dir / "outcome_registry_draft.csv"
    registry = pd.read_csv(registry_path)
    schema = pd.read_csv(audit_dir / "charls_schema.csv")

    rows = []
    for _, outcome in registry.iterrows():
        for _, variable in schema.iterrows():
            decision, reason, severity, usable_as_predictor = leakage_decision(outcome, variable)
            rows.append(
                {
                    "outcome_id": outcome["outcome_id"],
                    "outcome_name": outcome["outcome_name"],
                    "outcome_domain": outcome["domain"],
                    "outcome_usable": bool(outcome.get("usable", False)),
                    "variable": variable["variable"],
                    "label_zh": variable.get("label_zh"),
                    "variable_module": variable["module"],
                    "variable_type_model": variable["type_model"],
                    "missing_rate_overall": variable.get("missing_rate_overall"),
                    "decision": decision,
                    "reason": reason,
                    "severity": severity,
                    "usable_as_baseline_predictor_after_review": usable_as_predictor,
                }
            )
    matrix = pd.DataFrame(rows)
    matrix.to_csv(output_dir / "per_outcome_predictor_leakage_matrix.csv", index=False)
    exclusion = matrix[matrix["decision"].isin(["exclude", "review"])].copy()
    exclusion.to_csv(output_dir / "per_outcome_leakage_exclusion.csv", index=False)
    summary = (
        matrix.groupby(["outcome_id", "decision", "reason"])
        .size()
        .rename("variable_count")
        .reset_index()
        .sort_values(["outcome_id", "decision", "reason"])
    )
    summary.to_csv(output_dir / "per_outcome_predictor_inclusion_summary.csv", index=False)

    lines = [
        "# Leakage Exclusion Report",
        "",
        f"Outcomes: {registry['outcome_id'].nunique()}",
        f"Variables per outcome: {schema['variable'].nunique()}",
        f"Full matrix rows: {len(matrix)}",
        f"Exclude/review rows: {len(exclusion)}",
        "",
        "## Output Files",
        "",
        "- `outputs/04_leakage/per_outcome_predictor_leakage_matrix.csv`: all outcome-variable decisions.",
        "- `outputs/04_leakage/per_outcome_leakage_exclusion.csv`: variables excluded or requiring review before modeling.",
        "- `outputs/04_leakage/per_outcome_predictor_inclusion_summary.csv`: counts by outcome, decision, and reason.",
        "",
        "## Rule Notes",
        "",
        "- Identifiers, time variables, mortality timing fields, direct target variables, and direct target components are excluded.",
        "- Disease-specific diagnostic, measurement, treatment, diagnosis-year, and disease-count construct variables are excluded even when they live outside the outcome's nominal module.",
        "- Same-domain non-target variables are marked `review` unless they are direct components.",
        "- Variables with overall missingness >=80 percent are excluded for the first modeling pass as a quality guard.",
        "",
    ]
    (output_dir / "leakage_exclusion_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote leakage matrix and exclusion table to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
