#!/usr/bin/env python3
"""Build a draft CHARLS outcome registry from API-verified variables."""

from __future__ import annotations

import argparse
import json
import math
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_CONFIG = SCRIPT_PATH.parents[1] / "configs" / "data.yaml"


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def get_json(
    base_url: str, path: str, params: dict[str, Any] | None = None, timeout: int = 180
) -> dict[str, Any]:
    query = ""
    if params:
        query = "?" + urllib.parse.urlencode(params, doseq=False)
    url = base_url.rstrip("/") + path + query
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"HTTP {exc.code} for {url}: {body[:500]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach {url}: {exc}") from exc


def normalize_wave(value: Any) -> Any:
    if pd.isna(value):
        return value
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return value
    if math.isfinite(as_float) and as_float.is_integer():
        return int(as_float)
    return value


def available_columns(schema_path: Path) -> set[str]:
    schema = pd.read_csv(schema_path)
    return set(schema["variable"].astype(str))


def extract_columns(
    base_url: str, source_id: str, main_file: str, columns: Iterable[str], row_limit: int
) -> pd.DataFrame:
    columns = list(dict.fromkeys(columns))
    payload = get_json(
        base_url,
        "/extract",
        {
            "source": source_id,
            "file": main_file,
            "columns": ",".join(columns),
            "limit": row_limit,
            "format": "json",
        },
        timeout=300,
    )
    records = payload.get("records") or []
    if not records:
        raise RuntimeError("Outcome registry extract returned no records")
    return pd.DataFrame.from_records(records)


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def yes_mask(series: pd.Series) -> pd.Series:
    numeric = to_numeric(series)
    return numeric == 1


def no_mask(series: pd.Series) -> pd.Series:
    numeric = to_numeric(series)
    return numeric == 0


def summarize_binary_incident(
    df: pd.DataFrame,
    id_col: str,
    wave_col: str,
    baseline_wave: int,
    followup_waves: list[int],
    variables: list[str],
    event_name: str,
) -> dict[str, Any]:
    baseline = df[df[wave_col] == baseline_wave].copy()
    followup = df[df[wave_col].isin(followup_waves)].copy()
    if not variables:
        return {"n_baseline": 0, "n_followup": 0, "n_events": 0, "event_rate": None}

    baseline_status = baseline[[id_col] + variables].drop_duplicates(subset=[id_col])
    followup_status = followup[[id_col, wave_col] + variables]
    baseline_any_positive = pd.Series(False, index=baseline_status.index)
    baseline_any_nonmissing = pd.Series(False, index=baseline_status.index)
    for variable in variables:
        baseline_any_positive = baseline_any_positive | yes_mask(baseline_status[variable])
        baseline_any_nonmissing = baseline_any_nonmissing | baseline_status[variable].notna()
    eligible_ids = set(
        baseline_status.loc[baseline_any_nonmissing & ~baseline_any_positive, id_col]
    )

    follow = followup_status[followup_status[id_col].isin(eligible_ids)]
    if follow.empty:
        return {"n_baseline": len(eligible_ids), "n_followup": 0, "n_events": 0, "event_rate": None}
    event_by_row = pd.Series(False, index=follow.index)
    observed_by_row = pd.Series(False, index=follow.index)
    for variable in variables:
        event_by_row = event_by_row | yes_mask(follow[variable])
        observed_by_row = observed_by_row | follow[variable].notna()
    observed_ids = set(follow.loc[observed_by_row, id_col])
    events = follow.loc[event_by_row].groupby(id_col)[wave_col].min()
    n_followup = len(observed_ids)
    n_events = int(events.shape[0])
    return {
        "n_baseline": int(len(eligible_ids)),
        "n_followup": int(n_followup),
        "n_events": n_events,
        "event_rate": round(n_events / n_followup, 6) if n_followup else None,
        "minimum_event_wave": int(events.min()) if not events.empty else None,
        "event_name": event_name,
    }


def summarize_binary_onset_from_threshold(
    df: pd.DataFrame,
    id_col: str,
    wave_col: str,
    baseline_wave: int,
    followup_waves: list[int],
    variable: str,
    threshold: float,
    direction: str,
) -> dict[str, Any]:
    baseline = (
        df[df[wave_col] == baseline_wave][[id_col, variable]]
        .drop_duplicates(subset=[id_col])
        .copy()
    )
    followup = df[df[wave_col].isin(followup_waves)][[id_col, wave_col, variable]].copy()
    base_value = to_numeric(baseline[variable])
    if direction == "ge":
        baseline_negative = baseline.loc[base_value.notna() & (base_value < threshold), id_col]
        follow_event = to_numeric(followup[variable]) >= threshold
    else:
        baseline_negative = baseline.loc[base_value.notna() & (base_value > threshold), id_col]
        follow_event = to_numeric(followup[variable]) <= threshold
    eligible_ids = set(baseline_negative)
    follow = followup[followup[id_col].isin(eligible_ids)]
    observed = follow[to_numeric(follow[variable]).notna()]
    events = follow.loc[follow_event].groupby(id_col)[wave_col].min()
    n_followup = int(observed[id_col].nunique())
    n_events = int(events.shape[0])
    return {
        "n_baseline": int(len(eligible_ids)),
        "n_followup": n_followup,
        "n_events": n_events,
        "event_rate": round(n_events / n_followup, 6) if n_followup else None,
        "minimum_event_wave": int(events.min()) if not events.empty else None,
    }


def summarize_continuous_change(
    df: pd.DataFrame,
    id_col: str,
    wave_col: str,
    baseline_wave: int,
    followup_waves: list[int],
    variable: str,
) -> dict[str, Any]:
    baseline = df[df[wave_col] == baseline_wave][[id_col, variable]].drop_duplicates(
        subset=[id_col]
    )
    followup = df[df[wave_col].isin(followup_waves)][[id_col, wave_col, variable]].copy()
    base = baseline[to_numeric(baseline[variable]).notna()]
    follow = followup[to_numeric(followup[variable]).notna()]
    ids_with_follow = set(follow[id_col])
    usable_ids = set(base[id_col]) & ids_with_follow
    return {
        "n_baseline": int(base[id_col].nunique()),
        "n_followup": int(len(usable_ids)),
        "n_events": None,
        "event_rate": None,
        "minimum_event_wave": (
            int(follow.loc[follow[id_col].isin(usable_ids), wave_col].min()) if usable_ids else None
        ),
    }


def summarize_multimorbidity(
    df: pd.DataFrame,
    id_col: str,
    wave_col: str,
    baseline_wave: int,
    followup_waves: list[int],
    disease_vars: list[str],
) -> dict[str, Any]:
    baseline = (
        df[df[wave_col] == baseline_wave][[id_col] + disease_vars]
        .drop_duplicates(subset=[id_col])
        .copy()
    )
    followup = df[df[wave_col].isin(followup_waves)][[id_col, wave_col] + disease_vars].copy()
    base_count = sum(yes_mask(baseline[var]).astype(int) for var in disease_vars)
    base_observed = sum(baseline[var].notna().astype(int) for var in disease_vars)
    eligible_ids = set(baseline.loc[(base_observed >= 3) & (base_count < 2), id_col])
    follow = followup[followup[id_col].isin(eligible_ids)].copy()
    if follow.empty:
        return {"n_baseline": len(eligible_ids), "n_followup": 0, "n_events": 0, "event_rate": None}
    follow_count = sum(yes_mask(follow[var]).astype(int) for var in disease_vars)
    follow_observed = sum(follow[var].notna().astype(int) for var in disease_vars)
    observed = follow[follow_observed >= 3]
    events = (
        follow.loc[(follow_observed >= 3) & (follow_count >= 2)].groupby(id_col)[wave_col].min()
    )
    n_followup = int(observed[id_col].nunique())
    n_events = int(events.shape[0])
    return {
        "n_baseline": int(len(eligible_ids)),
        "n_followup": n_followup,
        "n_events": n_events,
        "event_rate": round(n_events / n_followup, 6) if n_followup else None,
        "minimum_event_wave": int(events.min()) if not events.empty else None,
    }


def summarize_mortality_review(
    df: pd.DataFrame,
    id_col: str,
    wave_col: str,
    baseline_wave: int,
    followup_waves: list[int],
    mortality_vars: list[str],
) -> dict[str, Any]:
    baseline_ids = set(df.loc[df[wave_col] == baseline_wave, id_col].dropna())
    followup = df[df[wave_col].isin(followup_waves)].copy()
    followup_ids = set(followup[id_col].dropna())
    review_ids = set()
    if "radyear" in mortality_vars:
        review_ids.update(followup.loc[followup["radyear"].notna(), id_col].dropna())
    if "radmonth" in mortality_vars:
        review_ids.update(followup.loc[followup["radmonth"].notna(), id_col].dropna())
    if "iwstat" in mortality_vars:
        review_ids.update(followup.loc[yes_mask(followup["iwstat"]), id_col].dropna())
    eligible_followup_ids = baseline_ids & followup_ids
    review_ids = review_ids & eligible_followup_ids
    n_followup = len(eligible_followup_ids)
    n_events = len(review_ids)
    return {
        "n_baseline": int(len(baseline_ids)),
        "n_followup": int(n_followup),
        "n_events": int(n_events),
        "event_rate": round(n_events / n_followup, 6) if n_followup else None,
        "minimum_event_wave": None,
    }


def missing_rates(
    df: pd.DataFrame,
    wave_col: str,
    baseline_wave: int,
    followup_waves: list[int],
    variables: list[str],
) -> tuple[float | None, float | None]:
    available = [v for v in variables if v in df]
    if not available:
        return None, None
    baseline = df[df[wave_col] == baseline_wave][available]
    followup = df[df[wave_col].isin(followup_waves)][available]
    base_rate = float(baseline.isna().mean().mean()) if not baseline.empty else None
    follow_rate = float(followup.isna().mean().mean()) if not followup.empty else None
    return (
        round(base_rate, 6) if base_rate is not None else None,
        round(follow_rate, 6) if follow_rate is not None else None,
    )


def event_tier(
    n_events: Any,
    event_rate: Any,
    missing_baseline: Any,
    missing_followup: Any,
    sensitivity_only: bool = False,
) -> tuple[str, str, str]:
    if sensitivity_only:
        return "exploratory_outcomes", "C", "usable_with_sensitivity"
    if n_events is None:
        return "secondary_outcomes", "B", "usable_continuous_change"
    if n_events >= 500 and (event_rate is not None and 0.01 <= event_rate <= 0.8):
        if (missing_baseline or 0) <= 0.5 and (missing_followup or 0) <= 0.5:
            return "secondary_outcomes", "B", "usable_draft"
    if n_events >= 100:
        return "exploratory_outcomes", "C", "usable_but_review"
    return "not_usable_outcomes", "D", "too_sparse_or_unstable"


def registry_row(
    outcome_id: str,
    outcome_name: str,
    domain: str,
    source_variables: list[str],
    definition_type: str,
    summary: dict[str, Any],
    missing_baseline: float | None,
    missing_followup: float | None,
    baseline_wave: int,
    followup_waves: list[int],
    event_definition: str = "",
    worsening_definition: str = "",
    continuous_change_definition: str = "",
    leakage_notes: str = "Requires per-outcome predictor exclusion before modeling.",
    recommended_metrics: str = "ROC-AUC; PR-AUC; calibration; risk quartile enrichment",
    definition_reference: str = "API-derived draft; literature check pending",
    supporting_literature: str = "pending",
    literature_support_level: str = "pending",
    internal_validation_checks: str = "event counts, age/sex gradients, wave timing, missingness",
    sensitivity_only: bool = False,
    force_registry_tier: str | None = None,
    force_evidence_tier: str | None = None,
    force_status: str | None = None,
    force_usable: bool | None = None,
) -> dict[str, Any]:
    tier, evidence, status = event_tier(
        summary.get("n_events"),
        summary.get("event_rate"),
        missing_baseline,
        missing_followup,
        sensitivity_only=sensitivity_only,
    )
    if force_registry_tier is not None:
        tier = force_registry_tier
    if force_evidence_tier is not None:
        evidence = force_evidence_tier
    if force_status is not None:
        status = force_status
    usable = evidence in {"B", "C"}
    if force_usable is not None:
        usable = force_usable
    return {
        "outcome_id": outcome_id,
        "outcome_name": outcome_name,
        "domain": domain,
        "source_variables": "|".join(source_variables),
        "definition_type": definition_type,
        "baseline_wave": baseline_wave,
        "followup_waves": "|".join(map(str, followup_waves)),
        "baseline_eligibility": "baseline wave observed and baseline-negative when incident outcome",
        "event_definition": event_definition,
        "worsening_definition": worsening_definition,
        "continuous_change_definition": continuous_change_definition,
        "n_baseline": summary.get("n_baseline"),
        "n_followup": summary.get("n_followup"),
        "n_events": summary.get("n_events"),
        "event_rate": summary.get("event_rate"),
        "missing_rate_baseline": missing_baseline,
        "missing_rate_followup": missing_followup,
        "usable": usable,
        "registry_tier": tier,
        "leakage_notes": leakage_notes,
        "recommended_metrics": recommended_metrics,
        "definition_reference": definition_reference,
        "supporting_literature": supporting_literature,
        "literature_support_level": literature_support_level,
        "internal_validation_checks": internal_validation_checks,
        "internal_validation_status": status,
        "evidence_tier": evidence,
        "notes": "Draft generated from API-backed CHARLS main table; not final until literature and leakage review.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--api-base")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--row-limit", type=int)
    args = parser.parse_args()

    config = load_config(args.config)
    api_base = args.api_base or config["api"]["base_url"]
    source_id = config["source"]["id"]
    main_file = config["source"]["main_file"]
    id_col = config["source"]["id_column"]
    wave_col = config["source"]["wave_column"]
    row_limit = args.row_limit or int(config["source"].get("full_row_limit", 100000))
    baseline_wave = int(config["outcomes"]["baseline_wave"])
    followup_waves = [int(w) for w in config["outcomes"]["followup_waves"]]
    cesd_cut = float(config["outcomes"]["cesd10_high_symptom_cutpoint"])
    low_cog_quantile = float(config["outcomes"]["low_cognition_quantile"])

    output_dir = args.output_dir or PROJECT_ROOT / config["outputs"].get(
        "outcome_registry_directory", config["outputs"]["directory"]
    )
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_dir = PROJECT_ROOT / config["outputs"].get(
        "data_audit_directory", config["outputs"]["directory"]
    )
    if not audit_dir.is_absolute():
        audit_dir = PROJECT_ROOT / audit_dir
    schema_path = audit_dir / "charls_schema.csv"
    if not schema_path.exists():
        raise RuntimeError(f"Missing {schema_path}; run data_audit/audit_charls_schema.py first")
    columns = available_columns(schema_path)

    chronic_candidates = [
        ("hibpe", "hypertension", "高血压"),
        ("dyslipe", "dyslipidemia", "血脂异常"),
        ("diabe", "diabetes", "糖尿病"),
        ("cancre", "cancer", "癌症"),
        ("lunge", "lung_disease", "肺病"),
        ("livere", "liver_disease", "肝脏疾病"),
        ("hearte", "heart_disease", "心脏病"),
        ("stroke", "stroke", "中风"),
        ("kidneye", "kidney_disease", "肾脏疾病"),
        ("digeste", "digestive_disease", "胃病"),
        ("psyche", "psychiatric_disease", "精神疾病"),
        ("memrye", "memory_disease", "记忆疾病"),
        ("arthre", "arthritis", "关节炎"),
        ("asthmae", "asthma", "哮喘病"),
    ]
    adl_items = [v for v in ["dressa", "batha", "eata", "beda", "toilta", "urina"] if v in columns]
    iadl_items = [
        v for v in ["housewka", "mealsa", "shopa", "phonea", "medsa", "moneya"] if v in columns
    ]
    continuous_outcomes = [
        ("total_cognition", "total_cognition_change", "cognition", "Total cognition change"),
        ("imrc", "immediate_recall_change", "cognition", "Immediate recall change"),
        ("dlrc", "delayed_recall_change", "cognition", "Delayed recall change"),
        ("cesd10", "cesd10_change", "mental_health", "CESD10 symptom score change"),
        ("bmi", "bmi_change", "physical_measures", "BMI change"),
        (
            "systo",
            "systolic_blood_pressure_change",
            "physical_measures",
            "Systolic blood pressure change",
        ),
    ]

    needed = {id_col, wave_col}
    needed.update([v for v, _, _ in chronic_candidates if v in columns])
    needed.update(adl_items)
    needed.update(iadl_items)
    needed.update([v for v, _, _, _ in continuous_outcomes if v in columns])
    for optional in ["hospital", "hospital_time", "doctor_time", "iwstat", "radyear", "radmonth"]:
        if optional in columns:
            needed.add(optional)
    df = extract_columns(api_base, source_id, main_file, sorted(needed), row_limit)
    df[wave_col] = df[wave_col].map(normalize_wave)

    rows: list[dict[str, Any]] = []
    disease_vars = []
    for variable, slug, label in chronic_candidates:
        if variable not in df:
            continue
        disease_vars.append(variable)
        summary = summarize_binary_incident(
            df, id_col, wave_col, baseline_wave, followup_waves, [variable], f"incident_{slug}"
        )
        miss_base, miss_follow = missing_rates(
            df, wave_col, baseline_wave, followup_waves, [variable]
        )
        rows.append(
            registry_row(
                outcome_id=f"incident_{slug}",
                outcome_name=f"Incident {label}",
                domain="chronic_disease",
                source_variables=[variable],
                definition_type="binary incident outcome",
                summary=summary,
                missing_baseline=miss_base,
                missing_followup=miss_follow,
                baseline_wave=baseline_wave,
                followup_waves=followup_waves,
                event_definition=f"Baseline {variable}=0 and any follow-up {variable}=1.",
                supporting_literature="CHARLS chronic disease self-report convention; exact citation pending",
                literature_support_level="needs citation",
            )
        )

    if disease_vars:
        summary = summarize_multimorbidity(
            df, id_col, wave_col, baseline_wave, followup_waves, disease_vars
        )
        miss_base, miss_follow = missing_rates(
            df, wave_col, baseline_wave, followup_waves, disease_vars
        )
        rows.append(
            registry_row(
                outcome_id="multimorbidity_onset",
                outcome_name="Multimorbidity onset",
                domain="multimorbidity",
                source_variables=disease_vars,
                definition_type="binary incident outcome",
                summary=summary,
                missing_baseline=miss_base,
                missing_followup=miss_follow,
                baseline_wave=baseline_wave,
                followup_waves=followup_waves,
                event_definition="Baseline chronic disease count <2 and any follow-up chronic disease count >=2.",
                supporting_literature="multimorbidity count threshold convention; exact CHARLS citation pending",
                literature_support_level="needs citation",
            )
        )

    if adl_items:
        summary = summarize_binary_incident(
            df, id_col, wave_col, baseline_wave, followup_waves, adl_items, "adl_new_difficulty"
        )
        miss_base, miss_follow = missing_rates(
            df, wave_col, baseline_wave, followup_waves, adl_items
        )
        rows.append(
            registry_row(
                outcome_id="adl_new_difficulty",
                outcome_name="ADL new difficulty",
                domain="function",
                source_variables=adl_items,
                definition_type="binary incident outcome",
                summary=summary,
                missing_baseline=miss_base,
                missing_followup=miss_follow,
                baseline_wave=baseline_wave,
                followup_waves=followup_waves,
                event_definition="No baseline ADL item difficulty and any follow-up ADL item difficulty.",
                supporting_literature="ADL item convention; exact CHARLS citation pending",
                literature_support_level="needs citation",
            )
        )

    if iadl_items:
        summary = summarize_binary_incident(
            df, id_col, wave_col, baseline_wave, followup_waves, iadl_items, "iadl_new_difficulty"
        )
        miss_base, miss_follow = missing_rates(
            df, wave_col, baseline_wave, followup_waves, iadl_items
        )
        rows.append(
            registry_row(
                outcome_id="iadl_new_difficulty",
                outcome_name="IADL new difficulty",
                domain="function",
                source_variables=iadl_items,
                definition_type="binary incident outcome",
                summary=summary,
                missing_baseline=miss_base,
                missing_followup=miss_follow,
                baseline_wave=baseline_wave,
                followup_waves=followup_waves,
                event_definition="No baseline IADL item difficulty and any follow-up IADL item difficulty.",
                supporting_literature="IADL item convention; exact CHARLS citation pending",
                literature_support_level="needs citation",
            )
        )

    if "cesd10" in df:
        summary = summarize_binary_onset_from_threshold(
            df, id_col, wave_col, baseline_wave, followup_waves, "cesd10", cesd_cut, "ge"
        )
        miss_base, miss_follow = missing_rates(
            df, wave_col, baseline_wave, followup_waves, ["cesd10"]
        )
        rows.append(
            registry_row(
                outcome_id="high_depressive_symptoms_onset",
                outcome_name="High depressive symptoms onset",
                domain="mental_health",
                source_variables=["cesd10"],
                definition_type="binary incident outcome",
                summary=summary,
                missing_baseline=miss_base,
                missing_followup=miss_follow,
                baseline_wave=baseline_wave,
                followup_waves=followup_waves,
                event_definition=f"Baseline CESD10 < {cesd_cut:g} and any follow-up CESD10 >= {cesd_cut:g}.",
                supporting_literature="CESD10 cutpoint convention in aging studies; exact CHARLS citation pending",
                literature_support_level="needs citation",
            )
        )

    if "total_cognition" in df:
        baseline_values = to_numeric(
            df.loc[df[wave_col] == baseline_wave, "total_cognition"]
        ).dropna()
        if not baseline_values.empty:
            threshold = float(baseline_values.quantile(low_cog_quantile))
            summary = summarize_binary_onset_from_threshold(
                df,
                id_col,
                wave_col,
                baseline_wave,
                followup_waves,
                "total_cognition",
                threshold,
                "le",
            )
            miss_base, miss_follow = missing_rates(
                df, wave_col, baseline_wave, followup_waves, ["total_cognition"]
            )
            rows.append(
                registry_row(
                    outcome_id="low_cognition_onset_data_driven",
                    outcome_name="Low cognition onset, data-driven threshold",
                    domain="cognition",
                    source_variables=["total_cognition"],
                    definition_type="binary worsening outcome",
                    summary=summary,
                    missing_baseline=miss_base,
                    missing_followup=miss_follow,
                    baseline_wave=baseline_wave,
                    followup_waves=followup_waves,
                    event_definition=f"Baseline total_cognition above wave-1 Q{int(low_cog_quantile * 100)} ({threshold:.3f}) and any follow-up at or below threshold.",
                    supporting_literature="data-driven draft threshold; replace with literature-supported definition before primary use",
                    literature_support_level="weak",
                )
            )

    for variable, outcome_id, domain, name in continuous_outcomes:
        if variable not in df:
            continue
        summary = summarize_continuous_change(
            df, id_col, wave_col, baseline_wave, followup_waves, variable
        )
        miss_base, miss_follow = missing_rates(
            df, wave_col, baseline_wave, followup_waves, [variable]
        )
        rows.append(
            registry_row(
                outcome_id=outcome_id,
                outcome_name=name,
                domain=domain,
                source_variables=[variable],
                definition_type="continuous change outcome",
                summary=summary,
                missing_baseline=miss_base,
                missing_followup=miss_follow,
                baseline_wave=baseline_wave,
                followup_waves=followup_waves,
                continuous_change_definition=f"Change in {variable} from baseline wave to follow-up waves.",
                recommended_metrics="RMSE; MAE; R2; Spearman correlation; clinically meaningful decline classification if threshold validated",
            )
        )

    if "hospital" in df:
        summary = summarize_binary_incident(
            df,
            id_col,
            wave_col,
            baseline_wave,
            followup_waves,
            ["hospital"],
            "hospitalization_onset",
        )
        miss_base, miss_follow = missing_rates(
            df, wave_col, baseline_wave, followup_waves, ["hospital"]
        )
        rows.append(
            registry_row(
                outcome_id="hospitalization_onset",
                outcome_name="Hospitalization onset",
                domain="medical_utilization",
                source_variables=["hospital"],
                definition_type="binary incident outcome",
                summary=summary,
                missing_baseline=miss_base,
                missing_followup=miss_follow,
                baseline_wave=baseline_wave,
                followup_waves=followup_waves,
                event_definition="Baseline hospital=0 and any follow-up hospital=1.",
            )
        )

    mortality_vars = [v for v in ["iwstat", "radyear", "radmonth"] if v in df]
    if mortality_vars:
        summary = summarize_mortality_review(
            df, id_col, wave_col, baseline_wave, followup_waves, mortality_vars
        )
        miss_base, miss_follow = missing_rates(
            df, wave_col, baseline_wave, followup_waves, mortality_vars
        )
        rows.append(
            registry_row(
                outcome_id="death_by_followup_sensitivity",
                outcome_name="Death by follow-up, sensitivity endpoint",
                domain="mortality",
                source_variables=mortality_vars,
                definition_type="time-to-event outcome",
                summary=summary,
                missing_baseline=miss_base,
                missing_followup=miss_follow,
                baseline_wave=baseline_wave,
                followup_waves=followup_waves,
                event_definition="Draft signal only: any follow-up radyear/radmonth non-missing or iwstat=1 among baseline-wave participants with follow-up.",
                recommended_metrics="C-index; Kaplan-Meier by risk quartile; Cox HR after timing validation",
                leakage_notes="Death year/month must be excluded from predictors for mortality models.",
                supporting_literature="mortality timing requires source-specific validation; iwstat and radyear/radmonth disagree under the current draft extraction",
                literature_support_level="pending",
                sensitivity_only=True,
                force_registry_tier="not_usable_outcomes",
                force_evidence_tier="D",
                force_status="timing_semantics_review_required",
                force_usable=False,
            )
        )

    registry = pd.DataFrame(rows)
    registry = registry.sort_values(["evidence_tier", "domain", "outcome_id"]).reset_index(
        drop=True
    )
    registry_path = output_dir / "outcome_registry_draft.csv"
    registry.to_csv(registry_path, index=False)

    report_lines = [
        "# Outcome Registry Draft Report",
        "",
        f"Source ID: `{source_id}`",
        f"API endpoint used: `GET /extract?source={source_id}&file={main_file}&columns=<candidate outcome variables>&limit={row_limit}&format=json`",
        f"Baseline wave: {baseline_wave}",
        f"Follow-up waves: {'|'.join(map(str, followup_waves))}",
        "",
        "## Draft Counts",
        "",
        f"- Candidate outcomes: {len(registry)}",
        f"- Evidence tier B: {int((registry['evidence_tier'] == 'B').sum())}",
        f"- Evidence tier C: {int((registry['evidence_tier'] == 'C').sum())}",
        f"- Evidence tier D: {int((registry['evidence_tier'] == 'D').sum())}",
        "",
        "## Important Caveat",
        "",
        "This file is a draft registry. It uses real CHARLS variables and observed wave timing, but literature references, cut points, leakage exclusions, and internal plausibility checks still need to be completed before model training.",
        "",
        "## Mortality Timing Warning",
        "",
        "`death_by_followup_sensitivity` is intentionally marked evidence tier D. The draft extraction finds death-timing signals in `radyear/radmonth`, while `iwstat` has very few positive rows, so mortality needs a source-specific timing review before use.",
        "",
    ]
    (output_dir / "outcome_registry_draft_report.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    print(f"Wrote {registry_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
