#!/usr/bin/env python3
"""Phase 0 discovery and crosswalk drafts for first-tier cohorts."""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
API_BASE = "http://127.0.0.1:8787"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "09_external_validation"
CHARLS_VARIABLE_MAP = PROJECT_ROOT / "outputs" / "05_tokenization" / "charls_variable_type_map.csv"
HRS_VARIABLE_CROSSWALK = OUTPUT_ROOT / "external_variable_crosswalk.csv"

SELECTED_OUTCOMES = [
    "adl_new_difficulty",
    "iadl_new_difficulty",
    "incident_diabetes",
    "incident_hypertension",
    "multimorbidity_onset",
    "high_depressive_symptoms_onset",
    "hospitalization_onset",
    "low_cognition_onset_data_driven",
]

OUTCOME_NAMES = {
    "adl_new_difficulty": "ADL new difficulty",
    "iadl_new_difficulty": "IADL new difficulty",
    "incident_diabetes": "Incident diabetes",
    "incident_hypertension": "Incident hypertension",
    "multimorbidity_onset": "Multimorbidity onset",
    "high_depressive_symptoms_onset": "High depressive symptoms onset",
    "hospitalization_onset": "Hospitalization onset",
    "low_cognition_onset_data_driven": "Low cognition onset, data-driven threshold",
}

OUTCOME_DOMAINS = {
    "adl_new_difficulty": "function",
    "iadl_new_difficulty": "function",
    "incident_diabetes": "chronic_disease",
    "incident_hypertension": "chronic_disease",
    "multimorbidity_onset": "multimorbidity",
    "high_depressive_symptoms_onset": "mental_health",
    "hospitalization_onset": "medical_utilization",
    "low_cognition_onset_data_driven": "cognition",
}

CHARLS_CHRONIC_14 = [
    "hibpe",
    "dyslipe",
    "diabe",
    "cancre",
    "lunge",
    "livere",
    "hearte",
    "stroke",
    "kidneye",
    "digeste",
    "psyche",
    "memrye",
    "arthre",
    "asthmae",
]

CHARLS_OUTCOME_SOURCES = {
    "adl_new_difficulty": "dressa|batha|eata|beda|toilta|urina",
    "iadl_new_difficulty": "housewka|mealsa|shopa|phonea|medsa|moneya",
    "incident_diabetes": "diabe",
    "incident_hypertension": "hibpe",
    "multimorbidity_onset": "|".join(CHARLS_CHRONIC_14),
    "high_depressive_symptoms_onset": "cesd10",
    "hospitalization_onset": "hospital",
    "low_cognition_onset_data_driven": "total_cognition",
}


@dataclass(frozen=True)
class CohortConfig:
    source_id: str
    label: str
    main_file: str
    id_column: str
    wave_column: str
    interview_year_column: str
    validation_waves: list[int]
    outcome_sources: dict[str, list[str]]
    common_chronic_used: list[str]
    manual_aliases: dict[str, str]
    depression_variable: str
    depression_aux_variable: str | None
    depression_note: str
    low_cognition_variable: str | None
    low_cognition_components: list[str]
    low_cognition_note: str
    extra_base_columns: list[str]
    notes: list[str]


COMMON_ALIASES = {
    "ID": "{id}",
    "age": "agey",
    "ragey_e": "agey",
    "raeduc_c": "raeducl",
    "rural2": "rural",
    "hrural": "rural",
    "hospital": "hosp1y",
    "hospital_time": "hsptim1y",
    "smokef": "smokef",
    "smoken": "smoken",
    "drinkl": "drink",
    "doctor": "doctor1y",
    "doctim": "doctim1y",
}

KLOSA_ALIASES = {
    **COMMON_ALIASES,
    "hospital": "",
    "hospital_time": "",
    "doctor": "doctor",
    "doctim": "doctim",
    "drink": "drink",
    "dressa": "dressb",
    "batha": "bathb",
    "eata": "eatb",
    "beda": "bedb_k",
    "toilta": "toiltb",
    "urina": "urinb",
    "housewka": "housewkb",
    "mealsa": "mealsb",
    "shopa": "shopb",
    "medsa": "medsb",
    "moneya": "moneyb",
    "phonea": "phoneb",
    "cesd10": "cesd10b",
    "total_cognition": "cog_total",
}

ELSA_ALIASES = {
    **COMMON_ALIASES,
    "hospital": "",
    "hospital_time": "",
    "doctor": "",
    "doctim": "",
    "drink": "drink",
    "sleepe": "",
    "sleep_night": "",
    "total_cognition": "tcog_z_z",
    "cesd10": "cesd",
}

SHARE_ALIASES = {
    **COMMON_ALIASES,
    "drink": "drinkx",
    "cesd10": "eurod",
    "total_cognition": "",
    "walksa": "walkra",
}

COHORTS = {
    "klosa": CohortConfig(
        source_id="klosa",
        label="KLoSA",
        main_file="analysis_data/klosa.parquet",
        id_column="pid",
        wave_column="wave",
        interview_year_column="iwy",
        validation_waves=[3, 4, 5, 6, 7, 8],
        outcome_sources={
            "adl_new_difficulty": ["dressb", "bathb", "eatb", "toiltb", "bedb_k", "urinb"],
            "iadl_new_difficulty": ["housewkb", "mealsb", "shopb", "phoneb", "medsb", "moneyb"],
            "incident_diabetes": ["diabe"],
            "incident_hypertension": ["hibpe"],
            "multimorbidity_onset": [
                "hibpe",
                "diabe",
                "cancre",
                "lunge",
                "hearte",
                "stroke",
                "psyche",
                "arthre",
                "livere",
            ],
            "high_depressive_symptoms_onset": ["cesd10a", "cesd10b"],
            "hospitalization_onset": [],
            "low_cognition_onset_data_driven": ["cog_total"],
        },
        common_chronic_used=[
            "hibpe",
            "diabe",
            "cancre",
            "lunge",
            "hearte",
            "stroke",
            "psyche",
            "arthre",
            "livere",
        ],
        manual_aliases=KLOSA_ALIASES,
        depression_variable="cesd10b",
        depression_aux_variable=None,
        depression_note="KLoSA exposes CESD-10 as cesd10a for waves 3-4 and cesd10b for waves 5-8; Phase 1 should coalesce them and confirm whether the onset cutpoint uses >=10 or a cohort-calibrated rule.",
        low_cognition_variable="cog_total",
        low_cognition_components=[],
        low_cognition_note="Use KLoSA cognition total with cohort-specific baseline Q25.",
        extra_base_columns=[],
        notes=[
            "KLoSA uses b-suffixed ADL/IADL item names in this harmonized file.",
            "KLoSA `digeste` exists but has no wave-3/wave-4 coverage, so it is excluded from the Phase 1 primary multimorbidity definition.",
        ],
    ),
    "elsa": CohortConfig(
        source_id="elsa",
        label="ELSA",
        main_file="analysis_data/elsa.parquet",
        id_column="idauniqc",
        wave_column="wave",
        interview_year_column="iwindy",
        validation_waves=[5, 6, 7, 8, 9],
        outcome_sources={
            "adl_new_difficulty": ["dressa", "batha", "eata", "beda", "toilta", "walkra"],
            "iadl_new_difficulty": ["housewka", "mealsa", "shopa", "phonea", "medsa", "moneya"],
            "incident_diabetes": ["diabe"],
            "incident_hypertension": ["hibpe"],
            "multimorbidity_onset": [
                "hibpe",
                "diabe",
                "cancre",
                "lunge",
                "hearte",
                "stroke",
                "psyche",
                "arthre",
                "asthmae",
                "memrye",
            ],
            "high_depressive_symptoms_onset": ["cesd", "depressive"],
            "hospitalization_onset": [],
            "low_cognition_onset_data_driven": ["tcog_z_z"],
        },
        common_chronic_used=[
            "hibpe",
            "diabe",
            "cancre",
            "lunge",
            "hearte",
            "stroke",
            "psyche",
            "arthre",
            "asthmae",
            "memrye",
        ],
        manual_aliases=ELSA_ALIASES,
        depression_variable="cesd",
        depression_aux_variable="depressive",
        depression_note="ELSA exposes CESD plus a derived depressive flag; infer the CESD cutpoint from observed flag concordance.",
        low_cognition_variable="tcog_z_z",
        low_cognition_components=[],
        low_cognition_note="Use ELSA total cognition z-score with cohort-specific baseline Q25.",
        extra_base_columns=[],
        notes=[
            "ELSA hospitalization is not present in the current analysis schema and is marked unmappable in Phase 0."
        ],
    ),
    "share": CohortConfig(
        source_id="share",
        label="SHARE",
        main_file="analysis_data/share.parquet",
        id_column="mergeid",
        wave_column="wave",
        interview_year_column="iwy",
        validation_waves=[4, 5, 6, 7, 8],
        outcome_sources={
            "adl_new_difficulty": ["dressa", "batha", "eata", "beda", "toilta", "walkra"],
            "iadl_new_difficulty": ["housewka", "mealsa", "shopa", "phonea", "medsa", "moneya"],
            "incident_diabetes": ["diabe"],
            "incident_hypertension": ["hibpe"],
            "multimorbidity_onset": [
                "hibpe",
                "diabe",
                "cancre",
                "lunge",
                "hearte",
                "stroke",
                "psyche",
                "arthre",
                "asthmae",
                "kidneye",
            ],
            "high_depressive_symptoms_onset": ["eurod"],
            "hospitalization_onset": ["hosp1y"],
            "low_cognition_onset_data_driven": ["imrc", "dlrc", "ser7", "orient"],
        },
        common_chronic_used=[
            "hibpe",
            "diabe",
            "cancre",
            "lunge",
            "hearte",
            "stroke",
            "psyche",
            "arthre",
            "asthmae",
            "kidneye",
        ],
        manual_aliases=SHARE_ALIASES,
        depression_variable="eurod",
        depression_aux_variable=None,
        depression_note="SHARE uses EURO-D rather than CESD; Phase 1 should treat the high-symptom rule as a cohort-specific decision.",
        low_cognition_variable=None,
        low_cognition_components=["imrc", "dlrc", "ser7", "orient"],
        low_cognition_note="Derive a simple cognition proxy from imrc+dlrc+ser7+orient, then use baseline Q25.",
        extra_base_columns=["country"],
        notes=[
            "SHARE wave 3 is SHARELIFE and is absent from this analysis file; validation uses waves 4-8."
        ],
    ),
}


def get_json(
    base_url: str, path: str, params: dict[str, Any] | None = None, timeout: int = 300
) -> dict[str, Any]:
    query = "?" + urllib.parse.urlencode(params, doseq=False) if params else ""
    url = base_url.rstrip("/") + path + query
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"HTTP {exc.code} for {url}: {body[:800]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach {url}: {exc}") from exc


def post_json(
    base_url: str, path: str, payload: dict[str, Any], timeout: int = 300
) -> dict[str, Any]:
    url = base_url.rstrip("/") + path
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"HTTP {exc.code} for {url}: {body_text[:800]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach {url}: {exc}") from exc


def fetch_schema(config: CohortConfig, api_base: str) -> list[str]:
    payload = get_json(
        api_base, "/schema", {"source": config.source_id, "file": config.main_file}, timeout=300
    )
    columns = payload.get("columns") or []
    if not columns:
        raise RuntimeError(f"{config.source_id} schema returned no columns")
    return [str(column) for column in columns]


def resolve_main_file(config: CohortConfig, api_base: str) -> dict[str, Any]:
    return get_json(
        api_base, "/resolve", {"source": config.source_id, "file": config.main_file}, timeout=300
    )


def extract_columns(
    config: CohortConfig, columns: list[str], api_base: str, row_limit: int
) -> pd.DataFrame:
    requested = list(
        dict.fromkeys(
            [
                config.id_column,
                config.wave_column,
                config.interview_year_column,
                *config.extra_base_columns,
                *columns,
            ]
        )
    )
    payload = get_json(
        api_base,
        "/extract",
        {
            "source": config.source_id,
            "file": config.main_file,
            "columns": ",".join(requested),
            "limit": row_limit,
            "format": "json",
        },
        timeout=900,
    )
    records = payload.get("records") or []
    if not records:
        raise RuntimeError(f"{config.source_id} extract returned no records")
    frame = pd.DataFrame.from_records(records)
    frame[config.wave_column] = pd.to_numeric(frame[config.wave_column], errors="coerce").astype(
        "Int64"
    )
    return frame


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def fmt(value: Any, digits: int = 4) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def generated_timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def write_text(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame[columns].iterrows():
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return lines


def validation_window(config: CohortConfig, frame: pd.DataFrame) -> pd.DataFrame:
    wave = pd.to_numeric(frame[config.wave_column], errors="coerce")
    return frame.loc[wave.isin(config.validation_waves)].copy()


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def candidate_alias(config: CohortConfig, variable: str) -> str:
    value = config.manual_aliases.get(variable, "")
    if value == "{id}":
        return config.id_column
    return value


def build_variable_crosswalk(config: CohortConfig, schema: set[str]) -> pd.DataFrame:
    charls_map = pd.read_csv(CHARLS_VARIABLE_MAP)
    rows: list[dict[str, Any]] = []
    for _, row in charls_map.iterrows():
        variable = str(row["variable"])
        is_safe = as_bool(row.get("baseline_predictor_general", False))
        include_pretraining = as_bool(row.get("include_in_pretraining_v1", False))
        if not is_safe and not include_pretraining:
            continue

        alias = candidate_alias(config, variable)
        if variable in schema:
            cohort_variable = variable
            status = "mapped"
            mapping_type = "same_name"
            reason = f"{config.label} schema contains the same harmonized variable name"
        elif alias and alias in schema:
            cohort_variable = alias
            status = "mapped"
            mapping_type = "manual"
            reason = f"Manual harmonized alias {variable}->{cohort_variable}"
        else:
            cohort_variable = ""
            status = "unmappable"
            mapping_type = "none"
            reason = f"No {config.label} schema variable found for this CHARLS predictor"

        rows.append(
            {
                "charls_variable": variable,
                "cohort": config.source_id,
                "cohort_variable": cohort_variable,
                "status": status,
                "mapping_type": mapping_type,
                "reason": reason,
                "charls_module": row.get("module", ""),
                "charls_type_model": row.get("type_model", ""),
                "charls_tokenization_type": row.get("tokenization_type", ""),
                "charls_baseline_predictor_general": is_safe,
                "api_schema_verified": bool(cohort_variable and cohort_variable in schema),
                "api_file": config.main_file if cohort_variable else "",
            }
        )
    return pd.DataFrame(rows).sort_values(["status", "charls_variable"], kind="stable")


def build_hrs_translation_summary(config: CohortConfig, schema: set[str]) -> dict[str, Any]:
    if not HRS_VARIABLE_CROSSWALK.exists():
        return {
            "hrs_safe_mapped_total": 0,
            "hrs_safe_direct_hits": 0,
            "hrs_safe_alias_hits": 0,
            "hrs_safe_unmapped": 0,
            "hrs_crosswalk_hit_rate": math.nan,
        }
    hrs = pd.read_csv(HRS_VARIABLE_CROSSWALK)
    safe = hrs[
        hrs["charls_baseline_predictor_general"].astype(bool) & hrs["status"].eq("mapped")
    ].copy()
    direct = 0
    alias_hits = 0
    for _, row in safe.iterrows():
        hrs_variable = str(row.get("hrs_variable", ""))
        charls_variable = str(row.get("charls_variable", ""))
        alias = candidate_alias(config, charls_variable) or candidate_alias(config, hrs_variable)
        if hrs_variable in schema:
            direct += 1
        elif charls_variable in schema or (alias and alias in schema):
            alias_hits += 1
    total = int(len(safe))
    hits = direct + alias_hits
    return {
        "hrs_safe_mapped_total": total,
        "hrs_safe_direct_hits": direct,
        "hrs_safe_alias_hits": alias_hits,
        "hrs_safe_unmapped": total - hits,
        "hrs_crosswalk_hit_rate": hits / total if total else math.nan,
    }


def build_outcome_crosswalk(
    config: CohortConfig,
    schema: set[str],
    depression_threshold_note: str,
    low_cog_threshold_note: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for outcome_id in SELECTED_OUTCOMES:
        variables = config.outcome_sources[outcome_id]
        missing = sorted(set(variables) - schema)
        if not variables:
            status = "unmappable"
        elif missing:
            status = "unmappable"
        else:
            status = "mapped"

        note = "Cohort-native definition in the Phase 0 validation window."
        if outcome_id == "multimorbidity_onset":
            note = "Use cohort intersection with CHARLS chronic flags: " + "|".join(
                config.common_chronic_used
            )
        elif outcome_id == "high_depressive_symptoms_onset":
            note = depression_threshold_note
        elif outcome_id == "low_cognition_onset_data_driven":
            note = low_cog_threshold_note
        elif outcome_id == "hospitalization_onset" and not variables:
            note = f"No hospitalization variable was found in {config.label} analysis schema during Phase 0."
        elif outcome_id == "adl_new_difficulty" and config.source_id == "klosa":
            note = "KLoSA ADL uses b-suffixed item variables."

        rows.append(
            {
                "outcome_id": outcome_id,
                "outcome_name": OUTCOME_NAMES[outcome_id],
                "domain": OUTCOME_DOMAINS[outcome_id],
                "charls_source_variables": CHARLS_OUTCOME_SOURCES[outcome_id],
                "cohort_source_variables": "|".join(variables),
                "status": status,
                "missing_cohort_variables": "|".join(missing),
                "definition_note": note,
                "api_schema_verified": bool(variables and not missing),
                "api_file": config.main_file if variables else "",
            }
        )
    return pd.DataFrame(rows)


def wave_summary(config: CohortConfig, window: pd.DataFrame) -> pd.DataFrame:
    year_col = config.interview_year_column
    out = (
        window.groupby(config.wave_column, as_index=False)
        .agg(
            interview_year_min=(year_col, "min"),
            interview_year_max=(year_col, "max"),
            interview_year_median=(year_col, "median"),
            rows=(config.id_column, "size"),
            ids=(config.id_column, "nunique"),
        )
        .rename(columns={config.wave_column: "wave"})
        .sort_values("wave", kind="stable")
    )
    out["wave"] = out["wave"].astype(int)
    return out[
        ["wave", "interview_year_min", "interview_year_max", "interview_year_median", "rows", "ids"]
    ]


def panel_summary(config: CohortConfig, window: pd.DataFrame) -> dict[str, Any]:
    waves_per_id = window.groupby(config.id_column)[config.wave_column].nunique()
    return {
        "cohort": config.source_id,
        "rows": int(len(window)),
        "ids": int(window[config.id_column].nunique()),
        "ids_ge2_waves": int(waves_per_id.ge(2).sum()),
        "pct_ids_ge2_waves": float(waves_per_id.ge(2).mean()) if len(waves_per_id) else math.nan,
        "waves": "|".join(
            str(wave) for wave in sorted(window[config.wave_column].dropna().astype(int).unique())
        ),
    }


def binary_status_summary(
    config: CohortConfig, window: pd.DataFrame, variables: list[str]
) -> dict[str, Any]:
    if not variables:
        return {"observed": 0, "positive": 0, "positive_rate": math.nan}
    values = pd.concat(
        [to_numeric(window[variable]) for variable in variables if variable in window.columns],
        axis=1,
    )
    if values.empty:
        return {"observed": 0, "positive": 0, "positive_rate": math.nan}
    observed = values.notna().any(axis=1)
    positive = values.eq(1).any(axis=1)
    observed_count = int(observed.sum())
    positive_count = int((observed & positive).sum())
    return {
        "observed": observed_count,
        "positive": positive_count,
        "positive_rate": positive_count / observed_count if observed_count else math.nan,
    }


def chronic_summary(config: CohortConfig, schema: set[str], window: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variable in CHARLS_CHRONIC_14:
        in_schema = variable in schema
        nonmissing = (
            int(window[variable].notna().sum()) if in_schema and variable in window.columns else 0
        )
        positive = (
            int(to_numeric(window[variable]).eq(1).sum())
            if in_schema and variable in window.columns
            else 0
        )
        rows.append(
            {
                "variable": variable,
                "in_schema": in_schema,
                "validation_nonmissing": nonmissing,
                "validation_positive": positive,
                "used_in_multimorbidity": variable in config.common_chronic_used and nonmissing > 0,
            }
        )
    return pd.DataFrame(rows)


def infer_depression_threshold(config: CohortConfig, window: pd.DataFrame) -> str:
    values = depression_series(config, window)
    variable = "cesd10a/cesd10b" if config.source_id == "klosa" else config.depression_variable
    if values.notna().sum() == 0:
        return "Not mapped in Phase 0."
    baseline_wave = min(config.validation_waves)
    baseline = values[pd.to_numeric(window[config.wave_column], errors="coerce").eq(baseline_wave)]
    value_range = (
        f"range {fmt(values.min())}-{fmt(values.max())}; baseline median {fmt(baseline.median())}"
    )
    if config.depression_aux_variable and config.depression_aux_variable in window.columns:
        aux = to_numeric(window[config.depression_aux_variable])
        positive_scores = values.loc[aux.eq(1)].dropna()
        if not positive_scores.empty:
            return f"{config.depression_note} Inferred candidate cutpoint: {variable} >= {fmt(positive_scores.min(), 0)} ({value_range})."
    if config.source_id == "klosa":
        return f"{config.depression_note} Candidate cutpoint for discovery: {variable} >= 10 ({value_range})."
    if config.source_id == "share":
        return f"{config.depression_note} Candidate cutpoint for discovery: EURO-D >= 4, to be reviewed before Phase 1 ({value_range})."
    return f"{config.depression_note} Distribution: {value_range}."


def depression_candidate_threshold(config: CohortConfig, window: pd.DataFrame) -> float:
    values = depression_series(config, window)
    if config.depression_aux_variable and config.depression_aux_variable in window.columns:
        aux = to_numeric(window[config.depression_aux_variable])
        positive_scores = values.loc[aux.eq(1)].dropna()
        if not positive_scores.empty:
            return float(positive_scores.min())
    if config.source_id == "share":
        return 4.0
    if config.source_id == "klosa":
        return 10.0
    return float("nan")


def depression_series(config: CohortConfig, window: pd.DataFrame) -> pd.Series:
    if config.source_id == "klosa":
        pieces = []
        for column in ["cesd10a", "cesd10b"]:
            if column in window.columns:
                pieces.append(to_numeric(window[column]))
        if pieces:
            out = pieces[0].copy()
            for piece in pieces[1:]:
                out = out.combine_first(piece)
            return out
    if config.depression_variable in window.columns:
        return to_numeric(window[config.depression_variable])
    return pd.Series(np.nan, index=window.index)


def low_cognition_series(config: CohortConfig, window: pd.DataFrame) -> pd.Series:
    if config.low_cognition_variable and config.low_cognition_variable in window.columns:
        return to_numeric(window[config.low_cognition_variable])
    pieces = [
        to_numeric(window[column])
        for column in config.low_cognition_components
        if column in window.columns
    ]
    if not pieces:
        return pd.Series(np.nan, index=window.index)
    values = pd.concat(pieces, axis=1)
    return values.sum(axis=1, min_count=max(1, min(2, len(pieces))))


def infer_low_cognition_threshold(config: CohortConfig, window: pd.DataFrame) -> str:
    values = low_cognition_series(config, window)
    baseline_wave = min(config.validation_waves)
    baseline = values[pd.to_numeric(window[config.wave_column], errors="coerce").eq(baseline_wave)]
    threshold = float(baseline.quantile(0.25))
    if not math.isfinite(threshold):
        return f"{config.low_cognition_note} Phase 0 could not calculate a finite baseline Q25."
    return f"{config.low_cognition_note} Baseline wave {baseline_wave} Q25 = {fmt(threshold)}."


def low_cognition_candidate_threshold(config: CohortConfig, window: pd.DataFrame) -> float:
    values = low_cognition_series(config, window)
    baseline_wave = min(config.validation_waves)
    baseline = values[pd.to_numeric(window[config.wave_column], errors="coerce").eq(baseline_wave)]
    return float(baseline.quantile(0.25))


def outcome_availability(
    config: CohortConfig, outcome_crosswalk: pd.DataFrame, window: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for _, row in outcome_crosswalk.iterrows():
        outcome_id = str(row["outcome_id"])
        variables = [
            part
            for part in str(row["cohort_source_variables"]).split("|")
            if part and part != "nan"
        ]
        if outcome_id == "multimorbidity_onset" and variables:
            values = pd.concat(
                [
                    to_numeric(window[variable])
                    for variable in variables
                    if variable in window.columns
                ],
                axis=1,
            )
            observed = values.notna().sum(axis=1).ge(min(3, len(variables)))
            positive = values.eq(1).sum(axis=1).ge(2)
            observed_count = int(observed.sum())
            positive_count = int((observed & positive).sum())
            positive_rate = positive_count / observed_count if observed_count else math.nan
        elif outcome_id == "high_depressive_symptoms_onset":
            series = depression_series(config, window)
            threshold = depression_candidate_threshold(config, window)
            observed = series.notna()
            positive = (
                series.ge(threshold)
                if math.isfinite(threshold)
                else pd.Series(False, index=window.index)
            )
            observed_count = int(observed.sum())
            positive_count = int((observed & positive).sum())
            positive_rate = positive_count / observed_count if observed_count else math.nan
        elif outcome_id == "low_cognition_onset_data_driven":
            series = low_cognition_series(config, window)
            threshold = low_cognition_candidate_threshold(config, window)
            observed = series.notna()
            positive = (
                series.le(threshold)
                if math.isfinite(threshold)
                else pd.Series(False, index=window.index)
            )
            observed_count = int(observed.sum())
            positive_count = int((observed & positive).sum())
            positive_rate = positive_count / observed_count if observed_count else math.nan
        else:
            stats = binary_status_summary(config, window, variables)
            observed_count = int(stats["observed"])
            positive_count = int(stats["positive"])
            positive_rate = (
                float(stats["positive_rate"]) if not pd.isna(stats["positive_rate"]) else math.nan
            )
        rows.append(
            {
                "outcome_id": outcome_id,
                "status": row["status"],
                "source_variables": row["cohort_source_variables"],
                "observed_rows": observed_count,
                "positive_rows_or_nonmissing_proxy": positive_count,
                "positive_rate": positive_rate,
            }
        )
    return pd.DataFrame(rows)


def search_checks(config: CohortConfig, api_base: str) -> dict[str, str]:
    queries = [
        "adl iadl",
        "cesd depressive eurod",
        "cognition memory",
        "hypertension diabetes",
        "hospital doctor",
    ]
    payload = post_json(
        api_base,
        "/search/batch",
        {"q": queries, "source": config.source_id, "kind": "auto", "limit": 8},
        timeout=300,
    )
    out: dict[str, str] = {}
    for result in payload.get("results", []):
        codes = []
        for hit in result.get("results", []):
            code = hit.get("code") or hit.get("physical_column") or hit.get("variable")
            if code and code not in codes:
                codes.append(str(code))
        out[str(result.get("query", ""))] = ", ".join(codes[:8]) if codes else "no hits"
    return out


def write_discovery_report(
    config: CohortConfig,
    output_dir: Path,
    resolve_payload: dict[str, Any],
    search: dict[str, str],
    panel: dict[str, Any],
    waves: pd.DataFrame,
    chronic: pd.DataFrame,
    variable_crosswalk: pd.DataFrame,
    outcome_crosswalk: pd.DataFrame,
    availability: pd.DataFrame,
    translation_summary: dict[str, Any],
    depression_note: str,
    low_cog_note: str,
) -> None:
    safe_mask = variable_crosswalk["charls_baseline_predictor_general"].astype(bool)
    safe_total = int(safe_mask.sum())
    safe_mapped = int((safe_mask & variable_crosswalk["status"].eq("mapped")).sum())
    mapped_outcomes = int(outcome_crosswalk["status"].eq("mapped").sum())
    lines = [
        f"# {config.label} Phase 0 Discovery Report",
        "",
        f"Generated: {generated_timestamp()}",
        "",
        "## API Evidence",
        "",
        f"- Source: `{config.source_id}`",
        f"- File key: `{config.main_file}`",
        f"- Resolve exists: `{resolve_payload.get('exists')}`",
        f"- API base: `{API_BASE}`",
        f"- ID column: `{config.id_column}`",
        f"- Wave column: `{config.wave_column}`",
        f"- Interview-year column: `{config.interview_year_column}`",
        f"- Phase 0 validation waves: {', '.join(str(wave) for wave in config.validation_waves)}",
        "",
        "## Panel Window",
        "",
        f"- Rows: {panel['rows']}",
        f"- Unique IDs: {panel['ids']}",
        f"- IDs with >=2 waves: {panel['ids_ge2_waves']} ({fmt(panel['pct_ids_ge2_waves'])})",
        f"- Waves: {panel['waves']}",
        "",
        "## Wave-To-Year Check",
        "",
    ]
    lines.extend(
        markdown_table(
            waves,
            [
                "wave",
                "interview_year_min",
                "interview_year_max",
                "interview_year_median",
                "rows",
                "ids",
            ],
        )
    )
    lines.extend(["", "## Search Checks", "", "| query | top API hits |", "|---|---|"])
    for query, codes in search.items():
        lines.append(f"| {query} | {codes} |")

    lines.extend(
        [
            "",
            "## Threshold Decisions For Review",
            "",
            f"- Depression: {depression_note}",
            f"- Low cognition: {low_cog_note}",
            "",
            "## Multimorbidity Disease Set",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            chronic,
            [
                "variable",
                "in_schema",
                "validation_nonmissing",
                "validation_positive",
                "used_in_multimorbidity",
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Crosswalk Summary",
            "",
            f"- CHARLS safe baseline predictors considered: {safe_total}",
            f"- Safe predictors mapped to {config.label}: {safe_mapped}",
            f"- Safe predictors unmappable: {safe_total - safe_mapped}",
            f"- HRS mapped safe predictors portable by same name: {translation_summary['hrs_safe_direct_hits']}/{translation_summary['hrs_safe_mapped_total']}",
            f"- HRS mapped safe predictors portable by alias: {translation_summary['hrs_safe_alias_hits']}/{translation_summary['hrs_safe_mapped_total']}",
            f"- HRS crosswalk portability hit rate: {fmt(translation_summary['hrs_crosswalk_hit_rate'])}",
            f"- Outcome definitions mapped: {mapped_outcomes}/{len(outcome_crosswalk)}",
            "",
            "Outputs:",
            "",
            f"- `{config.source_id}_variable_crosswalk.csv`",
            f"- `{config.source_id}_outcome_crosswalk.csv`",
            f"- `{config.source_id}_outcome_availability_audit.csv`",
            "",
            "## Outcome Crosswalk",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            outcome_crosswalk,
            ["outcome_id", "status", "cohort_source_variables", "definition_note"],
        )
    )
    lines.extend(["", "## Outcome Availability Proxy", ""])
    lines.extend(
        markdown_table(
            availability,
            [
                "outcome_id",
                "status",
                "observed_rows",
                "positive_rows_or_nonmissing_proxy",
                "positive_rate",
            ],
        )
    )
    if config.notes:
        lines.extend(["", "## Cohort Notes", ""])
        for note in config.notes:
            lines.append(f"- {note}")
    lines.extend(
        [
            "",
            "## Phase A Gate Items",
            "",
            "- Confirm mapped outcome count and whether hospitalization should remain excluded when unavailable.",
            "- Confirm the chronic-disease intersection used for multimorbidity.",
            "- Confirm depression and low-cognition threshold strategy before building person-period matrices.",
        ]
    )
    write_text(output_dir / f"{config.source_id}_discovery_report.md", lines)


def run_cohort(config: CohortConfig, api_base: str, row_limit: int) -> dict[str, Any]:
    output_dir = OUTPUT_ROOT / config.source_id
    output_dir.mkdir(parents=True, exist_ok=True)
    schema = set(fetch_schema(config, api_base))
    resolve_payload = resolve_main_file(config, api_base)

    requested = sorted(
        {
            *CHARLS_CHRONIC_14,
            *config.common_chronic_used,
            *[variable for variables in config.outcome_sources.values() for variable in variables],
            config.depression_variable,
            *(config.low_cognition_components or []),
            *([config.depression_aux_variable] if config.depression_aux_variable else []),
        }
        & schema
    )
    frame = extract_columns(config, requested, api_base, row_limit=row_limit)
    window = validation_window(config, frame)
    if window.empty:
        raise RuntimeError(f"{config.source_id} validation window is empty")

    depression_note = infer_depression_threshold(config, window)
    low_cog_note = infer_low_cognition_threshold(config, window)
    variable_crosswalk = build_variable_crosswalk(config, schema)
    outcome_crosswalk = build_outcome_crosswalk(config, schema, depression_note, low_cog_note)
    availability = outcome_availability(config, outcome_crosswalk, window)
    chronic = chronic_summary(config, schema, window)
    waves = wave_summary(config, window)
    panel = panel_summary(config, window)
    translation = build_hrs_translation_summary(config, schema)
    search = search_checks(config, api_base)

    variable_crosswalk.to_csv(
        output_dir / f"{config.source_id}_variable_crosswalk.csv", index=False
    )
    outcome_crosswalk.to_csv(output_dir / f"{config.source_id}_outcome_crosswalk.csv", index=False)
    availability.to_csv(
        output_dir / f"{config.source_id}_outcome_availability_audit.csv", index=False
    )
    chronic.to_csv(
        output_dir / f"{config.source_id}_multimorbidity_common_disease_audit.csv", index=False
    )
    waves.to_csv(output_dir / f"{config.source_id}_wave_year_audit.csv", index=False)
    write_discovery_report(
        config,
        output_dir,
        resolve_payload=resolve_payload,
        search=search,
        panel=panel,
        waves=waves,
        chronic=chronic,
        variable_crosswalk=variable_crosswalk,
        outcome_crosswalk=outcome_crosswalk,
        availability=availability,
        translation_summary=translation,
        depression_note=depression_note,
        low_cog_note=low_cog_note,
    )

    return {
        **panel,
        "label": config.label,
        "safe_predictors_total": int(
            variable_crosswalk["charls_baseline_predictor_general"].astype(bool).sum()
        ),
        "safe_predictors_mapped": int(
            (
                variable_crosswalk["charls_baseline_predictor_general"].astype(bool)
                & variable_crosswalk["status"].eq("mapped")
            ).sum()
        ),
        "outcomes_mapped": int(outcome_crosswalk["status"].eq("mapped").sum()),
        "outcomes_total": int(len(outcome_crosswalk)),
        "hrs_crosswalk_hit_rate": translation["hrs_crosswalk_hit_rate"],
        "discovery_report": str(
            (output_dir / f"{config.source_id}_discovery_report.md").relative_to(PROJECT_ROOT)
        ),
        "variable_crosswalk": str(
            (output_dir / f"{config.source_id}_variable_crosswalk.csv").relative_to(PROJECT_ROOT)
        ),
        "outcome_crosswalk": str(
            (output_dir / f"{config.source_id}_outcome_crosswalk.csv").relative_to(PROJECT_ROOT)
        ),
    }


def write_phase0_summary(summary: pd.DataFrame) -> None:
    path = OUTPUT_ROOT / "multicohort_phase0_summary.md"
    lines = [
        "# Multicohort Phase 0 Summary",
        "",
        f"Generated: {generated_timestamp()}",
        "",
        "Scope: KLoSA, ELSA, and SHARE discovery plus crosswalk draft validation only. No person-period matrices or model training were run in this phase.",
        "",
        "## Cohort Summary",
        "",
    ]
    display = summary.copy()
    display["pct_ids_ge2_waves"] = display["pct_ids_ge2_waves"].map(fmt)
    display["hrs_crosswalk_hit_rate"] = display["hrs_crosswalk_hit_rate"].map(fmt)
    lines.extend(
        markdown_table(
            display,
            [
                "cohort",
                "rows",
                "ids",
                "ids_ge2_waves",
                "pct_ids_ge2_waves",
                "waves",
                "safe_predictors_mapped",
                "safe_predictors_total",
                "outcomes_mapped",
                "outcomes_total",
                "hrs_crosswalk_hit_rate",
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Review Gate A",
            "",
            "Before Phase 1, review each cohort discovery report and approve:",
            "",
            "- mapped outcome count and unmappable outcomes;",
            "- multimorbidity disease intersection;",
            "- depression and low-cognition threshold strategy;",
            "- whether KLoSA should be the pilot full-flow cohort.",
        ]
    )
    write_text(path, lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", default=API_BASE)
    parser.add_argument("--row-limit", type=int, default=1_000_000)
    parser.add_argument(
        "--cohorts", nargs="*", default=["klosa", "elsa", "share"], choices=sorted(COHORTS)
    )
    args = parser.parse_args()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for cohort in args.cohorts:
        config = COHORTS[cohort]
        print(f"Running Phase 0 discovery for {config.label} ({config.source_id})")
        rows.append(run_cohort(config, api_base=args.api_base, row_limit=args.row_limit))
    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_ROOT / "multicohort_phase0_summary.csv", index=False)
    write_phase0_summary(summary)
    print(
        {
            "summary_csv": str(
                (OUTPUT_ROOT / "multicohort_phase0_summary.csv").relative_to(PROJECT_ROOT)
            ),
            "summary_md": str(
                (OUTPUT_ROOT / "multicohort_phase0_summary.md").relative_to(PROJECT_ROOT)
            ),
            "cohorts": args.cohorts,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
