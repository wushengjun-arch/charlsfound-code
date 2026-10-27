#!/usr/bin/env python3
"""Shared HRS external-validation utilities."""

from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]

API_BASE = "http://127.0.0.1:8787"
SOURCE_ID = "hrs"
MAIN_FILE = "analysis_data/hrs.parquet"
ID_COLUMN = "hhidpn"
WAVE_COLUMN = "wave"
INTERVIEW_YEAR_COLUMN = "iwendy"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "09_external_validation"

VALIDATION_WAVES = [10, 11, 12, 13, 14, 15]
ORIGIN_WAVES = [10, 11, 12, 13, 14]
WAVE_YEARS = {10: 2010, 11: 2012, 12: 2014, 13: 2016, 14: 2018, 15: 2020}

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

HRS_COMMON_CHRONIC_USED = [
    "hibpe",
    "diabe",
    "cancre",
    "lunge",
    "hearte",
    "stroke",
    "psyche",
    "arthre",
]

HRS_OUTCOME_SOURCES = {
    "adl_new_difficulty": ["dressa", "batha", "eata", "beda", "toilta", "walksa"],
    "iadl_new_difficulty": ["moneya", "phonea", "medsa", "mealsa", "shopa"],
    "incident_diabetes": ["diabe"],
    "incident_hypertension": ["hibpe"],
    "multimorbidity_onset": HRS_COMMON_CHRONIC_USED,
    "high_depressive_symptoms_onset": ["cesd", "depressive"],
    "hospitalization_onset": ["hosp"],
    "low_cognition_onset_data_driven": ["cog27"],
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


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def get_json(
    base_url: str, path: str, params: dict[str, Any] | None = None, timeout: int = 300
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


def fetch_schema(api_base: str = API_BASE) -> list[str]:
    payload = get_json(api_base, "/schema", {"source": SOURCE_ID, "file": MAIN_FILE}, timeout=300)
    columns = payload.get("columns") or []
    if not columns:
        raise RuntimeError("HRS schema returned no columns")
    return [str(column) for column in columns]


def resolve_main_file(api_base: str = API_BASE) -> dict[str, Any]:
    return get_json(api_base, "/resolve", {"source": SOURCE_ID, "file": MAIN_FILE}, timeout=300)


def chunked(values: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


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


def extract_columns(
    columns: Iterable[str],
    row_limit: int = 300000,
    api_base: str = API_BASE,
    chunk_size: int = 70,
    base_columns: list[str] | None = None,
) -> pd.DataFrame:
    schema = set(fetch_schema(api_base))
    base_columns = base_columns or [ID_COLUMN, WAVE_COLUMN]
    requested = list(dict.fromkeys([*base_columns, *columns]))
    missing = sorted(set(requested) - schema)
    if missing:
        raise RuntimeError(f"HRS columns missing from schema: {missing[:30]}")

    value_columns = [column for column in requested if column not in set(base_columns)]
    chunks = list(chunked(value_columns, max(1, chunk_size))) or [[]]
    merged: pd.DataFrame | None = None
    for index, chunk in enumerate(chunks, start=1):
        print(f"Extracting HRS chunk {index}/{len(chunks)}: {len(chunk)} variables")
        payload = get_json(
            api_base,
            "/extract",
            {
                "source": SOURCE_ID,
                "file": MAIN_FILE,
                "columns": ",".join([*base_columns, *chunk]),
                "limit": row_limit,
                "format": "json",
            },
            timeout=900,
        )
        records = payload.get("records") or []
        if not records:
            raise RuntimeError(f"HRS extract returned no records for columns: {chunk[:10]}")
        frame = pd.DataFrame.from_records(records)
        frame[WAVE_COLUMN] = frame[WAVE_COLUMN].map(normalize_wave)
        duplicate_count = int(frame.duplicated([ID_COLUMN, WAVE_COLUMN]).sum())
        if duplicate_count:
            raise RuntimeError(
                f"Duplicate {ID_COLUMN}+{WAVE_COLUMN} rows in HRS extract: {duplicate_count}"
            )
        if merged is None:
            merged = frame
        else:
            merged = merged.merge(frame, on=base_columns, how="left", validate="one_to_one")

    if merged is None:
        raise RuntimeError("No HRS data were extracted")
    merged[WAVE_COLUMN] = merged[WAVE_COLUMN].map(normalize_wave)
    return merged.sort_values([ID_COLUMN, WAVE_COLUMN], kind="stable").reset_index(drop=True)


def validation_window(frame: pd.DataFrame) -> pd.DataFrame:
    wave = pd.to_numeric(frame[WAVE_COLUMN], errors="coerce")
    out = frame.loc[wave.isin(VALIDATION_WAVES)].copy()
    out[WAVE_COLUMN] = pd.to_numeric(out[WAVE_COLUMN], errors="coerce").astype("Int64")
    return out


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def yes_mask(series: pd.Series) -> pd.Series:
    return to_numeric(series).eq(1)


def binary_any_positive(df: pd.DataFrame, variables: list[str]) -> tuple[pd.Series, pd.Series]:
    pieces = [to_numeric(df[variable]) for variable in variables]
    values = pd.concat(pieces, axis=1)
    observed = values.notna().any(axis=1)
    positive = values.eq(1).any(axis=1)
    return observed, positive


def infer_depression_threshold(frame: pd.DataFrame) -> int:
    base = frame.loc[
        pd.to_numeric(frame[WAVE_COLUMN], errors="coerce").eq(10), ["cesd", "depressive"]
    ].copy()
    base["cesd"] = to_numeric(base["cesd"])
    base["depressive"] = to_numeric(base["depressive"])
    positive_scores = base.loc[base["depressive"].eq(1), "cesd"].dropna()
    if not positive_scores.empty:
        return int(positive_scores.min())
    return 3


def infer_low_cognition_threshold(
    frame: pd.DataFrame, variable: str = "cog27", quantile: float = 0.25
) -> float:
    base = frame.loc[pd.to_numeric(frame[WAVE_COLUMN], errors="coerce").eq(10), variable]
    threshold = float(to_numeric(base).quantile(quantile))
    if not math.isfinite(threshold):
        raise RuntimeError(f"Could not infer HRS low-cognition threshold from {variable}")
    return threshold


def build_status_frame(
    frame: pd.DataFrame,
    outcome_id: str,
    depression_threshold: int,
    low_cognition_threshold: float,
) -> pd.DataFrame:
    variables = HRS_OUTCOME_SOURCES[outcome_id]
    missing = sorted(set(variables) - set(frame.columns))
    if missing:
        raise RuntimeError(f"{outcome_id} source variables were not extracted: {missing}")

    status = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    if outcome_id == "multimorbidity_onset":
        values = pd.concat([to_numeric(frame[variable]) for variable in variables], axis=1)
        observed = values.notna().sum(axis=1).ge(3)
        positive = values.eq(1).sum(axis=1).ge(2)
    elif outcome_id == "high_depressive_symptoms_onset":
        cesd = to_numeric(frame["cesd"])
        if "depressive" in frame.columns:
            derived = to_numeric(frame["depressive"])
            observed = cesd.notna() | derived.notna()
            positive = derived.eq(1) | cesd.ge(depression_threshold)
        else:
            observed = cesd.notna()
            positive = cesd.ge(depression_threshold)
    elif outcome_id == "low_cognition_onset_data_driven":
        value = to_numeric(frame["cog27"])
        observed = value.notna()
        positive = value.le(low_cognition_threshold)
    else:
        observed, positive = binary_any_positive(frame, variables)

    status.loc[observed & positive] = 1
    status.loc[observed & ~positive] = 0
    return frame[[ID_COLUMN, WAVE_COLUMN]].assign(outcome_status=status)


def add_outcome_status_columns(
    frame: pd.DataFrame,
    depression_threshold: int,
    low_cognition_threshold: float,
    outcomes: list[str] | None = None,
) -> pd.DataFrame:
    outcomes = outcomes or SELECTED_OUTCOMES
    out = frame.copy()
    for outcome_id in outcomes:
        status = build_status_frame(out, outcome_id, depression_threshold, low_cognition_threshold)
        out[f"status__{outcome_id}"] = status["outcome_status"]
    return out


def build_person_period_rows(
    person_wave: pd.DataFrame,
    selected_outcomes: list[str] | None = None,
    origin_waves: list[int] | None = None,
) -> pd.DataFrame:
    selected_outcomes = selected_outcomes or SELECTED_OUTCOMES
    origin_waves = origin_waves or ORIGIN_WAVES
    source_ids = person_wave[ID_COLUMN].dropna().drop_duplicates().tolist()
    rows: list[dict[str, Any]] = []

    for outcome_id in selected_outcomes:
        status_col = f"status__{outcome_id}"
        if status_col not in person_wave.columns:
            raise RuntimeError(f"Missing status column: {status_col}")
        status_lookup = {
            (row[ID_COLUMN], int(row[WAVE_COLUMN])): int(row[status_col])
            for _, row in person_wave.dropna(subset=[status_col]).iterrows()
        }
        for id_value in source_ids:
            exited_risk_set = False
            for origin_wave in origin_waves:
                if exited_risk_set:
                    break
                next_wave = origin_wave + 1
                origin_status = status_lookup.get((id_value, origin_wave))
                if origin_status == 1:
                    exited_risk_set = True
                    break
                if origin_status != 0:
                    continue
                next_status = status_lookup.get((id_value, next_wave))
                if next_status is None:
                    continue
                event_next_wave = int(next_status == 1)
                origin_year = WAVE_YEARS.get(origin_wave, origin_wave)
                next_year = WAVE_YEARS.get(next_wave, next_wave)
                rows.append(
                    {
                        ID_COLUMN: id_value,
                        "outcome_id": outcome_id,
                        "origin_wave": origin_wave,
                        "next_wave": next_wave,
                        "origin_year": origin_year,
                        "next_year": next_year,
                        "interval_years": next_year - origin_year,
                        "at_risk": True,
                        "event_next_wave": event_next_wave,
                    }
                )
                if event_next_wave:
                    exited_risk_set = True

    if not rows:
        raise RuntimeError("No HRS person-period rows were generated")
    matrix = pd.DataFrame(rows)
    matrix.insert(
        0,
        "row_id",
        matrix.apply(
            lambda row: f"{row['outcome_id']}|{row[ID_COLUMN]}|w{row['origin_wave']}", axis=1
        ),
    )
    return matrix


def make_group_split(
    ids: pd.Series, seed: int = 20260620, train_ratio: float = 0.8, validation_ratio: float = 0.1
) -> dict[Any, str]:
    unique_ids = pd.DataFrame({"id": ids.drop_duplicates().sort_values(kind="stable").to_numpy()})
    if unique_ids.empty:
        raise RuntimeError("Cannot split an empty ID set")
    test_ratio = 1.0 - train_ratio - validation_ratio
    if train_ratio <= 0 or validation_ratio <= 0 or test_ratio <= 0:
        raise ValueError("train, validation, and test ratios must all be positive")

    first_split = GroupShuffleSplit(n_splits=1, train_size=train_ratio, random_state=seed)
    train_idx, temp_idx = next(first_split.split(unique_ids, groups=unique_ids["id"]))
    train_ids = set(unique_ids.iloc[train_idx]["id"])
    temp = unique_ids.iloc[temp_idx].reset_index(drop=True)
    second_test_fraction = test_ratio / (validation_ratio + test_ratio)
    second_split = GroupShuffleSplit(
        n_splits=1, test_size=second_test_fraction, random_state=seed + 1
    )
    validation_idx, test_idx = next(second_split.split(temp, groups=temp["id"]))
    validation_ids = set(temp.iloc[validation_idx]["id"])
    test_ids = set(temp.iloc[test_idx]["id"])

    split_map: dict[Any, str] = {}
    for value in unique_ids["id"]:
        if value in train_ids:
            split_map[value] = "train"
        elif value in validation_ids:
            split_map[value] = "validation"
        elif value in test_ids:
            split_map[value] = "test"
        else:
            raise RuntimeError(f"ID did not receive a split: {value}")
    return split_map


def verify_group_splits(matrix: pd.DataFrame, id_column: str = ID_COLUMN) -> None:
    split_counts = matrix.groupby(id_column)["split"].nunique()
    leaked = split_counts[split_counts.gt(1)]
    if not leaked.empty:
        raise RuntimeError(f"ID-level split leakage detected for {len(leaked)} IDs")


def transition_audit(matrix: pd.DataFrame) -> pd.DataFrame:
    return (
        matrix.groupby(["outcome_id", "origin_wave", "next_wave"], as_index=False)
        .agg(
            rows=("event_next_wave", "size"),
            ids=(ID_COLUMN, "nunique"),
            events=("event_next_wave", "sum"),
            event_rate=("event_next_wave", "mean"),
        )
        .sort_values(["outcome_id", "origin_wave"], kind="stable")
    )


def build_time_to_event_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (outcome_id, id_value), group in matrix.sort_values(
        ["outcome_id", ID_COLUMN, "origin_wave"]
    ).groupby(["outcome_id", ID_COLUMN], sort=False):
        first = group.iloc[0]
        event_rows = group[group["event_next_wave"].astype(int).eq(1)]
        event_observed = not event_rows.empty
        terminal = event_rows.iloc[0] if event_observed else group.iloc[-1]
        rows.append(
            {
                ID_COLUMN: id_value,
                "outcome_id": outcome_id,
                "split": first["split"],
                "first_origin_wave": int(first["origin_wave"]),
                "first_origin_year": int(first["origin_year"]),
                "terminal_origin_wave": int(terminal["origin_wave"]),
                "terminal_wave": int(terminal["next_wave"]),
                "terminal_year": int(terminal["next_year"]),
                "event_observed": bool(event_observed),
                "time_to_event_years": float(terminal["next_year"] - first["origin_year"]),
                "person_period_rows": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def write_text(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
