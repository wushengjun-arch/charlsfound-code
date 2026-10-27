#!/usr/bin/env python3
"""Build formal CHARLS ID-outcome time alignment artifact."""

from __future__ import annotations

import argparse
import json
import math
import warnings
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


def load_yaml(path: Path) -> dict[str, Any]:
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
        raise RuntimeError("Time alignment extract returned no records")
    return pd.DataFrame.from_records(records)


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


def split_vars(value: Any) -> list[str]:
    if pd.isna(value) or str(value).strip() == "":
        return []
    return [part.strip() for part in str(value).split("|") if part.strip()]


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def yes_mask(series: pd.Series) -> pd.Series:
    return to_numeric(series) == 1


def any_observed(df: pd.DataFrame, variables: list[str]) -> pd.Series:
    if not variables:
        return pd.Series(False, index=df.index)
    return df[variables].notna().any(axis=1)


def any_positive(df: pd.DataFrame, variables: list[str]) -> pd.Series:
    if not variables:
        return pd.Series(False, index=df.index)
    positive = pd.Series(False, index=df.index)
    for variable in variables:
        positive = positive | yes_mask(df[variable])
    return positive


def count_positive(df: pd.DataFrame, variables: list[str]) -> pd.Series:
    total = pd.Series(0, index=df.index, dtype="int64")
    for variable in variables:
        total = total + yes_mask(df[variable]).astype(int)
    return total


def count_observed(df: pd.DataFrame, variables: list[str]) -> pd.Series:
    total = pd.Series(0, index=df.index, dtype="int64")
    for variable in variables:
        total = total + df[variable].notna().astype(int)
    return total


def infer_low_cognition_threshold(registry: pd.DataFrame) -> float | None:
    row = registry.loc[registry["outcome_id"] == "low_cognition_onset_data_driven"]
    if row.empty:
        return None
    text = str(row.iloc[0].get("event_definition", ""))
    if "(" in text and ")" in text:
        try:
            return float(text.split("(")[-1].split(")")[0])
        except ValueError:
            return None
    return None


def first_non_missing_time(row: pd.Series, wave_col: str, year_col: str | None) -> Any:
    if year_col and year_col in row and not pd.isna(row[year_col]):
        return row[year_col]
    return row[wave_col]


def build_binary_alignment(
    df: pd.DataFrame,
    row: pd.Series,
    id_col: str,
    wave_col: str,
    year_col: str | None,
    baseline_wave: int,
    followup_waves: list[int],
    low_cog_threshold: float | None,
) -> pd.DataFrame:
    outcome_id = row["outcome_id"]
    variables = [var for var in split_vars(row["source_variables"]) if var in df]
    baseline_cols = [id_col, wave_col] + ([year_col] if year_col else []) + variables
    baseline = (
        df[df[wave_col] == baseline_wave][baseline_cols].drop_duplicates(subset=[id_col]).copy()
    )
    follow_cols = [id_col, wave_col] + ([year_col] if year_col else []) + variables
    followup = df[df[wave_col].isin(followup_waves)][follow_cols].copy()

    if outcome_id == "multimorbidity_onset":
        base_count = count_positive(baseline, variables)
        base_observed = count_observed(baseline, variables)
        baseline["baseline_status"] = "baseline_negative"
        baseline.loc[base_observed < 3, "baseline_status"] = "baseline_missing"
        baseline.loc[(base_observed >= 3) & (base_count >= 2), "baseline_status"] = (
            "baseline_positive"
        )
        follow_observed = count_observed(followup, variables) >= 3
        follow_event = (count_positive(followup, variables) >= 2) & follow_observed
    elif outcome_id == "high_depressive_symptoms_onset":
        base_value = to_numeric(baseline[variables[0]])
        baseline["baseline_status"] = "baseline_negative"
        baseline.loc[base_value.isna(), "baseline_status"] = "baseline_missing"
        baseline.loc[base_value >= 10, "baseline_status"] = "baseline_positive"
        follow_value = to_numeric(followup[variables[0]])
        follow_observed = follow_value.notna()
        follow_event = follow_value >= 10
    elif outcome_id == "low_cognition_onset_data_driven" and low_cog_threshold is not None:
        base_value = to_numeric(baseline[variables[0]])
        baseline["baseline_status"] = "baseline_negative"
        baseline.loc[base_value.isna(), "baseline_status"] = "baseline_missing"
        baseline.loc[base_value <= low_cog_threshold, "baseline_status"] = "baseline_positive"
        follow_value = to_numeric(followup[variables[0]])
        follow_observed = follow_value.notna()
        follow_event = follow_value <= low_cog_threshold
    elif outcome_id == "death_by_followup_sensitivity":
        baseline["baseline_status"] = "timing_review_required"
        follow_observed = any_observed(followup, variables)
        follow_event = pd.Series(False, index=followup.index)
    else:
        baseline["baseline_status"] = "baseline_negative"
        baseline.loc[~any_observed(baseline, variables), "baseline_status"] = "baseline_missing"
        baseline.loc[any_positive(baseline, variables), "baseline_status"] = "baseline_positive"
        follow_observed = any_observed(followup, variables)
        follow_event = any_positive(followup, variables) & follow_observed

    followup = followup.copy()
    followup["follow_observed"] = follow_observed
    followup["event"] = follow_event
    eligible_ids = set(baseline.loc[baseline["baseline_status"] == "baseline_negative", id_col])
    followup_eligible = followup[followup[id_col].isin(eligible_ids)].copy()
    observed = followup_eligible[followup_eligible["follow_observed"]].copy()
    event_rows = (
        followup_eligible[followup_eligible["event"]].copy().sort_values([id_col, wave_col])
    )
    first_event = (
        event_rows.groupby(id_col)
        .head(1)[[id_col, wave_col] + ([year_col] if year_col else [])]
        .copy()
    )
    first_event = first_event.rename(columns={wave_col: "first_event_wave"})
    if year_col:
        first_event = first_event.rename(columns={year_col: "first_event_time"})
    else:
        first_event["first_event_time"] = first_event["first_event_wave"]

    last_obs = (
        observed.sort_values([id_col, wave_col])
        .groupby(id_col)
        .tail(1)[[id_col, wave_col] + ([year_col] if year_col else [])]
        .copy()
    )
    last_obs = last_obs.rename(columns={wave_col: "censor_wave"})
    if year_col:
        last_obs = last_obs.rename(columns={year_col: "censor_time"})
    else:
        last_obs["censor_time"] = last_obs["censor_wave"]

    out = baseline[
        [id_col, wave_col] + ([year_col] if year_col else []) + ["baseline_status"]
    ].copy()
    out = out.rename(columns={wave_col: "prediction_start_wave"})
    if year_col:
        out = out.rename(columns={year_col: "prediction_start_time"})
    else:
        out["prediction_start_time"] = out["prediction_start_wave"]
    out["outcome_id"] = outcome_id
    out = out.merge(first_event, on=id_col, how="left")
    out = out.merge(last_obs, on=id_col, how="left")
    out["event_observed"] = out["first_event_wave"].notna().fillna(False).astype(bool)
    out["has_followup_observed"] = out["censor_wave"].notna().fillna(False).astype(bool)
    outcome_usable = bool(row.get("usable", False))
    out["eligible_for_binary_prediction"] = (
        out["baseline_status"].eq("baseline_negative")
        & out["has_followup_observed"]
        & outcome_usable
        & (row["outcome_id"] != "death_by_followup_sensitivity")
    )
    out["eligible_for_survival_model"] = out["eligible_for_binary_prediction"]
    out["eligible_for_continuous_change"] = False
    out["continuous_baseline_value"] = pd.NA
    out["continuous_last_followup_value"] = pd.NA
    out["continuous_change_value"] = pd.NA

    out["reason_if_excluded"] = ""
    out.loc[out["baseline_status"].eq("baseline_missing"), "reason_if_excluded"] = (
        "baseline_missing"
    )
    out.loc[out["baseline_status"].eq("baseline_positive"), "reason_if_excluded"] = (
        "baseline_positive"
    )
    out.loc[
        out["baseline_status"].eq("baseline_negative") & ~out["has_followup_observed"],
        "reason_if_excluded",
    ] = "no_followup_observed"
    out.loc[out["eligible_for_binary_prediction"], "reason_if_excluded"] = ""
    if not outcome_usable:
        out.loc[:, "reason_if_excluded"] = "outcome_not_usable"
    out.loc[out["baseline_status"].eq("timing_review_required"), "reason_if_excluded"] = (
        "timing_semantics_review_required"
    )
    return out


def build_continuous_alignment(
    df: pd.DataFrame,
    row: pd.Series,
    id_col: str,
    wave_col: str,
    year_col: str | None,
    baseline_wave: int,
    followup_waves: list[int],
) -> pd.DataFrame:
    outcome_id = row["outcome_id"]
    variable = split_vars(row["source_variables"])[0]
    baseline_cols = [id_col, wave_col] + ([year_col] if year_col else []) + [variable]
    baseline = (
        df[df[wave_col] == baseline_wave][baseline_cols].drop_duplicates(subset=[id_col]).copy()
    )
    follow_cols = [id_col, wave_col] + ([year_col] if year_col else []) + [variable]
    followup = df[df[wave_col].isin(followup_waves)][follow_cols].copy()
    baseline_value = to_numeric(baseline[variable])
    baseline["baseline_status"] = "baseline_observed"
    baseline.loc[baseline_value.isna(), "baseline_status"] = "baseline_missing"

    observed = followup[to_numeric(followup[variable]).notna()].sort_values([id_col, wave_col])
    last_obs = (
        observed.groupby(id_col)
        .tail(1)[[id_col, wave_col] + ([year_col] if year_col else []) + [variable]]
        .copy()
    )
    last_obs = last_obs.rename(
        columns={wave_col: "censor_wave", variable: "continuous_last_followup_value"}
    )
    if year_col:
        last_obs = last_obs.rename(columns={year_col: "censor_time"})
    else:
        last_obs["censor_time"] = last_obs["censor_wave"]

    out = baseline[
        [id_col, wave_col] + ([year_col] if year_col else []) + ["baseline_status", variable]
    ].copy()
    out = out.rename(
        columns={wave_col: "prediction_start_wave", variable: "continuous_baseline_value"}
    )
    if year_col:
        out = out.rename(columns={year_col: "prediction_start_time"})
    else:
        out["prediction_start_time"] = out["prediction_start_wave"]
    out["outcome_id"] = outcome_id
    out = out.merge(last_obs, on=id_col, how="left")
    out["first_event_wave"] = pd.NA
    out["first_event_time"] = pd.NA
    out["event_observed"] = False
    out["has_followup_observed"] = out["censor_wave"].notna().fillna(False).astype(bool)
    out["continuous_change_value"] = to_numeric(out["continuous_last_followup_value"]) - to_numeric(
        out["continuous_baseline_value"]
    )
    out["eligible_for_binary_prediction"] = False
    out["eligible_for_survival_model"] = False
    outcome_usable = bool(row.get("usable", False))
    out["eligible_for_continuous_change"] = (
        out["baseline_status"].eq("baseline_observed")
        & out["has_followup_observed"]
        & outcome_usable
    )
    out["reason_if_excluded"] = "continuous_change_outcome"
    out.loc[out["baseline_status"].eq("baseline_missing"), "reason_if_excluded"] = (
        "baseline_missing"
    )
    out.loc[
        out["baseline_status"].eq("baseline_observed") & ~out["has_followup_observed"],
        "reason_if_excluded",
    ] = "no_followup_observed"
    out.loc[out["eligible_for_continuous_change"], "reason_if_excluded"] = ""
    if not outcome_usable:
        out.loc[:, "reason_if_excluded"] = "outcome_not_usable"
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--api-base")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--row-limit", type=int)
    args = parser.parse_args()

    config = load_yaml(args.config)
    api_base = args.api_base or config["api"]["base_url"]
    source_id = config["source"]["id"]
    main_file = config["source"]["main_file"]
    id_col = config["source"]["id_column"]
    wave_col = config["source"]["wave_column"]
    row_limit = args.row_limit or int(config["source"].get("full_row_limit", 100000))
    baseline_wave = int(config["outcomes"]["baseline_wave"])
    followup_waves = [int(w) for w in config["outcomes"]["followup_waves"]]

    output_dir = args.output_dir or PROJECT_ROOT / config["outputs"].get(
        "time_alignment_directory", config["outputs"]["directory"]
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
    schema = pd.read_csv(audit_dir / "charls_schema.csv")
    registry = pd.read_csv(registry_path)

    all_source_vars = sorted(
        {var for value in registry["source_variables"] for var in split_vars(value)}
    )
    available = set(schema["variable"])
    year_col = "iwy" if "iwy" in available else None
    helper_cols = [id_col, wave_col] + ([year_col] if year_col else [])
    columns = [c for c in helper_cols + all_source_vars if c in available]
    df = extract_columns(api_base, source_id, main_file, columns, row_limit)
    df[wave_col] = df[wave_col].map(normalize_wave)
    low_cog_threshold = infer_low_cognition_threshold(registry)

    frames = []
    for _, row in registry.iterrows():
        if "continuous" in str(row["definition_type"]):
            frames.append(
                build_continuous_alignment(
                    df, row, id_col, wave_col, year_col, baseline_wave, followup_waves
                )
            )
        else:
            frames.append(
                build_binary_alignment(
                    df,
                    row,
                    id_col,
                    wave_col,
                    year_col,
                    baseline_wave,
                    followup_waves,
                    low_cog_threshold,
                )
            )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        alignment = pd.concat(frames, ignore_index=True)
    required_cols = [
        id_col,
        "prediction_start_wave",
        "prediction_start_time",
        "outcome_id",
        "baseline_status",
        "first_event_wave",
        "first_event_time",
        "censor_wave",
        "censor_time",
        "eligible_for_binary_prediction",
        "eligible_for_survival_model",
        "reason_if_excluded",
    ]
    extra_cols = [c for c in alignment.columns if c not in required_cols]
    alignment = alignment[required_cols + extra_cols]

    parquet_path = output_dir / "charls_time_alignment.parquet"
    alignment.to_parquet(parquet_path, index=False)
    alignment.head(1000).to_csv(output_dir / "charls_time_alignment_preview.csv", index=False)

    summary = (
        alignment.groupby("outcome_id")
        .agg(
            rows=(id_col, "size"),
            eligible_binary=("eligible_for_binary_prediction", "sum"),
            eligible_survival=("eligible_for_survival_model", "sum"),
            eligible_continuous=("eligible_for_continuous_change", "sum"),
            events=("event_observed", "sum"),
            baseline_missing=("baseline_status", lambda x: int((x == "baseline_missing").sum())),
            baseline_positive=("baseline_status", lambda x: int((x == "baseline_positive").sum())),
            no_followup=("reason_if_excluded", lambda x: int((x == "no_followup_observed").sum())),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / "charls_time_alignment_summary.csv", index=False)

    lines = [
        "# CHARLS Time Alignment Report",
        "",
        f"Source ID: `{source_id}`",
        f"API endpoint used: `GET /extract?source={source_id}&file={main_file}&columns=<registry source variables + time>&limit={row_limit}&format=json`",
        f"Output parquet: `{parquet_path}`",
        f"Rows: {len(alignment)}",
        f"Outcomes: {alignment['outcome_id'].nunique()}",
        "",
        "## Required Columns",
        "",
        "\n".join(f"- `{col}`" for col in required_cols),
        "",
        "## Warnings",
        "",
        "- Mortality rows are retained for audit but are not eligible for binary or survival modeling until timing semantics are resolved.",
        "- Time fields use `iwy` when available and wave number otherwise.",
        "",
    ]
    (output_dir / "charls_time_alignment_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {parquet_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
