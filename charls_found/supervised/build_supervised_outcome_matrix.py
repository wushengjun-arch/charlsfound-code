#!/usr/bin/env python3
"""Build leakage-filtered CHARLS person-period supervised matrices."""

from __future__ import annotations

import argparse
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
import yaml
from sklearn.model_selection import GroupShuffleSplit

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_CONFIG = SCRIPT_PATH.parents[1] / "configs" / "supervised.yaml"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_path(path_value: str | Path) -> Path:
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
        raise RuntimeError(f"HTTP {exc.code} for {url}: {body[:500]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach {url}: {exc}") from exc


def extract_columns(
    base_url: str,
    source_id: str,
    main_file: str,
    columns: Iterable[str],
    row_limit: int,
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
        timeout=900,
    )
    records = payload.get("records") or []
    if not records:
        raise RuntimeError(
            f"Supervised matrix extract returned no records for columns: {columns[:5]}"
        )
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


def chunked(values: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def yes_mask(series: pd.Series) -> pd.Series:
    return to_numeric(series).eq(1)


def available_columns(data_config: dict[str, Any]) -> set[str]:
    output_dir = resolve_path(
        data_config["outputs"].get("data_audit_directory", data_config["outputs"]["directory"])
    )
    schema_path = output_dir / "charls_schema.csv"
    if not schema_path.exists():
        return set()
    schema = pd.read_csv(schema_path)
    return set(schema["variable"].astype(str))


def infer_low_cognition_threshold(registry: pd.DataFrame) -> float | None:
    row = registry.loc[registry["outcome_id"].eq("low_cognition_onset_data_driven")]
    if row.empty:
        return None
    text = str(row.iloc[0].get("event_definition", ""))
    if "(" in text and ")" in text:
        try:
            return float(text.split("(")[-1].split(")")[0])
        except ValueError:
            return None
    return None


def fetch_person_wave_data(
    data_config: dict[str, Any],
    variables: list[str],
    row_limit: int,
    chunk_size: int,
    api_base: str | None,
) -> pd.DataFrame:
    api_base = api_base or data_config["api"]["base_url"]
    source_id = data_config["source"]["id"]
    main_file = data_config["source"]["main_file"]
    id_column = data_config["source"]["id_column"]
    wave_column = data_config["source"]["wave_column"]
    available = available_columns(data_config)
    year_column = "iwy" if not available or "iwy" in available else None
    base_columns = [id_column, wave_column] + ([year_column] if year_column else [])
    variables = [variable for variable in variables if variable not in set(base_columns)]
    if available:
        missing = sorted(set(variables) - available)
        if missing:
            raise RuntimeError(f"Variables missing from audited schema: {missing[:20]}")

    merged: pd.DataFrame | None = None
    variable_chunks = list(chunked(variables, max(1, chunk_size))) or [[]]
    for chunk_index, variable_chunk in enumerate(variable_chunks, start=1):
        print(
            f"Extracting person-wave source data {chunk_index}/{len(variable_chunks)}: {len(variable_chunk)} variables"
        )
        chunk_df = extract_columns(
            api_base, source_id, main_file, base_columns + variable_chunk, row_limit
        )
        chunk_df[wave_column] = chunk_df[wave_column].map(normalize_wave)
        duplicate_count = int(chunk_df.duplicated([id_column, wave_column]).sum())
        if duplicate_count:
            raise RuntimeError(
                f"Duplicate {id_column}+{wave_column} rows in source extract: {duplicate_count}"
            )
        if merged is None:
            merged = chunk_df
        else:
            merged = merged.merge(chunk_df, on=base_columns, how="left", validate="one_to_one")

    if merged is None:
        raise RuntimeError("No person-wave data were extracted")
    return merged.sort_values([id_column, wave_column], kind="stable").reset_index(drop=True)


def safe_predictors(
    leakage: pd.DataFrame,
    selected_outcomes: list[str],
    safe_reason: str,
) -> dict[str, list[str]]:
    safe_rows = leakage[
        leakage["outcome_id"].isin(selected_outcomes)
        & leakage["outcome_usable"].astype(bool)
        & leakage["decision"].eq("include")
        & leakage["reason"].eq(safe_reason)
        & leakage["usable_as_baseline_predictor_after_review"].astype(bool)
    ].copy()
    if safe_rows.empty:
        raise RuntimeError("No safe predictors found for selected outcomes")
    predictors = {
        outcome_id: safe_rows.loc[safe_rows["outcome_id"].eq(outcome_id), "variable"]
        .drop_duplicates()
        .sort_values()
        .tolist()
        for outcome_id in selected_outcomes
    }
    empty_outcomes = [outcome_id for outcome_id, variables in predictors.items() if not variables]
    if empty_outcomes:
        raise RuntimeError(f"Outcomes without safe predictors: {empty_outcomes}")
    return predictors


def build_status_frame(
    df: pd.DataFrame,
    outcome_id: str,
    variables: list[str],
    id_column: str,
    wave_column: str,
    low_cog_threshold: float | None,
) -> pd.DataFrame:
    missing_variables = sorted(set(variables) - set(df.columns))
    if missing_variables:
        raise RuntimeError(f"{outcome_id} source variables not extracted: {missing_variables}")

    status = pd.Series(pd.NA, index=df.index, dtype="Int64")
    if outcome_id == "multimorbidity_onset":
        observed = sum(df[variable].notna().astype(int) for variable in variables) >= 3
        positive = sum(yes_mask(df[variable]).astype(int) for variable in variables) >= 2
    elif outcome_id == "high_depressive_symptoms_onset":
        value = to_numeric(df[variables[0]])
        observed = value.notna()
        positive = value.ge(10)
    elif outcome_id == "low_cognition_onset_data_driven":
        if low_cog_threshold is None:
            raise RuntimeError("Could not infer low-cognition threshold from registry")
        value = to_numeric(df[variables[0]])
        observed = value.notna()
        positive = value.le(low_cog_threshold)
    else:
        observed = df[variables].notna().any(axis=1)
        positive = pd.Series(False, index=df.index)
        for variable in variables:
            positive = positive | yes_mask(df[variable])

    status.loc[observed & positive] = 1
    status.loc[observed & ~positive] = 0
    return df[[id_column, wave_column]].assign(outcome_status=status)


def build_person_period_rows(
    source_df: pd.DataFrame,
    registry: pd.DataFrame,
    selected_outcomes: list[str],
    origin_waves: list[int],
    wave_years: dict[int, int],
    id_column: str,
    wave_column: str,
) -> pd.DataFrame:
    low_cog_threshold = infer_low_cognition_threshold(registry)
    source_ids = source_df[id_column].dropna().drop_duplicates().tolist()
    registry_by_id = registry.set_index("outcome_id")
    rows: list[dict[str, Any]] = []

    for outcome_id in selected_outcomes:
        outcome_row = registry_by_id.loc[outcome_id]
        variables = split_vars(outcome_row["source_variables"])
        status_frame = build_status_frame(
            source_df, outcome_id, variables, id_column, wave_column, low_cog_threshold
        )
        status_lookup = {
            (row[id_column], int(row[wave_column])): int(row["outcome_status"])
            for _, row in status_frame.dropna(subset=["outcome_status"]).iterrows()
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
                origin_year = wave_years.get(origin_wave, origin_wave)
                next_year = wave_years.get(next_wave, next_wave)
                rows.append(
                    {
                        id_column: id_value,
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
        raise RuntimeError("No person-period rows were generated")
    matrix = pd.DataFrame(rows)
    matrix.insert(
        0,
        "row_id",
        matrix.apply(
            lambda row: f"{row['outcome_id']}|{row[id_column]}|w{row['origin_wave']}", axis=1
        ),
    )
    return matrix


def make_group_split(
    ids: pd.Series,
    seed: int,
    train_ratio: float,
    validation_ratio: float,
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


def make_reference_group_split(
    ids: pd.Series,
    reference_matrix_path: Path,
    id_column: str,
    seed: int,
    train_ratio: float,
    validation_ratio: float,
) -> dict[Any, str]:
    if not reference_matrix_path.exists():
        raise RuntimeError(f"Reference split matrix not found: {reference_matrix_path}")
    reference = pd.read_parquet(reference_matrix_path, columns=[id_column, "split"])
    reference_counts = reference.groupby(id_column)["split"].nunique()
    leaked = reference_counts[reference_counts.gt(1)]
    if not leaked.empty:
        raise RuntimeError(f"Reference split has {len(leaked)} IDs in multiple folds")
    reference_map = reference.drop_duplicates(id_column).set_index(id_column)["split"].to_dict()

    unique_ids = ids.drop_duplicates().sort_values(kind="stable").tolist()
    split_map = {
        id_value: reference_map[id_value] for id_value in unique_ids if id_value in reference_map
    }
    new_ids = [id_value for id_value in unique_ids if id_value not in reference_map]
    if new_ids:
        new_split = make_group_split(
            pd.Series(new_ids),
            seed=seed,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
        )
        split_map.update(new_split)
    return split_map


def build_time_to_event_matrix(matrix: pd.DataFrame, id_column: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (outcome_id, id_value), group in matrix.sort_values(
        ["outcome_id", id_column, "origin_wave"]
    ).groupby(["outcome_id", id_column], sort=False):
        first = group.iloc[0]
        event_rows = group[group["event_next_wave"].astype(int).eq(1)]
        event_observed = not event_rows.empty
        terminal = event_rows.iloc[0] if event_observed else group.iloc[-1]
        rows.append(
            {
                id_column: id_value,
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


def transition_audit(matrix: pd.DataFrame, id_column: str) -> pd.DataFrame:
    return (
        matrix.groupby(["outcome_id", "origin_wave", "next_wave"], as_index=False)
        .agg(
            rows=("event_next_wave", "size"),
            ids=(id_column, "nunique"),
            events=("event_next_wave", "sum"),
            event_rate=("event_next_wave", "mean"),
        )
        .sort_values(["outcome_id", "origin_wave"], kind="stable")
    )


def write_transition_audit_report(path: Path, audit: pd.DataFrame) -> None:
    lines = [
        "# CHARLS Person-Period Transition Audit",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "| outcome_id | origin_wave | next_wave | rows | events | event_rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in audit.iterrows():
        lines.append(
            f"| {row['outcome_id']} | {int(row['origin_wave'])} | {int(row['next_wave'])} | {int(row['rows'])} | {int(row['events'])} | {row['event_rate']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Rows are adjacent transitions where the respondent is observed and still at risk at the origin wave, and the next wave label is observed.",
            "An observed event at `t+1` removes the respondent from later origin waves for the same outcome.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_matrix_report(
    path: Path,
    matrix: pd.DataFrame,
    time_to_event: pd.DataFrame,
    audit: pd.DataFrame,
    selected_outcomes: list[str],
    per_outcome_predictors: dict[str, list[str]],
    registry: pd.DataFrame,
    id_column: str,
) -> None:
    split_rows = (
        matrix.groupby("split")
        .agg(row_count=(id_column, "size"), ids=(id_column, "nunique"))
        .reset_index()
    )
    outcome_rows = []
    registry_by_id = registry.set_index("outcome_id")
    for outcome_id in selected_outcomes:
        subset = matrix[matrix["outcome_id"].eq(outcome_id)]
        survival = time_to_event[time_to_event["outcome_id"].eq(outcome_id)]
        outcome_rows.append(
            {
                "outcome_id": outcome_id,
                "domain": registry_by_id.loc[outcome_id, "domain"],
                "rows": int(len(subset)),
                "ids": int(subset[id_column].nunique()),
                "events": int(subset["event_next_wave"].sum()),
                "event_rate": float(subset["event_next_wave"].mean()),
                "survival_ids": int(survival[id_column].nunique()),
                "safe_predictors": len(per_outcome_predictors[outcome_id]),
            }
        )
    outcome_df = pd.DataFrame(outcome_rows)

    lines = [
        "# CHARLS Person-Period Supervised Matrix Report",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
        f"- Person-period rows: {len(matrix)}",
        f"- Unique respondent IDs: {matrix[id_column].nunique()}",
        f"- Selected outcomes: {len(selected_outcomes)}",
        f"- Feature columns: {len([col for col in matrix.columns if col.startswith('feature__')])}",
        "- Matrix unit: one at-risk adjacent transition per `(ID, outcome_id, origin_wave)`.",
        "- Split: `GroupShuffleSplit` by respondent ID; all origins and outcomes for an ID share one split.",
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
            "| outcome_id | domain | rows | ids | events | event_rate | survival_ids | safe_predictors |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in outcome_df.iterrows():
        lines.append(
            f"| {row['outcome_id']} | {row['domain']} | {row['rows']} | {row['ids']} | {row['events']} | {row['event_rate']:.4f} | {row['survival_ids']} | {row['safe_predictors']} |"
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
            f"| {row['outcome_id']} | {int(row['origin_wave'])} | {int(row['next_wave'])} | {int(row['rows'])} | {int(row['events'])} | {row['event_rate']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- Features are merged from the origin wave only; labels come from the adjacent next wave.",
            "- Per-outcome feature lists are still read from the leakage matrix, so disease diagnostic/treatment constructs stay excluded at every origin.",
            "- Respondents leave an outcome risk set after the first observed event.",
            "- Adjacent transitions with missing origin status or missing next-wave label are not treated as negatives.",
            "- `origin_wave` and `interval_years` are available for discrete-time hazard models; intervals are 2011-2013, 2013-2015, 2015-2018, and 2018-2020.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--api-base")
    parser.add_argument("--chunk-size", type=int, default=70)
    args = parser.parse_args()

    config = load_yaml(args.config)
    data_config = load_yaml(resolve_path(config["inputs"]["data_config"]))
    registry = pd.read_csv(resolve_path(config["inputs"]["outcome_registry"]))
    leakage = pd.read_csv(resolve_path(config["inputs"]["leakage_matrix"]))
    variable_map = pd.read_csv(resolve_path(config["inputs"]["variable_type_map"]))

    selected_outcomes = list(config["probe"]["selected_outcomes"])
    missing_outcomes = sorted(set(selected_outcomes) - set(registry["outcome_id"]))
    if missing_outcomes:
        raise RuntimeError(f"Selected outcomes not found in registry: {missing_outcomes}")
    registry = registry[registry["outcome_id"].isin(selected_outcomes)].copy()
    unusable = registry.loc[~registry["usable"].astype(bool), "outcome_id"].tolist()
    if unusable:
        raise RuntimeError(f"Selected outcomes are not marked usable: {unusable}")

    safe_reason = config["probe"]["safe_predictor_reason"]
    per_outcome_predictors = safe_predictors(leakage, selected_outcomes, safe_reason)
    predictor_variables = sorted(
        set().union(*[set(values) for values in per_outcome_predictors.values()])
    )
    id_column = data_config["source"]["id_column"]
    wave_column = data_config["source"]["wave_column"]
    predictor_variables = [
        variable for variable in predictor_variables if variable not in {id_column, wave_column}
    ]
    outcome_source_variables = sorted(
        {variable for value in registry["source_variables"] for variable in split_vars(value)}
    )
    extract_variables = sorted(set(predictor_variables) | set(outcome_source_variables))

    origin_waves = [int(wave) for wave in config["probe"].get("origin_waves", [1, 2, 3, 4])]
    wave_years = {
        int(wave): int(year) for wave, year in config["probe"].get("wave_years", {}).items()
    }
    row_limit = int(
        config["probe"].get("row_limit", data_config["source"].get("full_row_limit", 100000))
    )
    source_df = fetch_person_wave_data(
        data_config=data_config,
        variables=extract_variables,
        row_limit=row_limit,
        chunk_size=args.chunk_size,
        api_base=args.api_base,
    )

    matrix = build_person_period_rows(
        source_df=source_df,
        registry=registry,
        selected_outcomes=selected_outcomes,
        origin_waves=origin_waves,
        wave_years=wave_years,
        id_column=id_column,
        wave_column=wave_column,
    )

    feature_frame = source_df[[id_column, wave_column] + predictor_variables].copy()
    feature_frame = feature_frame.rename(
        columns={
            wave_column: "origin_wave",
            **{variable: f"feature__{variable}" for variable in predictor_variables},
        }
    )
    matrix = matrix.merge(
        feature_frame, on=[id_column, "origin_wave"], how="left", validate="many_to_one"
    )
    split_config = config["split"]
    reference_matrix = split_config.get("reference_supervised_matrix")
    if reference_matrix:
        split_map = make_reference_group_split(
            matrix[id_column],
            reference_matrix_path=resolve_path(reference_matrix),
            id_column=id_column,
            seed=int(split_config["seed"]),
            train_ratio=float(split_config["train"]),
            validation_ratio=float(split_config["validation"]),
        )
        split_method = (
            f"Reference split from {reference_matrix}; new IDs assigned by GroupShuffleSplit"
        )
    else:
        split_map = make_group_split(
            matrix[id_column],
            seed=int(split_config["seed"]),
            train_ratio=float(split_config["train"]),
            validation_ratio=float(split_config["validation"]),
        )
        split_method = "GroupShuffleSplit by respondent ID"
    matrix.insert(2, "split", matrix[id_column].map(split_map))
    if matrix["split"].isna().any():
        raise RuntimeError("Some person-period rows did not receive a split")

    time_to_event = build_time_to_event_matrix(matrix, id_column)
    audit = transition_audit(matrix, id_column)
    out_path = resolve_path(config["outputs"]["supervised_outcome_matrix"])
    time_to_event_path = resolve_path(config["outputs"]["supervised_time_to_event_matrix"])
    audit_path = resolve_path(config["outputs"]["person_period_transition_audit"])
    for path in [out_path, time_to_event_path, audit_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_parquet(out_path, index=False)
    time_to_event.to_parquet(time_to_event_path, index=False)
    audit.to_csv(audit_path, index=False)
    write_transition_audit_report(
        resolve_path(config["outputs"]["person_period_transition_audit_report"]), audit
    )

    variable_type_by_name = variable_map.set_index("variable")["type_model"].to_dict()
    tokenization_type_by_name = variable_map.set_index("variable")["tokenization_type"].to_dict()
    feature_type_by_variable = {
        variable: variable_type_by_name.get(variable) for variable in predictor_variables
    }
    tokenization_type_by_variable = {
        variable: tokenization_type_by_name.get(variable) for variable in predictor_variables
    }
    feature_type_by_variable.update({"origin_wave": "categorical", "interval_years": "continuous"})
    split_counts = {
        split: {
            "rows": int(subset.shape[0]),
            "ids": int(subset[id_column].nunique()),
        }
        for split, subset in matrix.groupby("split")
    }
    manifest = {
        "metadata": {
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "matrix_type": "person_period_discrete_time_hazard",
            "source_id": data_config["source"]["id"],
            "main_file": data_config["source"]["main_file"],
            "api_base": args.api_base or data_config["api"]["base_url"],
            "row_limit": row_limit,
            "id_column": id_column,
            "wave_column": wave_column,
            "origin_waves": origin_waves,
            "wave_years": wave_years,
            "safe_predictor_reason": safe_reason,
            "split_method": split_method,
        },
        "selected_outcomes": selected_outcomes,
        "feature_columns": [f"feature__{variable}" for variable in predictor_variables],
        "feature_variables": predictor_variables,
        "time_features": ["origin_wave", "interval_years"],
        "feature_type_by_variable": feature_type_by_variable,
        "tokenization_type_by_variable": tokenization_type_by_variable,
        "per_outcome_predictors": per_outcome_predictors,
        "target_columns": {
            outcome_id: {
                "row_filter": f"outcome_id == '{outcome_id}'",
                "event": "event_next_wave",
                "time": "interval_years",
            }
            for outcome_id in selected_outcomes
        },
        "split_counts": split_counts,
        "outputs": {
            "person_period_matrix": str(out_path.relative_to(PROJECT_ROOT)),
            "time_to_event_matrix": str(time_to_event_path.relative_to(PROJECT_ROOT)),
            "transition_audit": str(audit_path.relative_to(PROJECT_ROOT)),
        },
    }
    manifest_path = resolve_path(config["outputs"]["supervised_matrix_manifest"])
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    write_matrix_report(
        resolve_path(config["outputs"]["supervised_matrix_report"]),
        matrix=matrix,
        time_to_event=time_to_event,
        audit=audit,
        selected_outcomes=selected_outcomes,
        per_outcome_predictors=per_outcome_predictors,
        registry=registry,
        id_column=id_column,
    )

    summary = {
        "rows": int(len(matrix)),
        "unique_ids": int(matrix[id_column].nunique()),
        "selected_outcomes": len(selected_outcomes),
        "feature_union_count": len(predictor_variables),
        "split_counts": split_counts,
        "matrix": str(out_path.relative_to(PROJECT_ROOT)),
        "time_to_event": str(time_to_event_path.relative_to(PROJECT_ROOT)),
        "manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
