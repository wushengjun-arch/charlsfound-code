#!/usr/bin/env python3
"""Tokenize CHARLS person-wave rows with reviewed v1 vocabularies."""

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
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_TOKENIZATION_CONFIG = SCRIPT_PATH.parents[1] / "configs" / "tokenization.yaml"
DEFAULT_DATA_CONFIG = SCRIPT_PATH.parents[1] / "configs" / "data.yaml"

CATEGORICAL_TYPES = {
    "categorical_value_label",
    "categorical_raw_code",
    "categorical_high_cardinality_context",
}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
        timeout=600,
    )
    records = payload.get("records") or []
    if not records:
        raise RuntimeError(f"Person-wave extract returned no records for columns: {columns[:5]}")
    return pd.DataFrame.from_records(records)


def chunked(values: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def normalize_raw_value(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, bool | np.bool_):
        return "1" if bool(value) else "0"
    if isinstance(value, int | np.integer):
        return str(int(value))
    if isinstance(value, float | np.floating):
        if not math.isfinite(float(value)):
            return None
        if float(value).is_integer():
            return str(int(value))
        return format(float(value), ".12g")
    text = str(value).strip()
    if text == "":
        return None
    try:
        as_float = float(text)
    except ValueError:
        return text
    if math.isfinite(as_float) and as_float.is_integer():
        return str(int(as_float))
    if math.isfinite(as_float):
        return format(as_float, ".12g")
    return text


def json_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def build_value_entry_maps(value_vocab: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    maps: dict[str, dict[str, dict[str, Any]]] = {}
    for entry in value_vocab["values"]:
        variable = entry.get("variable")
        raw_value = entry.get("raw_value")
        if variable is None or raw_value is None:
            continue
        maps.setdefault(str(variable), {})[normalize_raw_value(raw_value) or str(raw_value)] = entry
    return maps


def build_bin_entry_maps(continuous_bins: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    return {
        str(variable): sorted(entry.get("bins", []), key=lambda item: int(item["bin_index"]))
        for variable, entry in continuous_bins["variables"].items()
    }


def fetch_person_wave_frame(
    api_base: str,
    source_id: str,
    main_file: str,
    id_column: str,
    wave_column: str,
    variables: list[str],
    row_limit: int,
    chunk_size: int,
) -> pd.DataFrame:
    base_columns = [id_column, wave_column]
    variable_chunks = list(chunked(variables, max(1, chunk_size)))
    merged: pd.DataFrame | None = None
    total_chunks = len(variable_chunks)

    for chunk_index, variable_chunk in enumerate(variable_chunks, start=1):
        columns = base_columns + variable_chunk
        print(
            f"Extracting tokenization columns {chunk_index}/{total_chunks}: {len(variable_chunk)} variables"
        )
        chunk_df = extract_columns(api_base, source_id, main_file, columns, row_limit)
        if merged is None:
            merged = chunk_df.copy()
            merged.insert(0, "source_row_number", np.arange(len(merged), dtype=np.int32))
        else:
            merged = merged.merge(chunk_df, on=base_columns, how="left", validate="one_to_one")

    if merged is None:
        raise RuntimeError("No tokenization variables were extracted")

    duplicate_count = int(merged.duplicated(base_columns).sum())
    if duplicate_count:
        raise RuntimeError(
            f"Found duplicate {id_column}+{wave_column} rows after extract: {duplicate_count}"
        )
    missing_keys = int(merged[id_column].isna().sum() + merged[wave_column].isna().sum())
    if missing_keys:
        raise RuntimeError(f"Found missing ID or wave keys after extract: {missing_keys}")

    merged = merged.sort_values([id_column, wave_column], kind="stable").reset_index(drop=True)
    return merged


def encode_categorical(
    series: pd.Series,
    variable: str,
    value_entries_by_raw: dict[str, dict[str, Any]],
    missing_value_id: int,
    allow_unknown_as_missing: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, list[str]]:
    normalized = series.map(normalize_raw_value)
    value_ids = np.full(len(series), missing_value_id, dtype=np.int32)
    is_missing = normalized.isna().to_numpy(dtype=bool)
    is_maskable = np.zeros(len(series), dtype=bool)
    is_non_scale = np.zeros(len(series), dtype=bool)
    unknown_examples: list[str] = []
    unknown_count = 0

    for raw_value, indices in normalized.dropna().groupby(normalized.dropna()).groups.items():
        entry = value_entries_by_raw.get(str(raw_value))
        if entry is None:
            unknown_count += len(indices)
            if len(unknown_examples) < 10:
                unknown_examples.append(str(raw_value))
            continue
        index_array = np.fromiter(indices, dtype=np.int64)
        value_ids[index_array] = int(entry["index"])
        maskable = bool(entry.get("include_as_masked_value_target_v1", False))
        non_scale = entry.get("valid_for_variable_scale") is False
        if maskable:
            is_maskable[index_array] = True
        if non_scale:
            is_non_scale[index_array] = True

    if unknown_count and not allow_unknown_as_missing:
        examples = ", ".join(unknown_examples)
        raise RuntimeError(
            f"Unknown observed values for {variable}: {unknown_count} rows; examples: {examples}"
        )
    return value_ids, is_missing, is_maskable, is_non_scale, unknown_count, unknown_examples


def encode_continuous(
    series: pd.Series,
    variable: str,
    bin_rows: list[dict[str, Any]],
    missing_value_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, list[str]]:
    numeric = pd.to_numeric(series, errors="coerce")
    finite = np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan))
    value_ids = np.full(len(series), missing_value_id, dtype=np.int32)
    is_missing = ~finite
    is_maskable = np.zeros(len(series), dtype=bool)
    is_non_scale = np.zeros(len(series), dtype=bool)
    if not bin_rows:
        return value_ids, is_missing, is_maskable, is_non_scale, 0, []

    cutpoints = [
        float(row["upper_bound_inclusive"])
        for row in bin_rows[:-1]
        if row.get("upper_bound_inclusive") is not None
    ]
    values = numeric.to_numpy(dtype=float, na_value=np.nan)
    bin_indices = np.searchsorted(np.asarray(cutpoints, dtype=float), values[finite], side="left")
    bin_value_ids = np.asarray([int(row["value_token_id"]) for row in bin_rows], dtype=np.int32)
    if int(bin_indices.max(initial=0)) >= len(bin_value_ids):
        raise RuntimeError(f"Continuous bin assignment exceeded bin count for {variable}")
    value_ids[finite] = bin_value_ids[bin_indices]
    is_maskable[finite] = True
    return value_ids, is_missing, is_maskable, is_non_scale, 0, []


def fixed_size_list_array(
    flat_values: np.ndarray, list_size: int, value_type: pa.DataType
) -> pa.FixedSizeListArray:
    return pa.FixedSizeListArray.from_arrays(pa.array(flat_values, type=value_type), list_size)


def build_token_metadata(
    trait_entries: list[dict[str, Any]],
    value_vocab: dict[str, Any],
    module_vocab: dict[str, Any],
    continuous_bins: dict[str, Any],
    out_path: Path,
) -> pd.DataFrame:
    trait_by_variable = {entry["variable"]: entry for entry in trait_entries}
    trait_position_by_variable = {entry["variable"]: pos for pos, entry in enumerate(trait_entries)}
    module_token_to_index = module_vocab["token_to_index"]
    continuous_by_variable = continuous_bins["variables"]

    rows: list[dict[str, Any]] = []
    for value_entry in value_vocab["values"]:
        variable = value_entry.get("variable")
        trait_entry = trait_by_variable.get(variable) if variable is not None else None
        continuous_entry = continuous_by_variable.get(variable) if variable is not None else None
        rows.append(
            {
                "value_token_id": int(value_entry["index"]),
                "value_token": value_entry["token"],
                "value_type": value_entry.get("value_type"),
                "variable": variable,
                "trait_position": trait_position_by_variable.get(variable),
                "trait_token_id": trait_entry.get("index") if trait_entry else None,
                "trait_token": trait_entry.get("token") if trait_entry else None,
                "module": trait_entry.get("module") if trait_entry else None,
                "module_token_id": (
                    module_token_to_index.get(trait_entry.get("module_token"))
                    if trait_entry
                    else None
                ),
                "module_token": trait_entry.get("module_token") if trait_entry else None,
                "tokenization_type": trait_entry.get("tokenization_type") if trait_entry else None,
                "raw_value": value_entry.get("raw_value"),
                "value_label": value_entry.get("value_label"),
                "bin_index": value_entry.get("bin_index"),
                "interval_label": value_entry.get("interval_label"),
                "include_as_masked_value_target_v1": bool(
                    value_entry.get("include_as_masked_value_target_v1", False)
                ),
                "valid_for_variable_scale": value_entry.get("valid_for_variable_scale"),
                "continuous_bin_status": value_entry.get("bin_status")
                or (continuous_entry.get("status") if continuous_entry else None),
                "continuous_target_bin_count": value_entry.get("target_bin_count")
                or (continuous_entry.get("target_bin_count") if continuous_entry else None),
                "continuous_actual_bin_count": value_entry.get("actual_bin_count")
                or (continuous_entry.get("actual_bin_count") if continuous_entry else None),
                "continuous_has_mass_point_aware_binning": (
                    value_entry.get("has_mass_point_aware_binning")
                    if value_entry.get("has_mass_point_aware_binning") is not None
                    else (
                        len(continuous_entry.get("mass_points", [])) > 0
                        if continuous_entry
                        else None
                    )
                ),
                "continuous_is_reduced_bin_variable": (
                    value_entry.get("is_reduced_bin_variable")
                    if value_entry.get("is_reduced_bin_variable") is not None
                    else (
                        int(continuous_entry.get("actual_bin_count", 0))
                        < int(continuous_entry.get("target_bin_count", 0))
                        if continuous_entry
                        else None
                    )
                ),
                "contains_mass_point": value_entry.get("contains_mass_point"),
                "mass_points": json_text(
                    value_entry.get("mass_points")
                    or (continuous_entry.get("mass_points") if continuous_entry else None)
                ),
                "contained_mass_points": json_text(value_entry.get("contained_mass_points")),
            }
        )

    metadata = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(out_path, index=False)
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_TOKENIZATION_CONFIG)
    parser.add_argument("--data-config", type=Path, default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--api-base")
    parser.add_argument("--row-limit", type=int)
    parser.add_argument("--chunk-size", type=int, default=70)
    parser.add_argument("--allow-unknown-as-missing", action="store_true")
    args = parser.parse_args()

    tokenization_config = load_yaml(args.config)
    data_config = load_yaml(args.data_config)
    trait_vocab = load_json(resolve_path(tokenization_config["outputs"]["trait_vocab"]))
    value_vocab = load_json(resolve_path(tokenization_config["outputs"]["value_vocab"]))
    module_vocab = load_json(resolve_path(tokenization_config["outputs"]["module_vocab"]))
    wave_vocab = load_json(resolve_path(tokenization_config["outputs"]["wave_vocab"]))
    continuous_bins = load_json(
        resolve_path(tokenization_config["outputs"]["continuous_bin_edges"])
    )

    trait_entries = trait_vocab["traits"]
    variables = [entry["variable"] for entry in trait_entries]
    trait_count = len(trait_entries)
    if trait_count == 0:
        raise RuntimeError("Trait vocab is empty")

    metadata_path = resolve_path(
        tokenization_config["outputs"].get(
            "tokenization_metadata",
            "outputs/05_tokenization/charls_tokenization_metadata.csv",
        )
    )
    token_metadata = build_token_metadata(
        trait_entries=trait_entries,
        value_vocab=value_vocab,
        module_vocab=module_vocab,
        continuous_bins=continuous_bins,
        out_path=metadata_path,
    )

    value_token_to_index = value_vocab["token_to_index"]
    missing_value_id = int(
        value_token_to_index[tokenization_config["special_tokens"]["value_missing"]]
    )
    wave_raw_to_entry = {
        normalize_raw_value(entry["raw_value"]): entry for entry in wave_vocab["waves"]
    }
    value_entries_by_variable = build_value_entry_maps(value_vocab)
    bin_entries_by_variable = build_bin_entry_maps(continuous_bins)
    value_id_by_token = value_vocab["token_to_index"]
    for bin_rows in bin_entries_by_variable.values():
        for bin_row in bin_rows:
            bin_row["value_token_id"] = int(value_id_by_token[bin_row["token"]])

    api_base = args.api_base or data_config["api"]["base_url"]
    source_id = data_config["source"]["id"]
    main_file = data_config["source"]["main_file"]
    id_column = data_config["source"]["id_column"]
    wave_column = data_config["source"]["wave_column"]
    row_limit = args.row_limit or int(data_config["source"].get("full_row_limit", 100000))

    df = fetch_person_wave_frame(
        api_base=api_base,
        source_id=source_id,
        main_file=main_file,
        id_column=id_column,
        wave_column=wave_column,
        variables=variables,
        row_limit=row_limit,
        chunk_size=args.chunk_size,
    )
    row_count = len(df)

    wave_values = df[wave_column].map(normalize_raw_value)
    unknown_waves = sorted(set(wave_values.dropna()) - set(wave_raw_to_entry))
    if unknown_waves:
        raise RuntimeError(f"Unknown wave values: {unknown_waves}")
    wave_token_ids = wave_values.map(lambda item: int(wave_raw_to_entry[item]["index"])).to_numpy(
        dtype=np.int16
    )

    trait_ids = np.asarray([int(entry["index"]) for entry in trait_entries], dtype=np.int32)
    module_token_to_index = module_vocab["token_to_index"]
    module_ids = np.asarray(
        [int(module_token_to_index[entry["module_token"]]) for entry in trait_entries],
        dtype=np.int16,
    )

    value_ids = np.full((row_count, trait_count), missing_value_id, dtype=np.int32)
    value_is_missing = np.ones((row_count, trait_count), dtype=bool)
    value_is_masked_target = np.zeros((row_count, trait_count), dtype=bool)
    value_is_non_scale = np.zeros((row_count, trait_count), dtype=bool)
    unknown_total = 0
    unknown_examples: list[dict[str, Any]] = []

    for position, trait_entry in enumerate(trait_entries):
        variable = trait_entry["variable"]
        tokenization_type = trait_entry["tokenization_type"]
        if variable not in df.columns:
            raise RuntimeError(f"Extract did not return tokenization variable: {variable}")

        if tokenization_type in CATEGORICAL_TYPES:
            encoded = encode_categorical(
                series=df[variable],
                variable=variable,
                value_entries_by_raw=value_entries_by_variable.get(variable, {}),
                missing_value_id=missing_value_id,
                allow_unknown_as_missing=args.allow_unknown_as_missing,
            )
        elif tokenization_type == "continuous_quantile_bin":
            encoded = encode_continuous(
                series=df[variable],
                variable=variable,
                bin_rows=bin_entries_by_variable.get(variable, []),
                missing_value_id=missing_value_id,
            )
        else:
            raise RuntimeError(f"Unsupported tokenization type for {variable}: {tokenization_type}")

        ids, is_missing, is_maskable, is_non_scale, unknown_count, examples = encoded
        value_ids[:, position] = ids
        value_is_missing[:, position] = is_missing
        value_is_masked_target[:, position] = is_maskable
        value_is_non_scale[:, position] = is_non_scale
        unknown_total += unknown_count
        if unknown_count:
            unknown_examples.append(
                {
                    "variable": variable,
                    "unknown_count": unknown_count,
                    "examples": examples,
                }
            )

    sequence_length = np.full(row_count, trait_count, dtype=np.int16)
    missing_counts = value_is_missing.sum(axis=1).astype(np.int16)
    observed_counts = (trait_count - missing_counts).astype(np.int16)
    maskable_counts = value_is_masked_target.sum(axis=1).astype(np.int16)
    non_scale_counts = value_is_non_scale.sum(axis=1).astype(np.int16)
    unknown_counts = np.zeros(row_count, dtype=np.int16)

    flat_trait_ids = np.tile(trait_ids, row_count)
    flat_module_ids = np.tile(module_ids, row_count)

    arrays = [
        pa.array(df["source_row_number"].to_numpy(dtype=np.int32), type=pa.int32()),
        pa.array(df[id_column].to_numpy(), type=pa.int64()),
        pa.array(df[wave_column].to_numpy(dtype=np.int16), type=pa.int16()),
        pa.array(wave_token_ids, type=pa.int16()),
        pa.array(sequence_length, type=pa.int16()),
        pa.array(observed_counts, type=pa.int16()),
        pa.array(missing_counts, type=pa.int16()),
        pa.array(maskable_counts, type=pa.int16()),
        pa.array(non_scale_counts, type=pa.int16()),
        pa.array(unknown_counts, type=pa.int16()),
        fixed_size_list_array(flat_trait_ids, trait_count, pa.int32()),
        fixed_size_list_array(value_ids.reshape(-1), trait_count, pa.int32()),
        fixed_size_list_array(flat_module_ids, trait_count, pa.int16()),
        fixed_size_list_array(value_is_missing.reshape(-1), trait_count, pa.bool_()),
        fixed_size_list_array(value_is_masked_target.reshape(-1), trait_count, pa.bool_()),
        fixed_size_list_array(value_is_non_scale.reshape(-1), trait_count, pa.bool_()),
    ]
    names = [
        "source_row_number",
        id_column,
        wave_column,
        "wave_token_id",
        "sequence_length",
        "observed_value_count",
        "missing_value_count",
        "maskable_value_count",
        "non_scale_value_count",
        "unknown_value_count",
        "trait_token_ids",
        "value_token_ids",
        "module_token_ids",
        "value_is_missing",
        "value_is_masked_target",
        "value_is_non_scale",
    ]
    table = pa.Table.from_arrays(arrays, names=names)
    schema_metadata = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_id": source_id,
        "main_file": main_file,
        "api_base": api_base,
        "api_endpoint": "GET /extract?source=<source>&file=<main_file>&columns=<chunked tokenization variables>&limit=<row_limit>&format=json",
        "row_limit": str(row_limit),
        "row_count": str(row_count),
        "id_column": id_column,
        "wave_column": wave_column,
        "trait_count": str(trait_count),
        "trait_vocab": tokenization_config["outputs"]["trait_vocab"],
        "value_vocab": tokenization_config["outputs"]["value_vocab"],
        "module_vocab": tokenization_config["outputs"]["module_vocab"],
        "wave_vocab": tokenization_config["outputs"]["wave_vocab"],
        "tokenization_metadata": str(metadata_path.relative_to(PROJECT_ROOT)),
        "missing_value_token_id": str(missing_value_id),
        "unknown_assignment_count": str(unknown_total),
        "unknown_assignment_examples": json.dumps(unknown_examples, ensure_ascii=False),
    }
    table = table.replace_schema_metadata(
        {key: value.encode("utf-8") for key, value in schema_metadata.items()}
    )

    out_path = resolve_path(tokenization_config["outputs"]["tokenized_person_wave"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path, compression="zstd", use_dictionary=True)

    satlife_non_scale = token_metadata[
        (token_metadata["variable"].eq("satlife"))
        & (token_metadata["raw_value"].astype(str).eq("6"))
        & (token_metadata["valid_for_variable_scale"].eq(False))
    ]
    summary = {
        "rows": row_count,
        "unique_ids": int(df[id_column].nunique()),
        "trait_count": trait_count,
        "value_vocab_size": int(value_vocab["summary"]["value_token_count"]),
        "wave_token_count": int(wave_vocab["summary"]["wave_token_count"]),
        "missing_value_assignments": int(value_is_missing.sum()),
        "maskable_value_assignments": int(value_is_masked_target.sum()),
        "non_scale_value_assignments": int(value_is_non_scale.sum()),
        "satlife_6_non_scale_metadata_rows": int(len(satlife_non_scale)),
        "unknown_assignment_count": unknown_total,
        "tokenized_person_wave": str(out_path.relative_to(PROJECT_ROOT)),
        "tokenization_metadata": str(metadata_path.relative_to(PROJECT_ROOT)),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
