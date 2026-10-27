#!/usr/bin/env python3
"""Prepare SHARE-native tokenized person-wave rows, MLM splits, and masks."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
sys.path.insert(0, str(SCRIPT_PATH.parent))

from share_utils import (  # noqa: E402
    API_BASE,
    ID_COLUMN,
    INTERVIEW_YEAR_COLUMN,
    OUTPUT_DIR,
    VALIDATION_WAVES,
    WAVE_COLUMN,
    extract_columns,
    generated_timestamp,
    validation_window,
    write_text,
)

DEFAULT_CONFIG = SCRIPT_PATH.parents[1] / "configs" / "share_mlm.yaml"
TARGET_CONTINUOUS_BINS = 10
MODULE_TOKEN = "module:share"


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


def jsonable(value: Any) -> Any:
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def safe_token_part(value: Any) -> str:
    text = str(value).strip()
    for old, new in [("\n", " "), ("\r", " "), ("\t", " "), (":", "_"), ("|", "_"), ("/", "_")]:
        text = text.replace(old, new)
    return "_".join(text.split()) or "<empty>"


def normalize_raw_value(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (bool, np.bool_)):
        return "1" if bool(value) else "0"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return None
        if float(value).is_integer():
            return str(int(value))
        return format(float(value), ".12g")
    text = str(value).strip()
    if not text:
        return None
    try:
        as_float = float(text)
    except ValueError:
        return text
    if math.isfinite(as_float) and as_float.is_integer():
        return str(int(as_float))
    return format(as_float, ".12g") if math.isfinite(as_float) else text


def sort_observed_values(values: list[str]) -> list[str]:
    def key(value: str) -> tuple[int, float | str]:
        try:
            return (0, float(value))
        except ValueError:
            return (1, value)

    return sorted(values, key=key)


def make_id(series: pd.Series) -> pd.Series:
    out = series.astype(str).str.strip()
    if out.eq("").any() or out.isna().any():
        raise RuntimeError("SHARE respondent ID contains missing values")
    return out


def build_vocabularies(
    frame: pd.DataFrame, variable_map: pd.DataFrame, config: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], pd.DataFrame]:
    variables = (
        variable_map.loc[variable_map["include_in_tokenization_v1"].astype(bool), "variable"]
        .astype(str)
        .tolist()
    )
    trait_entries = []
    for index, variable in enumerate(variables):
        row = variable_map.loc[variable_map["variable"].eq(variable)].iloc[0]
        trait_entries.append(
            {
                "index": index,
                "token": f"trait:{safe_token_part(variable)}",
                "variable": variable,
                "module": "share",
                "module_token": MODULE_TOKEN,
                "tokenization_type": str(row["tokenization_type"]),
                "type_model": str(row["type_model"]),
                "missing_rate_overall": float(row["missing_rate_overall"]),
                "unique_count": int(row["unique_count"]),
                "include_in_tokenization_v1": True,
                "include_in_pretraining_v1": bool(row.get("include_in_pretraining_v1", True)),
            }
        )

    value_entries: list[dict[str, Any]] = []
    special_tokens = [
        (config["masking"]["pad_token"], "pad"),
        (config["masking"]["mask_token"], "mask"),
        (config["masking"]["missing_token"], "missing"),
    ]
    for token, role in special_tokens:
        value_entries.append(
            {
                "index": len(value_entries),
                "token": token,
                "value_type": "special",
                "role": role,
                "include_as_masked_value_target_v1": False,
            }
        )

    continuous_bins: dict[str, list[dict[str, Any]]] = {}
    categorical_maps: dict[str, dict[str, int]] = {}
    for trait in trait_entries:
        variable = trait["variable"]
        tokenization_type = trait["tokenization_type"]
        series = frame[variable]
        if tokenization_type == "continuous_quantile_bin":
            numeric = pd.to_numeric(series, errors="coerce")
            values = numeric[np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan))].to_numpy(
                dtype=float
            )
            unique_count = int(np.unique(values).size) if values.size else 0
            bin_count = max(1, min(TARGET_CONTINUOUS_BINS, unique_count))
            if bin_count == 1:
                edges = (
                    np.asarray([np.nanmin(values), np.nanmax(values)])
                    if values.size
                    else np.asarray([0.0, 0.0])
                )
            else:
                edges = np.quantile(values, np.linspace(0, 1, bin_count + 1))
                edges = np.unique(edges)
                if edges.size < 2:
                    edges = np.asarray([float(np.nanmin(values)), float(np.nanmax(values))])
            bins: list[dict[str, Any]] = []
            actual_bins = max(1, int(edges.size - 1))
            for bin_index in range(actual_bins):
                lower = None if bin_index == 0 else float(edges[bin_index])
                upper = None if bin_index == actual_bins - 1 else float(edges[bin_index + 1])
                token = f"value:{safe_token_part(variable)}:bin{bin_index}"
                value_entries.append(
                    {
                        "index": len(value_entries),
                        "token": token,
                        "value_type": "continuous_bin",
                        "variable": variable,
                        "trait_token": trait["token"],
                        "bin_index": bin_index,
                        "interval_label": f"({lower if lower is not None else '-inf'}, {upper if upper is not None else 'inf'}]",
                        "lower_bound_exclusive": lower,
                        "upper_bound_inclusive": upper,
                        "target_bin_count": TARGET_CONTINUOUS_BINS,
                        "actual_bin_count": actual_bins,
                        "bin_status": (
                            "ok" if actual_bins == TARGET_CONTINUOUS_BINS else "reduced_bins"
                        ),
                        "include_as_masked_value_target_v1": True,
                        "valid_for_variable_scale": True,
                    }
                )
                bins.append(
                    {
                        "bin_index": bin_index,
                        "value_token_id": len(value_entries) - 1,
                        "lower_bound_exclusive": lower,
                        "upper_bound_inclusive": upper,
                    }
                )
            continuous_bins[variable] = bins
        else:
            normalized = series.map(normalize_raw_value)
            observed = sort_observed_values(
                [str(value) for value in normalized.dropna().drop_duplicates().tolist()]
            )
            value_map: dict[str, int] = {}
            for raw_value in observed:
                token = f"value:{safe_token_part(variable)}:{safe_token_part(raw_value)}"
                value_entries.append(
                    {
                        "index": len(value_entries),
                        "token": token,
                        "value_type": "categorical",
                        "variable": variable,
                        "trait_token": trait["token"],
                        "raw_value": raw_value,
                        "value_label": raw_value,
                        "include_as_masked_value_target_v1": True,
                        "valid_for_variable_scale": True,
                    }
                )
                value_map[raw_value] = len(value_entries) - 1
            categorical_maps[variable] = value_map

    trait_vocab = {
        "metadata": {
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source_id": "share",
        },
        "summary": {
            "trait_count": len(trait_entries),
            "tokenization_type_counts": dict(
                Counter(item["tokenization_type"] for item in trait_entries)
            ),
        },
        "ordered_tokens": [entry["token"] for entry in trait_entries],
        "token_to_index": {entry["token"]: int(entry["index"]) for entry in trait_entries},
        "traits": trait_entries,
    }
    value_vocab = {
        "metadata": {
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source_id": "share",
        },
        "summary": {"value_count": len(value_entries)},
        "ordered_tokens": [entry["token"] for entry in value_entries],
        "token_to_index": {entry["token"]: int(entry["index"]) for entry in value_entries},
        "values": value_entries,
    }
    module_vocab = {
        "metadata": {
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source_id": "share",
        },
        "ordered_tokens": [MODULE_TOKEN],
        "token_to_index": {MODULE_TOKEN: 0},
        "modules": [{"index": 0, "token": MODULE_TOKEN, "module": "share"}],
    }
    wave_tokens = [f"wave:{wave}" for wave in VALIDATION_WAVES]
    wave_vocab = {
        "metadata": {
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source_id": "share",
        },
        "ordered_tokens": wave_tokens,
        "token_to_index": {token: index for index, token in enumerate(wave_tokens)},
        "waves": [
            {"index": index, "token": token, "wave": int(wave)}
            for index, (token, wave) in enumerate(zip(wave_tokens, VALIDATION_WAVES))
        ],
    }
    metadata = build_token_metadata(value_vocab, trait_vocab, module_vocab)
    metadata.attrs["categorical_maps"] = categorical_maps
    metadata.attrs["continuous_bins"] = continuous_bins
    return trait_vocab, value_vocab, module_vocab, wave_vocab, metadata


def build_token_metadata(
    value_vocab: dict[str, Any], trait_vocab: dict[str, Any], module_vocab: dict[str, Any]
) -> pd.DataFrame:
    trait_by_variable = {entry["variable"]: entry for entry in trait_vocab["traits"]}
    trait_position_by_variable = {
        entry["variable"]: int(entry["index"]) for entry in trait_vocab["traits"]
    }
    rows = []
    for entry in value_vocab["values"]:
        variable = entry.get("variable")
        trait = trait_by_variable.get(variable) if variable is not None else None
        rows.append(
            {
                "value_token_id": int(entry["index"]),
                "value_token": entry["token"],
                "value_type": entry.get("value_type"),
                "variable": variable,
                "trait_position": trait_position_by_variable.get(variable),
                "trait_token_id": trait.get("index") if trait else None,
                "trait_token": trait.get("token") if trait else None,
                "module": trait.get("module") if trait else None,
                "module_token_id": 0 if trait else None,
                "module_token": MODULE_TOKEN if trait else None,
                "tokenization_type": trait.get("tokenization_type") if trait else None,
                "raw_value": entry.get("raw_value"),
                "value_label": entry.get("value_label"),
                "bin_index": entry.get("bin_index"),
                "interval_label": entry.get("interval_label"),
                "include_as_masked_value_target_v1": bool(
                    entry.get("include_as_masked_value_target_v1", False)
                ),
                "valid_for_variable_scale": entry.get("valid_for_variable_scale"),
                "continuous_bin_status": entry.get("bin_status"),
                "continuous_target_bin_count": entry.get("target_bin_count"),
                "continuous_actual_bin_count": entry.get("actual_bin_count"),
                "contains_mass_point": False,
                "mass_points": "",
                "contained_mass_points": "",
            }
        )
    return pd.DataFrame(rows)


def encode_frame(
    frame: pd.DataFrame,
    trait_vocab: dict[str, Any],
    value_vocab: dict[str, Any],
    wave_vocab: dict[str, Any],
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    variables = [entry["variable"] for entry in trait_vocab["traits"]]
    tokenization_by_variable = {
        entry["variable"]: entry["tokenization_type"] for entry in trait_vocab["traits"]
    }
    trait_ids = np.asarray([int(entry["index"]) for entry in trait_vocab["traits"]], dtype=np.int16)
    module_ids = np.zeros(len(variables), dtype=np.int16)
    missing_value_id = int(value_vocab["token_to_index"]["value:<missing>"])
    value_by_variable_raw: dict[str, dict[str, int]] = {}
    for entry in value_vocab["values"]:
        variable = entry.get("variable")
        raw_value = entry.get("raw_value")
        if variable is not None and raw_value is not None:
            value_by_variable_raw.setdefault(str(variable), {})[
                normalize_raw_value(raw_value) or str(raw_value)
            ] = int(entry["index"])
    continuous_bins: dict[str, list[dict[str, Any]]] = {}
    for _, row in metadata.dropna(subset=["variable", "bin_index"]).iterrows():
        variable = str(row["variable"])
        continuous_bins.setdefault(variable, []).append(
            {
                "bin_index": int(row["bin_index"]),
                "value_token_id": int(row["value_token_id"]),
                "upper_bound_inclusive": row["interval_label"],
            }
        )

    # Use value vocab entries for numeric bin bounds because metadata stores human labels only.
    continuous_bounds: dict[str, list[dict[str, Any]]] = {}
    for entry in value_vocab["values"]:
        if entry.get("value_type") == "continuous_bin":
            continuous_bounds.setdefault(str(entry["variable"]), []).append(entry)
    for bins in continuous_bounds.values():
        bins.sort(key=lambda item: int(item["bin_index"]))

    encoded_columns: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]] = {}
    for variable in variables:
        series = frame[variable]
        value_ids = np.full(len(frame), missing_value_id, dtype=np.int32)
        is_missing = series.isna().to_numpy(dtype=bool)
        is_maskable = np.zeros(len(frame), dtype=bool)
        is_non_scale = np.zeros(len(frame), dtype=bool)
        unknown_count = 0
        if tokenization_by_variable[variable] == "continuous_quantile_bin":
            numeric = pd.to_numeric(series, errors="coerce")
            values = numeric.to_numpy(dtype=float, na_value=np.nan)
            finite = np.isfinite(values)
            bins = continuous_bounds[variable]
            cutpoints = [
                float(item["upper_bound_inclusive"])
                for item in bins[:-1]
                if item.get("upper_bound_inclusive") is not None
            ]
            bin_indices = np.searchsorted(
                np.asarray(cutpoints, dtype=float), values[finite], side="left"
            )
            token_ids = np.asarray([int(item["index"]) for item in bins], dtype=np.int32)
            value_ids[finite] = token_ids[bin_indices]
            is_missing = ~finite
            is_maskable[finite] = True
        else:
            raw_map = value_by_variable_raw.get(variable, {})
            normalized = series.map(normalize_raw_value)
            for raw_value, indices in (
                normalized.dropna().groupby(normalized.dropna()).groups.items()
            ):
                token_id = raw_map.get(str(raw_value))
                if token_id is None:
                    unknown_count += len(indices)
                    continue
                idx = np.fromiter(indices, dtype=np.int64)
                value_ids[idx] = token_id
                is_maskable[idx] = True
        encoded_columns[variable] = (
            value_ids,
            is_missing,
            is_maskable,
            is_non_scale,
            unknown_count,
        )

    rows = []
    wave_token_by_wave = {int(item["wave"]): int(item["index"]) for item in wave_vocab["waves"]}
    for row_index in range(len(frame)):
        value_ids = np.asarray(
            [encoded_columns[variable][0][row_index] for variable in variables], dtype=np.int16
        )
        is_missing = np.asarray(
            [encoded_columns[variable][1][row_index] for variable in variables], dtype=bool
        )
        is_maskable = np.asarray(
            [encoded_columns[variable][2][row_index] for variable in variables], dtype=bool
        )
        is_non_scale = np.asarray(
            [encoded_columns[variable][3][row_index] for variable in variables], dtype=bool
        )
        wave = int(frame.iloc[row_index][WAVE_COLUMN])
        rows.append(
            {
                "source_row_number": int(row_index),
                "ID": str(frame.iloc[row_index]["ID"]),
                "pid": str(frame.iloc[row_index]["ID"]),
                "wave": wave,
                "wave_token_id": wave_token_by_wave[wave],
                "sequence_length": len(variables),
                "observed_value_count": int((~is_missing).sum()),
                "missing_value_count": int(is_missing.sum()),
                "maskable_value_count": int(is_maskable.sum()),
                "non_scale_value_count": int(is_non_scale.sum()),
                "unknown_value_count": int(
                    sum(encoded_columns[variable][4] for variable in variables)
                ),
                "trait_token_ids": trait_ids.tolist(),
                "value_token_ids": value_ids.tolist(),
                "module_token_ids": module_ids.tolist(),
                "value_is_missing": is_missing.tolist(),
                "value_is_masked_target": is_maskable.tolist(),
                "value_is_non_scale": is_non_scale.tolist(),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(
        [
            {"variable": variable, "unknown_count": int(encoded_columns[variable][4])}
            for variable in variables
        ]
    )


def build_split_outputs(
    tokenized: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    supervised = pd.read_parquet(
        resolve_path(config["inputs"]["supervised_outcome_matrix"]), columns=[ID_COLUMN, "split"]
    )
    split = supervised.assign(ID=make_id(supervised[ID_COLUMN]))[["ID", "split"]].drop_duplicates()
    leaked = split.groupby("ID")["split"].nunique()
    if leaked.gt(1).any():
        raise RuntimeError(
            f"Share supervised split has {int(leaked.gt(1).sum())} IDs in multiple folds"
        )
    retained = tokenized.merge(split, on="ID", how="inner", validate="many_to_one")
    omitted_ids = int(tokenized["ID"].nunique() - retained["ID"].nunique())
    retained = retained.sort_values(["split", "ID", "wave"], kind="stable").reset_index(drop=True)
    manifest = {
        "metadata": {
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source_id": "share",
            "split_source": config["inputs"]["supervised_outcome_matrix"],
            "tokenized_source": config["inputs"]["tokenized_person_wave"],
        },
        "rows_by_split": retained.groupby("split").size().astype(int).to_dict(),
        "ids_by_split": retained.groupby("split")["ID"].nunique().astype(int).to_dict(),
        "omitted_tokenized_ids": omitted_ids,
        "sequence_length": int(retained["sequence_length"].max()),
    }
    return retained, manifest


def build_outcome_masks(config: dict[str, Any], metadata: pd.DataFrame) -> dict[str, Any]:
    supervised_manifest = load_json(resolve_path(config["inputs"]["supervised_matrix_manifest"]))
    leakage = pd.read_csv(resolve_path(config["inputs"]["leakage_matrix"]))
    traits = metadata.dropna(subset=["variable", "trait_token_id", "trait_position"]).copy()
    traits["trait_token_id"] = traits["trait_token_id"].astype(int)
    traits["trait_position"] = traits["trait_position"].astype(int)
    traits = traits[["variable", "trait_token_id", "trait_position"]].drop_duplicates("variable")
    tokenized_variables = set(traits["variable"].astype(str))
    diagnostic_reasons = set(config["masking"]["diagnostic_exclusion_reasons"])

    def mask_entry(variables: list[str]) -> dict[str, Any]:
        subset = traits[traits["variable"].isin(variables)].sort_values("trait_position")
        return {
            "variables": subset["variable"].astype(str).tolist(),
            "trait_token_ids": subset["trait_token_id"].astype(int).tolist(),
            "trait_positions": subset["trait_position"].astype(int).tolist(),
        }

    outcome_masks: dict[str, Any] = {}
    for outcome_id in supervised_manifest["selected_outcomes"]:
        safe_variables = set(supervised_manifest["per_outcome_predictors"][outcome_id])
        diagnostic_variables = sorted(
            set(
                leakage.loc[
                    leakage["outcome_id"].eq(outcome_id)
                    & leakage["decision"].eq("exclude")
                    & leakage["reason"].isin(diagnostic_reasons),
                    "variable",
                ].astype(str)
            )
            & tokenized_variables
        )
        strict_safe_variables = sorted(tokenized_variables - safe_variables)
        outcome_masks[outcome_id] = {
            "diagnostic": mask_entry(diagnostic_variables),
            "strict_safe": mask_entry(strict_safe_variables),
        }
    return {
        "metadata": {
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source_id": "share",
            "diagnostic_exclusion_reasons": sorted(diagnostic_reasons),
            "strict_safe_definition": "mask every tokenized trait not present in the outcome-specific safe predictor list",
        },
        "outcomes": outcome_masks,
    }


def write_report(
    path: Path,
    tokenized: pd.DataFrame,
    retained: pd.DataFrame,
    masks: dict[str, Any],
    variables: list[str],
    unknown_audit: pd.DataFrame,
) -> None:
    row_counts = (
        retained.groupby("split").agg(rows=("ID", "size"), ids=("ID", "nunique")).reset_index()
    )
    lines = [
        "# Share MLM Input Report",
        "",
        f"Generated: {generated_timestamp()}",
        "",
        f"- Tokenized person-wave rows: {len(tokenized)}",
        f"- Retained split-aligned rows: {len(retained)}",
        f"- Retained IDs: {retained['ID'].nunique()}",
        f"- Tokenized variables: {len(variables)}",
        f"- Sequence length: {int(tokenized['sequence_length'].max())}",
        f"- Total unknown encoded values: {int(unknown_audit['unknown_count'].sum())}",
        "",
        "| split | rows | ids |",
        "|---|---:|---:|",
    ]
    for _, row in row_counts.iterrows():
        lines.append(f"| {row['split']} | {int(row['rows'])} | {int(row['ids'])} |")
    lines.extend(
        [
            "",
            "## Outcome Mask Modes",
            "",
            "| outcome_id | diagnostic_traits | strict_safe_traits |",
            "|---|---:|---:|",
        ]
    )
    for outcome_id, entry in masks["outcomes"].items():
        lines.append(
            f"| {outcome_id} | {len(entry['diagnostic']['variables'])} | {len(entry['strict_safe']['variables'])} |"
        )
    lines.extend(
        [
            "",
            "Share tokenization is source-backed through the local API on port 8787 and uses Share-native vocabularies, not CHARLS token IDs.",
            "",
        ]
    )
    write_text(path, lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--api-base", default=API_BASE)
    parser.add_argument("--row-limit", type=int, default=1000000)
    parser.add_argument("--chunk-size", type=int, default=70)
    args = parser.parse_args()

    config = load_yaml(args.config)
    output_dir = resolve_path(config["outputs"]["mlm_directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    variable_map = pd.read_csv(OUTPUT_DIR / "share_variable_type_map.csv")
    variable_map = (
        variable_map[variable_map["include_in_tokenization_v1"].astype(bool)]
        .copy()
        .reset_index(drop=True)
    )
    variables = variable_map["variable"].astype(str).tolist()
    base_columns = [ID_COLUMN, WAVE_COLUMN, INTERVIEW_YEAR_COLUMN]
    frame = extract_columns(
        variables,
        row_limit=args.row_limit,
        api_base=args.api_base,
        chunk_size=args.chunk_size,
        base_columns=base_columns,
    )
    frame = validation_window(frame)
    frame["ID"] = make_id(frame[ID_COLUMN])
    frame = frame.sort_values(["ID", WAVE_COLUMN], kind="stable").reset_index(drop=True)

    trait_vocab, value_vocab, module_vocab, wave_vocab, metadata = build_vocabularies(
        frame, variable_map, config
    )
    tokenized, unknown_audit = encode_frame(frame, trait_vocab, value_vocab, wave_vocab, metadata)
    retained, split_manifest = build_split_outputs(tokenized, config)
    split_manifest.update(
        {
            "value_vocab_size": int(len(value_vocab["values"])),
            "trait_vocab_size": int(len(trait_vocab["traits"])),
            "module_vocab_size": int(len(module_vocab["modules"])),
            "wave_vocab_size": int(len(wave_vocab["waves"])),
        }
    )
    masks = build_outcome_masks(config, metadata)

    tokenized_path = resolve_path(config["inputs"]["tokenized_person_wave"])
    metadata_path = resolve_path(config["inputs"]["tokenization_metadata"])
    tokenized_path.parent.mkdir(parents=True, exist_ok=True)
    tokenized.to_parquet(tokenized_path, index=False)
    retained.to_parquet(resolve_path(config["outputs"]["mlm_person_wave_dataset"]), index=False)
    metadata.to_csv(metadata_path, index=False)
    unknown_audit.to_csv(output_dir / "share_tokenization_unknown_audit.csv", index=False)
    (output_dir / "share_trait_vocab.json").write_text(
        json.dumps(trait_vocab, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "share_value_vocab.json").write_text(
        json.dumps(value_vocab, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "share_module_vocab.json").write_text(
        json.dumps(module_vocab, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "share_wave_vocab.json").write_text(
        json.dumps(wave_vocab, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    resolve_path(config["outputs"]["mlm_split_manifest"]).write_text(
        json.dumps(split_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    resolve_path(config["outputs"]["diagnostic_trait_masks"]).write_text(
        json.dumps(masks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_report(
        resolve_path(config["outputs"]["mlm_input_report"]),
        tokenized,
        retained,
        masks,
        variables,
        unknown_audit,
    )

    print(
        json.dumps(
            {
                "tokenized_rows": int(len(tokenized)),
                "mlm_rows": int(len(retained)),
                "ids": int(retained["ID"].nunique()),
                "sequence_length": int(split_manifest["sequence_length"]),
                "value_vocab_size": int(split_manifest["value_vocab_size"]),
                "dataset": str(
                    resolve_path(config["outputs"]["mlm_person_wave_dataset"]).relative_to(
                        PROJECT_ROOT
                    )
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
