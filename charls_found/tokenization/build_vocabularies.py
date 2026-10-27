#!/usr/bin/env python3
"""Build CHARLS trait, value, module, and wave vocabularies."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_CONFIG = SCRIPT_PATH.parents[1] / "configs" / "tokenization.yaml"

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


def jsonable(value: Any) -> Any:
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if not math.isfinite(float(value)):
            return None
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def safe_token_part(value: Any) -> str:
    if value is None:
        return "<none>"
    text = str(value).strip()
    for old, new in [("\n", " "), ("\r", " "), ("\t", " "), (":", "_"), ("|", "_"), ("/", "_")]:
        text = text.replace(old, new)
    text = "_".join(text.split())
    return text or "<empty>"


def trait_token(variable: str) -> str:
    return f"trait:{safe_token_part(variable)}"


def module_token(module: str) -> str:
    return f"module:{safe_token_part(module)}"


def ensure_unique_tokens(entries: list[dict[str, Any]], vocab_name: str) -> None:
    counts = Counter(str(entry["token"]) for entry in entries)
    duplicates = [token for token, count in counts.items() if count > 1]
    if duplicates:
        raise RuntimeError(f"{vocab_name} has duplicate tokens: {duplicates[:10]}")


def token_to_index(entries: list[dict[str, Any]]) -> dict[str, int]:
    return {str(entry["token"]): int(entry["index"]) for entry in entries}


def interval_contains_mass_point(bin_row: dict[str, Any], value: float) -> bool:
    lower = bin_row.get("lower_bound_exclusive")
    upper = bin_row.get("upper_bound_inclusive")
    lower_ok = True if lower is None else float(value) > float(lower)
    upper_ok = True if upper is None else float(value) <= float(upper)
    return lower_ok and upper_ok


def base_metadata(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "variable_type_map": config["outputs"]["variable_type_map"],
        "value_label_map": config["outputs"]["value_label_map"],
        "continuous_bin_edges": config["outputs"]["continuous_bin_edges"],
        "tokenization_config": str(DEFAULT_CONFIG.relative_to(PROJECT_ROOT)),
        "version": "v1",
    }


def build_trait_vocab(
    variable_map: pd.DataFrame,
    value_label_map: dict[str, Any],
    continuous_bins: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    included = variable_map[variable_map["include_in_tokenization_v1"].astype(bool)].copy()
    included = included.reset_index(drop=False).rename(columns={"index": "variable_map_row_index"})
    value_variables = value_label_map["variables"]
    continuous_variables = continuous_bins["variables"]

    entries: list[dict[str, Any]] = []
    missing_sources: list[str] = []
    for vocab_index, (_, row) in enumerate(included.iterrows()):
        variable = str(row["variable"])
        tokenization_type = str(row["tokenization_type"])
        value_source = None
        value_count = None
        continuous_meta: dict[str, Any] | None = None

        if tokenization_type in CATEGORICAL_TYPES:
            if variable not in value_variables:
                missing_sources.append(f"{variable}:value_label_map")
            else:
                value_source = "value_label_map"
                value_count = int(value_variables[variable]["value_count"])
        elif tokenization_type == "continuous_quantile_bin":
            if variable not in continuous_variables:
                missing_sources.append(f"{variable}:continuous_bin_edges")
            else:
                value_source = "continuous_bin_edges"
                bin_entry = continuous_variables[variable]
                value_count = int(bin_entry["actual_bin_count"])
                continuous_meta = {
                    "bin_strategy": bin_entry["strategy"],
                    "bin_status": bin_entry["status"],
                    "target_bin_count": int(bin_entry["target_bin_count"]),
                    "actual_bin_count": int(bin_entry["actual_bin_count"]),
                    "has_reduced_bins": int(bin_entry["actual_bin_count"])
                    < int(bin_entry["target_bin_count"]),
                    "has_mass_point_aware_binning": len(bin_entry.get("mass_points", [])) > 0,
                    "mass_points": bin_entry.get("mass_points", []),
                    "observed_min": bin_entry.get("observed_min"),
                    "observed_max": bin_entry.get("observed_max"),
                    "non_missing_count": int(bin_entry["non_missing_count"]),
                    "missing_count": int(bin_entry["missing_count"]),
                    "missing_rate": bin_entry["missing_rate"],
                }
        else:
            missing_sources.append(f"{variable}:unsupported_tokenization_type:{tokenization_type}")

        entry = {
            "index": vocab_index,
            "token": trait_token(variable),
            "variable": variable,
            "label_zh": jsonable(row.get("label_zh")),
            "module": jsonable(row.get("module")),
            "module_token": module_token(str(row.get("module"))),
            "tokenization_type": tokenization_type,
            "value_source": value_source,
            "value_count": value_count,
            "review_status": jsonable(row.get("review_status")),
            "include_in_tokenization_v1": bool(row.get("include_in_tokenization_v1", False)),
            "include_in_pretraining_v1": bool(row.get("include_in_pretraining_v1", False)),
            "baseline_predictor_general": bool(row.get("baseline_predictor_general", False)),
            "requires_per_outcome_filter": bool(row.get("requires_per_outcome_filter", False)),
            "direct_usable_outcome_source": bool(row.get("direct_usable_outcome_source", False)),
            "usable_outcome_source_ids": jsonable(row.get("usable_outcome_source_ids")),
            "any_outcome_source_ids": jsonable(row.get("any_outcome_source_ids")),
            "usable_outcome_include_count": int(row.get("usable_outcome_include_count", 0) or 0),
            "usable_outcome_review_count": int(row.get("usable_outcome_review_count", 0) or 0),
            "usable_outcome_exclude_count": int(row.get("usable_outcome_exclude_count", 0) or 0),
        }
        if continuous_meta is not None:
            entry["continuous_bin_metadata"] = continuous_meta
        entries.append(entry)

    if missing_sources:
        raise RuntimeError(f"Trait vocab has missing source artifacts: {missing_sources[:20]}")
    ensure_unique_tokens(entries, "trait_vocab")

    type_counts = Counter(entry["tokenization_type"] for entry in entries)
    module_counts = Counter(str(entry["module"]) for entry in entries)
    trait_vocab = {
        "metadata": {
            **base_metadata(config),
            "token_format": "trait:<variable>",
        },
        "summary": {
            "trait_count": len(entries),
            "tokenization_type_counts": dict(type_counts),
            "module_counts": dict(module_counts),
            "traits_requiring_per_outcome_filter": int(
                sum(entry["requires_per_outcome_filter"] for entry in entries)
            ),
            "direct_usable_outcome_source_traits": int(
                sum(entry["direct_usable_outcome_source"] for entry in entries)
            ),
            "continuous_traits": int(type_counts.get("continuous_quantile_bin", 0)),
            "categorical_traits": int(sum(type_counts.get(item, 0) for item in CATEGORICAL_TYPES)),
        },
        "ordered_tokens": [entry["token"] for entry in entries],
        "token_to_index": token_to_index(entries),
        "traits": entries,
    }
    return trait_vocab, included


def build_value_vocab(
    trait_rows: pd.DataFrame,
    trait_vocab: dict[str, Any],
    value_label_map: dict[str, Any],
    continuous_bins: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    value_variables = value_label_map["variables"]
    continuous_variables = continuous_bins["variables"]
    trait_by_variable = {entry["variable"]: entry for entry in trait_vocab["traits"]}

    special_order = [
        ("value_pad", "pad"),
        ("value_mask", "mask"),
        ("value_missing", "missing"),
    ]
    for key, role in special_order:
        if key not in config.get("special_tokens", {}):
            continue
        entries.append(
            {
                "index": len(entries),
                "token": config["special_tokens"][key],
                "value_type": "special",
                "role": role,
                "include_as_masked_value_target_v1": False,
            }
        )

    for _, row in trait_rows.iterrows():
        variable = str(row["variable"])
        tokenization_type = str(row["tokenization_type"])
        trait_entry = trait_by_variable[variable]
        if tokenization_type not in CATEGORICAL_TYPES:
            continue
        variable_entry = value_variables[variable]
        usage = variable_entry.get("usage", {})
        for value_row in variable_entry["values"]:
            value_label = value_row.get("value_label")
            valid_for_scale = value_row.get("valid_for_variable_scale", True)
            entry = {
                "index": len(entries),
                "token": value_row["token"],
                "value_type": "categorical",
                "variable": variable,
                "trait_token": trait_entry["token"],
                "module": trait_entry["module"],
                "module_token": trait_entry["module_token"],
                "tokenization_type": tokenization_type,
                "raw_value": value_row.get("raw_value"),
                "value_label": value_label,
                "label_source": value_row.get("label_source"),
                "value_order": value_row.get("value_order"),
                "observed_count": int(value_row.get("observed_count", 0)),
                "observed_rate_all_rows": value_row.get("observed_rate_all_rows"),
                "observed_rate_non_missing": value_row.get("observed_rate_non_missing"),
                "map_source": variable_entry.get("map_source"),
                "include_as_masked_value_target_v1": bool(
                    usage.get("include_as_masked_value_target_v1", False)
                )
                and bool(valid_for_scale),
                "valid_for_variable_scale": bool(valid_for_scale),
            }
            for key in [
                "source_question",
                "source_url",
                "source_lines",
                "evidence_note",
                "semantic_note",
            ]:
                if key in value_row:
                    entry[key] = value_row[key]
            entries.append(entry)

    for _, row in trait_rows.iterrows():
        variable = str(row["variable"])
        tokenization_type = str(row["tokenization_type"])
        if tokenization_type != "continuous_quantile_bin":
            continue
        trait_entry = trait_by_variable[variable]
        bin_entry = continuous_variables[variable]
        mass_points = bin_entry.get("mass_points", [])
        is_reduced = int(bin_entry["actual_bin_count"]) < int(bin_entry["target_bin_count"])
        has_mass_point = len(mass_points) > 0
        for bin_row in bin_entry["bins"]:
            contained_mass_points = [
                mass_point
                for mass_point in mass_points
                if interval_contains_mass_point(bin_row, float(mass_point["value"]))
            ]
            entries.append(
                {
                    "index": len(entries),
                    "token": bin_row["token"],
                    "value_type": "continuous_bin",
                    "variable": variable,
                    "trait_token": trait_entry["token"],
                    "module": trait_entry["module"],
                    "module_token": trait_entry["module_token"],
                    "tokenization_type": tokenization_type,
                    "bin_index": int(bin_row["bin_index"]),
                    "interval_label": bin_row["interval_label"],
                    "lower_bound_exclusive": bin_row.get("lower_bound_exclusive"),
                    "upper_bound_inclusive": bin_row.get("upper_bound_inclusive"),
                    "observed_count": int(bin_row.get("observed_count", 0)),
                    "observed_rate_non_missing": bin_row.get("observed_rate_non_missing"),
                    "bin_strategy": bin_entry["strategy"],
                    "bin_status": bin_entry["status"],
                    "target_bin_count": int(bin_entry["target_bin_count"]),
                    "actual_bin_count": int(bin_entry["actual_bin_count"]),
                    "has_mass_point_aware_binning": has_mass_point,
                    "is_reduced_bin_variable": is_reduced,
                    "mass_points": mass_points,
                    "contains_mass_point": len(contained_mass_points) > 0,
                    "contained_mass_points": contained_mass_points,
                    "include_as_masked_value_target_v1": True,
                    "valid_for_variable_scale": True,
                }
            )

    ensure_unique_tokens(entries, "value_vocab")
    type_counts = Counter(entry["value_type"] for entry in entries)
    non_scale_values = [
        entry
        for entry in entries
        if entry.get("variable") == "satlife" and entry.get("raw_value") == "6"
    ]
    value_vocab = {
        "metadata": {
            **base_metadata(config),
            "token_format": "value:<variable>:<raw/label or bin>",
            "special_token_order": [
                config["special_tokens"][key]
                for key, _ in special_order
                if key in config["special_tokens"]
            ],
        },
        "summary": {
            "value_token_count": len(entries),
            "value_type_counts": dict(type_counts),
            "categorical_value_tokens": int(type_counts.get("categorical", 0)),
            "continuous_bin_tokens": int(type_counts.get("continuous_bin", 0)),
            "special_value_tokens": int(type_counts.get("special", 0)),
            "masked_value_target_tokens": int(
                sum(entry.get("include_as_masked_value_target_v1", False) for entry in entries)
            ),
            "non_scale_value_tokens": int(
                sum(entry.get("valid_for_variable_scale") is False for entry in entries)
            ),
            "continuous_tokens_from_mass_point_aware_variables": int(
                sum(
                    entry.get("value_type") == "continuous_bin"
                    and entry.get("has_mass_point_aware_binning")
                    for entry in entries
                )
            ),
            "continuous_tokens_from_reduced_bin_variables": int(
                sum(
                    entry.get("value_type") == "continuous_bin"
                    and entry.get("is_reduced_bin_variable")
                    for entry in entries
                )
            ),
            "satlife_non_scale_token_present": len(non_scale_values) == 1
            and non_scale_values[0].get("valid_for_variable_scale") is False,
        },
        "ordered_tokens": [entry["token"] for entry in entries],
        "token_to_index": token_to_index(entries),
        "values": entries,
    }
    if not value_vocab["summary"]["satlife_non_scale_token_present"]:
        raise RuntimeError("satlife=6 non-scale token was not preserved in value_vocab")
    return value_vocab


def build_module_vocab(trait_vocab: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    first_seen: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for trait in trait_vocab["traits"]:
        module = str(trait["module"])
        first_seen.setdefault(module, []).append(trait)

    entries: list[dict[str, Any]] = []
    for module, traits in first_seen.items():
        tokenization_type_counts = Counter(trait["tokenization_type"] for trait in traits)
        entries.append(
            {
                "index": len(entries),
                "token": module_token(module),
                "module": module,
                "trait_count": len(traits),
                "variables": [trait["variable"] for trait in traits],
                "tokenization_type_counts": dict(tokenization_type_counts),
            }
        )
    ensure_unique_tokens(entries, "module_vocab")
    module_vocab = {
        "metadata": {
            **base_metadata(config),
            "token_format": "module:<module>",
        },
        "summary": {
            "module_count": len(entries),
            "trait_count": int(sum(entry["trait_count"] for entry in entries)),
        },
        "ordered_tokens": [entry["token"] for entry in entries],
        "token_to_index": token_to_index(entries),
        "modules": entries,
    }
    return module_vocab


def build_wave_vocab(value_label_map: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    wave_entry = value_label_map["variables"].get("wave")
    if not wave_entry:
        raise RuntimeError("value_label_map does not contain wave context values")
    values = sorted(
        wave_entry["values"], key=lambda item: int(item.get("value_order") or item.get("raw_value"))
    )
    entries = []
    for value_row in values:
        entries.append(
            {
                "index": len(entries),
                "token": value_row["token"],
                "wave": int(value_row["raw_value"]),
                "raw_value": value_row["raw_value"],
                "value_label": value_row["value_label"],
                "observed_count": int(value_row["observed_count"]),
                "observed_rate_all_rows": value_row.get("observed_rate_all_rows"),
                "observed_rate_non_missing": value_row.get("observed_rate_non_missing"),
                "include_as_masked_value_target_v1": False,
                "role": "wave_context",
            }
        )
    ensure_unique_tokens(entries, "wave_vocab")
    wave_vocab = {
        "metadata": {
            **base_metadata(config),
            "token_format": "wave:<wave>:<label>",
            "source_variable": "wave",
        },
        "summary": {
            "wave_token_count": len(entries),
            "waves": [entry["wave"] for entry in entries],
            "wave_is_masked_value_target": False,
        },
        "ordered_tokens": [entry["token"] for entry in entries],
        "token_to_index": token_to_index(entries),
        "waves": entries,
    }
    return wave_vocab


def write_report(
    report_path: Path,
    trait_vocab: dict[str, Any],
    value_vocab: dict[str, Any],
    module_vocab: dict[str, Any],
    wave_vocab: dict[str, Any],
    continuous_bins: dict[str, Any],
) -> None:
    reduced_variables = [
        (variable, entry)
        for variable, entry in continuous_bins["variables"].items()
        if int(entry["actual_bin_count"]) < int(entry["target_bin_count"])
    ]
    mass_point_variables = [
        (variable, entry)
        for variable, entry in continuous_bins["variables"].items()
        if len(entry.get("mass_points", [])) > 0
    ]
    satlife_non_scale = [
        entry
        for entry in value_vocab["values"]
        if entry.get("variable") == "satlife" and entry.get("raw_value") == "6"
    ]
    lines = [
        "# CHARLS Vocabulary Report",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
        f"- Trait tokens: {trait_vocab['summary']['trait_count']}",
        f"- Value tokens: {value_vocab['summary']['value_token_count']}",
        f"- Module tokens: {module_vocab['summary']['module_count']}",
        f"- Wave tokens: {wave_vocab['summary']['wave_token_count']}",
        f"- Non-scale value tokens: {value_vocab['summary']['non_scale_value_tokens']}",
        f"- Continuous tokens from mass-point-aware variables: {value_vocab['summary']['continuous_tokens_from_mass_point_aware_variables']}",
        f"- Continuous tokens from reduced-bin variables: {value_vocab['summary']['continuous_tokens_from_reduced_bin_variables']}",
        "",
        "## Guardrail Checks",
        "",
        f"- `satlife=6` preserved as non-scale: `{value_vocab['summary']['satlife_non_scale_token_present']}`",
        f"- `wave` masked target: `{wave_vocab['summary']['wave_is_masked_value_target']}`",
        f"- Duplicate tokens within each vocab: `0`",
        "",
        "## Trait Tokenization Type Counts",
        "",
        "| tokenization_type | trait_count |",
        "|---|---:|",
    ]
    for tokenization_type, count in trait_vocab["summary"]["tokenization_type_counts"].items():
        lines.append(f"| {tokenization_type} | {count} |")
    lines.extend(
        ["", "## Value Token Type Counts", "", "| value_type | token_count |", "|---|---:|"]
    )
    for value_type, count in value_vocab["summary"]["value_type_counts"].items():
        lines.append(f"| {value_type} | {count} |")
    lines.extend(["", "## Reduced Continuous Variables Carried Forward", ""])
    if not reduced_variables:
        lines.append("No reduced-bin variables.")
    else:
        lines.extend(["| variable | status | actual_bins | target_bins |", "|---|---|---:|---:|"])
        for variable, entry in reduced_variables:
            lines.append(
                f"| {variable} | {entry['status']} | {entry['actual_bin_count']} | {entry['target_bin_count']} |"
            )
    lines.extend(["", "## Mass-Point-Aware Variables Carried Forward", ""])
    if not mass_point_variables:
        lines.append("No mass-point-aware variables.")
    else:
        lines.extend(["| variable | status | actual_bins | mass_points |", "|---|---|---:|---|"])
        for variable, entry in mass_point_variables:
            mass_text = "; ".join(
                f"{point['position']}={point['value']} ({point['observed_rate_non_missing']:.3f})"
                for point in entry.get("mass_points", [])
            )
            lines.append(
                f"| {variable} | {entry['status']} | {entry['actual_bin_count']} | {mass_text} |"
            )
    lines.extend(["", "## Non-Scale Value Tokens", ""])
    if not satlife_non_scale:
        lines.append("No non-scale value token found.")
    else:
        lines.extend(["| variable | raw_value | token | semantic_note |", "|---|---:|---|---|"])
        for entry in satlife_non_scale:
            note = str(entry.get("semantic_note", "")).replace("|", "/")
            lines.append(
                f"| {entry['variable']} | {entry['raw_value']} | {entry['token']} | {note} |"
            )
    lines.extend(
        [
            "",
            "## Next Step",
            "",
            "Build `tokenization/tokenize_person_wave.py` using these vocabularies. The tokenizer must skip masked-target training for context/non-scale values while still preserving them in the sequence where appropriate.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    config = load_yaml(args.config)
    variable_map = pd.read_csv(resolve_path(config["outputs"]["variable_type_map"]))
    value_label_map = load_json(resolve_path(config["outputs"]["value_label_map"]))
    continuous_bins = load_json(resolve_path(config["outputs"]["continuous_bin_edges"]))

    trait_vocab, trait_rows = build_trait_vocab(
        variable_map, value_label_map, continuous_bins, config
    )
    value_vocab = build_value_vocab(
        trait_rows, trait_vocab, value_label_map, continuous_bins, config
    )
    module_vocab = build_module_vocab(trait_vocab, config)
    wave_vocab = build_wave_vocab(value_label_map, config)

    write_json(resolve_path(config["outputs"]["trait_vocab"]), trait_vocab)
    write_json(resolve_path(config["outputs"]["value_vocab"]), value_vocab)
    write_json(resolve_path(config["outputs"]["module_vocab"]), module_vocab)
    write_json(resolve_path(config["outputs"]["wave_vocab"]), wave_vocab)
    write_report(
        resolve_path(config["outputs"]["vocabulary_report"]),
        trait_vocab,
        value_vocab,
        module_vocab,
        wave_vocab,
        continuous_bins,
    )

    summary = {
        "trait_tokens": trait_vocab["summary"]["trait_count"],
        "value_tokens": value_vocab["summary"]["value_token_count"],
        "module_tokens": module_vocab["summary"]["module_count"],
        "wave_tokens": wave_vocab["summary"]["wave_token_count"],
        "satlife_non_scale_token_present": value_vocab["summary"][
            "satlife_non_scale_token_present"
        ],
        "continuous_tokens_from_mass_point_aware_variables": value_vocab["summary"][
            "continuous_tokens_from_mass_point_aware_variables"
        ],
        "continuous_tokens_from_reduced_bin_variables": value_vocab["summary"][
            "continuous_tokens_from_reduced_bin_variables"
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
