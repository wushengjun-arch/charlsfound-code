#!/usr/bin/env python3
"""Build CHARLS categorical value-label token map."""

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

import pandas as pd
import yaml

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_TOKENIZATION_CONFIG = SCRIPT_PATH.parents[1] / "configs" / "tokenization.yaml"
DEFAULT_DATA_CONFIG = SCRIPT_PATH.parents[1] / "configs" / "data.yaml"

VALUE_MAPPED_TYPES = {
    "categorical_value_label",
    "categorical_raw_code",
    "categorical_high_cardinality_context",
    "wave_context",
}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


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
        timeout=300,
    )
    records = payload.get("records") or []
    if not records:
        raise RuntimeError(f"Value-label extract returned no records for columns: {columns[:5]}")
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
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        if value.is_integer():
            return str(int(value))
        return format(value, ".12g")
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


def safe_token_part(value: Any) -> str:
    text = normalize_raw_value(value)
    if text is None:
        return "<missing>"
    for old, new in [("\n", " "), ("\r", " "), ("\t", " "), (":", "_"), ("|", "_")]:
        text = text.replace(old, new)
    text = "_".join(text.split())
    return text or "<empty>"


def raw_sort_key(value: str) -> tuple[int, float | str]:
    try:
        return (0, float(value))
    except ValueError:
        return (1, value)


def jsonable(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def observed_value_counts(
    api_base: str,
    source_id: str,
    main_file: str,
    variables: list[str],
    row_limit: int,
    chunk_size: int,
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, int]]]:
    counts_by_variable: dict[str, dict[str, int]] = {}
    stats_by_variable: dict[str, dict[str, int]] = {}
    total_chunks = math.ceil(len(variables) / chunk_size)
    for chunk_index, variable_chunk in enumerate(chunked(variables, chunk_size), start=1):
        print(
            f"Extracting observed categorical values {chunk_index}/{total_chunks}: {len(variable_chunk)} variables"
        )
        df = extract_columns(api_base, source_id, main_file, variable_chunk, row_limit)
        total_rows = len(df)
        for variable in variable_chunk:
            if variable not in df.columns:
                counts_by_variable[variable] = {}
                stats_by_variable[variable] = {
                    "rows_extracted": total_rows,
                    "observed_non_missing_count": 0,
                    "observed_missing_count": total_rows,
                    "observed_unique_count": 0,
                }
                continue
            normalized = df[variable].map(normalize_raw_value)
            non_missing = normalized.dropna()
            counts = non_missing.value_counts(dropna=True).to_dict()
            counts_by_variable[variable] = {str(k): int(v) for k, v in counts.items()}
            stats_by_variable[variable] = {
                "rows_extracted": total_rows,
                "observed_non_missing_count": int(non_missing.shape[0]),
                "observed_missing_count": int(total_rows - non_missing.shape[0]),
                "observed_unique_count": int(non_missing.nunique(dropna=True)),
            }
    return counts_by_variable, stats_by_variable


def load_supplemental_labels(config: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    path_value = config.get("inputs", {}).get("value_label_supplement")
    if not path_value:
        return {}
    path = resolve_path(path_value)
    if not path.exists():
        return {}
    payload = load_yaml(path) or {}
    supplements: dict[str, dict[str, dict[str, Any]]] = {}
    for variable, mapping in (payload.get("supplemental_labels") or {}).items():
        supplements[str(variable)] = {
            normalize_raw_value(code) or str(code): dict(details or {})
            for code, details in (mapping or {}).items()
        }
    return supplements


def supplemental_value_row(
    variable: str,
    raw_value: str,
    count: int,
    rows_extracted: int,
    observed_non_missing: int,
    supplement: dict[str, Any],
) -> dict[str, Any]:
    label = str(supplement["value_label"])
    row = {
        "raw_value": raw_value,
        "value_label": label,
        "label_source": str(supplement.get("label_source", "value_label_supplement")),
        "value_order": None,
        "observed_count": count,
        "observed_rate_all_rows": count / rows_extracted,
        "observed_rate_non_missing": count / observed_non_missing,
        "token": f"value:{variable}:{safe_token_part(raw_value)}:{safe_token_part(label)}",
    }
    for key in [
        "source_question",
        "source_url",
        "source_lines",
        "evidence_note",
        "semantic_note",
        "valid_for_variable_scale",
    ]:
        if key in supplement:
            row[key] = supplement[key]
    return row


def build_value_rows(
    variable: str,
    tokenization_type: str,
    label_rows: pd.DataFrame,
    observed_counts: dict[str, int],
    stats: dict[str, int],
    supplemental_labels: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]], str]:
    values: list[dict[str, Any]] = []
    unlabeled_observed: list[str] = []
    applied_supplements: list[dict[str, Any]] = []
    rows_extracted = max(int(stats.get("rows_extracted", 0)), 1)
    observed_non_missing = max(int(stats.get("observed_non_missing_count", 0)), 1)
    has_dictionary = not label_rows.empty

    if has_dictionary:
        label_rows = label_rows.sort_values(["value_order", "code"], kind="stable")
        dictionary_codes: set[str] = set()
        for _, label_row in label_rows.iterrows():
            raw_value = normalize_raw_value(label_row["code"])
            if raw_value is None:
                continue
            dictionary_codes.add(raw_value)
            label = str(label_row["value_label"])
            count = int(observed_counts.get(raw_value, 0))
            token_prefix = "wave" if tokenization_type == "wave_context" else f"value:{variable}"
            token = f"{token_prefix}:{safe_token_part(raw_value)}:{safe_token_part(label)}"
            values.append(
                {
                    "raw_value": raw_value,
                    "value_label": label,
                    "label_source": "value_label_dictionary",
                    "value_order": int(label_row["value_order"]),
                    "observed_count": count,
                    "observed_rate_all_rows": count / rows_extracted,
                    "observed_rate_non_missing": count / observed_non_missing,
                    "token": token,
                }
            )
        for raw_value in sorted(set(observed_counts) - dictionary_codes, key=raw_sort_key):
            count = int(observed_counts[raw_value])
            if raw_value in supplemental_labels:
                supplement = supplemental_labels[raw_value]
                values.append(
                    supplemental_value_row(
                        variable,
                        raw_value,
                        count,
                        rows_extracted,
                        observed_non_missing,
                        supplement,
                    )
                )
                applied_supplements.append(
                    {
                        "variable": variable,
                        "raw_value": raw_value,
                        "value_label": supplement["value_label"],
                        "source_question": supplement.get("source_question"),
                        "source_lines": supplement.get("source_lines"),
                        "semantic_note": supplement.get("semantic_note"),
                    }
                )
            else:
                unlabeled_observed.append(raw_value)
                values.append(
                    {
                        "raw_value": raw_value,
                        "value_label": raw_value,
                        "label_source": "observed_unlabeled_value",
                        "value_order": None,
                        "observed_count": count,
                        "observed_rate_all_rows": count / rows_extracted,
                        "observed_rate_non_missing": count / observed_non_missing,
                        "token": f"value:{variable}:{safe_token_part(raw_value)}",
                    }
                )
        map_source = (
            "value_label_dictionary+supplement" if applied_supplements else "value_label_dictionary"
        )
        return values, unlabeled_observed, applied_supplements, map_source

    for raw_value in sorted(observed_counts, key=raw_sort_key):
        count = int(observed_counts[raw_value])
        if raw_value in supplemental_labels:
            supplement = supplemental_labels[raw_value]
            values.append(
                supplemental_value_row(
                    variable,
                    raw_value,
                    count,
                    rows_extracted,
                    observed_non_missing,
                    supplement,
                )
            )
            applied_supplements.append(
                {
                    "variable": variable,
                    "raw_value": raw_value,
                    "value_label": supplement["value_label"],
                    "source_question": supplement.get("source_question"),
                    "source_lines": supplement.get("source_lines"),
                    "semantic_note": supplement.get("semantic_note"),
                }
            )
            continue
        values.append(
            {
                "raw_value": raw_value,
                "value_label": raw_value,
                "label_source": "observed_raw_value",
                "value_order": None,
                "observed_count": count,
                "observed_rate_all_rows": count / rows_extracted,
                "observed_rate_non_missing": count / observed_non_missing,
                "token": f"value:{variable}:{safe_token_part(raw_value)}",
            }
        )
    map_source = "observed_raw_values+supplement" if applied_supplements else "observed_raw_values"
    return values, unlabeled_observed, applied_supplements, map_source


def build_report(
    report_path: Path,
    tokenization_config: dict[str, Any],
    data_config: dict[str, Any],
    variable_entries: dict[str, dict[str, Any]],
    warnings: list[dict[str, Any]],
    applied_supplements: list[dict[str, Any]],
) -> None:
    variable_records = [
        {
            "variable": variable,
            "label_zh": entry["label_zh"],
            "module": entry["module"],
            "tokenization_type": entry["tokenization_type"],
            "map_source": entry["map_source"],
            "value_count": entry["value_count"],
            "observed_unique_count": entry["observed_unique_count"],
            "observed_non_missing_count": entry["observed_non_missing_count"],
            "unlabeled_observed_value_count": len(entry["unlabeled_observed_values"]),
            "include_as_masked_value_target_v1": entry["usage"][
                "include_as_masked_value_target_v1"
            ],
        }
        for variable, entry in variable_entries.items()
    ]
    variables_df = pd.DataFrame(variable_records)
    source_counts = variables_df["map_source"].value_counts().reset_index()
    type_counts = variables_df["tokenization_type"].value_counts().reset_index()
    high_cardinality = variables_df.sort_values(
        ["value_count", "variable"], ascending=[False, True]
    ).head(15)
    if warnings:
        warning_rows = pd.DataFrame(warnings).sort_values(
            ["unlabeled_observed_value_count", "variable"], ascending=[False, True]
        )
    else:
        warning_rows = pd.DataFrame(
            columns=["variable", "label_zh", "unlabeled_observed_value_count", "examples"]
        )

    lines = [
        "# CHARLS Value-Label Map Report",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Inputs",
        "",
        f"- Variable type map: `{tokenization_config['outputs']['variable_type_map']}`",
        f"- Value-label dictionary: `{tokenization_config['inputs']['value_labels']}`",
        f"- Value-label supplement: `{tokenization_config['inputs'].get('value_label_supplement')}`",
        f"- API extract: `GET /extract?source={data_config['source']['id']}&file={data_config['source']['main_file']}&columns=<categorical variables>&limit={data_config['source'].get('full_row_limit', 100000)}&format=json`",
        "",
        "## Summary",
        "",
        f"- Variables in value-label map: {len(variable_entries)}",
        f"- Value tokens/context tokens: {int(variables_df['value_count'].sum())}",
        f"- Variables eligible as masked value targets: {int(variables_df['include_as_masked_value_target_v1'].sum())}",
        f"- Supplemental labels applied: {len(applied_supplements)}",
        f"- Variables with unlabeled observed values despite dictionary labels: {len(warnings)}",
        "",
        "## Tokenization Type Counts",
        "",
        "| tokenization_type | variable_count |",
        "|---|---:|",
    ]
    for _, row in type_counts.iterrows():
        lines.append(f"| {row['tokenization_type']} | {row['count']} |")
    lines.extend(["", "## Map Source Counts", "", "| map_source | variable_count |", "|---|---:|"])
    for _, row in source_counts.iterrows():
        lines.append(f"| {row['map_source']} | {row['count']} |")
    lines.extend(
        [
            "",
            "## Highest Cardinality Value Maps",
            "",
            "| variable | label | tokenization_type | map_source | value_count | observed_unique_count |",
            "|---|---|---|---|---:|---:|",
        ]
    )
    for _, row in high_cardinality.iterrows():
        label = str(row["label_zh"]).replace("|", "/")
        lines.append(
            f"| {row['variable']} | {label} | {row['tokenization_type']} | {row['map_source']} | {row['value_count']} | {row['observed_unique_count']} |"
        )
    lines.extend(["", "## Supplemental Labels Applied", ""])
    if applied_supplements:
        lines.extend(
            [
                "| variable | raw_value | supplemental_label | source_question | source_lines | semantic_note |",
                "|---|---:|---|---|---|---|",
            ]
        )
        for item in applied_supplements:
            label = str(item.get("value_label", "")).replace("|", "/")
            semantic_note = str(item.get("semantic_note", "")).replace("|", "/")
            lines.append(
                f"| {item['variable']} | {item['raw_value']} | {label} | {item.get('source_question')} | {item.get('source_lines')} | {semantic_note} |"
            )
    else:
        lines.append("No supplemental labels were applied.")
    lines.extend(
        [
            "",
            "## Dictionary Coverage Warnings",
            "",
        ]
    )
    if warning_rows.empty:
        lines.append(
            "No observed categorical values were missing from the official value-label dictionary."
        )
    else:
        lines.extend(
            [
                "| variable | label | missing_label_value_count | examples |",
                "|---|---|---:|---|",
            ]
        )
        for _, row in warning_rows.head(30).iterrows():
            examples = ", ".join(str(value) for value in row["examples"])
            label = str(row["label_zh"]).replace("|", "/")
            lines.append(
                f"| {row['variable']} | {label} | {row['unlabeled_observed_value_count']} | {examples} |"
            )
    lines.extend(
        [
            "",
            "## Next Step",
            "",
            "Build `tokenization/build_bins.py` and generate global v1 continuous-bin edges for variables marked `continuous_quantile_bin`.",
            "Before final vocabulary construction, keep `charls_found/configs/value_label_supplement.yaml` under review as the auditable source for labels not present in the raw exported value-label dictionary.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_TOKENIZATION_CONFIG)
    parser.add_argument("--data-config", type=Path, default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--api-base")
    parser.add_argument("--row-limit", type=int)
    parser.add_argument("--chunk-size", type=int, default=80)
    args = parser.parse_args()

    tokenization_config = load_yaml(args.config)
    data_config = load_yaml(args.data_config)
    variable_map = pd.read_csv(resolve_path(tokenization_config["outputs"]["variable_type_map"]))
    value_labels = pd.read_csv(resolve_path(tokenization_config["inputs"]["value_labels"]))
    supplemental_labels = load_supplemental_labels(tokenization_config)

    api_base = args.api_base or data_config["api"]["base_url"]
    source_id = data_config["source"]["id"]
    main_file = data_config["source"]["main_file"]
    row_limit = args.row_limit or int(data_config["source"].get("full_row_limit", 100000))

    value_variables = variable_map[
        variable_map["tokenization_type"].isin(VALUE_MAPPED_TYPES)
    ].copy()
    value_variables = value_variables.sort_values("variable", kind="stable")
    variables = value_variables["variable"].astype(str).tolist()
    if not variables:
        raise RuntimeError("No categorical or wave-context variables found in variable type map")

    counts_by_variable, stats_by_variable = observed_value_counts(
        api_base=api_base,
        source_id=source_id,
        main_file=main_file,
        variables=variables,
        row_limit=row_limit,
        chunk_size=max(1, args.chunk_size),
    )

    labels_by_variable = {
        variable: group.copy() for variable, group in value_labels.groupby("variable", sort=False)
    }
    variable_entries: dict[str, dict[str, Any]] = {}
    warnings: list[dict[str, Any]] = []
    applied_supplements: list[dict[str, Any]] = []

    for _, row in value_variables.iterrows():
        variable = str(row["variable"])
        tokenization_type = str(row["tokenization_type"])
        stats = stats_by_variable.get(variable, {})
        observed_counts = counts_by_variable.get(variable, {})
        label_rows = labels_by_variable.get(variable, pd.DataFrame())
        values, unlabeled_observed, variable_supplements, map_source = build_value_rows(
            variable=variable,
            tokenization_type=tokenization_type,
            label_rows=label_rows,
            observed_counts=observed_counts,
            stats=stats,
            supplemental_labels=supplemental_labels.get(variable, {}),
        )
        applied_supplements.extend(variable_supplements)
        include_as_wave_context = bool(row.get("include_as_wave_context", False))
        include_in_tokenization = bool(row.get("include_in_tokenization_v1", False))
        include_as_masked_target = include_in_tokenization and tokenization_type not in {
            "wave_context",
            "categorical_high_cardinality_context",
        }
        entry = {
            "variable": variable,
            "label_zh": jsonable(row.get("label_zh")),
            "module": jsonable(row.get("module")),
            "tokenization_type": tokenization_type,
            "review_status": jsonable(row.get("review_status")),
            "map_source": map_source,
            "usage": {
                "include_in_tokenization_v1": include_in_tokenization,
                "include_as_wave_context": include_as_wave_context,
                "include_in_pretraining_v1": bool(row.get("include_in_pretraining_v1", False)),
                "include_as_masked_value_target_v1": include_as_masked_target,
                "requires_per_outcome_filter": bool(row.get("requires_per_outcome_filter", False)),
            },
            "observed_non_missing_count": int(stats.get("observed_non_missing_count", 0)),
            "observed_missing_count": int(stats.get("observed_missing_count", 0)),
            "observed_unique_count": int(stats.get("observed_unique_count", 0)),
            "value_count": len(values),
            "unlabeled_observed_values": unlabeled_observed,
            "supplemental_label_count": len(variable_supplements),
            "values": values,
        }
        variable_entries[variable] = entry
        if unlabeled_observed and not label_rows.empty:
            warnings.append(
                {
                    "variable": variable,
                    "label_zh": jsonable(row.get("label_zh")),
                    "unlabeled_observed_value_count": len(unlabeled_observed),
                    "examples": unlabeled_observed[:10],
                }
            )

    output = {
        "metadata": {
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source_id": source_id,
            "main_file": main_file,
            "api_base": api_base,
            "api_endpoint": "GET /extract?source=<source>&file=<main_file>&columns=<categorical variables>&limit=<row_limit>&format=json",
            "row_limit": row_limit,
            "variable_type_map": tokenization_config["outputs"]["variable_type_map"],
            "value_label_dictionary": tokenization_config["inputs"]["value_labels"],
            "value_label_supplement": tokenization_config["inputs"].get("value_label_supplement"),
            "included_tokenization_types": sorted(VALUE_MAPPED_TYPES),
            "special_tokens": tokenization_config.get("special_tokens", {}),
        },
        "summary": {
            "variable_count": len(variable_entries),
            "token_count": int(sum(entry["value_count"] for entry in variable_entries.values())),
            "masked_value_target_variable_count": int(
                sum(
                    entry["usage"]["include_as_masked_value_target_v1"]
                    for entry in variable_entries.values()
                )
            ),
            "variables_with_dictionary_labels": int(
                sum(
                    1
                    for entry in variable_entries.values()
                    if entry["map_source"].startswith("value_label_dictionary")
                )
            ),
            "variables_from_observed_raw_values": int(
                sum(
                    1
                    for entry in variable_entries.values()
                    if entry["map_source"].startswith("observed_raw_values")
                )
            ),
            "supplemental_label_count": len(applied_supplements),
            "variables_with_dictionary_coverage_warnings": len(warnings),
        },
        "variables": variable_entries,
        "supplemental_labels_applied": applied_supplements,
        "dictionary_coverage_warnings": warnings,
    }

    out_path = resolve_path(tokenization_config["outputs"]["value_label_map"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report_path = resolve_path(tokenization_config["outputs"]["value_label_map_report"])
    build_report(
        report_path,
        tokenization_config,
        data_config,
        variable_entries,
        warnings,
        applied_supplements,
    )

    print(f"Wrote {out_path}")
    print(f"Wrote {report_path}")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
