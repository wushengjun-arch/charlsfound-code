#!/usr/bin/env python3
"""Generate QC report for CHARLS person-wave tokenization."""

from __future__ import annotations

import argparse
import json
import time
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


def fixed_list_to_numpy(table: pa.Table, column: str, dtype: Any) -> np.ndarray:
    array = table[column].combine_chunks()
    list_size = array.type.list_size
    values = array.values.to_numpy(zero_copy_only=False).astype(dtype, copy=False)
    return values.reshape(len(array), list_size)


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> list[str]:
    if max_rows is not None:
        df = df.head(max_rows)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in df.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if pd.isna(value):
                values.append("")
            elif isinstance(value, int | np.integer):
                values.append(str(int(value)))
            elif isinstance(value, float | np.floating):
                numeric = float(value)
                if np.isfinite(numeric) and numeric.is_integer():
                    values.append(str(int(numeric)))
                else:
                    values.append(f"{numeric:.4f}")
            else:
                values.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def nullable_bool_sum(series: pd.Series) -> int:
    return int(
        series.map(
            lambda value: (
                False if pd.isna(value) else str(value).strip().lower() in {"true", "1", "yes"}
            )
        ).sum()
    )


def percentile_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "max": float(np.max(values)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_TOKENIZATION_CONFIG)
    parser.add_argument("--data-config", type=Path, default=DEFAULT_DATA_CONFIG)
    args = parser.parse_args()

    tokenization_config = load_yaml(args.config)
    data_config = load_yaml(args.data_config)
    trait_vocab = load_json(resolve_path(tokenization_config["outputs"]["trait_vocab"]))
    value_vocab = load_json(resolve_path(tokenization_config["outputs"]["value_vocab"]))
    module_vocab = load_json(resolve_path(tokenization_config["outputs"]["module_vocab"]))
    wave_vocab = load_json(resolve_path(tokenization_config["outputs"]["wave_vocab"]))
    value_label_map = load_json(resolve_path(tokenization_config["outputs"]["value_label_map"]))
    continuous_bins = load_json(
        resolve_path(tokenization_config["outputs"]["continuous_bin_edges"])
    )

    tokenized_path = resolve_path(tokenization_config["outputs"]["tokenized_person_wave"])
    metadata_path = resolve_path(
        tokenization_config["outputs"].get(
            "tokenization_metadata",
            "outputs/05_tokenization/charls_tokenization_metadata.csv",
        )
    )
    report_path = resolve_path(tokenization_config["outputs"]["tokenization_qc_report"])

    table = pq.read_table(tokenized_path)
    token_metadata = pd.read_csv(metadata_path)
    row_count = table.num_rows
    id_column = data_config["source"]["id_column"]
    wave_column = data_config["source"]["wave_column"]
    trait_count = int(trait_vocab["summary"]["trait_count"])

    sequence_lengths = table["sequence_length"].combine_chunks().to_numpy(zero_copy_only=False)
    missing_counts = table["missing_value_count"].combine_chunks().to_numpy(zero_copy_only=False)
    maskable_counts = table["maskable_value_count"].combine_chunks().to_numpy(zero_copy_only=False)
    non_scale_counts = (
        table["non_scale_value_count"].combine_chunks().to_numpy(zero_copy_only=False)
    )
    unknown_counts = table["unknown_value_count"].combine_chunks().to_numpy(zero_copy_only=False)
    waves = table[wave_column].combine_chunks().to_numpy(zero_copy_only=False)
    ids = table[id_column].combine_chunks().to_numpy(zero_copy_only=False)

    value_ids = fixed_list_to_numpy(table, "value_token_ids", np.int32)
    value_is_missing = fixed_list_to_numpy(table, "value_is_missing", bool)
    value_is_masked_target = fixed_list_to_numpy(table, "value_is_masked_target", bool)
    value_is_non_scale = fixed_list_to_numpy(table, "value_is_non_scale", bool)

    trait_entries = trait_vocab["traits"]
    trait_df = pd.DataFrame(
        [
            {
                "trait_position": pos,
                "variable": entry["variable"],
                "module": entry["module"],
                "tokenization_type": entry["tokenization_type"],
            }
            for pos, entry in enumerate(trait_entries)
        ]
    )
    per_trait = trait_df.copy()
    per_trait["missing_count"] = value_is_missing.sum(axis=0).astype(int)
    per_trait["missing_rate"] = per_trait["missing_count"] / row_count
    per_trait["maskable_count"] = value_is_masked_target.sum(axis=0).astype(int)
    per_trait["maskable_rate_all_rows"] = per_trait["maskable_count"] / row_count
    per_trait["non_scale_count"] = value_is_non_scale.sum(axis=0).astype(int)
    per_trait["observed_count"] = row_count - per_trait["missing_count"]

    wave_counts = pd.Series(waves).value_counts().sort_index().reset_index()
    wave_counts.columns = ["wave", "row_count"]
    wave_counts["row_rate"] = wave_counts["row_count"] / row_count

    module_rows = []
    for module, group in per_trait.groupby("module", sort=False):
        positions = group["trait_position"].to_numpy(dtype=int)
        module_rows.append(
            {
                "module": module,
                "trait_count": len(positions),
                "token_slots": int(len(positions) * row_count),
                "missing_rate": float(value_is_missing[:, positions].mean()),
                "maskable_rate": float(value_is_masked_target[:, positions].mean()),
                "non_scale_count": int(value_is_non_scale[:, positions].sum()),
            }
        )
    module_qc = pd.DataFrame(module_rows).sort_values(
        ["token_slots", "module"], ascending=[False, True]
    )

    categorical_traits = per_trait[
        per_trait["tokenization_type"].str.startswith("categorical")
    ].copy()
    categorical_unknown = int(unknown_counts.sum())
    dictionary_warnings = value_label_map.get("dictionary_coverage_warnings", [])

    continuous_rows = []
    value_entries_by_id = {int(entry["index"]): entry for entry in value_vocab["values"]}
    for _, trait_row in per_trait[
        per_trait["tokenization_type"].eq("continuous_quantile_bin")
    ].iterrows():
        position = int(trait_row["trait_position"])
        variable = str(trait_row["variable"])
        observed_ids = value_ids[~value_is_missing[:, position], position]
        counts = pd.Series(observed_ids).value_counts()
        bin_rates = counts / max(int(counts.sum()), 1)
        continuous_entry = continuous_bins["variables"][variable]
        continuous_rows.append(
            {
                "variable": variable,
                "status": continuous_entry["status"],
                "actual_bins": int(continuous_entry["actual_bin_count"]),
                "target_bins": int(continuous_entry["target_bin_count"]),
                "mass_point_aware": len(continuous_entry.get("mass_points", [])) > 0,
                "reduced_bins": int(continuous_entry["actual_bin_count"])
                < int(continuous_entry["target_bin_count"]),
                "non_missing_count": int(counts.sum()),
                "min_bin_rate": float(bin_rates.min()) if not bin_rates.empty else 0.0,
                "max_bin_rate": float(bin_rates.max()) if not bin_rates.empty else 0.0,
                "observed_bin_count": int(counts.shape[0]),
                "top_bin_token": (
                    value_entries_by_id[int(counts.index[0])]["token"] if not counts.empty else ""
                ),
            }
        )
    continuous_qc = pd.DataFrame(continuous_rows)
    continuous_imbalanced = continuous_qc.sort_values(
        ["max_bin_rate", "variable"], ascending=[False, True]
    )

    satlife_meta = token_metadata[
        (token_metadata["variable"].eq("satlife"))
        & (token_metadata["raw_value"].astype(str).eq("6"))
    ]
    satlife_non_scale_ok = False
    satlife_6_count = 0
    satlife_6_maskable_count = 0
    if not satlife_meta.empty:
        satlife_position = int(satlife_meta.iloc[0]["trait_position"])
        satlife_value_id = int(satlife_meta.iloc[0]["value_token_id"])
        satlife_6_rows = value_ids[:, satlife_position] == satlife_value_id
        satlife_6_count = int(satlife_6_rows.sum())
        satlife_6_maskable_count = int(
            value_is_masked_target[satlife_6_rows, satlife_position].sum()
        )
        satlife_non_scale_ok = (
            bool(satlife_meta.iloc[0]["valid_for_variable_scale"]) is False
            and bool(satlife_meta.iloc[0]["include_as_masked_value_target_v1"]) is False
            and satlife_6_maskable_count == 0
        )

    mass_point_variables = continuous_qc[continuous_qc["mass_point_aware"]].copy()
    reduced_variables = continuous_qc[continuous_qc["reduced_bins"]].copy()
    metadata_mass_point_rows = nullable_bool_sum(
        token_metadata["continuous_has_mass_point_aware_binning"]
    )
    metadata_reduced_rows = nullable_bool_sum(token_metadata["continuous_is_reduced_bin_variable"])

    seq_summary = percentile_summary(sequence_lengths.astype(float))
    missing_summary = percentile_summary(missing_counts.astype(float))
    maskable_summary = percentile_summary(maskable_counts.astype(float))

    lines = [
        "# CHARLS Person-Wave Tokenization QC Report",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Inputs",
        "",
        f"- Tokenized person-wave parquet: `{tokenization_config['outputs']['tokenized_person_wave']}`",
        f"- Token metadata: `{tokenization_config['outputs'].get('tokenization_metadata', 'outputs/05_tokenization/charls_tokenization_metadata.csv')}`",
        f"- Trait vocab: `{tokenization_config['outputs']['trait_vocab']}`",
        f"- Value vocab: `{tokenization_config['outputs']['value_vocab']}`",
        f"- API source: `GET /extract?source={data_config['source']['id']}&file={data_config['source']['main_file']}&columns=<chunked tokenization variables>&limit={data_config['source'].get('full_row_limit', 100000)}&format=json`",
        "",
        "## Summary",
        "",
        f"- Person-wave rows: {row_count}",
        f"- Unique IDs: {int(pd.Series(ids).nunique())}",
        f"- Trait slots per row: {trait_count}",
        f"- Total token slots: {row_count * trait_count}",
        f"- Unique trait tokens used: {trait_count}",
        f"- Value vocab size: {value_vocab['summary']['value_token_count']}",
        f"- Unique value token IDs observed: {int(pd.Series(value_ids.reshape(-1)).nunique())}",
        f"- Missing value assignments: {int(missing_counts.sum())} ({float(missing_counts.sum() / (row_count * trait_count)):.4f} of slots)",
        f"- Maskable value assignments: {int(maskable_counts.sum())} ({float(maskable_counts.sum() / (row_count * trait_count)):.4f} of slots)",
        f"- Non-scale value assignments: {int(non_scale_counts.sum())}",
        f"- Unknown value assignments: {int(unknown_counts.sum())}",
        "",
        "## Gate Checks",
        "",
        "| check | result | detail |",
        "|---|---|---|",
        f"| fixed sequence length equals trait vocab size | {bool(np.all(sequence_lengths == trait_count))} | min={int(sequence_lengths.min())}, max={int(sequence_lengths.max())} |",
        f"| wave is context only | {wave_vocab['summary']['wave_is_masked_value_target'] is False} | wave_token_count={wave_vocab['summary']['wave_token_count']} |",
        f"| no unknown value assignments | {int(unknown_counts.sum()) == 0} | unknown_assignments={int(unknown_counts.sum())} |",
        f"| satlife=6 preserved but not maskable | {satlife_non_scale_ok} | rows={satlife_6_count}, maskable_rows={satlife_6_maskable_count} |",
        f"| continuous mass-point metadata present | {metadata_mass_point_rows > 0} | metadata_rows={metadata_mass_point_rows}, variables={len(mass_point_variables)} |",
        f"| continuous reduced-bin metadata present | {metadata_reduced_rows > 0} | metadata_rows={metadata_reduced_rows}, variables={len(reduced_variables)} |",
        "",
        "## Sequence Length And Target Counts",
        "",
        "| metric | min | p25 | median | p75 | max |",
        "|---|---:|---:|---:|---:|---:|",
        f"| sequence_length | {seq_summary['min']:.0f} | {seq_summary['p25']:.0f} | {seq_summary['median']:.0f} | {seq_summary['p75']:.0f} | {seq_summary['max']:.0f} |",
        f"| missing_value_count | {missing_summary['min']:.0f} | {missing_summary['p25']:.0f} | {missing_summary['median']:.0f} | {missing_summary['p75']:.0f} | {missing_summary['max']:.0f} |",
        f"| maskable_value_count | {maskable_summary['min']:.0f} | {maskable_summary['p25']:.0f} | {maskable_summary['median']:.0f} | {maskable_summary['p75']:.0f} | {maskable_summary['max']:.0f} |",
        "",
        "## Wave Distribution",
        "",
    ]
    lines.extend(markdown_table(wave_counts, ["wave", "row_count", "row_rate"]))
    lines.extend(
        [
            "",
            "## Value-Label Coverage",
            "",
            f"- Categorical/context traits: {len(categorical_traits)}",
            f"- Categorical/context missing assignments: {int(per_trait.loc[categorical_traits.index, 'missing_count'].sum())}",
            f"- Unknown categorical assignments in tokenized parquet: {categorical_unknown}",
            f"- Dictionary coverage warnings carried from value-label map: {len(dictionary_warnings)}",
            "",
            "## Highest Missingness Traits",
            "",
        ]
    )
    high_missing_traits = per_trait.sort_values(
        ["missing_rate", "variable"], ascending=[False, True]
    )[["variable", "module", "tokenization_type", "missing_count", "missing_rate"]]
    lines.extend(
        markdown_table(high_missing_traits, list(high_missing_traits.columns), max_rows=20)
    )
    lines.extend(
        [
            "",
            "## Module Distribution",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            module_qc,
            [
                "module",
                "trait_count",
                "token_slots",
                "missing_rate",
                "maskable_rate",
                "non_scale_count",
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Continuous Bin Balance",
            "",
            f"- Continuous traits: {len(continuous_qc)}",
            f"- Mass-point-aware variables: {len(mass_point_variables)}",
            f"- Reduced-bin variables: {len(reduced_variables)}",
            "",
            "### Most Imbalanced Continuous Variables",
            "",
        ]
    )
    continuous_columns = [
        "variable",
        "status",
        "actual_bins",
        "target_bins",
        "mass_point_aware",
        "reduced_bins",
        "non_missing_count",
        "min_bin_rate",
        "max_bin_rate",
        "observed_bin_count",
    ]
    lines.extend(
        markdown_table(continuous_imbalanced[continuous_columns], continuous_columns, max_rows=20)
    )
    lines.extend(
        [
            "",
            "## Reduced Or Mass-Point-Aware Variables",
            "",
        ]
    )
    special_continuous = continuous_qc[
        continuous_qc["mass_point_aware"] | continuous_qc["reduced_bins"]
    ].sort_values("variable")
    if special_continuous.empty:
        lines.append("No continuous variables carried mass-point-aware or reduced-bin metadata.")
    else:
        lines.extend(markdown_table(special_continuous[continuous_columns], continuous_columns))
    lines.extend(
        [
            "",
            "## Next Step",
            "",
            "Proceed to Stage 4 only if the gate checks above remain true. Build the supervised matrix from baseline wave-1 rows, split by ID, and keep per-outcome leakage filters active.",
            "",
        ]
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")

    summary = {
        "rows": row_count,
        "unique_ids": int(pd.Series(ids).nunique()),
        "trait_count": trait_count,
        "missing_slot_rate": float(missing_counts.sum() / (row_count * trait_count)),
        "maskable_slot_rate": float(maskable_counts.sum() / (row_count * trait_count)),
        "unknown_assignments": int(unknown_counts.sum()),
        "satlife_6_non_scale_ok": satlife_non_scale_ok,
        "mass_point_aware_variables": int(len(mass_point_variables)),
        "reduced_bin_variables": int(len(reduced_variables)),
        "report": str(report_path.relative_to(PROJECT_ROOT)),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
