#!/usr/bin/env python3
"""Build global v1 quantile bin edges for CHARLS continuous variables."""

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

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_TOKENIZATION_CONFIG = SCRIPT_PATH.parents[1] / "configs" / "tokenization.yaml"
DEFAULT_DATA_CONFIG = SCRIPT_PATH.parents[1] / "configs" / "data.yaml"


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
        raise RuntimeError(f"Continuous-bin extract returned no records for columns: {columns[:5]}")
    return pd.DataFrame.from_records(records)


def chunked(values: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def finite_numeric(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric[np.isfinite(numeric)].dropna()


def round_float(value: float | int | np.number, digits: int = 12) -> float:
    return float(f"{float(value):.{digits}g}")


def jsonable(value: Any) -> Any:
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def make_token(variable: str, bin_index: int) -> str:
    return f"value:{variable}:bin_{bin_index:02d}"


def assign_bins(values: pd.Series, cutpoints: list[float]) -> np.ndarray:
    if not cutpoints:
        return np.zeros(len(values), dtype=int)
    cuts = np.asarray(cutpoints, dtype=float)
    return np.searchsorted(cuts, values.to_numpy(dtype=float), side="left")


def qcut_internal_cutpoints(values: pd.Series, requested_bins: int) -> list[float]:
    if requested_bins <= 1 or values.nunique(dropna=True) <= 1:
        return []
    try:
        _, raw_edges = pd.qcut(
            values, q=requested_bins, labels=False, retbins=True, duplicates="drop"
        )
        raw_cutpoints = raw_edges[1:-1]
    except ValueError:
        quantiles = [idx / requested_bins for idx in range(1, requested_bins)]
        raw_cutpoints = values.quantile(quantiles, interpolation="linear").tolist()
    observed_min = float(values.min())
    observed_max = float(values.max())
    return sorted(
        set(
            round_float(edge) for edge in raw_cutpoints if observed_min < float(edge) < observed_max
        )
    )


def mass_point_aware_cutpoints(
    numeric: pd.Series,
    target_bins: int,
    mass_point_threshold: float,
) -> tuple[list[float], str, list[dict[str, Any]]]:
    observed_min = float(numeric.min())
    observed_max = float(numeric.max())
    lower_share = float((numeric == observed_min).mean())
    upper_share = float((numeric == observed_max).mean())
    lower_mass = lower_share >= mass_point_threshold
    upper_mass = upper_share >= mass_point_threshold and observed_max > observed_min

    if not lower_mass and not upper_mass:
        cutpoints = qcut_internal_cutpoints(numeric, target_bins)
        status = (
            "ok" if len(cutpoints) + 1 == target_bins else "reduced_bins_due_to_duplicate_quantiles"
        )
        return cutpoints, status, []

    remaining = numeric
    mass_points: list[dict[str, Any]] = []
    if lower_mass:
        remaining = remaining[remaining > observed_min]
        mass_points.append(
            {
                "position": "minimum",
                "value": round_float(observed_min),
                "observed_count": int((numeric == observed_min).sum()),
                "observed_rate_non_missing": lower_share,
            }
        )
    if upper_mass:
        remaining = remaining[remaining < observed_max]
        mass_points.append(
            {
                "position": "maximum",
                "value": round_float(observed_max),
                "observed_count": int((numeric == observed_max).sum()),
                "observed_rate_non_missing": upper_share,
            }
        )

    remaining_bins = max(1, target_bins - len(mass_points))
    cutpoints: list[float] = []
    if lower_mass:
        cutpoints.append(round_float(observed_min))
    if not remaining.empty:
        cutpoints.extend(qcut_internal_cutpoints(remaining, remaining_bins))
        if upper_mass:
            cutpoints.append(round_float(float(remaining.max())))
    elif lower_mass and upper_mass:
        cutpoints.append(round_float(observed_min))

    cutpoints = sorted(set(edge for edge in cutpoints if observed_min <= edge < observed_max))
    status = (
        "mass_point_aware_quantile"
        if len(cutpoints) + 1 == target_bins
        else "mass_point_aware_quantile_reduced"
    )
    return cutpoints, status, mass_points


def build_bins_for_variable(
    row: pd.Series,
    series: pd.Series,
    target_bins: int,
    mass_point_threshold: float,
    total_rows: int,
) -> dict[str, Any]:
    variable = str(row["variable"])
    numeric = finite_numeric(series)
    non_missing_count = int(numeric.shape[0])
    missing_count = int(total_rows - non_missing_count)

    if non_missing_count == 0:
        return {
            "variable": variable,
            "label_zh": jsonable(row.get("label_zh")),
            "module": jsonable(row.get("module")),
            "strategy": "global_quantile_v1",
            "status": "no_numeric_values",
            "target_bin_count": target_bins,
            "actual_bin_count": 0,
            "non_missing_count": 0,
            "missing_count": missing_count,
            "missing_rate": missing_count / total_rows if total_rows else None,
            "unique_count_observed": 0,
            "observed_min": None,
            "observed_max": None,
            "mass_points": [],
            "cutpoints": [],
            "bins": [],
        }

    unique_count = int(numeric.nunique(dropna=True))
    observed_min = round_float(float(numeric.min()))
    observed_max = round_float(float(numeric.max()))

    status = "ok"
    mass_points: list[dict[str, Any]] = []
    if unique_count == 1:
        cutpoints: list[float] = []
        status = "single_unique_value"
    else:
        cutpoints, status, mass_points = mass_point_aware_cutpoints(
            numeric,
            target_bins=target_bins,
            mass_point_threshold=mass_point_threshold,
        )

    bin_index = assign_bins(numeric, cutpoints)
    actual_bin_count = len(cutpoints) + 1 if non_missing_count else 0
    counts = pd.Series(bin_index).value_counts().sort_index().to_dict()
    bins = []
    for idx in range(actual_bin_count):
        lower = None if idx == 0 else cutpoints[idx - 1]
        upper = None if idx == len(cutpoints) else cutpoints[idx]
        observed_count = int(counts.get(idx, 0))
        if lower is None and upper is None:
            interval_label = "all observed values"
        elif lower is None:
            interval_label = f"<= {upper}"
        elif upper is None:
            interval_label = f"> {lower}"
        else:
            interval_label = f"({lower}, {upper}]"
        bins.append(
            {
                "bin_index": idx,
                "token": make_token(variable, idx),
                "lower_bound_exclusive": lower,
                "upper_bound_inclusive": upper,
                "interval_label": interval_label,
                "observed_count": observed_count,
                "observed_rate_non_missing": (
                    observed_count / non_missing_count if non_missing_count else None
                ),
            }
        )

    return {
        "variable": variable,
        "label_zh": jsonable(row.get("label_zh")),
        "module": jsonable(row.get("module")),
        "review_status": jsonable(row.get("review_status")),
        "requires_per_outcome_filter": bool(row.get("requires_per_outcome_filter", False)),
        "strategy": "global_quantile_v1",
        "assignment_rule": "bin_index = number of cutpoints strictly below value; cutpoints are upper-bound inclusive",
        "status": status,
        "target_bin_count": target_bins,
        "actual_bin_count": actual_bin_count,
        "non_missing_count": non_missing_count,
        "missing_count": missing_count,
        "missing_rate": missing_count / total_rows if total_rows else None,
        "unique_count_observed": unique_count,
        "observed_min": observed_min,
        "observed_max": observed_max,
        "mass_points": mass_points,
        "cutpoints": cutpoints,
        "bins": bins,
    }


def build_report(
    report_path: Path,
    tokenization_config: dict[str, Any],
    data_config: dict[str, Any],
    variable_bins: dict[str, dict[str, Any]],
) -> None:
    records = [
        {
            "variable": variable,
            "label_zh": entry["label_zh"],
            "module": entry["module"],
            "status": entry["status"],
            "actual_bin_count": entry["actual_bin_count"],
            "target_bin_count": entry["target_bin_count"],
            "non_missing_count": entry["non_missing_count"],
            "missing_rate": entry["missing_rate"],
            "unique_count_observed": entry["unique_count_observed"],
            "observed_min": entry["observed_min"],
            "observed_max": entry["observed_max"],
            "mass_point_count": len(entry.get("mass_points", [])),
        }
        for variable, entry in variable_bins.items()
    ]
    df = pd.DataFrame(records)
    status_counts = df["status"].value_counts().reset_index()
    module_counts = df["module"].value_counts().reset_index()
    reduced = df[df["actual_bin_count"].lt(df["target_bin_count"])].sort_values(
        ["status", "variable"]
    )
    mass_point_aware = df[df["mass_point_count"].gt(0)].sort_values(["variable"])
    top_missing = df.sort_values(["missing_rate", "variable"], ascending=[False, True]).head(20)
    fewest_bins = df.sort_values(["actual_bin_count", "variable"], ascending=[True, True]).head(20)

    lines = [
        "# CHARLS Continuous Bin Edges Report",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Inputs",
        "",
        f"- Variable type map: `{tokenization_config['outputs']['variable_type_map']}`",
        f"- API extract: `GET /extract?source={data_config['source']['id']}&file={data_config['source']['main_file']}&columns=<continuous variables>&limit={data_config['source'].get('full_row_limit', 100000)}&format=json`",
        f"- Target quantile bins: {tokenization_config['thresholds'].get('continuous_quantile_bins', 10)}",
        f"- Mass-point threshold: {tokenization_config['thresholds'].get('continuous_mass_point_threshold', 0.2)}",
        "",
        "## Summary",
        "",
        f"- Continuous variables binned: {len(variable_bins)}",
        f"- Variables with target bin count: {int((df['actual_bin_count'] == df['target_bin_count']).sum())}",
        f"- Variables with reduced bins: {int((df['actual_bin_count'] < df['target_bin_count']).sum())}",
        f"- Variables using mass-point-aware binning: {int(df['mass_point_count'].gt(0).sum())}",
        f"- Minimum actual bin count: {int(df['actual_bin_count'].min())}",
        f"- Maximum actual bin count: {int(df['actual_bin_count'].max())}",
        "",
        "## Status Counts",
        "",
        "| status | variable_count |",
        "|---|---:|",
    ]
    for _, row in status_counts.iterrows():
        lines.append(f"| {row['status']} | {row['count']} |")
    lines.extend(["", "## Module Counts", "", "| module | variable_count |", "|---|---:|"])
    for _, row in module_counts.iterrows():
        module = str(row["module"]).replace("|", "/")
        lines.append(f"| {module} | {row['count']} |")
    lines.extend(
        [
            "",
            "## Mass-Point-Aware Variables",
            "",
        ]
    )
    if mass_point_aware.empty:
        lines.append("No variables required mass-point-aware binning.")
    else:
        lines.extend(
            [
                "| variable | label | status | actual_bins | mass_point_count | observed_min | observed_max |",
                "|---|---|---|---:|---:|---:|---:|",
            ]
        )
        for _, row in mass_point_aware.iterrows():
            label = str(row["label_zh"]).replace("|", "/")
            lines.append(
                f"| {row['variable']} | {label} | {row['status']} | {row['actual_bin_count']} | {row['mass_point_count']} | {row['observed_min']} | {row['observed_max']} |"
            )
    lines.extend(
        [
            "",
            "## Reduced Or Non-OK Bin Variables",
            "",
        ]
    )
    if reduced.empty:
        lines.append("All continuous variables reached the target bin count.")
    else:
        lines.extend(
            [
                "| variable | label | status | actual_bins | unique_count | missing_rate |",
                "|---|---|---|---:|---:|---:|",
            ]
        )
        for _, row in reduced.iterrows():
            label = str(row["label_zh"]).replace("|", "/")
            lines.append(
                f"| {row['variable']} | {label} | {row['status']} | {row['actual_bin_count']} | {row['unique_count_observed']} | {row['missing_rate']:.3f} |"
            )
    lines.extend(
        [
            "",
            "## Highest Missingness Continuous Variables",
            "",
            "| variable | label | actual_bins | missing_rate | non_missing_count |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for _, row in top_missing.iterrows():
        label = str(row["label_zh"]).replace("|", "/")
        lines.append(
            f"| {row['variable']} | {label} | {row['actual_bin_count']} | {row['missing_rate']:.3f} | {row['non_missing_count']} |"
        )
    lines.extend(
        [
            "",
            "## Fewest Actual Bins",
            "",
            "| variable | label | actual_bins | unique_count | observed_min | observed_max |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in fewest_bins.iterrows():
        label = str(row["label_zh"]).replace("|", "/")
        lines.append(
            f"| {row['variable']} | {label} | {row['actual_bin_count']} | {row['unique_count_observed']} | {row['observed_min']} | {row['observed_max']} |"
        )
    lines.extend(
        [
            "",
            "## Next Step",
            "",
            "Build `tokenization/build_vocabularies.py` from the reviewed variable map, value-label map, and continuous bin edges. Preserve supplemental value-label metadata, especially the non-scale flag for `satlife=6`.",
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
    parser.add_argument("--chunk-size", type=int, default=60)
    args = parser.parse_args()

    tokenization_config = load_yaml(args.config)
    data_config = load_yaml(args.data_config)
    variable_map = pd.read_csv(resolve_path(tokenization_config["outputs"]["variable_type_map"]))

    api_base = args.api_base or data_config["api"]["base_url"]
    source_id = data_config["source"]["id"]
    main_file = data_config["source"]["main_file"]
    row_limit = args.row_limit or int(data_config["source"].get("full_row_limit", 100000))
    target_bins = int(tokenization_config["thresholds"].get("continuous_quantile_bins", 10))
    mass_point_threshold = float(
        tokenization_config["thresholds"].get("continuous_mass_point_threshold", 0.2)
    )
    if target_bins < 2:
        raise RuntimeError("continuous_quantile_bins must be >= 2")
    if not 0 < mass_point_threshold < 1:
        raise RuntimeError("continuous_mass_point_threshold must be between 0 and 1")

    continuous = variable_map[
        variable_map["tokenization_type"].eq("continuous_quantile_bin")
    ].copy()
    continuous = continuous.sort_values("variable", kind="stable")
    variables = continuous["variable"].astype(str).tolist()
    if not variables:
        raise RuntimeError("No continuous_quantile_bin variables found in variable type map")

    row_lookup = {str(row["variable"]): row for _, row in continuous.iterrows()}
    variable_bins: dict[str, dict[str, Any]] = {}
    total_chunks = math.ceil(len(variables) / max(1, args.chunk_size))
    for chunk_index, variable_chunk in enumerate(
        chunked(variables, max(1, args.chunk_size)), start=1
    ):
        print(
            f"Extracting continuous values {chunk_index}/{total_chunks}: {len(variable_chunk)} variables"
        )
        df = extract_columns(api_base, source_id, main_file, variable_chunk, row_limit)
        total_rows = len(df)
        for variable in variable_chunk:
            if variable not in df.columns:
                raise RuntimeError(
                    f"Extract did not return requested continuous variable: {variable}"
                )
            variable_bins[variable] = build_bins_for_variable(
                row_lookup[variable],
                df[variable],
                target_bins=target_bins,
                mass_point_threshold=mass_point_threshold,
                total_rows=total_rows,
            )

    output = {
        "metadata": {
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source_id": source_id,
            "main_file": main_file,
            "api_base": api_base,
            "api_endpoint": "GET /extract?source=<source>&file=<main_file>&columns=<continuous variables>&limit=<row_limit>&format=json",
            "row_limit": row_limit,
            "variable_type_map": tokenization_config["outputs"]["variable_type_map"],
            "strategy": "global_quantile_v1",
            "target_bin_count": target_bins,
            "mass_point_threshold": mass_point_threshold,
        },
        "summary": {
            "continuous_variable_count": len(variable_bins),
            "target_bin_count": target_bins,
            "variables_with_target_bin_count": int(
                sum(entry["actual_bin_count"] == target_bins for entry in variable_bins.values())
            ),
            "variables_with_reduced_bin_count": int(
                sum(entry["actual_bin_count"] < target_bins for entry in variable_bins.values())
            ),
            "variables_using_mass_point_aware_binning": int(
                sum(len(entry.get("mass_points", [])) > 0 for entry in variable_bins.values())
            ),
            "status_counts": {
                str(status): int(count)
                for status, count in pd.Series(
                    [entry["status"] for entry in variable_bins.values()]
                )
                .value_counts()
                .items()
            },
        },
        "variables": variable_bins,
    }

    out_path = resolve_path(tokenization_config["outputs"]["continuous_bin_edges"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report_path = resolve_path(tokenization_config["outputs"]["continuous_bin_edges_report"])
    build_report(report_path, tokenization_config, data_config, variable_bins)

    print(f"Wrote {out_path}")
    print(f"Wrote {report_path}")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
