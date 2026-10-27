#!/usr/bin/env python3
"""Build the first API-backed CHARLS data audit artifacts."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_CONFIG = SCRIPT_PATH.parents[1] / "configs" / "data.yaml"


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def get_json(
    base_url: str, path: str, params: dict[str, Any] | None = None, timeout: int = 120
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


def post_json(
    base_url: str, path: str, payload: dict[str, Any], timeout: int = 60
) -> dict[str, Any]:
    url = base_url.rstrip("/") + path
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"HTTP {exc.code} for {url}: {body[:500]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach {url}: {exc}") from exc


def parse_physical_column(column: str) -> tuple[str, str]:
    if " (" in column and column.endswith(")"):
        code, label = column.split(" (", 1)
        return code, label[:-1]
    return column, ""


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


def classify_module(variable: str, label: str) -> str:
    v = variable.lower()
    text = f"{v} {label}"
    chronic = {
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
        "parkinson",
    }
    adl = {"dressa", "batha", "eata", "beda", "toilta", "urina", "adlab_c"}
    iadl = {"housewka", "mealsa", "shopa", "phonea", "medsa", "moneya", "iadl"}
    cognition = {"total_cognition", "imrc", "dlrc", "tr20", "orient", "ser7", "cog_status"}
    biomarkers = {
        "glu",
        "hba1c",
        "cho",
        "tg",
        "hdl",
        "ldl",
        "crp",
        "ua",
        "bun",
        "crea",
        "hct",
        "hgb",
        "mcv",
        "plt",
        "wbc",
    }

    if v in {"id", "householdid", "communityid"}:
        return "identifiers"
    if (
        v in {"wave", "iwy", "iwm", "rabyear", "rabmonth"}
        or "调查年份" in label
        or "调查月份" in label
    ):
        return "wave/time"
    if v in {"radyear", "radmonth", "iwstat"} or "死亡" in label:
        return "mortality/death timing"
    if v in {"age", "ragender", "nation", "marry", "hukou", "hrural", "raeduc_c", "raeducl"}:
        return "demographics"
    if v in {"city", "province"}:
        return "geographic variables"
    if (
        "17岁前" in label
        or "16岁前" in label
        or v.startswith("ra")
        and any(x in text for x in ["child", "mom", "dad", "educ"])
    ):
        return "early-life conditions"
    if v in chronic or v.endswith("_self") and any(x in v for x in chronic):
        return "chronic diseases"
    if v in adl or label.startswith("ADL/"):
        return "ADL"
    if v in iadl or label.startswith("IADL/"):
        return "IADL"
    if v in cognition or "认知" in label or "记忆" in label:
        return "cognition"
    if v == "cesd10" or "心理健康" in label or "抑郁" in label:
        return "mental health"
    if "睡眠" in label or v.startswith("sleep"):
        return "sleep"
    if "身体活动" in label or v.startswith(("vgact", "mdact", "ltact")):
        return "physical activity"
    if "社交" in label or v.startswith(("act_", "freq_act_", "social", "freq_social")):
        return "social activity"
    if v in {"work", "retire"} or "工作" in label or "退休" in label:
        return "work/retirement"
    if (
        "住院" in label
        or "门诊" in label
        or "医生" in label
        or "医疗" in label
        or "保险" in label
        or v in {"hospital", "hospital_time", "doctor_time"}
    ):
        return "medical utilization"
    if (
        v in {"bmi", "height", "weight", "systo", "diasto", "pulse"}
        or "身高" in label
        or "体重" in label
        or "血压" in label
    ):
        return "physical measures"
    if v in biomarkers or any(token in v for token in biomarkers) or "血" in label or "尿" in label:
        return "biomarkers"
    if (
        "收入" in label
        or "消费" in label
        or "资产" in label
        or "财务" in label
        or "养老金" in label
    ):
        return "household economy"
    if "住房" in label or "居住" in label or "环境" in label:
        return "housing/environment"
    if v.startswith(("frailty", "tyg")) or "指数" in label or "分组" in label or "状态" in label:
        return "derived scores"
    return "other"


def type_model(series: pd.Series, is_identifier: bool, is_time_variable: bool) -> str:
    if is_identifier:
        return "identifier"
    if is_time_variable:
        return "time"
    non_missing = series.dropna()
    unique_count = int(non_missing.nunique(dropna=True))
    if unique_count <= 2:
        return "binary_or_boolean"
    if pd.api.types.is_numeric_dtype(series):
        if unique_count <= 20:
            return "categorical_numeric"
        return "continuous"
    if unique_count <= 50:
        return "categorical_text"
    return "text_or_high_cardinality"


def leakage_class(module: str, is_identifier: bool, is_time_variable: bool, variable: str) -> str:
    if is_identifier:
        return "identifier"
    if is_time_variable:
        return "time_variable"
    if module == "mortality/death timing" or variable in {"radyear", "radmonth", "iwstat"}:
        return "future_information"
    if module in {
        "chronic diseases",
        "ADL",
        "IADL",
        "cognition",
        "mental health",
        "medical utilization",
        "derived scores",
    }:
        return "outcome_same_domain"
    return "safe_baseline_predictor"


def infer_outcome_candidate(module: str, variable: str) -> bool:
    if variable in {"age", "ragender", "wave", "ID"}:
        return False
    return module in {
        "chronic diseases",
        "ADL",
        "IADL",
        "cognition",
        "mental health",
        "medical utilization",
        "physical measures",
        "biomarkers",
        "mortality/death timing",
        "derived scores",
    }


def infer_predictor_candidate(module: str, leakage: str) -> bool:
    return leakage in {"safe_baseline_predictor", "outcome_same_domain"} and module not in {
        "identifiers",
        "wave/time",
    }


def fetch_index_metadata(
    base_url: str, source_id: str, variables: list[str]
) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for idx, variable in enumerate(variables, start=1):
        payload = {"q": variable, "source": source_id, "kind": "auto", "match": "any", "limit": 8}
        try:
            response = post_json(base_url, "/search", payload, timeout=30)
        except RuntimeError:
            continue
        rows = response.get("results") or []
        best: dict[str, Any] | None = None
        for row in rows:
            row_data = row.get("row") or {}
            code = row.get("code") or row_data.get("variable") or row_data.get("preferred_name")
            if code == variable:
                best = row
                break
        if best is None and rows:
            best = rows[0]
        if best:
            row_data = best.get("row") or {}
            metadata[variable] = {
                "index_kind": best.get("index_kind"),
                "index_file": best.get("index_file"),
                "mapping_status": row_data.get("mapping_status"),
                "type_index": row_data.get("type") or row_data.get("data_type"),
                "missing_count_index": row_data.get("missing_count"),
                "missing_rate_index": row_data.get("missing_rate"),
                "coverage_summary": row_data.get("coverage_summary"),
                "value_label_count": row_data.get("value_label_count"),
                "value_label_preview": row_data.get("value_label_preview"),
                "waves": best.get("waves"),
                "data_path": row_data.get("data_path")
                or row_data.get("data_paths")
                or best.get("parquet_rel"),
            }
        if idx % 100 == 0:
            print(
                f"Fetched variable index metadata for {idx}/{len(variables)} variables",
                file=sys.stderr,
            )
    return metadata


def fetch_optional_table(
    base_url: str, source_id: str, file_key: str, row_limit: int = 200000
) -> pd.DataFrame:
    try:
        payload = get_json(
            base_url,
            "/extract",
            {"source": source_id, "file": file_key, "limit": row_limit, "format": "json"},
            timeout=120,
        )
    except RuntimeError as exc:
        print(f"Warning: could not extract {file_key}: {exc}", file=sys.stderr)
        return pd.DataFrame()
    records = payload.get("records") or []
    return pd.DataFrame.from_records(records)


def build_wave_transition(
    df: pd.DataFrame, id_col: str, wave_col: str, expected_waves: list[int]
) -> pd.DataFrame:
    pairs = []
    grouped = df[[id_col, wave_col]].dropna().drop_duplicates().groupby(id_col)[wave_col].apply(set)
    for from_wave in expected_waves:
        for to_wave in expected_waves:
            count = int(sum(from_wave in waves and to_wave in waves for waves in grouped))
            pairs.append({"from_wave": from_wave, "to_wave": to_wave, "id_count": count})
    return pd.DataFrame(pairs)


def write_report(
    report_path: Path,
    base_url: str,
    source_id: str,
    main_file: str,
    health: dict[str, Any],
    resolve: dict[str, Any],
    schema: pd.DataFrame,
    panel_profile: pd.DataFrame,
    missing_by_variable: pd.DataFrame,
) -> None:
    profile = dict(zip(panel_profile["metric"], panel_profile["value"]))
    top_missing = missing_by_variable.sort_values("missing_rate", ascending=False).head(15)
    module_counts = schema["module"].value_counts().reset_index()
    lines = [
        "# CHARLS Data Audit Report",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## API Evidence",
        "",
        f"- Source ID: `{source_id}`",
        f"- API base: `{base_url}`",
        f"- Health endpoint: `GET /health`",
        f"- Manifest endpoint: `GET /manifest?source={source_id}`",
        f"- Resolve endpoint: `GET /resolve?source={source_id}&file={main_file}`",
        f"- Schema endpoint: `GET /schema?source={source_id}&file={main_file}`",
        f"- Extract endpoint: `GET /extract?source={source_id}&file={main_file}&limit={profile.get('extract_limit_used')}`",
        f"- Resolved exists: `{resolve.get('exists')}`",
        f"- Sidecar canonical URL: `{health.get('canonical_url')}`",
        "",
        "## Core Panel Facts",
        "",
        f"- Rows extracted: {profile.get('total_rows')}",
        f"- Columns: {profile.get('total_columns')}",
        f"- Unique respondent IDs: {profile.get('unique_ids')}",
        f"- Waves present: {profile.get('waves_present')}",
        f"- Duplicate ID-wave rows: {profile.get('duplicate_id_wave_rows')}",
        f"- Rows with missing ID: {profile.get('rows_missing_id')}",
        f"- Rows with missing wave: {profile.get('rows_missing_wave')}",
        "",
        "## Variable Modules",
        "",
        "| module | variable_count |",
        "|---|---:|",
    ]
    for _, row in module_counts.iterrows():
        lines.append(f"| {row['module']} | {row['count']} |")
    lines += [
        "",
        "## Highest Missingness Variables",
        "",
        "| variable | label | missing_rate | module |",
        "|---|---|---:|---|",
    ]
    for _, row in top_missing.iterrows():
        label = str(row.get("label_zh", "")).replace("|", "/")
        lines.append(
            f"| {row['variable']} | {label} | {row['missing_rate']:.3f} | {row['module']} |"
        )
    lines += [
        "",
        "## Next Step",
        "",
        "Use `outputs/02_outcome_registry/outcome_registry_draft.csv` as a draft candidate registry, then add literature evidence and internal validation checks before promoting any endpoint to primary or secondary status.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--api-base")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--row-limit", type=int)
    parser.add_argument("--skip-index-metadata", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    api_base = args.api_base or config["api"]["base_url"]
    source_id = config["source"]["id"]
    main_file = config["source"]["main_file"]
    id_col = config["source"]["id_column"]
    wave_col = config["source"]["wave_column"]
    expected_waves = [int(w) for w in config["source"]["expected_waves"]]
    output_dir = args.output_dir or PROJECT_ROOT / config["outputs"].get(
        "data_audit_directory", config["outputs"]["directory"]
    )
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using API base: {api_base}")
    health = get_json(api_base, "/health", timeout=20)
    manifest = get_json(api_base, "/manifest", {"source": source_id}, timeout=30)
    resolve = get_json(api_base, "/resolve", {"source": source_id, "file": main_file}, timeout=30)
    schema_payload = get_json(
        api_base, "/schema", {"source": source_id, "file": main_file}, timeout=30
    )
    if not resolve.get("exists"):
        raise RuntimeError(f"Resolved file does not exist for {source_id}:{main_file}")

    schema_rows = []
    for order, physical in enumerate(schema_payload["columns"], start=1):
        variable, label = parse_physical_column(physical)
        schema_rows.append(
            {
                "column_order": order,
                "variable": variable,
                "label_zh": label,
                "physical_column": physical,
            }
        )
    schema_df = pd.DataFrame(schema_rows)
    variables = schema_df["variable"].tolist()
    row_limit = args.row_limit or int(
        config["source"].get("full_row_limit") or schema_payload.get("metadata_num_rows") or 100000
    )
    extract = get_json(
        api_base,
        "/extract",
        {
            "source": source_id,
            "file": main_file,
            "columns": ",".join(variables),
            "limit": row_limit,
            "format": "json",
        },
        timeout=300,
    )
    records = extract.get("records") or []
    df = pd.DataFrame.from_records(records)
    if df.empty:
        raise RuntimeError("API extract returned no records")
    if wave_col in df:
        df[wave_col] = df[wave_col].map(normalize_wave)

    if int(extract.get("rows_returned", 0)) < int(schema_payload.get("metadata_num_rows") or 0):
        print(
            f"Warning: extracted {extract.get('rows_returned')} rows, schema metadata says {schema_payload.get('metadata_num_rows')}",
            file=sys.stderr,
        )

    fetch_index = (
        bool(config.get("audit", {}).get("fetch_variable_index_metadata", True))
        and not args.skip_index_metadata
    )
    index_metadata = fetch_index_metadata(api_base, source_id, variables) if fetch_index else {}

    enriched_rows = []
    total_rows = len(df)
    for row in schema_rows:
        variable = row["variable"]
        series = df[variable] if variable in df else pd.Series(dtype="object")
        missing_count = int(series.isna().sum())
        missing_rate = missing_count / total_rows if total_rows else 0.0
        non_missing = series.dropna()
        module = classify_module(variable, row["label_zh"])
        is_identifier = module == "identifiers" or variable == id_col
        is_time_variable = module == "wave/time" or variable == wave_col
        type_raw = str(series.dtype)
        model_type = type_model(series, is_identifier, is_time_variable)
        leakage = leakage_class(module, is_identifier, is_time_variable, variable)
        meta = index_metadata.get(variable, {})
        enriched_rows.append(
            {
                **row,
                "module": module,
                "type_raw": type_raw,
                "type_index": meta.get("type_index"),
                "type_model": model_type,
                "is_identifier": is_identifier,
                "is_time_variable": is_time_variable,
                "is_outcome_candidate": infer_outcome_candidate(module, variable),
                "is_predictor_candidate": infer_predictor_candidate(module, leakage),
                "leakage_risk_level": leakage,
                "missing_count_overall": missing_count,
                "missing_rate_overall": round(missing_rate, 6),
                "non_missing_count": int(non_missing.shape[0]),
                "unique_count": int(non_missing.nunique(dropna=True)),
                "value_label_count": meta.get("value_label_count"),
                "value_label_preview": meta.get("value_label_preview"),
                "waves_index": meta.get("waves"),
                "mapping_status": meta.get("mapping_status"),
                "data_path_index": meta.get("data_path"),
            }
        )
    schema_out = pd.DataFrame(enriched_rows)

    missing_by_variable = schema_out[
        [
            "variable",
            "label_zh",
            "module",
            "type_model",
            "missing_count_overall",
            "missing_rate_overall",
            "non_missing_count",
            "unique_count",
        ]
    ].rename(
        columns={"missing_count_overall": "missing_count", "missing_rate_overall": "missing_rate"}
    )

    missing_wave_rows = []
    if wave_col in df:
        for wave, wave_df in df.groupby(wave_col, dropna=False):
            wave_rows = len(wave_df)
            for variable in variables:
                series = wave_df[variable]
                missing_count = int(series.isna().sum())
                missing_wave_rows.append(
                    {
                        "wave": wave,
                        "variable": variable,
                        "label_zh": schema_out.loc[
                            schema_out["variable"] == variable, "label_zh"
                        ].iloc[0],
                        "module": schema_out.loc[schema_out["variable"] == variable, "module"].iloc[
                            0
                        ],
                        "rows": wave_rows,
                        "missing_count": missing_count,
                        "missing_rate": round(missing_count / wave_rows if wave_rows else 0.0, 6),
                    }
                )
    missing_by_wave = pd.DataFrame(missing_wave_rows)

    panel_metrics: list[dict[str, Any]] = [
        {"metric": "api_base", "value": api_base},
        {"metric": "source_id", "value": source_id},
        {"metric": "main_file", "value": main_file},
        {"metric": "resolved_exists", "value": resolve.get("exists")},
        {"metric": "resolved_format", "value": schema_payload.get("format")},
        {"metric": "metadata_num_rows", "value": schema_payload.get("metadata_num_rows")},
        {"metric": "metadata_num_columns", "value": schema_payload.get("num_columns")},
        {"metric": "extract_limit_used", "value": row_limit},
        {"metric": "rows_returned_by_extract", "value": extract.get("rows_returned")},
        {"metric": "total_rows", "value": total_rows},
        {"metric": "total_columns", "value": len(variables)},
    ]
    if id_col in df:
        panel_metrics.append(
            {"metric": "unique_ids", "value": int(df[id_col].nunique(dropna=True))}
        )
        panel_metrics.append({"metric": "rows_missing_id", "value": int(df[id_col].isna().sum())})
        rows_per_id = df.groupby(id_col, dropna=True).size()
        panel_metrics += [
            {"metric": "rows_per_id_min", "value": int(rows_per_id.min())},
            {"metric": "rows_per_id_median", "value": float(rows_per_id.median())},
            {"metric": "rows_per_id_max", "value": int(rows_per_id.max())},
        ]
    if wave_col in df:
        waves_present = sorted([int(w) for w in df[wave_col].dropna().unique()])
        panel_metrics.append(
            {"metric": "waves_present", "value": "|".join(map(str, waves_present))}
        )
        panel_metrics.append(
            {"metric": "rows_missing_wave", "value": int(df[wave_col].isna().sum())}
        )
        if id_col in df:
            duplicate_id_wave = int(df.duplicated([id_col, wave_col], keep=False).sum())
            panel_metrics.append({"metric": "duplicate_id_wave_rows", "value": duplicate_id_wave})
            waves_per_id = df[[id_col, wave_col]].dropna().drop_duplicates().groupby(id_col).size()
            panel_metrics += [
                {"metric": "waves_per_id_min", "value": int(waves_per_id.min())},
                {"metric": "waves_per_id_median", "value": float(waves_per_id.median())},
                {"metric": "waves_per_id_max", "value": int(waves_per_id.max())},
            ]
        for wave, wave_df in df.groupby(wave_col, dropna=False):
            panel_metrics.append({"metric": f"wave_{wave}_rows", "value": int(len(wave_df))})
            if id_col in wave_df:
                panel_metrics.append(
                    {
                        "metric": f"wave_{wave}_unique_ids",
                        "value": int(wave_df[id_col].nunique(dropna=True)),
                    }
                )
            if "age" in wave_df:
                panel_metrics.append(
                    {
                        "metric": f"wave_{wave}_age_mean",
                        "value": round(
                            float(pd.to_numeric(wave_df["age"], errors="coerce").mean()), 3
                        ),
                    }
                )
            if "ragender" in wave_df:
                panel_metrics.append(
                    {
                        "metric": f"wave_{wave}_female_rate_raw0",
                        "value": round(float((wave_df["ragender"] == 0).mean()), 6),
                    }
                )

    panel_profile = pd.DataFrame(panel_metrics)

    coverage = pd.DataFrame()
    transition = pd.DataFrame()
    if id_col in df and wave_col in df:
        unique_id_wave = df[[id_col, wave_col]].dropna().drop_duplicates()
        coverage = unique_id_wave.assign(present=1).pivot_table(
            index=id_col, columns=wave_col, values="present", fill_value=0
        )
        coverage = coverage.rename(
            columns={wave: f"wave_{wave}_present" for wave in coverage.columns}
        ).reset_index()
        wave_cols = [c for c in coverage.columns if str(c).startswith("wave_")]
        coverage["n_waves_observed"] = coverage[wave_cols].sum(axis=1).astype(int)
        observed = unique_id_wave.groupby(id_col)[wave_col].agg(["min", "max"]).reset_index()
        observed = observed.rename(columns={"min": "first_wave", "max": "last_wave"})
        coverage = coverage.merge(observed, on=id_col, how="left")
        transition = build_wave_transition(df, id_col, wave_col, expected_waves)

    schema_out.to_csv(output_dir / "charls_schema.csv", index=False)
    schema_out.to_csv(output_dir / "charls_variable_dictionary.csv", index=False)
    missing_by_variable.to_csv(output_dir / "charls_missingness_by_variable.csv", index=False)
    missing_by_wave.to_csv(output_dir / "charls_missingness_by_wave.csv", index=False)
    panel_profile.to_csv(output_dir / "charls_panel_profile.csv", index=False)
    if not coverage.empty:
        coverage.to_csv(output_dir / "charls_id_wave_coverage.csv", index=False)
    if not transition.empty:
        transition.to_csv(output_dir / "charls_wave_transition_matrix.csv", index=False)

    labels_table = fetch_optional_table(
        api_base, source_id, "metadata/02_reference_dictionary/labels.csv"
    )
    if not labels_table.empty:
        labels_table.to_csv(output_dir / "charls_label_dictionary.csv", index=False)
    value_labels_table = fetch_optional_table(
        api_base, source_id, "metadata/02_reference_dictionary/value_labels.csv"
    )
    if not value_labels_table.empty:
        value_labels_table.to_csv(output_dir / "charls_value_label_dictionary.csv", index=False)

    (output_dir / "charls_api_manifest_summary.json").write_text(
        json.dumps(
            {
                "health": health,
                "manifest_source": manifest.get("source"),
                "manifest_query_files": manifest.get("query_files", []),
                "resolve": resolve,
                "schema": {
                    k: schema_payload.get(k)
                    for k in [
                        "source",
                        "file_size_bytes",
                        "format",
                        "metadata_num_rows",
                        "num_columns",
                        "num_row_groups",
                    ]
                },
                "extract": {
                    k: extract.get(k)
                    for k in ["source", "file_size_bytes", "limit", "offset", "rows_returned"]
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    write_report(
        output_dir / "data_audit_report.md",
        api_base,
        source_id,
        main_file,
        health,
        resolve,
        schema_out,
        panel_profile,
        missing_by_variable,
    )
    print(f"Wrote audit outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
