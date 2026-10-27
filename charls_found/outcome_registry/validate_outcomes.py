#!/usr/bin/env python3
"""Attach literature evidence and internal consistency checks to draft outcomes."""

from __future__ import annotations

import argparse
import json
import math
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_DATA_CONFIG = SCRIPT_PATH.parents[1] / "configs" / "data.yaml"
DEFAULT_LIT_CONFIG = SCRIPT_PATH.parents[1] / "configs" / "outcome_literature.yaml"


CHRONIC_VARS = {
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
}


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
        raise RuntimeError("Outcome validation extract returned no records")
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
    if not variables:
        return pd.Series(0, index=df.index)
    total = pd.Series(0, index=df.index, dtype="int64")
    for variable in variables:
        total = total + yes_mask(df[variable]).astype(int)
    return total


def count_observed(df: pd.DataFrame, variables: list[str]) -> pd.Series:
    if not variables:
        return pd.Series(0, index=df.index, dtype="int64")
    total = pd.Series(0, index=df.index, dtype="int64")
    for variable in variables:
        total = total + df[variable].notna().astype(int)
    return total


def infer_low_cognition_threshold(registry: pd.DataFrame) -> float | None:
    row = registry.loc[registry["outcome_id"] == "low_cognition_onset_data_driven"]
    if row.empty:
        return None
    text = str(row.iloc[0].get("event_definition", ""))
    marker = "at or below threshold."
    if "(" in text and ")" in text:
        try:
            return float(text.split("(")[-1].split(")")[0])
        except ValueError:
            return None
    if marker in text:
        return None
    return None


def binary_event_summary(
    df: pd.DataFrame,
    outcome_id: str,
    variables: list[str],
    id_col: str,
    wave_col: str,
    baseline_wave: int,
    followup_waves: list[int],
    low_cog_threshold: float | None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    baseline = (
        df[df[wave_col] == baseline_wave][[id_col, "age", "ragender"] + variables]
        .drop_duplicates(subset=[id_col])
        .copy()
    )
    followup = df[df[wave_col].isin(followup_waves)][[id_col, wave_col] + variables].copy()

    if outcome_id == "multimorbidity_onset":
        base_positive_count = count_positive(baseline, variables)
        base_observed_count = count_observed(baseline, variables)
        baseline["baseline_status"] = "baseline_negative"
        baseline.loc[base_observed_count < 3, "baseline_status"] = "baseline_missing"
        baseline.loc[(base_observed_count >= 3) & (base_positive_count >= 2), "baseline_status"] = (
            "baseline_positive"
        )
        follow_observed = count_observed(followup, variables) >= 3
        follow_event = (count_positive(followup, variables) >= 2) & follow_observed
    elif outcome_id == "high_depressive_symptoms_onset":
        value = to_numeric(baseline[variables[0]])
        baseline["baseline_status"] = "baseline_negative"
        baseline.loc[value.isna(), "baseline_status"] = "baseline_missing"
        baseline.loc[value >= 10, "baseline_status"] = "baseline_positive"
        follow_value = to_numeric(followup[variables[0]])
        follow_observed = follow_value.notna()
        follow_event = follow_value >= 10
    elif outcome_id == "low_cognition_onset_data_driven" and low_cog_threshold is not None:
        value = to_numeric(baseline[variables[0]])
        baseline["baseline_status"] = "baseline_negative"
        baseline.loc[value.isna(), "baseline_status"] = "baseline_missing"
        baseline.loc[value <= low_cog_threshold, "baseline_status"] = "baseline_positive"
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

    eligible_ids = set(baseline.loc[baseline["baseline_status"] == "baseline_negative", id_col])
    follow = followup[followup[id_col].isin(eligible_ids)].copy()
    observed_ids = set(follow.loc[follow_observed.reindex(follow.index, fill_value=False), id_col])
    event_rows = follow.loc[follow_event.reindex(follow.index, fill_value=False)]
    first_event = event_rows.groupby(id_col)[wave_col].min().rename("first_event_wave")

    person = baseline[[id_col, "age", "ragender", "baseline_status"]].copy()
    person["has_followup_observed"] = person[id_col].isin(observed_ids)
    person = person.merge(first_event, on=id_col, how="left")
    person["event"] = person["first_event_wave"].notna()

    wave_counts = event_rows.groupby(wave_col)[id_col].nunique().to_dict()
    return person, {int(k): int(v) for k, v in wave_counts.items()}


def continuous_summary(
    df: pd.DataFrame,
    variable: str,
    id_col: str,
    wave_col: str,
    baseline_wave: int,
    followup_waves: list[int],
) -> pd.DataFrame:
    baseline = (
        df[df[wave_col] == baseline_wave][[id_col, "age", "ragender", variable]]
        .drop_duplicates(subset=[id_col])
        .copy()
    )
    baseline = baseline.rename(columns={variable: "baseline_value"})
    followup = df[df[wave_col].isin(followup_waves)][[id_col, wave_col, variable]].copy()
    followup = followup[to_numeric(followup[variable]).notna()]
    last_idx = followup.sort_values(wave_col).groupby(id_col).tail(1).index
    last = followup.loc[last_idx, [id_col, wave_col, variable]].rename(
        columns={wave_col: "last_wave", variable: "last_value"}
    )
    person = baseline.merge(last, on=id_col, how="left")
    person["baseline_status"] = "baseline_observed"
    person.loc[to_numeric(person["baseline_value"]).isna(), "baseline_status"] = "baseline_missing"
    person["has_followup_observed"] = person["last_value"].notna()
    person["change"] = to_numeric(person["last_value"]) - to_numeric(person["baseline_value"])
    return person


def age_group(value: Any) -> str:
    if pd.isna(value):
        return "missing"
    try:
        age = float(value)
    except (TypeError, ValueError):
        return "missing"
    if age < 55:
        return "45_54"
    if age < 65:
        return "55_64"
    if age < 75:
        return "65_74"
    return "75_plus"


def sex_label(value: Any) -> str:
    if pd.isna(value):
        return "missing"
    try:
        numeric = int(float(value))
    except (TypeError, ValueError):
        return "other"
    if numeric == 0:
        return "female"
    if numeric == 1:
        return "male"
    return "other"


def trend_label(rates: dict[str, float | None]) -> str:
    ordered = [rates.get("45_54"), rates.get("55_64"), rates.get("65_74"), rates.get("75_plus")]
    values = [v for v in ordered if v is not None]
    if len(values) < 2:
        return "insufficient"
    if values[-1] > values[0] * 1.1:
        return "increasing"
    if values[-1] < values[0] * 0.9:
        return "decreasing"
    return "flat"


def validation_status(
    row: pd.Series, internal: dict[str, Any], literature_level: str
) -> tuple[str, str, str, bool]:
    if row["outcome_id"] == "death_by_followup_sensitivity":
        return "not_usable_outcomes", "D", "timing_semantics_review_required", False
    if row["outcome_id"] == "low_cognition_onset_data_driven":
        return "exploratory_outcomes", "C", "threshold_review_required", True
    definition_type = str(row["definition_type"])
    n_followup = int(internal.get("n_followup_observed") or 0)
    n_events = internal.get("n_events")
    if "continuous" in definition_type:
        if n_followup >= 1000 and literature_level in {"strong", "moderate"}:
            return "secondary_outcomes", "B", "validated_for_continuous_change", True
        return "exploratory_outcomes", "C", "continuous_change_review_needed", True
    if n_events is None:
        return "exploratory_outcomes", "C", "event_logic_missing", False
    event_rate = internal.get("event_rate")
    if (
        n_events >= 500
        and event_rate is not None
        and 0.01 <= event_rate <= 0.8
        and literature_level in {"strong", "moderate"}
    ):
        return "secondary_outcomes", "B", "validated_secondary_ready", True
    if n_events >= 100:
        return "exploratory_outcomes", "C", "usable_but_sparse_or_review", True
    return "not_usable_outcomes", "D", "too_sparse_or_unstable", False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--literature-config", type=Path, default=DEFAULT_LIT_CONFIG)
    parser.add_argument("--api-base")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--row-limit", type=int)
    args = parser.parse_args()

    data_config = load_yaml(args.config)
    literature_config = load_yaml(args.literature_config)
    api_base = args.api_base or data_config["api"]["base_url"]
    source_id = data_config["source"]["id"]
    main_file = data_config["source"]["main_file"]
    id_col = data_config["source"]["id_column"]
    wave_col = data_config["source"]["wave_column"]
    row_limit = args.row_limit or int(data_config["source"].get("full_row_limit", 100000))
    baseline_wave = int(data_config["outcomes"]["baseline_wave"])
    followup_waves = [int(w) for w in data_config["outcomes"]["followup_waves"]]

    output_dir = args.output_dir or PROJECT_ROOT / data_config["outputs"].get(
        "outcome_registry_directory", data_config["outputs"]["directory"]
    )
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_dir = PROJECT_ROOT / data_config["outputs"].get(
        "data_audit_directory", data_config["outputs"]["directory"]
    )
    if not audit_dir.is_absolute():
        audit_dir = PROJECT_ROOT / audit_dir
    registry_path = output_dir / "outcome_registry_draft.csv"
    schema_path = audit_dir / "charls_schema.csv"
    if not registry_path.exists():
        raise RuntimeError(f"Missing {registry_path}; run build_candidate_outcomes.py first")
    if not schema_path.exists():
        raise RuntimeError(f"Missing {schema_path}; run data audit first")

    registry = pd.read_csv(registry_path)
    schema = pd.read_csv(schema_path)
    all_source_vars = sorted(
        {var for value in registry["source_variables"] for var in split_vars(value)}
    )
    helper_cols = [id_col, wave_col, "age", "ragender"]
    columns = [c for c in helper_cols + all_source_vars if c in set(schema["variable"])]
    df = extract_columns(api_base, source_id, main_file, columns, row_limit)
    df[wave_col] = df[wave_col].map(normalize_wave)
    low_cog_threshold = infer_low_cognition_threshold(registry)

    evidence_rows: list[dict[str, Any]] = []
    internal_rows: list[dict[str, Any]] = []
    validated = registry.copy()

    references = literature_config["references"]
    domain_rules = literature_config["domain_rules"]

    for idx, row in registry.iterrows():
        outcome_id = row["outcome_id"]
        domain = row["domain"]
        variables = [var for var in split_vars(row["source_variables"]) if var in df]
        rule = domain_rules.get(domain, {})
        ref_ids = rule.get("references", [])
        literature_level = rule.get("literature_support_level", "pending")
        ref_labels = []
        ref_urls = []
        for ref_id in ref_ids:
            ref = references[ref_id]
            ref_labels.append(ref["citation"])
            ref_urls.append(ref["url"])
            evidence_rows.append(
                {
                    "outcome_id": outcome_id,
                    "domain": domain,
                    "reference_id": ref_id,
                    "citation": ref["citation"],
                    "url": ref["url"],
                    "supports": ref["supports"],
                }
            )

        definition_type = str(row["definition_type"])
        if "continuous" in definition_type and variables:
            person = continuous_summary(
                df, variables[0], id_col, wave_col, baseline_wave, followup_waves
            )
            n_baseline_observed = int((person["baseline_status"] == "baseline_observed").sum())
            n_followup = int(
                (
                    person["has_followup_observed"]
                    & (person["baseline_status"] == "baseline_observed")
                ).sum()
            )
            change = pd.to_numeric(
                person.loc[person["has_followup_observed"], "change"], errors="coerce"
            ).dropna()
            internal = {
                "outcome_id": outcome_id,
                "definition_type": definition_type,
                "n_baseline_observed": n_baseline_observed,
                "n_followup_observed": n_followup,
                "n_events": None,
                "event_rate": None,
                "mean_change": round(float(change.mean()), 6) if not change.empty else None,
                "median_change": round(float(change.median()), 6) if not change.empty else None,
                "age_gradient": "not_applicable_continuous",
                "event_wave_counts": "{}",
            }
        else:
            person, wave_counts = binary_event_summary(
                df,
                outcome_id,
                variables,
                id_col,
                wave_col,
                baseline_wave,
                followup_waves,
                low_cog_threshold,
            )
            analysis = person[
                (person["baseline_status"] == "baseline_negative") & person["has_followup_observed"]
            ].copy()
            analysis["age_group"] = analysis["age"].map(age_group)
            analysis["sex"] = analysis["ragender"].map(sex_label)
            age_rates: dict[str, float | None] = {}
            for group in ["45_54", "55_64", "65_74", "75_plus"]:
                group_df = analysis[analysis["age_group"] == group]
                age_rates[group] = (
                    round(float(group_df["event"].mean()), 6) if not group_df.empty else None
                )
            sex_rates: dict[str, float | None] = {}
            for group in ["female", "male"]:
                group_df = analysis[analysis["sex"] == group]
                sex_rates[group] = (
                    round(float(group_df["event"].mean()), 6) if not group_df.empty else None
                )
            n_followup = int(analysis.shape[0])
            n_events = int(analysis["event"].sum())
            internal = {
                "outcome_id": outcome_id,
                "definition_type": definition_type,
                "n_baseline_observed": int((person["baseline_status"] != "baseline_missing").sum()),
                "n_baseline_negative": int(
                    (person["baseline_status"] == "baseline_negative").sum()
                ),
                "n_baseline_positive": int(
                    (person["baseline_status"] == "baseline_positive").sum()
                ),
                "n_followup_observed": n_followup,
                "n_events": n_events,
                "event_rate": round(n_events / n_followup, 6) if n_followup else None,
                "age_rate_45_54": age_rates["45_54"],
                "age_rate_55_64": age_rates["55_64"],
                "age_rate_65_74": age_rates["65_74"],
                "age_rate_75_plus": age_rates["75_plus"],
                "age_gradient": trend_label(age_rates),
                "female_event_rate": sex_rates["female"],
                "male_event_rate": sex_rates["male"],
                "event_wave_counts": json.dumps(wave_counts, ensure_ascii=False, sort_keys=True),
            }

        registry_tier, evidence_tier, status, usable = validation_status(
            row, internal, literature_level
        )
        internal["registry_tier_validated"] = registry_tier
        internal["evidence_tier_validated"] = evidence_tier
        internal["internal_validation_status_validated"] = status
        internal["usable_validated"] = usable
        internal_rows.append(internal)

        validated.loc[idx, "definition_reference"] = rule.get(
            "definition_reference", row.get("definition_reference")
        )
        validated.loc[idx, "supporting_literature"] = "; ".join(ref_labels)
        validated.loc[idx, "literature_support_level"] = literature_level
        validated.loc[idx, "literature_reference_urls"] = "; ".join(ref_urls)
        validated.loc[idx, "n_followup"] = internal.get("n_followup_observed")
        validated.loc[idx, "n_events"] = internal.get("n_events")
        validated.loc[idx, "event_rate"] = internal.get("event_rate")
        validated.loc[idx, "registry_tier"] = registry_tier
        validated.loc[idx, "evidence_tier"] = evidence_tier
        validated.loc[idx, "internal_validation_status"] = status
        validated.loc[idx, "usable"] = usable
        validated.loc[idx, "internal_validation_checks"] = (
            "sample size, follow-up observed, event count/rate, age gradient, sex rates, wave event distribution"
        )
        validated.loc[idx, "notes"] = (
            "Validated draft from API-backed CHARLS data plus literature evidence; still requires final manuscript-level citation review."
        )

    evidence = pd.DataFrame(evidence_rows).drop_duplicates()
    internal_df = pd.DataFrame(internal_rows)
    validated.to_csv(output_dir / "outcome_registry_validated.csv", index=False)
    evidence.to_csv(output_dir / "outcome_literature_evidence.csv", index=False)
    internal_df.to_csv(output_dir / "outcome_internal_validation.csv", index=False)

    lines = [
        "# Outcome Validation Report",
        "",
        f"Source ID: `{source_id}`",
        f"API endpoint used: `GET /extract?source={source_id}&file={main_file}&columns=<outcome variables + age + sex>&limit={row_limit}&format=json`",
        f"Baseline wave: {baseline_wave}",
        f"Follow-up waves: {'|'.join(map(str, followup_waves))}",
        "",
        "## Validated Registry Counts",
        "",
        validated.groupby(["evidence_tier", "registry_tier", "usable"]).size().to_string(),
        "",
        "## Literature Evidence Files",
        "",
        "- `outputs/02_outcome_registry/outcome_literature_evidence.csv` stores reference-to-outcome mappings.",
        "- `outputs/02_outcome_registry/outcome_internal_validation.csv` stores event counts, age/sex rates, wave event counts, and continuous-change checks.",
        "",
        "## Timing Warning",
        "",
        "Mortality remains not usable for modeling until `iwstat`, `radyear`, and `radmonth` timing semantics are resolved.",
        "",
    ]
    (output_dir / "outcome_validation_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote validated registry and validation outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
