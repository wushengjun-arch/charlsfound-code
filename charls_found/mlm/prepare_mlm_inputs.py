#!/usr/bin/env python3
"""Prepare split-aligned MLM inputs and outcome-specific trait masks."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_CONFIG = SCRIPT_PATH.parents[1] / "configs" / "mlm.yaml"


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


def unique_id_split(supervised: pd.DataFrame, id_column: str) -> pd.DataFrame:
    split_counts = supervised.groupby(id_column)["split"].nunique()
    leaked = split_counts[split_counts.gt(1)]
    if not leaked.empty:
        raise RuntimeError(f"Supervised split has {len(leaked)} IDs in multiple folds")
    return supervised[[id_column, "split"]].drop_duplicates().reset_index(drop=True)


def trait_lookup(metadata: pd.DataFrame) -> pd.DataFrame:
    traits = metadata.dropna(subset=["variable", "trait_token_id", "trait_position"]).copy()
    traits["trait_token_id"] = traits["trait_token_id"].astype(int)
    traits["trait_position"] = traits["trait_position"].astype(int)
    return traits[
        ["variable", "trait_token_id", "trait_position", "trait_token", "module"]
    ].drop_duplicates("variable")


def mask_entry(variables: list[str], traits: pd.DataFrame) -> dict[str, Any]:
    subset = traits[traits["variable"].isin(variables)].sort_values("trait_position")
    return {
        "variables": subset["variable"].tolist(),
        "trait_token_ids": subset["trait_token_id"].astype(int).tolist(),
        "trait_positions": subset["trait_position"].astype(int).tolist(),
    }


def build_outcome_masks(config: dict[str, Any], traits: pd.DataFrame) -> dict[str, Any]:
    supervised_manifest = load_json(resolve_path(config["inputs"]["supervised_matrix_manifest"]))
    leakage = pd.read_csv(resolve_path(config["inputs"]["leakage_matrix"]))
    tokenized_variables = set(traits["variable"])
    diagnostic_reasons = set(config["masking"]["diagnostic_exclusion_reasons"])
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
                ]
            )
            & tokenized_variables
        )
        strict_safe_mask_variables = sorted(tokenized_variables - safe_variables)
        outcome_masks[outcome_id] = {
            "diagnostic": mask_entry(diagnostic_variables, traits),
            "strict_safe": mask_entry(strict_safe_mask_variables, traits),
        }

    return {
        "metadata": {
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "diagnostic_exclusion_reasons": sorted(diagnostic_reasons),
            "strict_safe_definition": "mask every tokenized trait not present in the outcome-specific safe predictor list",
        },
        "outcomes": outcome_masks,
    }


def write_report(
    path: Path,
    dataset: pd.DataFrame,
    split_by_id: pd.DataFrame,
    masks: dict[str, Any],
    omitted_ids: int,
) -> None:
    row_counts = (
        dataset.groupby("split").agg(rows=("ID", "size"), ids=("ID", "nunique")).reset_index()
    )
    lines = [
        "# CHARLS MLM Input Report",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Split Alignment",
        "",
        f"- Person-wave rows retained: {len(dataset)}",
        f"- IDs retained: {dataset['ID'].nunique()}",
        f"- Tokenized IDs omitted because they are absent from the supervised rolling split: {omitted_ids}",
        "- Split source: `outputs/06_supervised/supervised_outcome_matrix.parquet`",
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
            "- `diagnostic`: masks direct target and diagnostic/treatment/count constructs from the leakage table.",
            "- `strict_safe`: masks every tokenized trait outside that outcome's safe predictor list.",
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
            "MLM pretraining may use all retained person-wave rows. Downstream disease-outcome embedding export should use at least `diagnostic` masking, and formal fair comparison can use `strict_safe` masking.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    config = load_yaml(args.config)
    tokenized = pd.read_parquet(resolve_path(config["inputs"]["tokenized_person_wave"]))
    supervised_manifest = load_json(resolve_path(config["inputs"]["supervised_matrix_manifest"]))
    id_column = supervised_manifest.get("metadata", {}).get("id_column", "ID")
    supervised = pd.read_parquet(
        resolve_path(config["inputs"]["supervised_outcome_matrix"]), columns=[id_column, "split"]
    )
    supervised = supervised.rename(columns={id_column: "ID"})
    metadata = pd.read_csv(resolve_path(config["inputs"]["tokenization_metadata"]))
    split_by_id = unique_id_split(supervised, "ID")

    retained = tokenized.merge(split_by_id, on="ID", how="inner", validate="many_to_one")
    omitted_ids = int(tokenized["ID"].nunique() - retained["ID"].nunique())
    retained = retained.sort_values(["split", "ID", "wave"], kind="stable").reset_index(drop=True)

    dataset_path = resolve_path(config["outputs"]["mlm_person_wave_dataset"])
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    retained.to_parquet(dataset_path, index=False)

    traits = trait_lookup(metadata)
    masks = build_outcome_masks(config, traits)
    masks_path = resolve_path(config["outputs"]["diagnostic_trait_masks"])
    masks_path.write_text(json.dumps(masks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    split_manifest = {
        "metadata": {
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "split_source": config["inputs"]["supervised_outcome_matrix"],
            "tokenized_source": config["inputs"]["tokenized_person_wave"],
        },
        "rows_by_split": retained.groupby("split").size().astype(int).to_dict(),
        "ids_by_split": retained.groupby("split")["ID"].nunique().astype(int).to_dict(),
        "omitted_tokenized_ids": omitted_ids,
        "sequence_length": int(retained["sequence_length"].max()),
        "value_vocab_size": int(metadata["value_token_id"].max() + 1),
        "trait_vocab_size": int(metadata["trait_token_id"].dropna().max() + 1),
        "module_vocab_size": int(metadata["module_token_id"].dropna().max() + 1),
        "wave_vocab_size": int(retained["wave_token_id"].max() + 1),
    }
    manifest_path = resolve_path(config["outputs"]["mlm_split_manifest"])
    manifest_path.write_text(
        json.dumps(split_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    write_report(
        resolve_path(config["outputs"]["mlm_input_report"]),
        retained,
        split_by_id,
        masks,
        omitted_ids,
    )
    print(
        json.dumps(
            {
                "rows": len(retained),
                "ids": int(retained["ID"].nunique()),
                "dataset": str(dataset_path.relative_to(PROJECT_ROOT)),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
