#!/usr/bin/env python3
"""Export KLoSA unmasked and diagnostic-masked person-wave embeddings."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_CONFIG = SCRIPT_PATH.parents[1] / "configs" / "klosa_mlm.yaml"
EXPORTER = PROJECT_ROOT / "charls_found" / "mlm" / "export_person_wave_embeddings.py"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def disease_outcomes(config: dict[str, Any]) -> list[str]:
    registry = pd.read_csv(resolve_path(config["inputs"]["outcome_registry"]))
    disease_domains = set(config["comparison"]["disease_domains"])
    selected = json.loads(
        resolve_path(config["inputs"]["supervised_matrix_manifest"]).read_text(encoding="utf-8")
    )["selected_outcomes"]
    selected_set = set(selected)
    rows = registry[
        registry["outcome_id"].isin(selected_set) & registry["domain"].isin(disease_domains)
    ]
    return [outcome for outcome in selected if outcome in set(rows["outcome_id"])]


def run_export(
    config_path: Path,
    run_name: str | None,
    batch_size: int,
    device: str,
    splits: list[str],
    mask_mode: str,
    outcome_id: str | None = None,
) -> None:
    command = [
        sys.executable,
        str(EXPORTER),
        "--config",
        str(config_path),
        "--batch-size",
        str(batch_size),
        "--device",
        device,
        "--mask-mode",
        mask_mode,
        "--splits",
        *splits,
    ]
    if run_name:
        command.extend(["--run-name", run_name])
    if outcome_id:
        command.extend(["--outcome-id", outcome_id])
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-name")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument(
        "--only-outcome",
        action="append",
        help="Limit diagnostic exports to one or more outcome IDs.",
    )
    parser.add_argument("--skip-none", action="store_true")
    args = parser.parse_args()

    config = load_yaml(args.config)
    outcomes = disease_outcomes(config)
    if args.only_outcome:
        requested = set(args.only_outcome)
        outcomes = [outcome for outcome in outcomes if outcome in requested]

    if not args.skip_none:
        run_export(args.config, args.run_name, args.batch_size, args.device, args.splits, "none")
    for outcome_id in outcomes:
        run_export(
            args.config,
            args.run_name,
            args.batch_size,
            args.device,
            args.splits,
            "diagnostic",
            outcome_id,
        )

    print(json.dumps({"diagnostic_outcomes": outcomes}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
