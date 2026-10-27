#!/usr/bin/env python3
"""Stability checks for the KLoSA multimorbidity embedding-vs-tree delta."""

from __future__ import annotations

import importlib.util
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_CONFIG = SCRIPT_PATH.parents[1] / "configs" / "klosa_mlm.yaml"


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


STABILITY = load_module(
    "klosa_stability_shared",
    PROJECT_ROOT / "charls_found" / "external_validation" / "hrs_multimorbidity_stability.py",
)
BASELINE = load_module(
    "klosa_stability_baseline",
    PROJECT_ROOT / "charls_found" / "external_validation" / "klosa_train_baselines.py",
)
KLOSA_COMPARE = load_module(
    "klosa_stability_compare",
    PROJECT_ROOT / "charls_found" / "external_validation" / "klosa_embedding_vs_tree.py",
)


def write_report(
    path: Path,
    original: dict[str, Any],
    bootstrap: pd.DataFrame,
    repeated: pd.DataFrame,
    threshold: float,
) -> None:
    bootstrap_delta = bootstrap["delta_auc"].to_numpy(dtype=float)
    repeated_delta = (
        repeated["delta_auc"].to_numpy(dtype=float)
        if not repeated.empty
        else np.asarray([], dtype=float)
    )
    bootstrap_summary = STABILITY.summarize(bootstrap_delta)
    repeated_summary = STABILITY.summarize(repeated_delta)
    lines = [
        "# KLoSA Multimorbidity Stability Check",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Original Split",
        "",
        f"- Tree: `{original['tree_model']}`",
        f"- Embedding: `{original['embedding_model']}` / `{original['embedding_pooling']}`",
        f"- Test per-origin AUC delta: {original['delta_auc']:.4f}",
        f"- Test C-index delta: {original['delta_c_index']:.4f}",
        "",
        "## ID-Cluster Bootstrap On Original Test Split",
        "",
        f"- Iterations: {len(bootstrap)}",
        f"- Mean delta AUC: {bootstrap_summary['mean']:.4f}",
        f"- Median delta AUC: {bootstrap_summary['median']:.4f}",
        f"- 95% percentile interval: [{bootstrap_summary['ci_low']:.4f}, {bootstrap_summary['ci_high']:.4f}]",
        f"- P(delta > 0): {float(np.mean(bootstrap_delta > 0)):.3f}",
        f"- P(delta > {threshold:.3f}): {float(np.mean(bootstrap_delta > threshold)):.3f}",
        "",
        "## Repeated ID-Level Splits",
        "",
        f"- Completed repeats: {len(repeated)}",
        f"- Mean delta AUC: {repeated_summary['mean']:.4f}",
        f"- Median delta AUC: {repeated_summary['median']:.4f}",
        f"- 95% empirical interval: [{repeated_summary['ci_low']:.4f}, {repeated_summary['ci_high']:.4f}]",
        f"- Share delta > 0: {float(np.mean(repeated_delta > 0)) if repeated_delta.size else math.nan:.3f}",
        f"- Share delta > {threshold:.3f}: {float(np.mean(repeated_delta > threshold)) if repeated_delta.size else math.nan:.3f}",
        "",
    ]
    if not repeated.empty:
        lines.extend(
            [
                "## Selected Models Across Repeats",
                "",
                "| selected_tree_model | count |",
                "|---|---:|",
            ]
        )
        for model_name, count in (
            repeated["selected_tree_model"].value_counts().sort_index().items()
        ):
            lines.append(f"| {model_name} | {int(count)} |")
        lines.extend(["", "| selected_embedding | count |", "|---|---:|"])
        selected_embedding = (
            repeated["selected_embedding_model"] + "/" + repeated["selected_embedding_pooling"]
        )
        for model_name, count in selected_embedding.value_counts().sort_index().items():
            lines.append(f"| {model_name} | {int(count)} |")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    STABILITY.DEFAULT_CONFIG = DEFAULT_CONFIG
    STABILITY.BASELINE = BASELINE
    STABILITY.HRS_COMPARE = KLOSA_COMPARE
    STABILITY.SHARED_COMPARE = KLOSA_COMPARE.COMPARE
    STABILITY.write_report = write_report
    return STABILITY.main()


if __name__ == "__main__":
    raise SystemExit(main())
