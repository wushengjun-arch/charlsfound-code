#!/usr/bin/env python3
"""Export person-wave embeddings from a trained CHARLS-found-lite checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from charls_found.mlm.data import PersonWaveMLMDataset, collate_batch  # noqa: E402
from charls_found.mlm.modeling import CharlsFoundLite, CharlsFoundLiteConfig  # noqa: E402

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


def select_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_directory(config: dict[str, Any], run_name: str) -> Path:
    return resolve_path(config["outputs"]["mlm_directory"]) / "runs" / run_name


def mask_positions(config: dict[str, Any], outcome_id: str | None, mask_mode: str) -> list[int]:
    if mask_mode == "none":
        return []
    if not outcome_id:
        raise ValueError("--outcome-id is required when --mask-mode is not none")
    masks = load_json(resolve_path(config["outputs"]["diagnostic_trait_masks"]))
    try:
        return [int(value) for value in masks["outcomes"][outcome_id][mask_mode]["trait_positions"]]
    except KeyError as exc:
        raise ValueError(
            f"Unknown outcome/mask mode combination: {outcome_id}/{mask_mode}"
        ) from exc


def write_embedding_output(
    records: list[dict[str, Any]],
    dataset: pd.DataFrame,
    run_dir: Path,
    checkpoint_path: Path,
    mask_mode: str,
    outcome_id: str | None,
    splits: list[str],
    max_rows: int | None,
) -> Path:
    out_name = f"person_wave_embeddings_{mask_mode}"
    if outcome_id:
        out_name += f"_{outcome_id}"
    if max_rows is not None:
        out_name += "_sample"
    output_path = run_dir / f"{out_name}.parquet"
    output = pd.DataFrame(records)
    split_cols = dataset[["ID", "wave", "split"]].drop_duplicates(["ID", "wave"])
    output = output.merge(split_cols, on=["ID", "wave"], how="left", validate="one_to_one")
    output["embedding_mask_mode"] = mask_mode
    output["embedding_mask_outcome_id"] = outcome_id or ""
    output["checkpoint"] = str(checkpoint_path.relative_to(PROJECT_ROOT))
    output.to_parquet(output_path, index=False)

    report_path = run_dir / f"{out_name}_report.md"
    report_path.write_text(
        "\n".join(
            [
                "# CHARLS Person-Wave Embedding Export Report",
                "",
                f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                f"- Rows: {len(output)}",
                f"- Splits: {', '.join(splits)}",
                f"- Mask mode: `{mask_mode}`",
                f"- Outcome ID: `{outcome_id or ''}`",
                f"- Output: `{output_path.relative_to(PROJECT_ROOT)}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return output_path


def batch_raw_id(batch: dict[str, Any], index: int) -> Any:
    if "raw_ID" in batch:
        value = batch["raw_ID"][index]
        if isinstance(value, np.generic):
            return value.item()
        return value
    return int(batch["ID"][index].item())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-name")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--outcome-id")
    parser.add_argument("--outcome-ids", nargs="+")
    parser.add_argument(
        "--mask-mode", choices=["none", "diagnostic", "strict_safe"], default="none"
    )
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    config = load_yaml(args.config)
    run_name = args.run_name or config["training"]["run_name"]
    run_dir = run_directory(config, run_name)
    checkpoint_path = (
        resolve_path(args.checkpoint) if args.checkpoint else (run_dir / "checkpoint_best.pt")
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_cfg = CharlsFoundLiteConfig(**checkpoint["model_config"])
    device = select_device(args.device)
    model = CharlsFoundLite(model_cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    dataset = pd.read_parquet(resolve_path(config["outputs"]["mlm_person_wave_dataset"]))
    dataset = dataset[dataset["split"].isin(args.splits)].copy()
    if args.max_rows is not None and len(dataset) > args.max_rows:
        dataset = dataset.sample(
            n=args.max_rows, random_state=int(config["training"]["seed"])
        ).sort_values(["ID", "wave"])
    dataset = dataset.reset_index(drop=True)
    target_outcomes = args.outcome_ids or ([args.outcome_id] if args.outcome_id else [])
    if args.mask_mode == "none" and target_outcomes:
        raise ValueError("--outcome-id/--outcome-ids cannot be used with --mask-mode none")
    if args.mask_mode != "none" and not target_outcomes:
        raise ValueError("--outcome-id or --outcome-ids is required when --mask-mode is not none")
    positions_by_outcome = {
        outcome_id: mask_positions(config, outcome_id, args.mask_mode)
        for outcome_id in target_outcomes
    }
    positions = [] if args.mask_mode == "none" else positions_by_outcome[target_outcomes[0]]
    loader = DataLoader(
        PersonWaveMLMDataset(dataset),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )
    mask_token_id = int(checkpoint["mask_token_id"])

    records: list[dict[str, Any]] = []
    records_by_outcome: dict[str, list[dict[str, Any]]] = {
        outcome_id: [] for outcome_id in target_outcomes
    }
    with torch.no_grad():
        for batch in loader:
            value_ids = batch["value_ids"].to(device)
            if records_by_outcome:
                value_batches = []
                ordered_outcomes = list(records_by_outcome)
                for outcome_id in ordered_outcomes:
                    masked = value_ids.clone()
                    valid_positions = [
                        position
                        for position in positions_by_outcome[outcome_id]
                        if position < value_ids.shape[1]
                    ]
                    if valid_positions:
                        masked[:, valid_positions] = mask_token_id
                    value_batches.append(masked)
                stacked_value_ids = torch.cat(value_batches, dim=0)
                repeat_count = len(ordered_outcomes)
                cls_hidden, token_hidden = model.encode(
                    stacked_value_ids,
                    batch["trait_ids"].to(device).repeat(repeat_count, 1),
                    batch["module_ids"].to(device).repeat(repeat_count, 1),
                    batch["wave_token_id"].to(device).repeat(repeat_count),
                )
                cls_array = cls_hidden.detach().cpu().numpy().astype("float32")
                mean_array = token_hidden.detach().mean(dim=1).cpu().numpy().astype("float32")
                batch_size = int(value_ids.shape[0])
                for outcome_index, outcome_id in enumerate(ordered_outcomes):
                    offset = outcome_index * batch_size
                    for i in range(batch_size):
                        records_by_outcome[outcome_id].append(
                            {
                                "ID": batch_raw_id(batch, i),
                                "wave": int(batch["wave"][i].item()),
                                "wave_token_id": int(batch["wave_token_id"][i].item()),
                                "cls_embedding": cls_array[offset + i].tolist(),
                                "mean_embedding": mean_array[offset + i].tolist(),
                            }
                        )
                continue

            if positions:
                valid_positions = [
                    position for position in positions if position < value_ids.shape[1]
                ]
                if valid_positions:
                    value_ids[:, valid_positions] = mask_token_id
            cls_hidden, token_hidden = model.encode(
                value_ids,
                batch["trait_ids"].to(device),
                batch["module_ids"].to(device),
                batch["wave_token_id"].to(device),
            )
            cls_array = cls_hidden.detach().cpu().numpy().astype("float32")
            mean_array = token_hidden.detach().mean(dim=1).cpu().numpy().astype("float32")
            for i in range(cls_array.shape[0]):
                records.append(
                    {
                        "ID": batch_raw_id(batch, i),
                        "wave": int(batch["wave"][i].item()),
                        "wave_token_id": int(batch["wave_token_id"][i].item()),
                        "cls_embedding": cls_array[i].tolist(),
                        "mean_embedding": mean_array[i].tolist(),
                    }
                )

    if records_by_outcome:
        outputs = {}
        for outcome_id, outcome_records in records_by_outcome.items():
            output_path = write_embedding_output(
                records=outcome_records,
                dataset=dataset,
                run_dir=run_dir,
                checkpoint_path=checkpoint_path,
                mask_mode=args.mask_mode,
                outcome_id=outcome_id,
                splits=args.splits,
                max_rows=args.max_rows,
            )
            outputs[outcome_id] = str(output_path.relative_to(PROJECT_ROOT))
        print(
            json.dumps(
                {
                    "rows_per_outcome": {
                        key: len(value) for key, value in records_by_outcome.items()
                    },
                    "outputs": outputs,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    output_path = write_embedding_output(
        records=records,
        dataset=dataset,
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        mask_mode=args.mask_mode,
        outcome_id=args.outcome_id,
        splits=args.splits,
        max_rows=args.max_rows,
    )
    print(
        json.dumps(
            {"rows": len(records), "output": str(output_path.relative_to(PROJECT_ROOT))},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
