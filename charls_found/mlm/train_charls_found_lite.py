#!/usr/bin/env python3
"""Train CHARLS-found-lite masked value model."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
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


def run_directory(config: dict[str, Any], run_name: str) -> Path:
    return resolve_path(config["outputs"]["mlm_directory"]) / "runs" / run_name


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def value_token_id(metadata: pd.DataFrame, token: str) -> int:
    row = metadata[metadata["value_token"].eq(token)]
    if row.empty:
        raise RuntimeError(f"Value token not found: {token}")
    return int(row.iloc[0]["value_token_id"])


def sample_masked_values(
    value_ids: torch.Tensor,
    maskable: torch.Tensor,
    mask_token_id: int,
    mask_probability: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mask = (torch.rand(value_ids.shape, device=value_ids.device) < mask_probability) & maskable
    rows_without_mask = torch.where(mask.sum(dim=1).eq(0) & maskable.any(dim=1))[0]
    for row in rows_without_mask.tolist():
        first_maskable = torch.where(maskable[row])[0][0]
        mask[row, first_maskable] = True
    inputs = value_ids.clone()
    targets = torch.full_like(value_ids, fill_value=-100)
    targets[mask] = value_ids[mask]
    inputs[mask] = mask_token_id
    return inputs, targets, mask


def batch_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    masked_total = int(targets.numel())
    if masked_total == 0:
        return {"masked_tokens": 0, "top1": math.nan, "top5": math.nan}
    predictions = logits.argmax(dim=-1)
    top1 = int((predictions == targets).sum().item())
    topk = logits.topk(k=min(5, logits.shape[-1]), dim=-1).indices
    top5 = int((topk == targets.unsqueeze(-1)).any(dim=-1).sum().item())
    return {"masked_tokens": masked_total, "top1": top1 / masked_total, "top5": top5 / masked_total}


def module_lookup(metadata: pd.DataFrame) -> dict[int, str]:
    rows = metadata.dropna(subset=["module_token_id", "module_token"])[
        ["module_token_id", "module_token"]
    ].drop_duplicates()
    return {int(row["module_token_id"]): str(row["module_token"]) for _, row in rows.iterrows()}


def run_epoch(
    model: CharlsFoundLite,
    loader: DataLoader,
    device: torch.device,
    mask_token_id: int,
    mask_probability: float,
    optimizer: torch.optim.Optimizer | None = None,
    collect_module_metrics: bool = False,
    progress_label: str | None = None,
    progress_every: int = 100,
) -> tuple[dict[str, float], pd.DataFrame]:
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss = 0.0
    total_masked = 0
    total_top1 = 0.0
    total_top5 = 0.0
    module_totals: dict[int, dict[str, int]] = {}

    started_at = time.time()
    for batch_index, batch in enumerate(loader, start=1):
        value_ids = batch["value_ids"].to(device)
        trait_ids = batch["trait_ids"].to(device)
        module_ids = batch["module_ids"].to(device)
        wave_token_ids = batch["wave_token_id"].to(device)
        maskable = batch["maskable"].to(device)
        masked_inputs, targets, mask = sample_masked_values(
            value_ids, maskable, mask_token_id, mask_probability
        )
        masked_targets = targets[mask]

        with torch.set_grad_enabled(train_mode):
            _, token_hidden = model.encode(masked_inputs, trait_ids, module_ids, wave_token_ids)
            masked_hidden = token_hidden[mask]
            logits = model.lm_head(masked_hidden)
            loss = F.cross_entropy(logits, masked_targets)
            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        metrics = batch_metrics(logits.detach(), masked_targets.detach())
        masked_count = int(metrics["masked_tokens"])
        total_loss += float(loss.detach().item()) * max(masked_count, 1)
        total_masked += masked_count
        total_top1 += (
            float(metrics["top1"]) * masked_count if not math.isnan(metrics["top1"]) else 0.0
        )
        total_top5 += (
            float(metrics["top5"]) * masked_count if not math.isnan(metrics["top5"]) else 0.0
        )

        if collect_module_metrics and masked_count:
            masked_modules = module_ids[mask].detach().cpu().numpy()
            masked_targets_detached = masked_targets.detach()
            masked_predictions = logits.detach().argmax(dim=-1)
            masked_top5 = logits.detach().topk(k=min(5, logits.shape[-1]), dim=-1).indices
            for module_id in np.unique(masked_modules).tolist():
                module_mask_np = masked_modules == module_id
                module_mask = torch.as_tensor(
                    module_mask_np, dtype=torch.bool, device=masked_targets_detached.device
                )
                total = int(module_mask.sum().item())
                correct1 = int(
                    (masked_predictions[module_mask] == masked_targets_detached[module_mask])
                    .sum()
                    .item()
                )
                correct5 = int(
                    (masked_top5[module_mask] == masked_targets_detached[module_mask].unsqueeze(-1))
                    .any(dim=-1)
                    .sum()
                    .item()
                )
                item = module_totals.setdefault(int(module_id), {"total": 0, "top1": 0, "top5": 0})
                item["total"] += total
                item["top1"] += correct1
                item["top5"] += correct5

        if progress_label and (
            batch_index == 1 or batch_index % progress_every == 0 or batch_index == len(loader)
        ):
            elapsed = time.time() - started_at
            print(
                json.dumps(
                    {
                        "phase": progress_label,
                        "batch": batch_index,
                        "batches": len(loader),
                        "elapsed_sec": round(elapsed, 1),
                        "loss": round(float(loss.detach().item()), 4),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    denom = max(total_masked, 1)
    summary = {
        "loss": total_loss / denom,
        "masked_tokens": float(total_masked),
        "top1": total_top1 / denom,
        "top5": total_top5 / denom,
    }
    module_rows = [
        {
            "module_token_id": module_id,
            "masked_tokens": values["total"],
            "top1": values["top1"] / values["total"] if values["total"] else math.nan,
            "top5": values["top5"] / values["total"] if values["total"] else math.nan,
        }
        for module_id, values in sorted(module_totals.items())
    ]
    return summary, pd.DataFrame(module_rows)


def maybe_sample(frame: pd.DataFrame, max_rows: int | None, seed: int) -> pd.DataFrame:
    if max_rows is None or len(frame) <= max_rows:
        return frame
    return (
        frame.sample(n=max_rows, random_state=seed)
        .sort_values(["ID", "wave"], kind="stable")
        .reset_index(drop=True)
    )


def write_report(
    path: Path,
    metrics: pd.DataFrame,
    best_epoch: int,
    checkpoint_path: Path,
    args: argparse.Namespace,
) -> None:
    best = metrics[(metrics["split"].eq("validation")) & (metrics["epoch"].eq(best_epoch))].iloc[0]
    lines = [
        "# CHARLS-found-lite MLM Training Report",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"- Run name: `{args.run_name}`",
        f"- Best epoch: {best_epoch}",
        f"- Best validation loss: {best['loss']:.4f}",
        f"- Best validation top1/top5: {best['top1']:.4f} / {best['top5']:.4f}",
        f"- Checkpoint: `{checkpoint_path.relative_to(PROJECT_ROOT)}`",
        "",
        "This run uses respondent-level splits inherited from the rolling supervised matrix.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-name")
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-validation-rows", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--num-heads", type=int)
    parser.add_argument("--device")
    args = parser.parse_args()

    config = load_yaml(args.config)
    training_config = config["training"]
    model_config_raw = config["model"]
    run_name = args.run_name or training_config["run_name"]
    args.run_name = run_name
    seed = int(training_config["seed"])
    set_seed(seed)

    metadata = pd.read_csv(resolve_path(config["inputs"]["tokenization_metadata"]))
    split_manifest = load_json(resolve_path(config["outputs"]["mlm_split_manifest"]))
    dataset = pd.read_parquet(resolve_path(config["outputs"]["mlm_person_wave_dataset"]))
    train_frame = maybe_sample(
        dataset[dataset["split"].eq("train")].copy(), args.max_train_rows, seed
    )
    validation_frame = maybe_sample(
        dataset[dataset["split"].eq("validation")].copy(), args.max_validation_rows, seed + 1
    )
    if train_frame.empty or validation_frame.empty:
        raise RuntimeError("Train and validation MLM splits must both be non-empty")

    train_loader = DataLoader(
        PersonWaveMLMDataset(train_frame),
        batch_size=args.batch_size or int(training_config["batch_size"]),
        shuffle=True,
        num_workers=int(training_config.get("num_workers", 0)),
        collate_fn=collate_batch,
    )
    validation_loader = DataLoader(
        PersonWaveMLMDataset(validation_frame),
        batch_size=args.batch_size or int(training_config["batch_size"]),
        shuffle=False,
        num_workers=int(training_config.get("num_workers", 0)),
        collate_fn=collate_batch,
    )

    model_cfg = CharlsFoundLiteConfig(
        value_vocab_size=int(split_manifest["value_vocab_size"]),
        trait_vocab_size=int(split_manifest["trait_vocab_size"]),
        module_vocab_size=int(split_manifest["module_vocab_size"]),
        wave_vocab_size=int(split_manifest["wave_vocab_size"]),
        max_sequence_length=int(split_manifest["sequence_length"]),
        hidden_size=args.hidden_size or int(model_config_raw["hidden_size"]),
        num_layers=args.num_layers or int(model_config_raw["num_layers"]),
        num_heads=args.num_heads or int(model_config_raw["num_heads"]),
        dropout=float(model_config_raw["dropout"]),
    )
    device = select_device(args.device or training_config["device"])
    model = CharlsFoundLite(model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    mask_token_id = value_token_id(metadata, config["masking"]["mask_token"])
    mask_probability = float(training_config["mask_probability"])
    epochs = args.epochs or int(training_config["max_epochs"])
    patience = int(training_config["patience"])

    run_dir = run_directory(config, run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "checkpoint_best.pt"
    metrics_rows: list[dict[str, Any]] = []
    best_validation_loss = math.inf
    best_epoch = 0
    best_module_metrics = pd.DataFrame()
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        train_metrics, _ = run_epoch(
            model,
            train_loader,
            device,
            mask_token_id,
            mask_probability,
            optimizer=optimizer,
            progress_label=f"epoch_{epoch}_train",
        )
        validation_metrics, module_metrics = run_epoch(
            model,
            validation_loader,
            device,
            mask_token_id,
            mask_probability,
            optimizer=None,
            collect_module_metrics=True,
            progress_label=f"epoch_{epoch}_validation",
        )
        metrics_rows.extend(
            [
                {"epoch": epoch, "split": "train", **train_metrics},
                {"epoch": epoch, "split": "validation", **validation_metrics},
            ]
        )
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": round(train_metrics["loss"], 4),
                    "validation_loss": round(validation_metrics["loss"], 4),
                    "validation_top1": round(validation_metrics["top1"], 4),
                    "validation_top5": round(validation_metrics["top5"], 4),
                },
                ensure_ascii=False,
            )
        )
        if validation_metrics["loss"] < best_validation_loss:
            best_validation_loss = validation_metrics["loss"]
            best_epoch = epoch
            stale_epochs = 0
            best_module_metrics = module_metrics.copy()
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": asdict(model_cfg),
                    "mask_token_id": mask_token_id,
                    "run_name": run_name,
                    "epoch": epoch,
                    "validation_loss": validation_metrics["loss"],
                    "config": config,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    metrics = pd.DataFrame(metrics_rows)
    metrics_path = run_dir / "training_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    module_names = module_lookup(metadata)
    if not best_module_metrics.empty:
        best_module_metrics["module_token"] = best_module_metrics["module_token_id"].map(
            module_names
        )
        best_module_metrics.to_csv(run_dir / "validation_module_metrics.csv", index=False)
    write_report(run_dir / "training_report.md", metrics, best_epoch, checkpoint_path, args)

    print(
        json.dumps(
            {
                "run_name": run_name,
                "device": str(device),
                "train_rows": int(len(train_frame)),
                "validation_rows": int(len(validation_frame)),
                "best_epoch": best_epoch,
                "best_validation_loss": best_validation_loss,
                "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
