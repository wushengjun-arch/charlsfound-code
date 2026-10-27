#!/usr/bin/env python3
"""Train the KLoSA-native CHARLS-found-lite encoder using the KLoSA MLM config."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_CONFIG = SCRIPT_PATH.parents[1] / "configs" / "klosa_mlm.yaml"
TRAINER = PROJECT_ROOT / "charls_found" / "mlm" / "train_charls_found_lite.py"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-name")
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-validation-rows", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device")
    args = parser.parse_args()

    command = [sys.executable, str(TRAINER), "--config", str(args.config)]
    for flag, value in [
        ("--run-name", args.run_name),
        ("--max-train-rows", args.max_train_rows),
        ("--max-validation-rows", args.max_validation_rows),
        ("--epochs", args.epochs),
        ("--batch-size", args.batch_size),
        ("--device", args.device),
    ]:
        if value is not None:
            command.extend([flag, str(value)])
    return subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
