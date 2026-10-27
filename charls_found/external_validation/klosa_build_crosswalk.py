#!/usr/bin/env python3
"""Refresh KLoSA Phase 0 discovery/crosswalk artifacts."""

from __future__ import annotations

import argparse
import subprocess
import sys

from klosa_utils import API_BASE, PROJECT_ROOT


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", default=API_BASE)
    parser.add_argument("--row-limit", type=int, default=1000000)
    args = parser.parse_args()

    script = PROJECT_ROOT / "charls_found" / "external_validation" / "build_crosswalks.py"
    command = [
        sys.executable,
        str(script),
        "--api-base",
        args.api_base,
        "--row-limit",
        str(args.row_limit),
        "--cohorts",
        "klosa",
        "elsa",
        "share",
    ]
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
