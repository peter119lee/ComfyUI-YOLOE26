from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_checks() -> list[str]:
    failures: list[str] = []

    contracts_path = ROOT / ".plans" / "yolo26-node-pack" / "docs" / "api-contracts.md"
    if not contracts_path.exists():
        failures.append("Missing docs/api-contracts source file in .plans/yolo26-node-pack/docs/api-contracts.md")

    return failures


def main() -> int:
    failures = run_checks()
    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}")
        return 1

    print("[PASS] CI skeleton checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
