from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS_PATH = ROOT / "requirements.txt"
MINIMUM_ULTRALYTICS_SPEC = "ultralytics>=8.3.200"


@dataclass(frozen=True)
class CommandStep:
    name: str
    command: tuple[str, ...]


def validate_requirements_pin(requirements_path: Path = REQUIREMENTS_PATH) -> list[str]:
    if not requirements_path.exists():
        return [f"Missing requirements file: {requirements_path}"]

    requirement_lines = [
        line.strip()
        for line in requirements_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    if MINIMUM_ULTRALYTICS_SPEC not in requirement_lines:
        return [
            "requirements.txt must include the tested minimum ultralytics floor "
            f"'{MINIMUM_ULTRALYTICS_SPEC}'."
        ]

    return []


def build_ci_steps() -> list[CommandStep]:
    python = sys.executable
    return [
        CommandStep(
            name="Compile Python sources",
            command=(
                python,
                "-m",
                "compileall",
                str(ROOT / "__init__.py"),
                str(ROOT / "nodes.py"),
                str(ROOT / "scripts"),
                str(ROOT / "tests"),
            ),
        ),
        CommandStep(
            name="Run pytest suite",
            command=(python, "-m", "pytest"),
        ),
    ]


def run_step(step: CommandStep) -> int:
    print(f"[RUN] {step.name}: {' '.join(step.command)}")
    completed = subprocess.run(step.command, cwd=ROOT)
    if completed.returncode == 0:
        print(f"[PASS] {step.name}")
    else:
        print(f"[FAIL] {step.name} (exit {completed.returncode})")
    return int(completed.returncode)


def run_checks() -> list[str]:
    return validate_requirements_pin()


def main() -> int:
    failures = run_checks()
    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}")
        return 1

    for step in build_ci_steps():
        exit_code = run_step(step)
        if exit_code != 0:
            return exit_code

    print("[PASS] CI checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
