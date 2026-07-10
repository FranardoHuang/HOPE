#!/usr/bin/env python3
"""Read-only inventory for the Pod2 Stage-1 terminal-checkpoint recovery.

The 2026-07-10 audit found eleven completed Pod2 runs whose terminal
``model_16999.pt`` checkpoints were present, while their latest judge artifacts
still referred to ``model_16400.pt``.  The queue and revive-command ledger had
also been deleted.  This tool deliberately does *not* reconstruct a training
command and never writes, deletes, launches, or kills anything.  It only:

* maps the eleven semantic arm labels to run directories, requiring a unique
  regex match for every arm;
* checks the exact expected checkpoint sequence, ``params/env.yaml`` / ``env.pkl``, terminal
  checkpoint size, reports, and exported sidecars; and
* prints shell-quoted dry-run and real ``judge.sh`` commands bound explicitly
  to ``model_16999.pt``.

Run on a pod clone, for example::

    python scripts/audit_runpod_terminal_runs.py \
      --run-root logs/rsl_rl/agibot_a3_hope_virtualball \
      --judge-steps 15000 --judge-gpu 0

The default patterns are intentionally narrow.  A missing or duplicate match is
a hard error; inspect the candidate list and pass an audited replacement with
``--arm LABEL=REGEX`` rather than guessing or taking the newest directory.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_ARM_PATTERNS: tuple[tuple[str, str], ...] = (
    (
        "R5b_seed2",
        r"(?i)(?:^|[_-])R5b(?:[_-]|$).*(?:seed[_-]?2|s2)(?:[_-]|$)",
    ),
    ("ST1", r"(?i)(?:^|[_-])ST1(?:[_-]|$)"),
    ("G1", r"(?i)(?:^|[_-])G1(?:[_-]|$)"),
    ("G2", r"(?i)(?:^|[_-])G2(?:[_-]|$)"),
    ("C1", r"(?i)(?:^|[_-])C1(?:[_-]|$)"),
    ("C2", r"(?i)(?:^|[_-])C2(?:[_-]|$)"),
    ("C3", r"(?i)(?:^|[_-])C3(?:[_-]|$)"),
    ("C4", r"(?i)(?:^|[_-])C4(?:[_-]|$)"),
    ("N1", r"(?i)(?:^|[_-])N1(?:[_-]|$)"),
    ("T3", r"(?i)(?:^|[_-])T3(?:[_-]|$)"),
    (
        "R8b_seed2",
        r"(?i)(?:^|[_-])R8b(?:[_-]|$).*(?:seed[_-]?2|s2)(?:[_-]|$)",
    ),
)

_MODEL_RE = re.compile(r"^model_(\d+)\.pt$")
_REPORT_RE = re.compile(r"judge_report_model_(\d+)_.*\.md$")


@dataclass(frozen=True)
class ArmSpec:
    label: str
    pattern: str

    def matches(self, name: str) -> bool:
        return re.search(self.pattern, name) is not None


@dataclass(frozen=True)
class RunAudit:
    label: str
    pattern: str
    run_dir: str
    terminal_checkpoint: str
    checkpoint_count: int
    checkpoint_min: int | None
    checkpoint_max: int | None
    checkpoint_sequence_exact: bool
    terminal_checkpoint_exists: bool
    terminal_checkpoint_bytes: int
    env_yaml_exists: bool
    env_pickle_exists: bool
    agent_pickle_exists: bool
    report_checkpoints: list[int]
    terminal_report_exists: bool
    latest_report_checkpoint: int | None
    policy_onnx_exists: bool
    learned_std_exists: bool
    obs_norm_exists: bool
    status: str
    errors: list[str]
    warnings: list[str]
    dry_run_command: str
    judge_command: str


class MappingError(RuntimeError):
    """Raised when an arm label does not map to exactly one run directory."""


def _parse_arm(text: str) -> ArmSpec:
    if "=" not in text:
        raise argparse.ArgumentTypeError("--arm must be LABEL=REGEX")
    label, pattern = text.split("=", 1)
    label = label.strip()
    pattern = pattern.strip()
    if not label or not pattern:
        raise argparse.ArgumentTypeError("--arm must have a non-empty LABEL and REGEX")
    try:
        re.compile(pattern)
    except re.error as exc:
        raise argparse.ArgumentTypeError(f"invalid regex for {label}: {exc}") from exc
    return ArmSpec(label, pattern)


def default_specs() -> list[ArmSpec]:
    return [ArmSpec(label, pattern) for label, pattern in DEFAULT_ARM_PATTERNS]


def discover_run_dirs(run_root: Path) -> list[Path]:
    if not run_root.is_dir():
        raise FileNotFoundError(f"run root is not a directory: {run_root}")
    return sorted((p.resolve() for p in run_root.iterdir() if p.is_dir()), key=lambda p: p.name)


def map_runs(specs: Sequence[ArmSpec], candidates: Sequence[Path]) -> dict[str, Path]:
    mapped: dict[str, Path] = {}
    failures: list[str] = []
    for spec in specs:
        hits = [p for p in candidates if spec.matches(p.name)]
        if len(hits) != 1:
            names = ", ".join(p.name for p in hits) if hits else "<none>"
            failures.append(
                f"{spec.label}: expected exactly one match for /{spec.pattern}/, "
                f"found {len(hits)}: {names}"
            )
            continue
        mapped[spec.label] = hits[0]
    if failures:
        terminal_candidates = [
            p.name for p in candidates if (p / "model_16999.pt").is_file()
        ]
        suffix = "\nterminal-checkpoint candidates:\n  " + (
            "\n  ".join(terminal_candidates) if terminal_candidates else "<none>"
        )
        raise MappingError("\n".join(failures) + suffix)
    return mapped


def expected_checkpoint_numbers(start: int, terminal: int, interval: int) -> list[int]:
    if start < 0 or terminal < start or interval <= 0:
        raise ValueError("checkpoint start/terminal/interval are inconsistent")
    nums = list(range(start, terminal, interval))
    if not nums or nums[-1] != terminal:
        nums.append(terminal)
    return nums


def checkpoint_numbers(run_dir: Path) -> list[int]:
    nums: list[int] = []
    for path in run_dir.glob("model_*.pt"):
        match = _MODEL_RE.match(path.name)
        if match:
            nums.append(int(match.group(1)))
    return sorted(nums)


def report_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    reports: list[tuple[int, Path]] = []
    judge_dir = run_dir / "judge"
    if not judge_dir.is_dir():
        return reports
    for path in judge_dir.rglob("judge_report_model_*.md"):
        match = _REPORT_RE.match(path.name)
        if match:
            reports.append((int(match.group(1)), path))
    return sorted(reports, key=lambda item: (item[1].stat().st_mtime_ns, item[1].name))


def _shell_command(parts: Iterable[str]) -> str:
    return shlex.join([str(part) for part in parts])


def audit_run(
    spec: ArmSpec,
    run_dir: Path,
    *,
    checkpoint_start: int,
    terminal: int,
    checkpoint_interval: int,
    judge_script: Path,
    judge_steps: int,
    judge_seed: int,
    judge_gpu: int,
) -> RunAudit:
    errors: list[str] = []
    warnings: list[str] = []
    expected = expected_checkpoint_numbers(checkpoint_start, terminal, checkpoint_interval)
    actual = checkpoint_numbers(run_dir)
    sequence_exact = actual == expected
    if not sequence_exact:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        errors.append(f"checkpoint sequence mismatch; missing={missing}, extra={extra}")

    terminal_path = run_dir / f"model_{terminal}.pt"
    terminal_exists = terminal_path.is_file()
    terminal_bytes = terminal_path.stat().st_size if terminal_exists else 0
    if not terminal_exists:
        errors.append(f"terminal checkpoint missing: {terminal_path.name}")
    elif terminal_bytes <= 0:
        errors.append(f"terminal checkpoint is empty: {terminal_path.name}")

    env_path = run_dir / "params" / "env.yaml"
    env_exists = env_path.is_file()
    if not env_exists:
        errors.append("params/env.yaml missing; judge.sh must fail closed")
    env_pickle_exists = (run_dir / "params" / "env.pkl").is_file()
    if not env_pickle_exists:
        warnings.append(
            "params/env.pkl missing; terminal judge can use env.yaml, but formal Isaac "
            "scorecard cannot restore the exact training env config"
        )
    agent_pickle_exists = (run_dir / "params" / "agent.pkl").is_file()
    if not agent_pickle_exists:
        warnings.append(
            "params/agent.pkl missing; formal Isaac scorecard cannot restore the exact PPO "
            "runner/normalizer config"
        )

    reports = report_checkpoints(run_dir)
    report_tags = [tag for tag, _ in reports]
    terminal_report = terminal in report_tags
    latest_report = reports[-1][0] if reports else None
    if not reports:
        warnings.append("no judge report exists")
    elif not terminal_report:
        warnings.append(
            f"terminal checkpoint is unjudged; latest report is model_{latest_report}"
        )

    exported = run_dir / "exported"
    onnx_exists = (exported / "policy.onnx").is_file()
    std_exists = (exported / "learned_std.npy").is_file()
    norm_exists = (exported / "obs_norm.npz").is_file()
    if not (onnx_exists and std_exists):
        warnings.append("current exported policy/learned_std pair is incomplete")
    if not norm_exists:
        warnings.append(
            "obs_norm.npz absent (valid only when empirical normalization was disabled)"
        )
    if latest_report != terminal:
        warnings.append(
            "do not attribute current exported/ sidecars to the terminal checkpoint"
        )

    command_base = [
        "bash",
        str(judge_script),
        str(run_dir),
        str(terminal_path),
        "--steps",
        str(judge_steps),
        "--seed",
        str(judge_seed),
        "--gpu",
        str(judge_gpu),
    ]
    dry_run_command = _shell_command([*command_base, "--dry-run"])
    judge_command = _shell_command(command_base)

    if errors:
        status = "INVALID"
    elif terminal_report:
        status = "TERMINAL_JUDGED"
    else:
        status = "NEEDS_TERMINAL_JUDGE"

    return RunAudit(
        label=spec.label,
        pattern=spec.pattern,
        run_dir=str(run_dir),
        terminal_checkpoint=str(terminal_path),
        checkpoint_count=len(actual),
        checkpoint_min=min(actual) if actual else None,
        checkpoint_max=max(actual) if actual else None,
        checkpoint_sequence_exact=sequence_exact,
        terminal_checkpoint_exists=terminal_exists,
        terminal_checkpoint_bytes=terminal_bytes,
        env_yaml_exists=env_exists,
        env_pickle_exists=env_pickle_exists,
        agent_pickle_exists=agent_pickle_exists,
        report_checkpoints=report_tags,
        terminal_report_exists=terminal_report,
        latest_report_checkpoint=latest_report,
        policy_onnx_exists=onnx_exists,
        learned_std_exists=std_exists,
        obs_norm_exists=norm_exists,
        status=status,
        errors=errors,
        warnings=warnings,
        dry_run_command=dry_run_command,
        judge_command=judge_command,
    )


def audit_all(
    run_root: Path,
    specs: Sequence[ArmSpec],
    **kwargs: object,
) -> list[RunAudit]:
    candidates = discover_run_dirs(run_root)
    mapped = map_runs(specs, candidates)
    return [
        audit_run(spec, mapped[spec.label], **kwargs)  # type: ignore[arg-type]
        for spec in specs
    ]


def _print_human(audits: Sequence[RunAudit]) -> None:
    print(
        "label\tstatus\trun_dir\tckpts\tlatest_report\t"
        "onnx/std/norm\tterminal_bytes"
    )
    for item in audits:
        latest = (
            f"model_{item.latest_report_checkpoint}"
            if item.latest_report_checkpoint is not None
            else "none"
        )
        sidecars = "/".join(
            "Y" if value else "N"
            for value in (
                item.policy_onnx_exists,
                item.learned_std_exists,
                item.obs_norm_exists,
            )
        )
        print(
            f"{item.label}\t{item.status}\t{item.run_dir}\t"
            f"{item.checkpoint_count}\t{latest}\t{sidecars}\t"
            f"{item.terminal_checkpoint_bytes}"
        )
        for error in item.errors:
            print(f"  ERROR: {error}")
        for warning in item.warnings:
            print(f"  WARN: {warning}")
        print(f"  DRY-RUN: {item.dry_run_command}")
        print(f"  JUDGE:   {item.judge_command}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--arm",
        action="append",
        type=_parse_arm,
        default=None,
        metavar="LABEL=REGEX",
        help="replace the built-in eleven-arm map with audited custom patterns",
    )
    parser.add_argument("--checkpoint-start", type=int, default=13000)
    parser.add_argument("--terminal", type=int, default=16999)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--judge-steps", type=int, default=15000)
    parser.add_argument("--judge-seed", type=int, default=0)
    parser.add_argument("--judge-gpu", type=int, default=0)
    parser.add_argument(
        "--judge-script",
        type=Path,
        default=Path(__file__).resolve().with_name("judge.sh"),
    )
    parser.add_argument("--format", choices=("human", "json"), default="human")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    specs = args.arm if args.arm is not None else default_specs()
    labels = [spec.label for spec in specs]
    if len(set(labels)) != len(labels):
        print("[terminal-audit][FATAL] duplicate arm labels", file=sys.stderr)
        return 2
    if args.judge_steps <= 0 or args.judge_seed < 0 or args.judge_gpu < 0:
        print("[terminal-audit][FATAL] judge steps/gpu must be positive and seed non-negative", file=sys.stderr)
        return 2
    if not args.judge_script.is_file():
        print(f"[terminal-audit][FATAL] judge script missing: {args.judge_script}", file=sys.stderr)
        return 2

    try:
        audits = audit_all(
            args.run_root.resolve(),
            specs,
            checkpoint_start=args.checkpoint_start,
            terminal=args.terminal,
            checkpoint_interval=args.checkpoint_interval,
            judge_script=args.judge_script.resolve(),
            judge_steps=args.judge_steps,
            judge_seed=args.judge_seed,
            judge_gpu=args.judge_gpu,
        )
    except (FileNotFoundError, MappingError, ValueError) as exc:
        print(f"[terminal-audit][FATAL] {exc}", file=sys.stderr)
        return 2

    if args.format == "json":
        print(json.dumps([asdict(item) for item in audits], indent=2, sort_keys=True))
    else:
        _print_human(audits)
    return 1 if any(item.errors for item in audits) else 0


if __name__ == "__main__":
    raise SystemExit(main())
