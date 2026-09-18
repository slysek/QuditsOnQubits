"""Run the local E_theta gate and graph-state continuation benchmark."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from uuid import uuid4


_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from qudits_on_qubits.benchmarks.theta_continuation.artifacts import RunStore
from qudits_on_qubits.benchmarks.theta_continuation.models import ThetaBenchmarkConfig
from qudits_on_qubits.benchmarks.theta_continuation.report import generate_report


def run_theta_benchmark(*args, **kwargs):
    """Import the runner lazily so CLI parsing never starts synthesis runtimes."""
    from qudits_on_qubits.benchmarks.theta_continuation.runner import run_theta_benchmark as run

    return run(*args, **kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("gates", "circuits", "all"), default="all")
    parser.add_argument("--resume", type=Path, metavar="RUN_DIR", help="Resume the immutable stored configuration.")
    parser.add_argument("--output-dir", type=Path, help="New run directory; an existing run for --mode circuits.")
    # Suppressed defaults distinguish omitted options from explicitly supplied
    # values, including values equal to today's defaults during a resume.
    for name, kind in (
        ("theta-points", int), ("limit-points", int), ("max-nfev", int),
        ("max-subdivisions", int), ("f3-tolerance", float), ("cz3-tolerance", float),
        ("optimization-level", int), ("baseline-qpy", str), ("baseline-sha256", str),
    ):
        parser.add_argument(f"--{name}", type=kind, default=argparse.SUPPRESS)
    parser.add_argument("--transpiler-seeds", type=int, nargs="+", default=argparse.SUPPRESS)
    parser.add_argument("--states", nargs="+", choices=("two_qutrit", "ghz3", "ame43"), default=argparse.SUPPRESS)
    return parser


def _configuration(args: argparse.Namespace) -> tuple[ThetaBenchmarkConfig, Path, bool]:
    supplied = {name: value for name, value in vars(args).items()
                if name not in {"mode", "resume", "output_dir"}}
    if "baseline_qpy" in supplied:
        supplied["baseline_qpy"] = str(Path(supplied["baseline_qpy"]).resolve())
    if args.resume is not None and args.output_dir is not None:
        raise ValueError("--resume and --output-dir cannot be combined")
    existing = args.resume
    if args.mode == "circuits":
        existing = args.resume if args.resume is not None else args.output_dir
        if existing is None:
            raise ValueError("--mode circuits requires --resume RUN_DIR or --output-dir RUN_DIR")
    if existing is not None:
        store = RunStore.open(existing)
        config = ThetaBenchmarkConfig.from_dict(store.manifest["config"])
        proposed = ThetaBenchmarkConfig.from_dict({**config.to_dict(), **supplied})
        conflicts = []
        for name in supplied:
            previous, requested = getattr(config, name), getattr(proposed, name)
            if name == "baseline_qpy":
                previous, requested = Path(previous).resolve(), Path(requested).resolve()
            if previous != requested:
                conflicts.append("--" + name.replace("_", "-"))
        if conflicts:
            raise ValueError("Configuration conflicts with stored run: " + ", ".join(conflicts))
        return config, store.root, True
    config = ThetaBenchmarkConfig(**supplied)
    directory = args.output_dir or (
        _ROOT / "artifacts/theta_continuation"
        / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    )
    return config, directory.resolve(), False


def _progress(event) -> None:
    if isinstance(event, dict):
        print(json.dumps(event, ensure_ascii=True, allow_nan=False, default=str), flush=True)
    else:
        print(str(event), flush=True)


def main(argv=None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code)
    try:
        config, output_dir, resume = _configuration(args)
        print(f"stage=run mode={args.mode} output_dir={output_dir}", flush=True)
        summary = run_theta_benchmark(config, output_dir, mode=args.mode, resume=resume, progress=_progress)
        report_path = generate_report(summary.get("output_dir", output_dir), full_circuits=summary.get("full_circuits"))
        print(json.dumps({"stage": "completed", "counts": summary.get("counts", {}),
                          "report": str(report_path)}, ensure_ascii=True, allow_nan=False), flush=True)
        # A completed numerical protocol may contain unsuccessful points. Their
        # status belongs in the data; the process itself still completed.
        return 0
    except KeyboardInterrupt:
        print("Interrupted; completed point artifacts remain resumable.", file=sys.stderr)
        return 130
    except (ValueError, TypeError, KeyError, OSError) as exc:
        print(f"Configuration/artifact error ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Execution error ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
