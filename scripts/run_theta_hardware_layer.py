"""Append offline IQM topology costs to a saved theta benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--indices', type=int, nargs='+')
    parser.add_argument('--states', nargs='+', choices=['two_qutrit','ghz3','ame43'])
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    from qudits_on_qubits.benchmarks.theta_continuation.hardware_layer import run_hardware_layer
    result = run_hardware_layer(args.source_run, args.output_dir, args.snapshot,
        seeds=args.seeds, indices=args.indices, states=args.states, resume=args.resume,
        progress=lambda message: print(message, flush=True))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
