from __future__ import annotations

import argparse

from basis_direct_encoding_benchmarks.comparison import (
    compare_old_vs_direct,
    print_comparison_summary,
    timestamped_comparison_path,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare old append-W benchmark CSV with direct-basis CSV.",
    )
    parser.add_argument("--old-csv", required=True)
    parser.add_argument("--direct-csv", required=True)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    output_csv = args.output_csv or timestamped_comparison_path(output_dir=args.output_dir)

    comparison, summary = compare_old_vs_direct(
        args.old_csv,
        args.direct_csv,
        output_csv,
    )
    print(f"Comparison CSV: {output_csv}")
    print_comparison_summary(comparison, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
