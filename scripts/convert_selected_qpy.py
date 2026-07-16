"""Convert only selected candidate directories to an older QPY format."""

import argparse
import shutil
from pathlib import Path

from qiskit import qpy


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = (
    REPO_ROOT
    / "artifacts"
    / "iqm_runs"
    / "raw"
    / "quantum_circuits"
    / "garnet"
    / "ghz3"
)
DEFAULT_OUTPUT_ROOT = DEFAULT_SOURCE_ROOT.with_name("ghz3_qpy13")

# Pierwsze 10 pozycji z ghz_best_list w best_garnet_ghz.ipynb.
SELECTED_CANDIDATES = ["sup012_P012_ph022", "sup012_P012_ph121", "sup012_P012_ph221", "sup012_P012_ph222", "sup012_P021_ph012", "sup012_P021_ph022", "sup012_P021_ph212", "sup012_P021_ph222", "sup123_P120_ph222", "sup123_P210_ph121", "sup123_P210_ph222"]


def _directory_name(candidate):
    prefix = "monomial_full__"
    return candidate if candidate.startswith(prefix) else f"{prefix}{candidate}"


def convert_candidates(source_root, output_root, candidates, target_version=13):
    source_root = Path(source_root)
    output_root = Path(output_root)
    candidate_directories = [
        (source_root / _directory_name(candidate), _directory_name(candidate))
        for candidate in candidates
    ]

    missing = [str(path) for path, _ in candidate_directories if not path.is_dir()]
    if missing:
        raise FileNotFoundError("Missing candidate directories:\n" + "\n".join(missing))

    converted = 0
    for source_dir, directory_name in candidate_directories:
        destination_dir = output_root / directory_name
        for source_path in source_dir.rglob("*"):
            if not source_path.is_file():
                continue
            destination_path = destination_dir / source_path.relative_to(source_dir)
            destination_path.parent.mkdir(parents=True, exist_ok=True)

            if source_path.suffix.lower() == ".qpy":
                with source_path.open("rb") as handle:
                    circuits = qpy.load(handle)
                with destination_path.open("wb") as handle:
                    qpy.dump(circuits, handle, version=target_version)
                converted += 1
            else:
                shutil.copy2(source_path, destination_path)

        print(f"Converted: {directory_name}")

    return converted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--candidate",
        action="append",
        dest="candidates",
        help="Candidate name; repeat to override the built-in top-10 list.",
    )
    parser.add_argument("--target-version", type=int, default=13)
    args = parser.parse_args()

    count = convert_candidates(
        args.source_root,
        args.output_root,
        args.candidates or SELECTED_CANDIDATES,
        target_version=args.target_version,
    )
    print(f"Done: {count} QPY files -> {args.output_root}")


if __name__ == "__main__":
    main()
