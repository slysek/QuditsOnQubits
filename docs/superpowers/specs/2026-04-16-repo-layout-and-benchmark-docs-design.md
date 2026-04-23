# Repo Layout And Benchmark Docs Design

## Scope

This change reorganizes benchmark artifacts and their documentation so the
repository root is less cluttered, while keeping code execution stable.

## Goals

- Move benchmark CSV outputs under `data/benchmarks/`.
- Move benchmark analysis markdown files under `docs/benchmarks/`.
- Move exported benchmark circuits under `data/benchmarks/circuits/`.
- Make Python code resolve asset and output paths from the repository root
  instead of assuming the current working directory.
- Add a polished Markdown summary for the full benchmark results table.

## Non-Goals

- Moving `quantum_circuits/`, because it is already a clear top-level asset
  directory and several existing notebooks/scripts rely on it.
- Moving the older research notebooks for now, because many of them likely
  depend on being opened from the repository root.

## Design

### Directory layout

- `data/benchmarks/`
  - `benchmark_encoding_bases_results.csv`
  - `benchmark_encoding_bases_extended_results.csv`
  - `benchmark_encoding_bases_full_results.csv`
  - `benchmark_-full_results.csv`
  - `circuits/`
- `docs/benchmarks/`
  - `benchmark_encoding_bases_extended_results.md`
  - `benchmark_encoding_bases_full_results_analysis.md`
  - `benchmark_encoding_bases_full_results_overview.md`

### Code changes

- Add a small path helper module inside the package so code can build
  repository-relative paths in one place.
- Update `create_ame_circuit.py` to load `.qpy` gate assets using repository
  paths instead of plain relative paths.
- Update `benchmark_encoding_bases.py` so default CSV and circuit output
  locations point to the new `data/benchmarks` directories.

### Validation

- Add tests covering path helper behavior.
- Add a regression test proving `create_ame_circuit(...)` works even when the
  current working directory is outside the repository root.
