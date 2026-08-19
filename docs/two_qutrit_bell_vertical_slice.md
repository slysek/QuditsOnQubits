# Two-Qutrit Bell Vertical Slice

This reference run demonstrates the public qudit-to-qubit architecture. It is the smallest audited benchmark, not a limit on supported circuit domains or local dimensions.

The command performs this complete path:

1. Load the logical two-qutrit Bell reference.
2. Apply the explicit `canonical_ez` qutrit-to-two-qubit isometry.
3. Build one four-qubit source circuit and nine measured qubit circuits.
4. Compile and execute them on local ideal Qiskit Aer.
5. Decode physical counts to logical qutrit outcomes and evaluate the Bell functional.
6. Write a versioned `run-manifest-v1` with SHA-256 references to every result artifact.

## Example notebook

Open [`notebooks/two_qutrit_bell_vertical_slice.ipynb`](../notebooks/two_qutrit_bell_vertical_slice.ipynb) after installing the project. The notebook uses only the public library API and exposes each boundary separately: logical Bell specification, `canonical_ez` encoding, generated qubit circuits, Aer execution, decoded logical result, and integrity-checked manifest reload.

The committed notebook is clean: it contains no credentials, absolute user paths, execution counts, or saved outputs. Its default run uses `2048` shots and writes under `artifacts/two_qutrit_bell_notebook`.

## Clean installation

Python 3.11 through 3.13 is supported.

PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install .
qoq-two-qutrit-bell --shots 2048 --seed 42 --output-root artifacts/vertical_slice_runs
```

Linux or macOS:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
qoq-two-qutrit-bell --shots 2048 --seed 42 --output-root artifacts/vertical_slice_runs
```

Automated PowerShell verification builds a wheel, creates a new virtual environment, installs the wheel, runs the command, and validates the manifest:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/verify_two_qutrit_clean_room.ps1
```

For an offline local smoke test that reuses already installed dependencies:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/verify_two_qutrit_clean_room.ps1 -ReuseSystemPackages
```

`-ReuseSystemPackages` is not the clean-install acceptance check. The default mode is.

## Expected output

For `2048` shots and seed `42`, output has this shape:

```text
status=completed
benchmark=two_qutrit
encoding=canonical_ez
circuit_count=9
shots_per_circuit=2048
bell_unconditional=6.00...
bell_conditional=6.00...
leakage_rate=0.0000000000
manifest=<output-root>/<UTC-date>/<run-id>/run-manifest.json
```

Finite sampling changes the last digits. Acceptance range for `bell_unconditional` is `6.0 ± 0.15` at `2048` shots with the documented seed. Ideal canonical encoding has zero leakage.

## Artifacts

Each run directory contains:

```text
run-manifest.json
source-circuits.qpy
encoding.json
logical-measurements.qpy
postprocessing.json
compiled-circuits.qpy
counts.json
result.json
```

`run-manifest.json` records the experiment and encoding snapshots and hashes, backend identity and `ideal_simulator` execution mode, Python/package/dependency provenance, status history, job ID, decoded result, and SHA-256 for the other seven files. `load_run_manifest(run_dir)` verifies schema, run identity, containment, existence, and every artifact hash before returning the manifest.

## Troubleshooting

- `Python version is not supported`: use Python `3.11`, `3.12`, or `3.13`.
- `No module named qiskit_aer`: reinstall the project without `--no-deps`, or install a compatible `qiskit-aer>=0.17,<0.18`.
- wheel build tries to download build tools: install `setuptools>=69` and `wheel`, then use `python -m pip wheel . --no-build-isolation`.
- PowerShell blocks activation: invoke the verifier directly, or allow the current process with `Set-ExecutionPolicy -Scope Process Bypass`.
- output directory permission error: pass `--output-root` pointing to a writable directory.
- Bell value outside tolerance: confirm `--shots 2048 --seed 42`, supported dependency versions, and an unmodified `canonical_ez` encoding.
- artifact hash mismatch: preserve the run directory unchanged; rerun the experiment instead of editing QPY, counts, result, or encoding files.
