# SZY-42 Two-Qutrit Bell Vertical Slice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver an installable, reproducible two-qutrit Bell run that traverses logical qutrit state, explicit qubit encoding, measured qubit circuits, ideal Aer execution, decoded Bell result, and a versioned integrity-linked `RunManifest`.

**Architecture:** Add a small generic vertical-slice contract package beside the legacy experiment runner, preserving its public API. Core models know only encodings, circuit/postprocessor protocols, execution, backend snapshots, and manifests; the Bell adapter implements the domain-specific preparation and decoding. The runner reuses existing backend adapters and `ExperimentStore` for safe atomic artifacts, while a packaged CLI is the clean-room entry point.

**Tech Stack:** Python 3.11+, dataclasses/protocols, NumPy, Qiskit/QPY, Qiskit Aer, pytest, setuptools console scripts.

---

## File Structure

- Create `src/qudits_on_qubits/vertical_slice/models.py`: immutable generic contracts, isometric encoding, experiment spec, manifest, result.
- Create `src/qudits_on_qubits/vertical_slice/bell.py`: `BellReferenceCircuitSpec`, Bell postprocessor, canonical encoding factory.
- Create `src/qudits_on_qubits/vertical_slice/runner.py`: backend-neutral orchestration for this public slice and hashed artifact persistence.
- Create `src/qudits_on_qubits/vertical_slice/cli.py`: installed `qoq-two-qutrit-bell` command.
- Create `src/qudits_on_qubits/vertical_slice/__init__.py`: focused public exports.
- Modify `src/qudits_on_qubits/__init__.py`: lazy top-level exports without importing Aer at package import time.
- Modify `pyproject.toml`: console entry point.
- Create `tests/test_vertical_slice_models.py`: encoding/spec/manifest invariants.
- Create `tests/test_vertical_slice_bell.py`: domain boundary and preparation tests.
- Create `tests/test_two_qutrit_vertical_slice_e2e.py`: real ideal-Aer vertical slice and artifact integrity.
- Create `tests/test_two_qutrit_vertical_slice_cli.py`: public installed-command behavior through CLI main.
- Create `scripts/verify_two_qutrit_clean_room.ps1`: build wheel, create fresh venv, install, run CLI, validate manifest.
- Create `docs/two_qutrit_bell_vertical_slice.md`: clean-room commands, expected output, artifact meaning, troubleshooting.

### Task 1: Generic encoding and manifest contracts

**Files:**
- Create: `tests/test_vertical_slice_models.py`
- Create: `src/qudits_on_qubits/vertical_slice/models.py`

- [ ] **Step 1: Write failing encoding and manifest tests**

```python
def test_isometric_encoding_decodes_code_space_and_leakage():
    encoding = canonical_qutrit_encoding()
    assert encoding.decode((0, 0)) == LogicalOutcome(0, False)
    assert encoding.decode((1, 1)) == LogicalOutcome(None, True)
    assert encoding.stable_hash() == IsometricQuditEncoding.from_manifest_dict(
        encoding.to_manifest_dict()
    ).stable_hash()


def test_manifest_rejects_unknown_schema_version():
    with pytest.raises(ManifestValidationError, match="schema version"):
        RunManifest.from_safe_dict({"schema_version": "run-manifest-v2"})
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/test_vertical_slice_models.py -q`

Expected: collection fails because `qudits_on_qubits.vertical_slice` does not exist.

- [ ] **Step 3: Implement immutable contracts**

Implement these public signatures with strict JSON-safe validation, copied read-only matrices, canonical SHA-256, legal manifest transitions, and round-trip loaders:

```python
@runtime_checkable
class QuditEncoding(Protocol):
    kind: str
    encoding_id: str
    logical_dimension: int
    physical_qubits: int
    def isometry(self) -> np.ndarray: ...
    def decode(self, physical_bits: Sequence[int]) -> LogicalOutcome: ...
    def to_manifest_dict(self) -> Mapping[str, JsonValue]: ...
    def stable_hash(self) -> str: ...


@dataclass(frozen=True)
class QuditExperimentSpec:
    circuit: CircuitSpec
    encoding: QuditEncoding
    backend: AerIdeal
    execution: ExecutionSpec
    output_root: Path = Path("artifacts/vertical_slice_runs")
    tags: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RunManifest:
    schema_version: Literal["run-manifest-v1"]
    run_id: str
    experiment_spec: Mapping[str, JsonValue]
    experiment_hash: str
    encoding: Mapping[str, JsonValue]
    encoding_hash: str
    backend: BackendSnapshot | None
    software: SoftwareProvenance
    status: str
    timestamps: Mapping[str, str]
    status_history: tuple[Mapping[str, JsonValue], ...]
    jobs: Mapping[str, Mapping[str, JsonValue]]
    artifacts: tuple[ArtifactRef, ...]
    result: Mapping[str, JsonValue] | None
    warnings: tuple[str, ...]
    failure: Mapping[str, JsonValue] | None
```

- [ ] **Step 4: Run tests and verify GREEN**

Run: `python -m pytest tests/test_vertical_slice_models.py -q`

Expected: all model tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/qudits_on_qubits/vertical_slice/models.py tests/test_vertical_slice_models.py
git commit -m "feat: add vertical slice contracts"
```

### Task 2: Bell domain adapter

**Files:**
- Create: `tests/test_vertical_slice_bell.py`
- Create: `src/qudits_on_qubits/vertical_slice/bell.py`
- Create: `src/qudits_on_qubits/vertical_slice/__init__.py`

- [ ] **Step 1: Write failing Bell preparation tests**

```python
def test_two_qutrit_bell_prepares_measured_qubit_circuits():
    prepared = BellReferenceCircuitSpec("two_qutrit").prepare(
        canonical_qutrit_encoding()
    )
    assert len(prepared.source_circuits) == 1
    assert prepared.source_circuits[0].num_qubits == 4
    assert len(prepared.executable_circuits) == 9
    assert all(circuit.num_clbits == 4 for circuit in prepared.executable_circuits)
    assert prepared.postprocessor.kind == "bell.reference"


def test_bell_adapter_rejects_wrong_logical_dimension():
    with pytest.raises(SpecValidationError, match="logical dimension"):
        BellReferenceCircuitSpec("two_qutrit").prepare(ququart_encoding())
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/test_vertical_slice_bell.py -q`

Expected: import fails because Bell adapter is absent.

- [ ] **Step 3: Implement Bell circuit and postprocessor specs**

```python
@dataclass(frozen=True)
class BellReferenceCircuitSpec:
    reference_id: str

    @property
    def kind(self) -> str:
        return "bell.reference"

    @property
    def logical_dimensions(self) -> tuple[int, ...]:
        spec = get_reference_experiment(self.reference_id)
        return (spec.state.local_dimension,) * spec.state.num_parties

    def prepare(self, encoding: QuditEncoding) -> PreparedExperiment:
        # Build encoded state circuit, then existing audited Bell measurement circuits.
        ...


@dataclass(frozen=True)
class BellPostprocessorSpec:
    reference_id: str
    settings: tuple[tuple[str | None, ...], ...]
    qutrit_bit_indices: tuple[tuple[tuple[int, int], ...], ...]

    def evaluate(self, counts_by_circuit):
        # Reconstruct setting-indexed counts and return JSON-safe complex components.
        ...
```

- [ ] **Step 4: Run tests and verify GREEN**

Run: `python -m pytest tests/test_vertical_slice_bell.py -q`

Expected: all Bell adapter tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/qudits_on_qubits/vertical_slice tests/test_vertical_slice_bell.py
git commit -m "feat: adapt Bell references to vertical slice contracts"
```

### Task 3: Real Aer run and versioned artifacts

**Files:**
- Create: `tests/test_two_qutrit_vertical_slice_e2e.py`
- Create: `src/qudits_on_qubits/vertical_slice/runner.py`
- Modify: `src/qudits_on_qubits/vertical_slice/__init__.py`
- Modify: `src/qudits_on_qubits/__init__.py`

- [ ] **Step 1: Write failing real-backend E2E test**

```python
def test_two_qutrit_bell_runs_end_to_end_on_ideal_aer(tmp_path):
    spec = QuditExperimentSpec(
        circuit=BellReferenceCircuitSpec("two_qutrit"),
        encoding=canonical_qutrit_encoding(),
        backend=AerIdeal(seed_simulator=42),
        execution=ExecutionSpec(shots=2048, seed=42),
        output_root=tmp_path,
    )
    completed = run_vertical_slice(spec)
    assert completed.manifest.status == "completed"
    assert completed.result["bell_unconditional"]["real"] == pytest.approx(6.0, abs=0.15)
    assert completed.result["leakage_rate"] == 0.0
    assert completed.result["circuit_count"] == 9
    loaded = load_run_manifest(completed.artifact_dir)
    assert loaded == completed.manifest
    for artifact in loaded.artifacts:
        assert sha256((completed.artifact_dir / artifact.path).read_bytes()).hexdigest() == artifact.sha256
```

- [ ] **Step 2: Run test and verify RED**

Run: `python -m pytest tests/test_two_qutrit_vertical_slice_e2e.py -q`

Expected: import fails because runner functions are absent.

- [ ] **Step 3: Implement orchestration and persistence**

```python
def run_vertical_slice(spec: QuditExperimentSpec) -> QuditExperimentResult:
    store = ExperimentStore(spec.output_root)
    run = store.create_run("two-qutrit-bell")
    manifest = create_manifest(spec, run.name)
    write_manifest(store, run, manifest)
    prepared = spec.circuit.prepare(spec.encoding)
    # Persist source, logical, encoding, postprocessing; compile and execute via adapter;
    # persist compiled circuits and counts; evaluate; link every artifact by SHA-256;
    # atomically replace run-manifest.json after each legal status transition.
    ...


def load_run_manifest(run_dir: Path | str) -> RunManifest:
    # Validate schema, run ID, manifest hashes, and every referenced artifact.
    ...
```

- [ ] **Step 4: Run focused and legacy integration tests**

Run: `python -m pytest tests/test_two_qutrit_vertical_slice_e2e.py tests/test_experiment_runner.py tests/test_reference_measurement_integration.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/qudits_on_qubits tests/test_two_qutrit_vertical_slice_e2e.py
git commit -m "feat: run two-qutrit Bell vertical slice"
```

### Task 4: Installed CLI, clean-room verifier, and troubleshooting

**Files:**
- Create: `tests/test_two_qutrit_vertical_slice_cli.py`
- Create: `src/qudits_on_qubits/vertical_slice/cli.py`
- Modify: `pyproject.toml`
- Create: `scripts/verify_two_qutrit_clean_room.ps1`
- Create: `docs/two_qutrit_bell_vertical_slice.md`
- Modify: `README.md`

- [ ] **Step 1: Write failing CLI test**

```python
def test_cli_prints_result_and_manifest(tmp_path, capsys):
    exit_code = main(["--shots", "256", "--seed", "42", "--output-root", str(tmp_path)])
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "status=completed" in output
    assert "benchmark=two_qutrit" in output
    assert "manifest=" in output
```

- [ ] **Step 2: Run test and verify RED**

Run: `python -m pytest tests/test_two_qutrit_vertical_slice_cli.py -q`

Expected: import fails because CLI is absent.

- [ ] **Step 3: Implement installed command**

```toml
[project.scripts]
qoq-two-qutrit-bell = "qudits_on_qubits.vertical_slice.cli:entrypoint"
```

```python
def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    completed = run_vertical_slice(two_qutrit_bell_spec(
        shots=args.shots,
        seed=args.seed,
        output_root=args.output_root,
    ))
    print("status=completed")
    print("benchmark=two_qutrit")
    print(f"bell_unconditional={completed.result['bell_unconditional']['real']:.10f}")
    print(f"leakage_rate={completed.result['leakage_rate']:.10f}")
    print(f"manifest={completed.artifact_dir / 'run-manifest.json'}")
    return 0
```

- [ ] **Step 4: Add clean-room verification and docs**

`scripts/verify_two_qutrit_clean_room.ps1` must build a wheel, create a new venv, install the wheel, invoke `qoq-two-qutrit-bell`, parse the emitted manifest path, and assert `schema_version == "run-manifest-v1"`, `status == "completed"`, `circuit_count == 9`, zero leakage, and Bell real part within `0.15` of `6.0`.

`docs/two_qutrit_bell_vertical_slice.md` must include exact Linux/macOS and PowerShell setup commands, expected output fields/ranges, artifact list, interpretation, and fixes for Python version, Aer installation, wheel build, permissions, and numerical tolerance failures.

- [ ] **Step 5: Run CLI and clean-room checks**

Run: `python -m pytest tests/test_two_qutrit_vertical_slice_cli.py -q`

Expected: all CLI tests pass.

Run: `powershell -ExecutionPolicy Bypass -File scripts/verify_two_qutrit_clean_room.ps1`

Expected: exits 0 and prints `clean_room_vertical_slice=passed`.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml README.md docs/two_qutrit_bell_vertical_slice.md scripts/verify_two_qutrit_clean_room.ps1 src/qudits_on_qubits/vertical_slice/cli.py tests/test_two_qutrit_vertical_slice_cli.py
git commit -m "docs: add clean-room Bell vertical slice"
```

### Task 5: Full verification and coverage

**Files:**
- Modify only files required by failures found below.

- [ ] **Step 1: Run focused coverage**

Run: `python -m pytest tests/test_vertical_slice_models.py tests/test_vertical_slice_bell.py tests/test_two_qutrit_vertical_slice_e2e.py tests/test_two_qutrit_vertical_slice_cli.py --cov=qudits_on_qubits.vertical_slice --cov-report=term-missing --cov-fail-under=80`

Expected: all tests pass and vertical-slice coverage is at least 80%.

- [ ] **Step 2: Run full suite**

Run: `python -m pytest -q`

Expected: all tests pass.

- [ ] **Step 3: Review diff and SZY-42 acceptance criteria**

Run: `git diff --check && git diff --stat && git status --short`

Expected: no whitespace errors; only SZY-42 files plus preserved pre-existing `M2.1_API_DEFINITIONS.md` appear.

Acceptance checklist:

- clean environment can install built wheel and run packaged command;
- flow visibly includes logical qutrit reference, explicit encoding, qubit source/measurement/compiled circuits, ideal Aer, decoded Bell result;
- expected output documented and tested;
- JSON artifacts and `run-manifest-v1` link through SHA-256;
- troubleshooting is public;
- implementation labels two-qutrit Bell as reference benchmark, not framework boundary.
