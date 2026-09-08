"""Raw IBM SamplerV2 execution with explicit per-circuit shots and joint counts."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
from io import BytesIO
from itertools import islice
import math
import re
from typing import Any, Mapping, Sequence

from qiskit import QuantumCircuit, qpy, transpile
from qiskit.primitives.containers.sampler_pub import SamplerPub

from ..._ibm_runtime import runtime_account_options
from ..errors import (
    BackendCompatibilityError,
    BackendUnavailableError,
    JobResultError,
    JobSubmissionError,
    OptionalDependencyError,
)
from ..ibm_spec import IBMHardware
from ..models import TranspilationConfig
from ..raw_evidence import attach_raw_evidence
from ..safety import unsafe_persisted_text, validate_persisted_strings
from .base import (
    Availability,
    BackendCapabilities,
    BackendIdentity,
    BaseBackendAdapter,
    CompiledBatch,
    ExecutionResult,
    SubmittedJob,
    _exception_name,
    _extract_job_id,
    _positive_integer,
    _result_status,
    _safe_identifier,
    _validate_counts,
    _validated_circuit_tuple,
)


def _sampler_factory(**kwargs: Any) -> Any:
    from qiskit_ibm_runtime import SamplerV2

    return SamplerV2(**kwargs)


def _positive_limit(value: Any) -> int | None:
    return value if type(value) is int and value > 0 else None


def _name(backend: Any) -> str | None:
    try:
        value = getattr(backend, "name", None)
        return value() if callable(value) else value
    except MemoryError:
        raise
    except Exception as error:
        raise BackendUnavailableError(
            f"IBM backend identity is unavailable ({_exception_name(error)})"
        ) from None


def _configuration(backend: Any) -> Any:
    try:
        configuration = getattr(backend, "configuration", None)
        return configuration() if callable(configuration) else backend
    except MemoryError:
        raise
    except Exception as error:
        raise BackendUnavailableError(
            f"IBM backend limits are unavailable ({_exception_name(error)})"
        ) from None


def _registers(circuit: QuantumCircuit) -> tuple[dict[str, Any], ...]:
    registers = tuple(
        {"name": register.name, "clbits": tuple(circuit.find_bit(bit).index for bit in register)}
        for register in circuit.cregs
    )
    indices = [index for register in registers for index in register["clbits"]]
    if sorted(indices) != list(range(circuit.num_clbits)):
        raise BackendCompatibilityError("IBM circuits require disjoint registers covering all classical bits")
    validate_persisted_strings(
        registers, description="IBM register names", error_type=BackendCompatibilityError
    )
    return registers


def _circuit_batch(circuits: Sequence[Any]) -> tuple[QuantumCircuit, ...]:
    batch = _validated_circuit_tuple(circuits)
    for circuit in batch:
        if not isinstance(circuit, QuantumCircuit):
            raise BackendCompatibilityError("IBM requires QuantumCircuit inputs")
        if circuit.num_parameters:
            raise BackendCompatibilityError("IBM raw execution requires fully bound circuits")
        if not circuit.num_clbits or not any(item.operation.name == "measure" for item in circuit.data):
            raise BackendCompatibilityError("IBM Sampler circuits must contain measurements")
        _registers(circuit)
    return batch


def _circuit_digest(circuit: QuantumCircuit) -> str:
    # Provider metadata and transpiler layout may be enriched after submission.
    # Compare the actual instructions/registers, without those volatile fields.
    comparable = QuantumCircuit(*circuit.qregs, *circuit.cregs, name="provenance")
    comparable.global_phase = circuit.global_phase
    for instruction in circuit.data:
        comparable.append(
            instruction.operation,
            [comparable.qubits[circuit.find_bit(bit).index] for bit in instruction.qubits],
            [comparable.clbits[circuit.find_bit(bit).index] for bit in instruction.clbits],
        )
    buffer = BytesIO()
    qpy.dump(comparable, buffer)
    return sha256(buffer.getvalue()).hexdigest()


def _provenance(circuits: Sequence[QuantumCircuit]) -> dict[str, Any]:
    return {
        "register_layouts": [_registers(circuit) for circuit in circuits],
        "circuit_digests": [_circuit_digest(circuit) for circuit in circuits],
    }


def _raw_options(options: Mapping[str, Any] | None, backend: Any) -> dict[str, Any]:
    """Allow only documented execution controls that preserve raw classified shots."""
    if options is None:
        options = {}
    if not isinstance(options, Mapping):
        raise BackendCompatibilityError("IBM run options must be a mapping")
    validate_persisted_strings(
        options, description="IBM run options", error_type=BackendCompatibilityError
    )
    if set(options) - {"execution", "max_execution_time", "dynamical_decoupling", "twirling"}:
        raise BackendCompatibilityError("IBM raw execution received unsupported run options or duplicate shots")
    raw: dict[str, Any] = {
        "dynamical_decoupling": {"enable": False},
        "twirling": {"enable_gates": False, "enable_measure": False},
        "execution": {"meas_type": "classified", "init_qubits": True},
    }
    for group in ("dynamical_decoupling", "twirling"):
        supplied = options.get(group, {})
        if not isinstance(supplied, Mapping) or any(
            key not in raw[group] or value is not False for key, value in supplied.items()
        ):
            raise BackendCompatibilityError("IBM raw execution forbids mitigation, decoupling, and twirling")
    if "max_execution_time" in options:
        value = options["max_execution_time"]
        if type(value) is not int or value <= 0:
            raise BackendCompatibilityError("IBM max_execution_time must be a positive integer")
        raw["max_execution_time"] = value
    execution = options.get("execution", {})
    if not isinstance(execution, Mapping) or set(execution) - {"init_qubits", "rep_delay", "meas_type"}:
        raise BackendCompatibilityError("IBM execution options contain unsupported fields")
    if "init_qubits" in execution and execution["init_qubits"] is not True:
        raise BackendCompatibilityError("IBM independent shots require init_qubits=True")
    if "meas_type" in execution and execution["meas_type"] != "classified":
        raise BackendCompatibilityError("IBM raw counts require classified measurements")
    if "rep_delay" in execution:
        delay = execution["rep_delay"]
        if isinstance(delay, bool) or not isinstance(delay, (int, float)) or not math.isfinite(delay) or delay < 0:
            raise BackendCompatibilityError("IBM rep_delay must be a finite non-negative number")
        config = _configuration(backend)
        interval = getattr(config, "rep_delay_range", None)
        if getattr(config, "dynamic_reprate_enabled", None) is not True or interval is None:
            raise BackendCompatibilityError("IBM backend does not advertise repetition delay controls")
        if len(interval) != 2 or not interval[0] <= delay <= interval[1]:
            raise BackendCompatibilityError("IBM rep_delay is outside the backend range")
        raw["execution"]["rep_delay"] = delay
    return raw


class IBMAdapter(BaseBackendAdapter):
    """One lazily connected IBM backend, with injectable local provider doubles."""

    def __init__(
        self,
        spec: IBMHardware,
        *,
        service: Any = None,
        sampler_factory: Any = None,
        backend: Any = None,
        transpiler: Any = None,
    ) -> None:
        if not isinstance(spec, IBMHardware):
            raise BackendCompatibilityError("IBMAdapter requires an IBMHardware specification")
        for factory in (sampler_factory, transpiler):
            if factory is not None and not callable(factory):
                raise BackendCompatibilityError("IBM factories must be callable")
        self._spec = spec
        self._service = service
        self._backend = backend
        self._sampler_factory = sampler_factory or _sampler_factory
        self._transpiler = transpiler or transpile
        self._identity: BackendIdentity | None = None

    def _service_instance(self) -> Any:
        if self._service is None:
            try:
                from qiskit_ibm_runtime import QiskitRuntimeService

                self._service = QiskitRuntimeService(**runtime_account_options(
                    account_name=self._spec.account_name, instance=self._spec.instance
                ))
            except (ImportError, ModuleNotFoundError) as error:
                raise OptionalDependencyError(
                    f"IBMHardware requires qiskit-ibm-runtime ({_exception_name(error)})"
                ) from None
            except MemoryError:
                raise
            except Exception as error:
                raise BackendUnavailableError(
                    f"could not initialize IBM Runtime service ({_exception_name(error)})"
                ) from None
        return self._service

    @property
    def backend(self) -> Any:
        if self._backend is None:
            try:
                self._backend = self._service_instance().backend(self._spec.device)
            except (BackendUnavailableError, OptionalDependencyError, MemoryError):
                raise
            except Exception as error:
                raise BackendUnavailableError(
                    f"could not load IBM backend ({_exception_name(error)})"
                ) from None
        if self._backend is None or _name(self._backend) != self._spec.device:
            raise BackendCompatibilityError("resolved IBM backend does not match requested device")
        return self._backend

    def resolve(self) -> BackendIdentity:
        backend = self.backend
        if self._identity is None:
            version = getattr(backend, "backend_version", None)
            version = version if (
                isinstance(version, str) and version.strip() and len(version) <= 512
                and not unsafe_persisted_text(version)
            ) else None
            self._identity = BackendIdentity(
                "ibm", self._spec.device, provider="ibm", version=version,
                metadata={"primitive": "SamplerV2", "raw_counts": True},
            )
        return self._identity

    def capabilities(self) -> BackendCapabilities:
        backend = self.backend
        return BackendCapabilities(
            local=False, supports_resume=True,
            max_circuits=_positive_limit(getattr(backend, "max_circuits", None)),
            metadata={"primitive": "SamplerV2", "explicit_pub_shots": True},
        )

    def availability(self) -> Availability:
        try:
            backend = self.backend
            status = getattr(backend, "status", None)
            if callable(status) and getattr(status(), "operational", None) is False:
                return Availability(False, "IBM backend is not operational")
        except MemoryError:
            raise
        except Exception as error:
            return Availability(False, f"IBM backend is unavailable ({_exception_name(error)})")
        return Availability(True)

    def preflight(self, circuits: Sequence[Any], shots: int) -> None:
        batch = _circuit_batch(circuits)
        super().preflight(batch, shots)
        backend = self.backend
        capacity = _positive_limit(getattr(backend, "num_qubits", None))
        if capacity is not None and any(circuit.num_qubits > capacity for circuit in batch):
            raise BackendCompatibilityError("circuit exceeds IBM backend qubit capacity")
        max_shots = _positive_limit(getattr(_configuration(backend), "max_shots", None))
        if max_shots is not None and shots > max_shots:
            raise BackendCompatibilityError("shots exceed IBM backend shot limit")

    def compile(self, circuits: Sequence[Any], config: TranspilationConfig) -> CompiledBatch:
        batch = _circuit_batch(circuits)
        if not isinstance(config, TranspilationConfig):
            raise BackendCompatibilityError("compile requires TranspilationConfig")
        options = {key: value for key, value in config.to_safe_dict().items() if value is not None}
        try:
            compiled = self._transpiler(list(batch), backend=self.backend, **options)
            compiled = _circuit_batch(compiled if isinstance(compiled, (tuple, list)) else (compiled,))
        except (BackendCompatibilityError, BackendUnavailableError, OptionalDependencyError, MemoryError):
            raise
        except Exception as error:
            raise BackendCompatibilityError(f"could not compile IBM ISA circuits ({_exception_name(error)})") from None
        if len(compiled) != len(batch) or any(
            _registers(before) != _registers(after) for before, after in zip(batch, compiled)
        ):
            raise BackendCompatibilityError("IBM compilation changed circuit count or classical mapping")
        return CompiledBatch(compiled, self.resolve(), {"transpilation": options, **_provenance(compiled)})

    def submit(
        self, circuits: Sequence[Any], shots: int, options: Mapping[str, Any] | None = None
    ) -> SubmittedJob:
        batch = _circuit_batch(circuits)
        self.preflight(batch, shots)
        raw_options = _raw_options(options, self.backend)
        metadata = {"raw_options": raw_options, **_provenance(batch)}
        try:
            sampler = self._sampler_factory(mode=self.backend, options=raw_options)
            handle = sampler.run([(circuit, None, shots) for circuit in batch])
            job_id = _extract_job_id(handle, allow_local_fallback=False)
        except MemoryError:
            raise
        except (ImportError, ModuleNotFoundError) as error:
            raise OptionalDependencyError(f"IBM SamplerV2 is unavailable ({_exception_name(error)})") from None
        except Exception as error:
            sanitized = JobSubmissionError(f"IBM SamplerV2 submission failed ({_exception_name(error)})")
            sanitized.provider_exception_type = _exception_name(error)
            raise sanitized from None
        return SubmittedJob(job_id, handle, self.resolve(), len(batch), shots, metadata)

    def validate_run_options(self, options: Mapping[str, Any] | None) -> None:
        """Validate raw provider controls before a durable submission attempt starts."""
        _raw_options(options, self.backend)

    def restore_job(
        self, job_id: str, *, circuit_count: int | None = None, shots: int | None = None,
        circuits: Sequence[Any] | None = None,
    ) -> SubmittedJob:
        _safe_identifier(job_id, "job_id")
        _positive_integer(circuit_count, "circuit_count", optional=True)
        _positive_integer(shots, "shots", optional=True)
        expected = _circuit_batch(circuits) if circuits is not None else None
        if expected is not None:
            if circuit_count is not None and circuit_count != len(expected):
                raise BackendCompatibilityError("expected IBM circuit count is inconsistent")
            circuit_count = len(expected)
        try:
            handle = self._service_instance().job(job_id)
            if _extract_job_id(handle, allow_local_fallback=False) != job_id:
                raise JobResultError("restored IBM job ID does not match requested job")
            backend = getattr(handle, "backend", None)
            backend = backend() if callable(backend) else backend
            if backend is None or _name(backend) != self._spec.device:
                raise JobResultError("restored IBM job backend does not match requested device")
            primitive = getattr(handle, "primitive_id", None)
            if primitive is not None and primitive != "sampler":
                raise JobResultError("restored IBM job is not a Sampler job")
            inputs = getattr(handle, "inputs", None)
            inputs = inputs() if callable(inputs) else inputs
            raw_verified = _restored_raw_options(inputs, self.backend)
            remote_circuits, remote_shots = _input_pubs(inputs)
            circuits_verified = remote_circuits is not None and expected is not None
            if remote_circuits is not None:
                if circuit_count is not None and len(remote_circuits) != circuit_count:
                    raise JobResultError("restored IBM PUB count does not match expected circuit count")
                circuit_count = len(remote_circuits)
                if any(value is None for value in remote_shots):
                    raise JobResultError("restored IBM PUBs lack explicit shot provenance")
                if len(set(remote_shots)) != 1 or (shots is not None and remote_shots[0] != shots):
                    raise JobResultError("restored IBM PUB shots do not match expected shots")
                shots = remote_shots[0]
                if expected is not None and any(
                    _circuit_digest(left) != _circuit_digest(right)
                    for left, right in zip(expected, remote_circuits)
                ):
                    raise JobResultError("restored IBM circuit provenance does not match expected circuits")
                expected = remote_circuits
            metadata = {
                "restored": True, "input_provenance_available": remote_circuits is not None,
                "circuit_provenance_verified": circuits_verified, "raw_options_verified": raw_verified,
            }
            if expected is not None:
                metadata.update(_provenance(expected))
        except (JobResultError, OptionalDependencyError, BackendUnavailableError, MemoryError):
            raise
        except Exception as error:
            raise JobResultError(f"could not restore IBM job ({_exception_name(error)})") from None
        return SubmittedJob(job_id, handle, self.resolve(), circuit_count, shots, metadata)

    def verify_restored_job(
        self, submitted: SubmittedJob, *, circuits: Sequence[Any], shots: int
    ) -> None:
        """Require provider evidence before attaching a formerly unknown submission."""
        if not isinstance(submitted, SubmittedJob) or submitted.target_identity != self.resolve():
            raise BackendCompatibilityError("IBM recovery requires a matching restored job")
        verified = self.restore_job(submitted.job_id, circuits=circuits, shots=shots)
        if not verified.metadata["circuit_provenance_verified"] or not verified.metadata["raw_options_verified"]:
            raise BackendCompatibilityError(
                "IBM unknown-job recovery requires provider evidence of circuits, shots, backend, and raw options"
            )

    def result(self, submitted: SubmittedJob, timeout: float | None = None) -> ExecutionResult:
        if not isinstance(submitted, SubmittedJob):
            raise JobResultError("result requires a SubmittedJob")
        if submitted.target_identity != self.resolve():
            raise BackendCompatibilityError("submitted job target does not match this adapter")
        if timeout is not None and (
            isinstance(timeout, bool) or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout) or timeout <= 0
        ):
            raise JobResultError("result timeout must be a positive finite number or None")
        try:
            raw_result = submitted.handle.result(**({"timeout": timeout} if timeout is not None else {}))
            entries = tuple(raw_result)
        except MemoryError:
            raise
        except Exception as error:
            raise JobResultError(f"could not retrieve IBM raw result ({_exception_name(error)})") from None
        counts = _decoded_entries(entries, submitted)
        return ExecutionResult(
            counts, submitted.job_id, submitted.target_identity,
            status=_result_status(submitted.handle, raw_result),
            metadata={
                "primitive": "SamplerV2", "raw_counts": True,
                "joint_register_counts": True, "execution_order": "unknown",
            },
        )


def _decoded_entries(entries: tuple[Any, ...], submitted: SubmittedJob) -> tuple[Mapping[str, int], ...]:
    """Decode actual shot records before checking the requested cardinalities."""
    layouts = submitted.metadata.get("register_layouts")
    failure = None
    if not entries or (submitted.circuit_count is not None and len(entries) != submitted.circuit_count):
        failure = JobResultError("IBM result PUB count does not match expected circuit count")
    elif layouts is not None and len(layouts) != len(entries):
        failure = JobResultError("IBM result register provenance has the wrong circuit count")
    counts = []
    diagnostics = {"pubs": [], "joint_counts_unavailable": False}
    diagnostic_budget = {"bits": 65536, "registers": 64}
    for index, entry in enumerate(entries):
        layout = layouts[index] if layouts is not None and index < len(layouts) else None
        try:
            # In particular, fewer returned shots must remain observable evidence.
            histogram = _joint_counts(entry, layout, expected_shots=None)
            counts.append(histogram)
            if submitted.shots is not None and sum(histogram.values()) != submitted.shots:
                failure = failure or JobResultError("IBM result counts do not sum to expected shots")
        except MemoryError:
            raise
        except Exception as error:
            failure = failure or (
                error if isinstance(error, JobResultError)
                else JobResultError(f"could not decode IBM raw result ({_exception_name(error)})")
            )
            # A null preserves this PUB's position without inventing correlations.
            counts.append(None)
            diagnostics["joint_counts_unavailable"] = True
            if len(diagnostics["pubs"]) < 64:
                diagnostics["pubs"].append(_register_diagnostics(entry, index, diagnostic_budget))
    if failure is not None:
        raise attach_raw_evidence(
            failure, job_id=submitted.job_id, target_identity=submitted.target_identity,
            counts=counts, diagnostics=diagnostics,
        ) from None
    return tuple(counts)


def _register_diagnostics(entry: Any, pub_index: int, budget: dict[str, int]) -> dict[str, Any]:
    """Bounded binary-only snapshots; register shots remain explicitly separate."""
    output: dict[str, Any] = {"pub_index": pub_index, "registers": []}
    try:
        data = getattr(entry, "data", None)
        keys = getattr(data, "keys", None)
        names = islice(keys() if callable(keys) else vars(data), 64)
        for register_index, name in enumerate(names):
            if budget["registers"] <= 0 or budget["bits"] <= 0:
                output["truncated"] = True
                break
            if not isinstance(name, str) or name.startswith("_"):
                continue
            record: dict[str, Any] = {"register_index": register_index}
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", name):
                record["name"] = name
            register = data[name] if hasattr(data, "__getitem__") else getattr(data, name)
            for attribute in ("num_bits", "num_shots"):
                value = getattr(register, attribute, None)
                if type(value) is int and 0 <= value <= 2**31:
                    record[attribute] = value
            getter = getattr(register, "get_bitstrings", None)
            if callable(getter) and record.get("num_bits", 1) <= 1024:
                # BitArray supports slicing before allocating Python shot strings.
                slicer = getattr(register, "slice_shots", None)
                if callable(slicer) and "num_shots" in record:
                    register = slicer(range(min(256, record["num_shots"])))
                    getter = register.get_bitstrings
                observed = []
                for bits in islice(getter(), 256):
                    if not isinstance(bits, str) or not bits or len(bits) > 1024 or set(bits) - {"0", "1"}:
                        record["unsafe_or_malformed_data_omitted"] = True
                        continue
                    if len(bits) > budget["bits"]:
                        record["truncated"] = True
                        break
                    observed.append(bits)
                    budget["bits"] -= len(bits)
                record["bitstrings"] = observed
                if record.get("num_shots", len(observed)) > len(observed):
                    record["truncated"] = True
            else:
                record["shot_bitstrings_unavailable"] = True
            output["registers"].append(record)
            budget["registers"] -= 1
    except MemoryError:
        raise
    except Exception:
        # No provider exception messages or opaque SDK objects cross this boundary.
        output["additional_register_data_unavailable"] = True
    return output


def _restored_raw_options(inputs: Any, backend: Any) -> bool:
    if not isinstance(inputs, Mapping) or "options" not in inputs:
        return False
    options = inputs["options"]
    try:
        _raw_options(options, backend)
    except BackendCompatibilityError:
        raise JobResultError("restored IBM job options are incompatible with raw execution") from None
    return (
        isinstance(options, Mapping)
        and options.get("dynamical_decoupling", {}).get("enable") is False
        and options.get("twirling", {}).get("enable_gates") is False
        and options.get("twirling", {}).get("enable_measure") is False
        and options.get("execution", {}).get("meas_type") == "classified"
        and options.get("execution", {}).get("init_qubits") is True
    )


def _input_pubs(inputs: Any) -> tuple[tuple[QuantumCircuit, ...] | None, tuple[int | None, ...]]:
    if inputs is None or (isinstance(inputs, Mapping) and "pubs" not in inputs):
        return None, ()
    if not isinstance(inputs, Mapping) or not isinstance(inputs["pubs"], (tuple, list)) or not inputs["pubs"]:
        raise JobResultError("restored IBM job has invalid PUB provenance")
    circuits, shot_values = [], []
    for pub in inputs["pubs"]:
        try:
            pub = SamplerPub.coerce(pub)
            pub.validate()
        except MemoryError:
            raise
        except Exception as error:
            raise JobResultError(f"restored IBM PUB is invalid ({_exception_name(error)})") from None
        circuit, shots = pub.circuit, pub.shots
        if pub.shape:
            raise JobResultError("restored IBM PUB contains multiple parameter bindings")
        _circuit_batch((circuit,))
        if shots is not None and (type(shots) is not int or shots <= 0):
            raise JobResultError("restored IBM PUB has invalid shots")
        circuits.append(circuit)
        shot_values.append(shots)
    return tuple(circuits), tuple(shot_values)


def _joint_counts(entry: Any, layout: Any, expected_shots: int | None) -> Mapping[str, int]:
    data = getattr(entry, "data", None)
    if data is None:
        raise JobResultError("IBM result PUB has no data")
    keys = getattr(data, "keys", None)
    names = tuple(keys()) if callable(keys) else tuple(
        name for name in vars(data) if not name.startswith("_")
    )
    if not names:
        raise JobResultError("IBM result PUB has no classical registers")
    if layout is None:
        if len(names) != 1:
            raise JobResultError("IBM multiple-register results require circuit register provenance")
        register = data[names[0]] if hasattr(data, "__getitem__") else getattr(data, names[0])
        width = getattr(register, "num_bits", None)
        if _positive_limit(width) is None:
            raise JobResultError("IBM result register width is unavailable")
        layout = ({"name": names[0], "clbits": tuple(range(width))},)
    if set(names) != {register["name"] for register in layout}:
        raise JobResultError("IBM result registers do not match circuit provenance")
    width = sum(len(register["clbits"]) for register in layout)
    per_register = []
    for record in layout:
        name = record["name"]
        register = data[name] if hasattr(data, "__getitem__") else getattr(data, name)
        if getattr(register, "shape", ()) != ():
            raise JobResultError("IBM raw result PUB must contain exactly one circuit execution")
        if getattr(register, "num_bits", len(record["clbits"])) != len(record["clbits"]):
            raise JobResultError("IBM result register width does not match circuit provenance")
        getter = getattr(register, "get_bitstrings", None)
        if not callable(getter):
            if len(layout) != 1 or tuple(record["clbits"]) != tuple(range(width)):
                raise JobResultError("IBM joint counts require corresponding per-shot register bitstrings")
            getter = getattr(register, "get_counts", None)
            if not callable(getter):
                raise JobResultError("IBM result register does not expose raw counts")
            counts = _validate_counts(getter())
            if any(len(key) != width for key in counts):
                raise JobResultError("IBM count bit width does not match circuit provenance")
            break
        strings = tuple(getter())
        if not strings or any(
            not isinstance(value, str) or len(value) != len(record["clbits"])
            or set(value) - {"0", "1"} for value in strings
        ):
            raise JobResultError("IBM register contains malformed bitstrings")
        per_register.append(strings)
    else:
        lengths = {len(strings) for strings in per_register}
        if len(lengths) != 1:
            raise JobResultError("IBM classical registers have inconsistent shot counts")
        counts = Counter()
        for shot in zip(*per_register):
            global_bits = ["0"] * width
            for record, register_bits in zip(layout, shot):
                for index, bit in zip(record["clbits"], reversed(register_bits)):
                    global_bits[index] = bit
            counts["".join(reversed(global_bits))] += 1
        counts = _validate_counts(counts)
    if expected_shots is not None and sum(counts.values()) != expected_shots:
        raise JobResultError("IBM result counts do not sum to expected shots")
    return counts


__all__ = ["IBMAdapter"]
