"""Select a verified circuit from previous campaigns without resuming their searches."""

import json
from pathlib import Path
import tempfile
from typing import Iterable

import numpy as np

from .artifacts import file_hash
from .checkpoints import latest_checkpoint
from .oe_agent import builder_source, with_builder
from .oe_candidates import MAX_BYTES, load_skeleton, skeleton_program
from .oe_evaluator import _warm_start, assess, rank_key
from .records import Record, _load_circuit, _matched, _READ_ERRORS, measure
from .skeleton_fit import skeleton_of
from .state_workloads import layout, party_encodings

VERIFY_SECONDS = 30


def _completed(directory: Path, config: dict) -> Record:
    state = config["state"]
    n, _, q = layout(state)
    summary = json.loads((directory / "summary.json").read_text())
    names = ["best.qpy", "best_encodings.npy"]
    if (directory / "decoder.qpy").is_file():
        names.append("decoder.qpy")
    elif config["encoding_mode"] == "free":
        raise ValueError("The saved free encoding has no decoder")
    for name in names:
        if file_hash((directory / name).read_bytes()) != summary["files"][name]:
            raise ValueError("The previous campaign file hash does not match")
    circuit = _load_circuit(directory / "best.qpy", n * q)
    encodings = party_encodings(state, np.load(directory / "best_encodings.npy", allow_pickle=False))
    decoder = _load_circuit(directory / "decoder.qpy", n * q) if "decoder.qpy" in names else None
    metrics = measure(state, circuit, encodings, decoder, config.get("architecture"), config["objective"])
    if not metrics["is_valid"]:
        raise ValueError("The previous circuit does not prepare the state")
    program = None
    path = directory / "best_program.py"
    if config.get("representation") == "skeleton" and path.is_file() and path.stat().st_size <= MAX_BYTES:
        code = path.read_bytes()
        if (file_hash(code) == summary["files"].get(path.name)
                and tuple(load_skeleton(path, expected_qubits=n * q)) == tuple(skeleton_of(circuit))):
            program = code.decode("utf-8")
    return Record(directory, circuit, encodings, decoder, metrics, config, program)


def _population(directory: Path, config: dict) -> Record | None:
    checkpoint, _ = latest_checkpoint(directory, update_metadata=False)
    if checkpoint is None:
        return None
    metadata = json.loads((checkpoint / "metadata.json").read_text())
    saved = json.loads((checkpoint / "programs" / f"{metadata['best_program_id']}.json").read_text())
    if not saved["metrics"].get("is_valid"):
        return None
    code = saved["code"]
    if not isinstance(code, str) or len(code.encode()) > MAX_BYTES:
        raise ValueError("The previous program is invalid or too large")
    state, representation = config["state"], config["representation"]
    encoding = np.load(directory / "encoding.npy", allow_pickle=False)
    options = {"state": state, "encoding": encoding, "representation": representation,
               "encoding_mode": config["encoding_mode"], "objective": config["objective"],
               "architecture": config.get("architecture")}
    with tempfile.TemporaryDirectory(prefix="qoq-previous-") as temporary:
        path = Path(temporary) / "candidate.py"
        path.write_text(code, encoding="utf-8")
        result = assess(path, **options, x0=_warm_start(path, options, directory), max_nfev=1,
                        restarts=1, timeout_seconds=VERIFY_SECONDS, cpu_seconds=VERIFY_SECONDS)
        if not result.metrics["is_valid"]:
            return None
        program = None
        if representation == "skeleton" and result.angles is not None:
            program = with_builder(skeleton_program(result.circuit.num_qubits,
                load_skeleton(path, expected_qubits=result.circuit.num_qubits), result.angles), builder_source(code))
    return Record(directory, result.circuit, result.encodings, result.decoder, result.metrics, config, program)


def best_previous(roots: Iterable[Path], state: str, processor: str | None, settings: dict,
                  current: Record | None = None) -> Record | None:
    """Keep the best compatible result. Verify saved circuits and checkpoint programs before use."""
    paths = set()
    for root in dict.fromkeys(Path(root).resolve() for root in roots):
        for pattern in ("*/configuration.json", "*/*/configuration.json"):
            paths.update(path for path in root.glob(pattern) if path.resolve().is_relative_to(root))
    best = current
    best_score = (rank_key(measure(state, current.circuit, current.encodings, current.decoder,
                                   processor, settings["objective"]), settings["objective"])
                  if current is not None else None)
    for path in sorted(paths, key=lambda path: path.stat().st_mtime_ns, reverse=True):
        try:
            config = json.loads(path.read_text())
            if (config.get("state") != state or config.get("architecture") != processor
                    or _matched(config) != _matched(settings)):
                continue
            directory = path.parent
            record = (_completed(directory, config) if (directory / "summary.json").is_file()
                      else _population(directory, config))
            if record is None:
                continue
            score = rank_key(record.metrics, settings["objective"])
            if (best_score is None or score > best_score
                    or (score == best_score and best.program is None and record.program is not None)):
                best = record
                best_score = score
        except (*_READ_ERRORS, EOFError, TimeoutError, RuntimeError, AttributeError, OverflowError) as exc:
            print(f"Previous campaign {path.parent}: skipped ({type(exc).__name__}: {exc})", flush=True)
    return best
