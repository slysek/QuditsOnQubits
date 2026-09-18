"""Data loading and plotting for the short encoding benchmark notebook."""
from dataclasses import dataclass, field
import hashlib
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import zipfile

from qudits_on_qubits.benchmarks import SchmidtTheta, ThetaContinuationSynthesis
from qudits_on_qubits.benchmarks.theta_continuation.artifacts import RunStore
from qudits_on_qubits.benchmarks.theta_continuation.models import ThetaBenchmarkConfig

ARCHIVE_SHA256 = "c7b346cd4c118c02a0e3d2008dca38ca2a64a165309bb1b3d114f81ae2fc6f1b"


@dataclass
class ThetaDemo:
    """Keep the verified source alive while its synthesis strategy is in use."""
    family: SchmidtTheta
    synthesis: ThetaContinuationSynthesis
    _directory: TemporaryDirectory = field(repr=False)

    def close(self):
        self._directory.cleanup()


def load_theta_demo(archive_path):
    """Load the bundled original gates without changing or resynthesizing them."""
    content = Path(archive_path).read_bytes()
    if hashlib.sha256(content).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("Bundled theta archive SHA256 mismatch.")
    directory = TemporaryDirectory()
    source = Path(directory.name).resolve()
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            for member in archive.infolist():
                if not (source / member.filename).resolve().is_relative_to(source):
                    raise ValueError("Archive member escapes its destination.")
            archive.extractall(source)
        store = RunStore.open(source)
        config = ThetaBenchmarkConfig.from_dict(store.manifest["config"])
        synthesis = ThetaContinuationSynthesis(config, source_run=source)
        return ThetaDemo(SchmidtTheta(points=config.theta_points), synthesis, directory)
    except BaseException:
        directory.cleanup()
        raise


def plot_pareto(result):
    """Show a 2D projection of the pipeline's three-objective Pareto front."""
    import matplotlib.pyplot as plt
    from IPython.display import Image

    accepted = result.statistics.loc[result.statistics.pareto_eligible]
    baseline = accepted.loc[accepted.is_baseline]
    front = result.pareto_front
    figure, axis = plt.subplots(figsize=(7, 4), layout="constrained")
    try:
        axis.scatter(accepted.mean_two_qubit_gate_count, accepted.mean_depth,
                     color="lightgray", label="Candidates", s=40)
        axis.scatter(front.mean_two_qubit_gate_count, front.mean_depth,
                     color="tab:blue", label="Pareto front", s=70)
        axis.scatter(baseline.mean_two_qubit_gate_count, baseline.mean_depth,
                     color="black", marker="*", label="Canonical baseline", s=160)
        axis.set(xlabel="Mean native 2q gates after routing", ylabel="Mean depth",
                 title=result.manifest["backend"]["name"])
        axis.legend()
        axis.grid(alpha=0.2)
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=140)
        return Image(data=buffer.getvalue())
    finally:
        plt.close(figure)
