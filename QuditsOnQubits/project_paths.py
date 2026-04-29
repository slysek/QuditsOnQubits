from pathlib import Path
import re


_REPO_ROOT = Path(__file__).resolve().parent.parent


def repo_root():
    return str(_REPO_ROOT)


def repo_path(*parts):
    return str(_REPO_ROOT.joinpath(*parts))


def quantum_circuits_path(*parts):
    return repo_path("quantum_circuits", *parts)


def data_dir(*parts):
    return repo_path("data", *parts)


def benchmark_data_dir(*parts):
    return data_dir("benchmarks", *parts)


def benchmark_circuits_dir(*parts):
    return benchmark_data_dir("circuits", *parts)


def benchmark_results_path(mode):
    filenames = {
        "full": "benchmark_encoding_bases_full_results.csv",
        "original": "benchmark_encoding_bases_results.csv",
        "extended": "benchmark_encoding_bases_extended_results.csv",
    }
    return benchmark_data_dir(filenames.get(mode, f"benchmark_{mode}_results.csv"))


def benchmark_docs_dir(*parts):
    return repo_path("docs", "benchmarks", *parts)


_VALID_STATE_NAMES = {"two_qutrit", "ghz3", "ame43"}
_GHZ_STAR_STATE_RE = re.compile(r"^ghz_star_\d+$")


def benchmark_state_slug(state_name):
    if state_name in _VALID_STATE_NAMES:
        return state_name
    if _GHZ_STAR_STATE_RE.match(str(state_name)):
        return state_name
    if state_name not in _VALID_STATE_NAMES:
        raise ValueError(f"Unknown benchmark state: {state_name}")


def benchmark_state_results_path(state_name, mode):
    if benchmark_state_slug(state_name) == "ghz3":
        return benchmark_results_path(mode)
    filename = f"benchmark_encoding_bases_{state_name}_{mode}_results.csv"
    return benchmark_data_dir(filename)


def benchmark_state_circuits_dir(state_name, *parts):
    return benchmark_circuits_dir(benchmark_state_slug(state_name), *parts)


def multi_state_benchmark_report_path():
    return benchmark_docs_dir("benchmark_encoding_bases_multi_state_analysis.md")


def prepared_w_benchmark_data_dir(*parts):
    """Output directory for 'prepared_w_then_conjugated_entanglers' benchmark results."""
    return benchmark_data_dir("prepared_w_then_conjugated_entanglers_results", *parts)


def prepared_w_benchmark_results_path(state_name, mode):
    """CSV results path for the second-stage benchmark."""
    slug = benchmark_state_slug(state_name)
    filename = f"benchmark_prepared_w_{slug}_{mode}_results.csv"
    return prepared_w_benchmark_data_dir(filename)
