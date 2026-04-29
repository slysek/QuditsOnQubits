import os
from pathlib import Path


_PACKAGE_ROOT = Path(__file__).resolve().parent


def package_root():
    return str(_PACKAGE_ROOT)


def default_results_root():
    return str(_PACKAGE_ROOT / "results")


def stage_label(stage):
    if isinstance(stage, str) and stage.startswith("stage"):
        return stage
    return f"stage{int(stage)}"


def stage_output_dir(state_name, stage, output_root=None):
    root = output_root or default_results_root()
    return os.path.join(root, state_name, stage_label(stage))


def stage_file_prefix(state_name, stage):
    return f"encoding_search_v2_{state_name}_{stage_label(stage)}"


def stage_results_csv_path(state_name, stage, output_root=None):
    output_dir = stage_output_dir(state_name, stage, output_root=output_root)
    return os.path.join(output_dir, f"{stage_file_prefix(state_name, stage)}_results.csv")
