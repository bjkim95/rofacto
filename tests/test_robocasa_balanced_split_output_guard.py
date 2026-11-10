from pathlib import Path
import sys

import pytest


ROBOCASA_DIR = Path(__file__).resolve().parents[1] / "data_processing" / "robocasa"
sys.path.insert(0, str(ROBOCASA_DIR))
import build_dit4dit_clipstart_task_balanced_split as balanced_split


def test_default_output_is_outside_data_root() -> None:
    assert balanced_split.DEFAULT_OUTPUT_DIR == (
        balanced_split.REPO_ROOT
        / "outputs"
        / "dataset_files"
        / "robocasa_dit4dit_clipstart_480x832_balanced24_task_success_balanced_split"
    )


def test_output_guard_rejects_data_root_and_symlink_target(tmp_path: Path) -> None:
    source_root = tmp_path / "world_model"
    source = source_root / "data"
    source.mkdir(parents=True)
    destination_root = tmp_path / "rofacto"
    destination_root.mkdir()
    data_link = destination_root / "data"
    data_link.symlink_to(source, target_is_directory=True)

    with pytest.raises(ValueError, match="must not be under data root"):
        balanced_split.validate_output_dir(data_link / "dataset_files", data_link)
    with pytest.raises(ValueError, match="must not be under data root"):
        balanced_split.validate_output_dir(source_root / "outputs", data_link)

    assert balanced_split.validate_output_dir(destination_root / "outputs", data_link) == (
        destination_root / "outputs"
    ).resolve()
