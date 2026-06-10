from __future__ import annotations

from pathlib import Path

import pytest

from Evo_1.dataset.config_utils import (
    resolve_dataset_config_paths,
    resolve_dataset_path,
    validate_dataset_config_structure,
)


def minimal_dataset_config(path: str = "../Evo1_training_dataset/Evo1_MetaWorld_Dataset") -> dict:
    return {
        "max_action_dim": 24,
        "max_state_dim": 24,
        "max_views": 3,
        "data_groups": {
            "metaworld_sawyer": {
                "Evo1_MetaWorld": {
                    "path": path,
                    "view_map": {"image_1": "observation.images.image"},
                }
            }
        },
    }


def test_resolve_dataset_path_uses_explicit_base_dir(tmp_path):
    resolved = resolve_dataset_path("../data/example", tmp_path / "Evo_1")

    assert resolved == (tmp_path / "data" / "example").resolve()


def test_resolve_dataset_config_paths_does_not_mutate_input(tmp_path):
    config = minimal_dataset_config()

    resolved = resolve_dataset_config_paths(config, tmp_path / "Evo_1")

    original_path = config["data_groups"]["metaworld_sawyer"]["Evo1_MetaWorld"]["path"]
    resolved_path = resolved["data_groups"]["metaworld_sawyer"]["Evo1_MetaWorld"]["path"]
    assert original_path == "../Evo1_training_dataset/Evo1_MetaWorld_Dataset"
    assert Path(resolved_path) == (tmp_path / "Evo1_training_dataset" / "Evo1_MetaWorld_Dataset").resolve()


def test_validate_dataset_config_structure_reports_dataset_count():
    assert validate_dataset_config_structure(minimal_dataset_config()) == 1


def test_validate_dataset_config_structure_rejects_missing_path():
    config = minimal_dataset_config()
    del config["data_groups"]["metaworld_sawyer"]["Evo1_MetaWorld"]["path"]

    with pytest.raises(ValueError, match="has no path"):
        validate_dataset_config_structure(config)
