from pathlib import Path
from argparse import Namespace

import pytest
import yaml

from training.wan.config_loader import (
    ExperimentConfigLoader,
    load_experiment_config,
    normalize_legacy_config,
)
from training.wan.inference import derive_condition_paths, resolve_output_dir
from training.wan.train_dwm_wan import build_dataset_init_kwargs


FIXTURES = Path(__file__).parent / "fixtures/wan"
SNAPSHOTS = [FIXTURES / "legacy_mesh.yaml", FIXTURES / "legacy_depth.yaml"]
LEGACY_DATA_KEYS = {
    "hand_video_subdir": "robot_mesh_video_subdir",
    "hand_video_latents_subdir": "robot_mesh_video_latents_subdir",
    "hand_disparity_subdir": "eef_depth_subdir",
    "hand_disparity_latents_subdir": "eef_depth_latents_subdir",
    "hand_disparity_npz_mode": "eef_depth_npz_mode",
}


@pytest.mark.parametrize("snapshot", SNAPSHOTS)
def test_real_snapshot_schema_is_normalized(snapshot):
    raw = yaml.safe_load(snapshot.read_text())
    config = load_experiment_config(str(snapshot), overrides=[])

    for legacy_key, canonical_key in LEGACY_DATA_KEYS.items():
        assert legacy_key not in config["data"]
        if legacy_key in raw["data"]:
            assert config["data"][canonical_key] == raw["data"][legacy_key]
        else:
            assert canonical_key not in config["data"]

    assert config["training"]["robot_dropout_prob"] == raw["training"]["hand_dropout_prob"]
    assert "hand_dropout_prob" not in config["training"]
    assert config["data"]["static_video_subdir"] == raw["data"]["static_video_subdir"]
    assert config["pipeline"]["type"] == "wan2.1_fun_inp_hand_concat"


def test_legacy_canonical_collision_rules():
    identical = normalize_legacy_config(
        {"data": {"hand_video_subdir": "mesh", "robot_mesh_video_subdir": "mesh"}}
    )
    assert identical == {"data": {"robot_mesh_video_subdir": "mesh"}}

    with pytest.raises(ValueError, match="Conflicting data configuration"):
        normalize_legacy_config(
            {"data": {"hand_video_subdir": "legacy", "robot_mesh_video_subdir": "canonical"}}
        )


def test_bare_inference_fallback_config_is_normalized():
    fallback_config = {
        "data": {
            "hand_video_subdir": "mesh",
            "hand_video_latents_subdir": "mesh_latents",
            "hand_disparity_subdir": "depth",
            "hand_disparity_latents_subdir": "depth_latents",
            "hand_disparity_npz_mode": "per_frame",
            "static_video_subdir": "static",
        },
        "training": {"hand_dropout_prob": 0.25},
        "pipeline": {"type": "wan2.1_fun_inp_hand_concat"},
    }

    normalized = normalize_legacy_config(fallback_config)
    assert normalized["data"] == {
        "robot_mesh_video_subdir": "mesh",
        "robot_mesh_video_latents_subdir": "mesh_latents",
        "eef_depth_subdir": "depth",
        "eef_depth_latents_subdir": "depth_latents",
        "eef_depth_npz_mode": "per_frame",
        "static_video_subdir": "static",
    }
    assert normalized["training"] == {"robot_dropout_prob": 0.25}
    assert normalized["pipeline"]["type"] == "wan2.1_fun_inp_hand_concat"
    assert fallback_config["data"]["hand_video_subdir"] == "mesh"
def test_explicit_output_override_only_rebases_output_role(tmp_path):
    snapshot = {
        "experiment": {"name": "snapshot", "output_dir": "snapshot-output"},
        "training": {"resume_from_checkpoint": "/read-only/checkpoint"},
        "pipeline": {"base_model_name_or_path": "/read-only/model"},
        "data": {"data_root": "/read-only/data"},
    }
    config_path = tmp_path / "snapshot.yaml"
    config_path.write_text(yaml.safe_dump(snapshot))

    config = load_experiment_config(str(config_path), output_dir="runtime-output")

    assert config["experiment"]["output_dir"] == "runtime-output"
    assert config["training"]["resume_from_checkpoint"] == "/read-only/checkpoint"
    assert config["pipeline"]["base_model_name_or_path"] == "/read-only/model"
    assert config["data"]["data_root"] == "/read-only/data"
@pytest.mark.parametrize("snapshot", SNAPSHOTS)
def test_legacy_overrides_update_normalized_snapshot_values(snapshot):
    config = load_experiment_config(str(snapshot), overrides=[])
    loader = ExperimentConfigLoader()

    updated = loader.apply_overrides(
        config,
        [
            "data.hand_video_subdir=runtime-mesh",
            "training.hand_dropout_prob=0.35",
        ],
    )

    assert updated["data"]["robot_mesh_video_subdir"] == "runtime-mesh"
    assert "hand_video_subdir" not in updated["data"]
    assert updated["training"]["robot_dropout_prob"] == 0.35
    assert "hand_dropout_prob" not in updated["training"]


def test_legacy_override_collision_in_one_layer_is_rejected():
    with pytest.raises(ValueError, match="Conflicting runtime override values"):
        ExperimentConfigLoader().apply_overrides(
            {"data": {"robot_mesh_video_subdir": "snapshot-mesh"}},
            [
                "data.hand_video_subdir=legacy-mesh",
                "data.robot_mesh_video_subdir=canonical-mesh",
            ],
        )


@pytest.mark.parametrize("snapshot", SNAPSHOTS)
def test_snapshot_condition_subdirs_reach_train_and_inference_consumers(snapshot):
    config = load_experiment_config(str(snapshot), overrides=[])
    data_config = config["data"]
    dataset_kwargs = build_dataset_init_kwargs(
        data_config,
        load_tensors=False,
        conditioning_mode=config["training"].get("conditioning_mode", "dwm"),
        use_preencoded_prompt_embeds=False,
        align_width_to_32=False,
    )

    canonical_keys = [
        canonical_key
        for canonical_key in LEGACY_DATA_KEYS.values()
        if canonical_key in data_config
    ]
    assert canonical_keys
    for canonical_key in canonical_keys:
        assert dataset_kwargs[canonical_key] == data_config[canonical_key]

    video_path = "droid/videos_clips81/example.mp4"
    condition_paths = derive_condition_paths(config, "/dataset-root", video_path)
    condition_root = Path("/dataset-root/droid")
    for canonical_key, path_key in (
        ("robot_mesh_video_subdir", "hand_video_path"),
        ("eef_depth_subdir", "eef_depth_video_path"),
    ):
        subdir = data_config.get(canonical_key)
        expected_path = condition_root / subdir / "example.mp4" if subdir else None
        assert condition_paths[path_key] == expected_path


def test_default_output_dir_is_local_and_checkpoint_outputs_are_rejected(tmp_path, monkeypatch):
    checkpoint_path = Path(
        "/example/outputs/checkpoint-100"
    )
    monkeypatch.chdir(tmp_path)
    args = Namespace(checkpoint_path=str(checkpoint_path), output_dir=None)

    assert resolve_output_dir(args) == tmp_path / "outputs/inference/checkpoint-100"

    args.output_dir = str(checkpoint_path / "eval")
    with pytest.raises(ValueError, match="must not be inside the checkpoint_path"):
        resolve_output_dir(args)
