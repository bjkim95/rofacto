import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
from copy import deepcopy


_LEGACY_DATA_KEYS = {
    "hand_video_subdir": "robot_mesh_video_subdir",
    "hand_video_latents_subdir": "robot_mesh_video_latents_subdir",
    "hand_disparity_subdir": "eef_depth_subdir",
    "hand_disparity_latents_subdir": "eef_depth_latents_subdir",
    "hand_disparity_npz_mode": "eef_depth_npz_mode",
}
_LEGACY_TRAINING_KEYS = {
    "hand_dropout_prob": "robot_dropout_prob",
}


def _move_legacy_keys(section: Dict[str, Any], key_map: Dict[str, str], section_name: str) -> None:
    for legacy_key, canonical_key in key_map.items():
        if legacy_key not in section:
            continue
        legacy_value = section[legacy_key]
        if canonical_key in section and section[canonical_key] != legacy_value:
            raise ValueError(
                f"Conflicting {section_name} configuration values for "
                f"'{legacy_key}' and '{canonical_key}'"
            )
        section[canonical_key] = legacy_value
        del section[legacy_key]


def normalize_legacy_config(config: dict) -> dict:
    """Return a copy with legacy condition names rewritten to canonical names.

    Legacy and canonical spellings may coexist only when their values are identical.
    Static condition keys are intentionally left unchanged.  The
    ``wan2.1_fun_inp_hand_concat`` pipeline type is a stable registry identifier,
    not a legacy data key, so it must never be rewritten.
    """
    if not isinstance(config, dict):
        raise TypeError("Experiment configuration must be a dictionary")

    normalized = deepcopy(config)
    data = normalized.get("data")
    if isinstance(data, dict):
        _move_legacy_keys(data, _LEGACY_DATA_KEYS, "data")
    training = normalized.get("training")
    if isinstance(training, dict):
        _move_legacy_keys(training, _LEGACY_TRAINING_KEYS, "training")
    return normalized


class ExperimentConfigLoader:
    """
    YAML-based configuration loader for WAN training experiments.
    
    Supports:
    - Pipeline-specific configurations (base config)
    - Experiment-specific configurations (overrides)
    - Configuration merging and validation
    - Runtime overrides
    """
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
    
    def load_pipeline_config(self, pipeline_type: str) -> Dict[str, Any]:
        """Load pipeline configuration from YAML."""
        config_path = self.config_dir / "pipelines" / f"{pipeline_type}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Pipeline config not found: {config_path}")
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        print(f"📋 Loaded pipeline config: {pipeline_type}")
        return normalize_legacy_config(config)
    
    def load_experiment_config(
        self, experiment_file: str, output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Load, normalize, and validate an experiment YAML configuration.

        ``output_dir`` is an explicit runtime output override and takes precedence
        over snapshot output values. Input paths (data_root, base model, resume)
        are left unchanged.
        """
        experiment_path = Path(experiment_file)
        if not experiment_path.exists():
            raise FileNotFoundError(f"Experiment config not found: {experiment_path}")

        with open(experiment_path, "r") as f:
            experiment_config = normalize_legacy_config(yaml.safe_load(f) or {})

        print(f"📋 Loaded experiment config: {experiment_path}")

        # Load and merge pipeline config if pipeline type is specified
        if "pipeline" in experiment_config and "type" in experiment_config["pipeline"]:
            pipeline_type = experiment_config["pipeline"]["type"]
            try:
                pipeline_config = self.load_pipeline_config(pipeline_type)
                merged_config = self._merge_configs(pipeline_config, experiment_config)
            except FileNotFoundError:
                print(f"⚠️  Pipeline config for '{pipeline_type}' not found, using experiment config only")
                merged_config = normalize_legacy_config(experiment_config)
        else:
            merged_config = normalize_legacy_config(experiment_config)

        if output_dir is not None:
            merged_config.setdefault("experiment", {})["output_dir"] = output_dir

        merged_config = normalize_legacy_config(merged_config)
        self.validate_config(merged_config)
        return merged_config
    
    def _merge_configs(self, pipeline_config: Dict, experiment_config: Dict) -> Dict:
        """
        Merge pipeline and experiment configurations.
        Pipeline config is the base, experiment config overrides.
        """
        merged = normalize_legacy_config(pipeline_config)
        experiment_config = normalize_legacy_config(experiment_config)

        for key, value in experiment_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value

        return normalize_legacy_config(merged)
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration."""
        print("🔍 Validating configuration...")
        
        # Check required sections
        required_sections = ["experiment", "training"]
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required section: {section}")
        
        # Validate pipeline type if specified
        if "pipeline" in config and "type" in config["pipeline"]:
            pipeline_type = config["pipeline"]["type"]
            supported_pipeline_types = [
                # WAN pipelines
                "wan2.1_fun_inpaint",
                "wan2.1_fun_inp_hand_concat",
                "wan2.1_fun_inp_hand_concat_i2v_diffusers",
                "wan2.1_vace_dwm_hand_stream",
                "wan2.2_fun_inpaint",
                "wan2.2_fun_inp_hand_concat",
                "wan2.2_fun_control_i2v_mesh",
            ]
            if pipeline_type not in supported_pipeline_types:
                print(f"⚠️  Pipeline type '{pipeline_type}' not in predefined list, but continuing...")
        
        print("✅ Configuration validation passed")
        return True
    
    def _canonicalize_override_key(self, key: str) -> str:
        """Rewrite supported legacy dotted override keys before assignment."""
        keys = key.split(".")
        if len(keys) != 2:
            return key

        section, target_key = keys
        key_map = {
            "data": _LEGACY_DATA_KEYS,
            "training": _LEGACY_TRAINING_KEYS,
        }.get(section)
        if key_map is not None:
            keys[-1] = key_map.get(target_key, target_key)
        return ".".join(keys)

    @staticmethod
    def _parse_override_value(value: str) -> Any:
        """Parse a command-line override value using the established coercions."""
        try:
            if value.lower() in ["true", "false"]:
                return value.lower() == "true"
            if "." in value or "e" in value.lower():
                return float(value)
            if value.isdigit():
                return int(value)
        except ValueError:
            pass
        return value

    def apply_overrides(self, config: Dict[str, Any], overrides: list) -> Dict[str, Any]:
        """Apply runtime overrides to configuration."""
        if not overrides:
            return config

        print("🔧 Applying runtime overrides...")
        parsed_overrides = []
        override_values = {}
        for override in overrides:
            if "=" not in override:
                raise ValueError(f"Invalid override format: {override}. Use key=value")

            key, raw_value = override.split("=", 1)
            canonical_key = self._canonicalize_override_key(key)
            value = self._parse_override_value(raw_value)
            previous = override_values.get(canonical_key)
            if previous is not None and previous[0] != key and previous[1] != value:
                raise ValueError(
                    f"Conflicting runtime override values for '{previous[0]}' "
                    f"and '{key}'"
                )
            override_values[canonical_key] = (key, value)
            parsed_overrides.append((canonical_key, value))

        for key, value in parsed_overrides:
            keys = key.split(".")
            current = config
            for section_key in keys[:-1]:
                if section_key not in current:
                    current[section_key] = {}
                current = current[section_key]

            target_key = keys[-1]
            current[target_key] = value
            print(f"   {key} = {current[target_key]}")

        return normalize_legacy_config(config)
    
    def print_config_summary(self, config: Dict[str, Any]):
        """Print a summary of the configuration."""
        print("\n" + "="*80)
        print("🎯 EXPERIMENT CONFIGURATION SUMMARY")
        print("="*80)
        
        # Experiment info
        experiment = config.get("experiment", {})
        print(f"📝 Experiment: {experiment.get('name', 'Unknown')}")
        print(f"   Description: {experiment.get('description', 'No description')}")
        print(f"   Author: {experiment.get('author', 'Unknown')}")
        print(f"   Date: {experiment.get('date', 'Unknown')}")
        
        # Pipeline info
        pipeline = config.get("pipeline", {})
        if pipeline:
            print(f"\n🔧 Pipeline: {pipeline.get('type', 'Unknown')}")
            print(f"   Class: {pipeline.get('class', 'Unknown')}")
            if 'base_model_name_or_path' in pipeline:
                print(f"   Base Model: {pipeline.get('base_model_name_or_path', 'Unknown')}")
        
        # Transformer info
        transformer = config.get("transformer", {})
        if transformer:
            print(f"\n🔩 Transformer:")
            print(f"   Class: {transformer.get('class', 'Unknown')}")
            if 'condition_channels' in transformer:
                print(f"   Condition Channels: {transformer.get('condition_channels', 'Unknown')}")
        
        # Training info
        training = config.get("training", {})
        print(f"\n🚀 Training Mode: {training.get('mode', 'Unknown')}")
        print(f"   Learning Rate: {training.get('learning_rate', 'Unknown')}")
        print(f"   Batch Size: {training.get('batch_size', 'Unknown')}")
        if 'max_train_steps' in training:
            print(f"   Max Train Steps: {training.get('max_train_steps', 'Unknown')}")
        if 'lora_rank' in training:
            print(f"   LoRA Rank: {training.get('lora_rank', 'Unknown')}")
        if 'lora_alpha' in training:
            print(f"   LoRA Alpha: {training.get('lora_alpha', 'Unknown')}")
        
        # Data info
        data = config.get("data", {})
        print(f"\n📊 Data:")
        print(f"   Dataset: {data.get('dataset_file', 'Unknown')}")
        print(f"   Data Root: {data.get('data_root', 'Unknown')}")
        
        print("="*80 + "\n")


def load_experiment_config(
    experiment_file: str,
    overrides: Optional[list] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Load, normalize, and validate an experiment configuration.

    Pipeline and experiment YAML layers plus every CLI override are normalized
    before use. ``output_dir`` only rebases the output role; snapshot input paths
    such as ``data_root``, base-model paths, and resume paths remain unchanged.
    """
    config_loader = ExperimentConfigLoader()
    config = config_loader.load_experiment_config(experiment_file, output_dir=output_dir)

    if overrides:
        config = config_loader.apply_overrides(config, overrides)

    config_loader.print_config_summary(config)
    return config
