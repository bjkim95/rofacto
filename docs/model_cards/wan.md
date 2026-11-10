---
license: apache-2.0
base_model: alibaba-pai/Wan2.1-Fun-V1.1-14B-InP
base_model_relation: adapter
pipeline_tag: image-to-video
tags:
  - rofacto
  - robotics
  - world-model
  - lora
  - video-generation
---

# RoFacto · Wan2.1-14B

Inference adapters for **Robot-Factored World Models via Robot Rendering**.
RoFacto conditions video generation on a static scene and a camera-aligned robot
rendering of the nominal trajectory. The depth model additionally uses scene and
end-effector depth rendered in the training representation.

[Paper](https://arxiv.org/abs/2607.22535) ·
[Code and installation](https://github.com/bjkim95/rofacto) ·
[Checkpoint guide](https://github.com/bjkim95/rofacto/blob/main/docs/checkpoints.md)

| Directory | Training data | Conditions | Update |
| --- | --- | --- | ---: |
| `rgb-droid` | DROID | Static RGB + nominal robot mesh RGB (16 added latent channels) | 8,000 |
| `depth-droid-robocasa` | DROID + RoboCasa DiT4DiT | RGB/mesh + scene depth + EEF depth (48 added latent channels) | 8,000 |

Both variants use the checkpoint at 8,000 optimizer updates. Their training
datasets differ; this release is not a matched-data RGB/depth ablation.
Each directory contains rank-64 LoRA weights, the trained extended patch embedding,
and an inference configuration. Both safetensors files are required. This repository
does not contain a full base model or optimizer state.

## Download and use

```bash
hf download byungjun-kim/RoFacto-Wan2.1-14B --local-dir models/rofacto-wan
hf download alibaba-pai/Wan2.1-Fun-V1.1-14B-InP --local-dir models/wan-base
```

Install the Wan environment with `setup_env.sh` in the code repository. Run from
the repository root, with prepared 832×480 conditions and an 81-frame robot video:

```bash
python -m training.wan.inference \
  --checkpoint_path models/rofacto-wan/rgb-droid \
  --experiment_config models/rofacto-wan/rgb-droid/inference.yaml \
  --base_model_path models/wan-base \
  --video_path inputs/static.mp4 --static_video_path inputs/static.mp4 \
  --robot_mesh_video_path inputs/mesh.mp4 \
  --prompt_mode custom --custom_prompt 'A robot picks up the object.' \
  --height 480 --width 832 --num_frames 81 \
  --num_inference_steps 50 --guidance_scale 6 \
  --output_dir outputs/rofacto-wan
```

For the depth model, select `depth-droid-robocasa` and also supply
`--static_disparity_video_path` and `--eef_depth_video_path`.
Use the repository's depth preprocessing; arbitrary grayscale depth is not an
equivalent input. LightX2V four-step inference is optional and requires its
separately downloaded acceleration LoRA; see the code README.

The weights generate videos, not executable robot commands. H2R use requires
human removal, motion retargeting and camera-aligned robot rendering beforehand.
Model outputs depend on the accuracy of those inputs.

## Provenance and license

Base: Alibaba PAI's Wan2.1-Fun-V1.1-14B-InP, from the Wan2.1 family.
RoFacto adds learned LoRA parameters and replaces the patch embedding to accept
robot/geometry conditions. Distributed under Apache-2.0; see `LICENSE` and `NOTICE`.
`SHA256SUMS` and `release_metadata.json` identify the exact files and code revision.

```bibtex
@inproceedings{kim2026rofacto,
  title={Robot-Factored World Models via Robot Rendering},
  author={Kim, Byungjun and Kim, Taeksoo and Cha, Hyunsoo and Joo, Hanbyul},
  booktitle={Conference on Robot Learning},
  year={2026}
}
```
