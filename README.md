<div align="center">

<h1>RoFacto: Robot-Factored World Models via Robot Rendering</h1>

<p>
  <a href="https://bjkim95.github.io/rofacto/"><img src="https://img.shields.io/badge/Project_Page-2563EB?style=for-the-badge&amp;logo=homepage&amp;logoColor=white" alt="Project page" height="28"></a>
  &nbsp;
  <a href="https://arxiv.org/abs/2607.22535"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B?style=for-the-badge&amp;logo=arxiv&amp;logoColor=white" alt="Read the paper on arXiv" height="28"></a>
  &nbsp;
  <a href="docs/checkpoints.md"><img src="https://img.shields.io/badge/Hugging_Face-Checkpoints-FFD21E?style=for-the-badge&amp;logo=huggingface&amp;logoColor=FFD21E" alt="Hugging Face checkpoints" height="28"></a>
</p>

</div>

Official implementation of **Robot-Factored World Models via Robot Rendering**.

<p align="center">
  <b>TL;DR:</b> RoFacto moves two robot-specific factors out of the world model — action
  realization (commands → a deployment-available nominal trajectory) and robot rendering
  (that trajectory → explicit robot geometry through the URDF) — so the model only ever
  sees the action as visible robot geometry and learns how objects respond to it.
</p>

## Model backends

The paper implementation uses **Wan2.1**. RoFacto also supports **Cosmos3 Nano/Super**
with aligned static-scene and robot-render video tokens, using a separate runtime.

| Backend | Installation | Inference/training |
| --- | --- | --- |
| Wan2.1 (paper) | `setup_env.sh` | Existing `training/wan` commands below |
| Cosmos3 Nano/Super (additional backend) | [Cosmos3 installation](docs/cosmos3/installation.md) | [Inference](docs/cosmos3/inference.md) · [Training](docs/cosmos3/training.md) · [Checkpoints](docs/cosmos3/checkpoints.md) |

Cosmos3 Nano/Super inference has passed single-A100 GPU checks. The ported
distributed training/resume path is experimental and awaits GPU acceptance tests. Start with the
[inference quickstart](docs/cosmos3/quickstart.md). See the
[verification record](docs/cosmos3/verification.md) for tested profiles and pending
release items. Existing Wan and DexYCB preprocessing remain in their original
locations. Prepared H2R conditions use the same Cosmos3 inference entrypoint;
see the [input example](examples/cosmos3/h2r/README.md).

[Wan RGB/depth adapters](https://huggingface.co/byungjun-kim/RoFacto-Wan2.1-14B)
and the [Cosmos3-Super adapter](https://huggingface.co/byungjun-kim/RoFacto-Cosmos3-Super)
are available on Hugging Face. Checkpoint variants and download commands are
listed in the [checkpoint guide](docs/checkpoints.md).

## Overview

Action-conditioned video world models predict future observations from an initial
observation and an action signal. Conditioning directly on action commands asks the model
to learn action realization itself; conditioning on logged future states leaks the very
interaction outcomes it should predict. RoFacto factors both out:

1. **Action realization** — each command is rolled through the robot's own controller and
   kinematics into a *nominal trajectory*, a middle signal available at deployment time.
2. **Robot rendering** — the nominal trajectory is rendered through the robot URDF, moving
   the robot's geometry, kinematics, and appearance out of the model and into explicit
   rendered geometry.

End-effector depth is paired with scene depth to resolve depth ambiguity, giving geometric
cues for contact and occlusion beyond image-plane overlap. Together with camera-aware
static RGB/depth context, this forms a shared visual interface that stays consistent across
viewpoints and robot embodiments.

The same interface lets the model **generate robot manipulation videos from human
demonstrations**, by retargeting the hand motion and rendering it as robot geometry
(see [`data_processing/dexycb/README_pipeline.md`](data_processing/dexycb/README_pipeline.md)).

## Conditioning interface

| Stream | Meaning |
| --- | --- |
| `videos` | target RGB clip |
| `static_videos` | camera-aware static scene RGB (clip-first frame) |
| `static_depth_videos` | static scene depth |
| `robot_mesh_videos_nominal` | robot geometry rendered from the nominal trajectory |
| `eef_depth_videos` | end-effector depth |

Two trained Wan variants are available: **16ch** (RGB + rendered robot geometry) and
**48ch** (+ scene depth + EEF depth). The depth pair is what resolves contact/occlusion
ambiguity, and it matters most out of domain — on DexYCB human demonstrations the 16ch
variant fails to form a coherent arm while 48ch does not.

Robot model: **Panda + Robotiq FT300 force-torque sensor + Robotiq 2F-85**
(`data_processing/droid_urdf_overlay/assets/panda_arm_robotiq_2f85_fts300.urdf`)
is used for DROID rendering. The DexYCB guide also describes Panda + Inspire hand retargeting.

## Installation

```bash
git clone https://github.com/bjkim95/rofacto
cd rofacto

bash setup_env.sh rofacto     # Python 3.10 + CUDA 12.8 PyTorch + pinned deps
conda activate rofacto
# Optional: needed for dataset preprocessing / robot rendering, not prepared-input inference
bash fetch_assets.sh
```

The repository tracks code only. `fetch_assets.sh` downloads the upstream artifacts the
pipelines read at runtime — DROID annotations from
[`KarlP/droid`](https://huggingface.co/KarlP/droid) (~215 MB) and third-party meshes/code at
selected upstream revisions. Existing downloads are reused on subsequent runs.

Retargeting and IK use their own specialized envs (`spider`, `pyroki`); prompt captioning
uses `qwen3vl` (vLLM). Details and the pinned dependency rationale are in
[`README_ENV.md`](README_ENV.md).

Model weights are external: Wan2.1-Fun-14B-InP base, LightX2V step-distill LoRA, SAM3,
MiniMax-Remover, Qwen3-VL, and the trained RoFacto checkpoints.

All commands below assume you run them from the repository root.

## Getting Started

### Data Preparation

Per-pipeline guides:

- [`data_processing/README.md`](data_processing/README.md) — index of all pipelines
- [`data_processing/dexycb/README_pipeline.md`](data_processing/dexycb/README_pipeline.md) — human demo → robot video
- [`data_processing/video_caption/README.md`](data_processing/video_caption/README.md) — Qwen3-VL scene captions

A processed DROID sample (WAN latents are the `*_latents_480x832_wan/` siblings):

```text
data/droid/render_v2/<lab>/<hash>/
├── videos_clips81/                     # RGB clips (81 frames)
├── static_videos_clips81/              # static scene RGB
├── static_depth_videos_clips81/        # static scene depth
├── robot_mesh_videos_nominal_clips81/  # rendered robot geometry
├── eef_depth_videos_clips81/           # end-effector depth
├── prompts/                            # scene captions
├── *_latents_480x832_wan/              # WAN VAE latents (per modality)
└── prompt_embeds_*_wan/                # WAN T5 prompt embeds
```

Encode videos/prompts → WAN latents:

```bash
python3 data_processing/encode_with_wan.py \
  --root_dir <processed_root> --dataset_type droid \
  --modalities videos static_videos hand_videos prompts \
  --hand_videos_dir_name robot_mesh_videos_nominal_clips81 \
  --hand_videos_output_dir_stem robot_mesh_videos_nominal_clips81 \
  --target_height 480 --target_width 832 --wan_version 2.1 \
  --rank 0 --world_size 1
```

### Training

```bash
accelerate launch --config_file accelerate_configs/deepspeed_4.yaml \
  training/wan/train_dwm_wan.py \
  --experiment_config exps/smoke/wan21_14b_smoke_48ch.yaml --mode batch
```

`exps/smoke/wan21_14b_smoke_{16,48}ch.yaml` are short smoke configs for both variants.
The 14B LoRA path needs 4×80 GB-class GPUs with DeepSpeed ZeRO-2 and
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

### Inference

For standard single-GPU inference with the public RGB/depth adapters, follow
the [Wan checkpoint quickstart](docs/model_cards/wan.md#download-and-use).

Optional LightX2V four-step inference (requires a separate acceleration LoRA):

```bash
torchrun --standalone --nproc_per_node=4 -m training.wan.inference \
  --checkpoint_path <ckpt> --experiment_config <config.yaml> \
  --base_model_path <Wan2.1-Fun-14B-InP> --dataset_file <shard.txt> \
  --data_root <processed_root> --lightx2v_lora_path <lora.safetensors> \
  --num_inference_steps 4 --guidance_scale 1.0 \
  --lightx2v_compat_mode force_compat --lightx2v_compat_strict \
  --height 480 --width 832 --output_dir outputs/<run>/
```

### Human demonstration → robot video

The DexYCB application runs the full chain — hand retargeting, Panda IK, robot mesh + EEF
depth rendering, conditioning assembly, and generation. See
[`data_processing/dexycb/README_pipeline.md`](data_processing/dexycb/README_pipeline.md);
robot base placement is an explicit user decision made in viser via
`data_processing/dexycb/interactive_panda_base.py`.

## Notes

- `data/` can be a local dataset directory or a symlink; write run products under
  `outputs/`, which is git-ignored.
- Reference checkpoints by absolute path in an experiment config. Never write into a source
  checkpoint directory.
- `pipeline.type: wan2.1_fun_inp_hand_concat` is a stable registry identifier and is exempt
  from the `robot_mesh_video_*` / `eef_depth_*` naming.

## Acknowledgements

We thank the contributors to [Wan](https://github.com/Wan-Video/Wan2.1),
[VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun),
[LightX2V](https://github.com/ModelTC/lightx2v),
[robosuite](https://github.com/ARISE-Initiative/robosuite),
[SPIDER](https://github.com/facebookresearch/spider),
[dex-retargeting](https://github.com/dexsuite/dex-retargeting),
[PyRoki](https://github.com/chungmin99/pyroki),
[SAM 3](https://github.com/facebookresearch/sam3), and
[MiniMax-Remover](https://github.com/zibojia/MiniMax-Remover) for open-sourcing their work,
and the [DROID](https://droid-dataset.github.io/) and
[DexYCB](https://dex-ycb.github.io/) teams for their datasets.

## License

Original RoFacto code is released under [Apache-2.0](LICENSE). Adapted components
retain their upstream licenses and attribution; see [NOTICE](NOTICE) and
[third-party notices](THIRD_PARTY_NOTICES.md). Model weights and separately
downloaded datasets, robot assets and dependencies have their own licenses.
Wan adapters use Apache-2.0; the Cosmos3-Super adapter uses OpenMDW-1.1.

## Citation

```bibtex
@article{kim2026rofacto,
  title={Robot-Factored World Models via Robot Rendering},
  author={Kim, Byungjun and Kim, Taeksoo and Cha, Hyunsoo and Joo, Hanbyul},
  journal={arXiv preprint arXiv:2607.22535},
  year={2026}
}
```
