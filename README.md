<div align="center">

<h1>RoFacto: Robot-Factored World Models via Robot Rendering</h1>

<p>Byungjun Kim · Taeksoo Kim · Hyunsoo Cha · Hanbyul Joo</p>
<p><strong>CoRL 2026</strong></p>

<p>
  <a href="https://bjkim95.github.io/rofacto/"><img src="https://img.shields.io/badge/Project_Page-2563EB?style=for-the-badge&amp;logo=homepage&amp;logoColor=white" alt="Project page" height="28"></a>
  &nbsp;
  <a href="https://arxiv.org/abs/2607.22535"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B?style=for-the-badge&amp;logo=arxiv&amp;logoColor=white" alt="Read the paper on arXiv" height="28"></a>
  &nbsp;
  <a href="docs/checkpoints.md"><img src="https://img.shields.io/badge/Hugging_Face-Checkpoints-FFD21E?style=for-the-badge&amp;logo=huggingface&amp;logoColor=FFD21E" alt="Hugging Face checkpoints" height="28"></a>
</p>

</div>

**RoFacto predicts how a scene responds to robot actions using rendered robot motion.**
The same visual interface supports new robot embodiments and turns retargeted
human demonstrations into robot manipulation videos.

This repository contains the paper's **Wan2.1** implementation and an additional
**Cosmos3** version, with pretrained checkpoints for both.

## News

- **November 2026:** Code and checkpoints released, including Cosmos3-Super for human-to-robot video generation.

## Overview

<p align="center">
  <img src="https://bjkim95.github.io/rofacto/static/image/overview.png" alt="RoFacto overview: robot actions are converted into nominal motion and rendered geometry, which condition the world model together with scene context." width="100%">
</p>

RoFacto first converts robot commands into a **nominal trajectory** using the
robot's controller and kinematics, then renders that motion through its URDF.
The video model takes these renderings together with static scene context and
predicts the resulting scene response. Scene and end-effector depth provide
additional cues for contact and occlusion.

See the [project page](https://bjkim95.github.io/rofacto/) for DROID and RoboCasa
results, action editing, and zero-shot embodiment generalization. Additional
Cosmos3 human-to-robot examples are featured in the
[RLWRLD × NVIDIA blog](https://www.rlwrld.ai/en/insight/blog/116).

## Installation

```bash
git clone https://github.com/bjkim95/rofacto.git
cd rofacto
```

### Wan2.1

```bash
bash setup_env.sh rofacto
conda activate rofacto
```

Requires Linux, Conda, an NVIDIA GPU, and FFmpeg. See
[environment setup](README_ENV.md) for dependencies and preprocessing tools.

### Cosmos3

Cosmos3 uses a separate environment:

```bash
bash setup_cosmos3.sh
source .deps/cosmos-framework/.venv/bin/activate
```

See the [Cosmos3 setup guide](docs/cosmos3/installation.md) for prerequisites.

## Pretrained Models

| Model | Training data | Conditioning | Download |
| --- | --- | --- | --- |
| Wan2.1-14B · RGB | DROID | Static RGB + robot mesh | [Hugging Face](https://huggingface.co/byungjun-kim/RoFacto-Wan2.1-14B/tree/main/rgb-droid) |
| Wan2.1-14B · RGB + Depth | DROID + RoboCasa | Static RGB + robot mesh + scene/EEF depth | [Hugging Face](https://huggingface.co/byungjun-kim/RoFacto-Wan2.1-14B/tree/main/depth-droid-robocasa) |
| Cosmos3-Super | DROID + RoboCasa | Static RGB + robot mesh | [Hugging Face](https://huggingface.co/byungjun-kim/RoFacto-Cosmos3-Super) |

Download the RoFacto adapter and its corresponding base model separately.
[Checkpoint instructions](docs/checkpoints.md) list the required files for each model.

## Inference

### Wan2.1

Download the adapter and base model:

```bash
hf download byungjun-kim/RoFacto-Wan2.1-14B --local-dir models/rofacto-wan
hf download alibaba-pai/Wan2.1-Fun-V1.1-14B-InP --local-dir models/wan-base
```

Prepare a static scene video and a camera-aligned robot mesh video at **832×480,
81 frames**, then run:

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

For the depth model, use `depth-droid-robocasa` and provide
`--static_disparity_video_path` and `--eef_depth_video_path`. See the
[Wan model card](docs/model_cards/wan.md#download-and-use) for details.

<details>
<summary>Faster inference with LightX2V</summary>

Four-step inference requires a separately downloaded LightX2V acceleration LoRA.
Add the following options to the command above, replacing the step and guidance settings:

```bash
--lightx2v_lora_path /path/to/lightx2v.safetensors \
--num_inference_steps 4 --guidance_scale 1.0 \
--lightx2v_compat_mode force_compat --lightx2v_compat_strict
```

</details>

### Cosmos3

Follow the [Cosmos3 quickstart](docs/cosmos3/quickstart.md) to download the
Super base model, VAE and RoFacto adapter, then generate from a static scene
image/video, a robot mesh video and a text prompt. The guide includes a
single-GPU configuration and [example inputs](examples/cosmos3/h2r/README.md).

## Human-to-Robot Video Generation

Retarget a human demonstration to the desired robot, render its motion, and
combine it with the scene image after removing the human. RoFacto uses these
conditions to generate the robot interaction video.

- [DexYCB preprocessing](data_processing/dexycb/README_pipeline.md): hand retargeting, arm IK, robot rendering and conditioning preparation.
- [Cosmos3 H2R quickstart](docs/cosmos3/quickstart.md): generation from prepared conditions, including longer clips.

## Training

Prepare robot renderings, scene context and captions following the
[data preparation guide](data_processing/README.md). For Wan, preencode the
videos and prompts into VAE latents and text embeddings before training.

The Wan example configurations use four GPUs. Set the dataset and model paths
in the configuration, then launch:

```bash
accelerate launch --config_file accelerate_configs/deepspeed_4.yaml \
  training/wan/train_dwm_wan.py \
  --experiment_config exps/smoke/wan21_14b_smoke_48ch.yaml --mode batch
```

The supplied smoke configs run 80 updates; adjust the training duration for a
full run. Use `wan21_14b_smoke_16ch.yaml` for RGB-only conditioning.

For Cosmos3, see the [training guide](docs/cosmos3/training.md) for preencoding,
LoRA training and resuming. The distributed training code is experimental.

## Acknowledgements

We thank the contributors to [Wan](https://github.com/Wan-Video/Wan2.1),
[NVIDIA Cosmos](https://github.com/nvidia/cosmos),
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

RoFacto code is released under [Apache-2.0](LICENSE). See
[third-party notices](THIRD_PARTY_NOTICES.md) for adapted components.
Checkpoints and external assets retain the licenses listed in their model cards.

## Citation

If you find RoFacto useful in your research, please cite:

```bibtex
@inproceedings{kim2026rofacto,
  title={Robot-Factored World Models via Robot Rendering},
  author={Kim, Byungjun and Kim, Taeksoo and Cha, Hyunsoo and Joo, Hanbyul},
  booktitle={Conference on Robot Learning},
  year={2026}
}
```
