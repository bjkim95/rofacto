# Data Processing

All commands below assume you run them from the repository root.

RoFacto data processing turns raw robot/human demonstrations into the video +
latent + prompt layout the Wan/DWM trainer and inference expect. Each dataset has
its own guide; the shared WAN encoder produces the latents/prompt-embeds consumed by
training.

## Pipelines

| Pipeline | Guide | What it produces |
| --- | --- | --- |
| **DexYCB** (human demo → robot video) | [`dexycb/README_pipeline.md`](dexycb/README_pipeline.md) | retarget → IK → robot-mesh + EEF-depth render → 5-stream conditioning |
| **DROID** (render_v2) | `droid/` | export → robot-mesh render → 81-frame clips |
| **RoboCasa** (DiT4DiT) | `robocasa/` | DiT4DiT clip-start rollout + dataset builders |
| **Scene captioning** | [`video_caption/README.md`](video_caption/README.md) | Qwen3-VL scene-only prompts (`qwen3vl` env) |
| **WAN latent encoding** | this file, §Encoding | video/prompt → WAN VAE latents + T5 embeds |

## Output Layout

Processed data lives under the input `data/` root (or any `--root_dir` you pass).
A processed DROID sample:

```text
data/droid/render_v2/<lab>/<hash>/
├── videos_clips81/                     # RGB clips (81 frames, 480x832)
├── static_videos_clips81/              # static (first-frame) clips
├── robot_mesh_videos_nominal_clips81/  # rendered robot mesh
├── eef_depth_videos_clips81/           # end-effector depth
├── static_depth_videos_clips81/        # scene depth
└── prompts/                            # scene caption .txt (per clip)
```

WAN preencoded siblings (created by the encoder, consumed by training):

```text
<sample>/
├── videos_clips81_latents_480x832_wan/                     # target video latents
├── static_videos_clips81_latents_480x832_wan/
├── robot_mesh_videos_nominal_clips81_latents_480x832_wan/  # robot-mesh latents
├── eef_depth_videos_clips81_latents_480x832_wan/
├── static_depth_videos_clips81_latents_480x832_wan/
└── prompt_embeds_*_wan/                                    # WAN T5 prompt embeds
```

## Encoding (video/prompt → WAN latents)

`encode_with_wan.py` loads the WAN 2.1/2.2 VAE + T5 (external weights via
`WAN_MODEL_PATH_2_1` / `--model_path`) and writes `<modality>_latents_*_wan/` and
`prompt_embeds_*_wan/` siblings next to each input modality.

Single-shard example (`rofacto` env):

```bash
python3 data_processing/encode_with_wan.py \
  --root_dir <processed_root> \
  --dataset_type droid \
  --modalities videos static_videos hand_videos prompts \
  --hand_videos_dir_name robot_mesh_videos_nominal_clips81 \
  --hand_videos_output_dir_stem robot_mesh_videos_nominal_clips81 \
  --target_height 480 --target_width 832 --wan_version 2.1 \
  --rank 0 --world_size 1
```

For parallel encoding, launch one process per GPU with a distinct `--rank`
in `[0, world_size)` and a shared `--world_size`. Assign each process its GPU
using `CUDA_VISIBLE_DEVICES`. Scheduler wrappers depend on your cluster.

## Notes

- Reads are input-only; write all products under a local `--root_dir` / `outputs/`.
- Latents are deterministic up to bf16/GPU nondeterminism (a re-encode matches an
  existing latent to `mean|Δ| ~ 1e-5`).
- Model weights (WAN base, SAM3, MiniMax-Remover, Qwen3-VL) are external checkpoints;
  each pipeline guide lists the env var / download for them.
