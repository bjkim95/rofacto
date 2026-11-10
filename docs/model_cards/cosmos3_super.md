---
license: other
license_name: openmdw1.1-license
license_link: LICENSE
base_model: nvidia/Cosmos3-Super
base_model_relation: adapter
pipeline_tag: image-to-video
tags:
  - rofacto
  - cosmos3
  - robotics
  - world-model
  - lora
---

# RoFacto · Cosmos3-Super

RoFacto's **static RGB + robot mesh RGB** adapter for NVIDIA Cosmos3-Super.
Camera/time-aligned static, robot-render and generated video tokens share
spatiotemporal positions and interact through joint attention. This is an
additional RoFacto backend; the original paper implementation uses Wan2.1.

[Paper](https://arxiv.org/abs/2607.22535) ·
[Code](https://github.com/bjkim95/rofacto) ·
[Inference quickstart](https://github.com/bjkim95/rofacto/blob/main/docs/cosmos3/quickstart.md)

| Property | Value |
| --- | --- |
| Base revision | `f543c56225b2e04d0ad141e29655be3a45d9c455` |
| Training data | DROID + RoboCasa DiT4DiT |
| Released update / effective batch | 5,000 / 32 |
| Adapter | Generator attention Q/K/V/output LoRA, rank/alpha 64 |
| Trainable parameters | 159,383,552 |
| Training resolution / length | 832×480 / 81 frames |
| VAE | Wan2.2 TI2V-5B VAE, obtained separately |

This export was used for the ALLEX/RB-Y1 H2R examples. It is an RGB-conditioned
checkpoint; scene/EEF depth adapters are not part of this release. Keep all four
export files together. Base weights and optimizer state are not included.

## Download and use

```bash
hf download byungjun-kim/RoFacto-Cosmos3-Super --local-dir models/rofacto-super
hf download nvidia/Cosmos3-Super \
  --revision f543c56225b2e04d0ad141e29655be3a45d9c455 --local-dir models/cosmos3-super
hf download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth \
  --revision 921dbaf3f1674a56f47e83fb80a34bac8a8f203e --local-dir models/wan22-vae
```

Install with `setup_cosmos3.sh` in the code repository. With its environment
active and a prepared input JSON/JSONL:

```bash
CUDA_VISIBLE_DEVICES=0 python -m training.cosmos3.inference \
  --model super --base models/cosmos3-super --vae models/wan22-vae/Wan2.2_VAE.pth \
  --adapter models/rofacto-super --inputs inputs/input.json \
  --config exps_infer/cosmos3/h2r_super.yaml \
  --resident-layers 20 --pin-cpu-memory --output outputs/rofacto-super
```

This A100 80GB profile uses CPU offload and 35 steps / CFG6 / shift10. Provision
at least 150 GiB host RAM. Tested peak allocated VRAM: 54 GiB for 133 frames and
72 GiB for 397 frames. Native FPS is preserved. Inputs must be 832×480 and
`4n+1` frames; static and robot-render conditions must align in time and camera.
Human removal/inpainting and retargeting are separate preparation steps.

Short/long inference, repeated generation and completed-output restart passed
the integration checks. Distributed training/resume in the refactored code and
Super equivalence against the original native recipe remain unverified.

## License and provenance

NVIDIA's Cosmos3 base and these adapter materials use OpenMDW-1.1. See `LICENSE`
and `NOTICE`. No private human demonstrations or robot assets are included.
`release_metadata.json` and `SHA256SUMS` identify the release.
