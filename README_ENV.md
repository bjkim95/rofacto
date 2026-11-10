# Environments

Run commands from the repository root. Linux, an NVIDIA CUDA-capable GPU, Git,
Conda, and FFmpeg/ffprobe are required for the Wan inference workflow.

## Wan

```bash
bash setup_env.sh rofacto
conda activate rofacto
python -m training.wan.inference --help
```

The installer creates Python 3.10.19, installs PyTorch 2.9.0 / CUDA 12.8 from
the PyTorch index, then installs `requirements-lock.txt`. Key pins include
diffusers 0.37.1, transformers 4.51.3, accelerate 1.2.1, and peft 0.17.0.
`requirements.txt` is a historical import inventory, **not** an installation
recipe; use `setup_env.sh` for the tested dependency set.

The installer handles pyrender's older PyOpenGL requirement explicitly. A local
FlashAttention wheel matching Python 3.10 / PyTorch 2.9 / CUDA 12.8 is optional;
without it the attention implementation uses its available fallback. GPU memory
and speed can differ from the FlashAttention profile.

Download the model weights separately as described in [checkpoints](docs/checkpoints.md).
Single-GPU Wan inference was checked on an A100 80GB. Training smoke configurations
target four 80GB-class GPUs; configure your own dataset paths before launching.

## Cosmos3

Use the separate [Cosmos3 environment](docs/cosmos3/installation.md), installed
by `setup_cosmos3.sh`. Do not install its dependencies into the Wan environment.
See the [quickstart](docs/cosmos3/quickstart.md) for public weights and H2R inputs.

## Optional preprocessing environments and assets

Prepared-input inference does not require these dependencies:

- Rendering: `bash fetch_assets.sh --third-party-only` downloads robot assets
  and retargeting repositories. Install the corresponding upstream dependencies.
- Hand retargeting: [dex-retargeting](https://github.com/dexsuite/dex-retargeting)
  in a separate environment. Existing scripts refer to this environment as `spider`.
- Arm IK / interactive placement: [PyRoki](https://github.com/chungmin99/pyroki)
  and its JAX dependencies in a separate `pyroki` environment. The Panda URDF
  references `example-robot-data`; install that model's meshes as well.
- MANO visualization: install [dex-ycb-toolkit](https://github.com/NVlabs/dex-ycb-toolkit)
  under `data_processing/third_party/dex-ycb-toolkit`, with its `manopth` dependency.
  Obtain the separately licensed MANO model from its provider and set `MANO_ROOT`.
- Human removal: install [SAM3](https://github.com/facebookresearch/sam3) and
  [MiniMax-Remover](https://github.com/zibojia/MiniMax-Remover) under
  `data_processing/third_party/`. Set `SAM3_CKPT` and `MINIMAX_MODEL_ROOT`
  to separately downloaded weights.
- Optional estimated depth: [MoGe](https://github.com/microsoft/MoGe); set `MOGE_ROOT`.
- Captioning: a separate vLLM environment supporting Qwen3-VL; see the
  [captioning guide](data_processing/video_caption/README.md).

Data, MANO models, meshes and model weights retain their upstream license terms.
These specialized preprocessing environments are not installed by `setup_env.sh`.

## CPU checks

With the Wan environment active and pytest installed:

```bash
python -m pytest -q tests
python -m training.wan.inference --help
python -m training.cosmos3.inference --help
```

Cosmos schema/checkpoint CPU tests do not load the native model. Actual Cosmos
generation requires the separate runtime and the base/VAE/adapter files.
