# Third-party notices

Original RoFacto additions are Apache-2.0. Existing source-file notices and the
licenses below continue to apply to adapted components. Repository-wide licensing
does not replace the licenses of external model weights, data or assets.

| Adapted component | Upstream | License / notice |
| --- | --- | --- |
| Wan models, VAE, T5, distributed utilities and VACE conditioning | [Wan2.1](https://github.com/Wan-Video/Wan2.1), [VACE](https://github.com/ali-vilab/VACE) | [Apache-2.0](LICENSE); Alibaba Wan / VACE authors, notices retained in source |
| Wan pipelines / LoRA integration | [VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun) | [Apache-2.0](LICENSE) |
| Diffusion schedulers and training utilities | [Diffusers](https://github.com/huggingface/diffusers) | [Apache-2.0](LICENSE); Hugging Face and scheduler authors |
| Discrete sampler | [CogVideo](https://github.com/zai-org/CogVideo) | [Apache-2.0](LICENSE) |
| Rotary extrapolation / cache utilities | [RIFLEx](https://github.com/thu-ml/DiT-Extrapolation), [TeaCache](https://github.com/ali-vilab/TeaCache) | [Apache-2.0](LICENSE) |
| `training/wan/models/wan_image_encoder.py` | [OpenAI CLIP](https://github.com/openai/CLIP), [OpenCLIP](https://github.com/mlfoundations/open_clip) | [CLIP MIT](licenses/OpenAI-CLIP-MIT.txt), [OpenCLIP MIT](licenses/OpenCLIP-MIT.txt), Wan notices |
| `training/wan/utils/lora_utils.py` references | [LoRA](https://github.com/microsoft/LoRA), [lora-diffusion](https://github.com/cloneofsimo/lora), [kohya_ss](https://github.com/bmaltais/kohya_ss) | [LoRA MIT](licenses/LoRA-MIT.txt), [lora-diffusion](licenses/lora-diffusion-Apache-2.0.txt), [kohya_ss](licenses/kohya_ss-Apache-2.0.txt) |
| `training/wan/utils/prope_attention.py` | PRoPE / [HY-WorldPlay `camera_rope.py`](https://github.com/Tencent-Hunyuan/HY-WorldPlay/blob/1588e1336e842b03b0a7860c654ebd7c46bb065e/hyvideo/prope/camera_rope.py) | This specific upstream module is [MIT](licenses/PRoPE-MIT.txt) |
| `data_processing/dexycb/pyroki_snippets/` | [PyRoki](https://github.com/chungmin99/pyroki) | [MIT](licenses/Pyroki-MIT.txt), Copyright 2025 Chung Min Kim |
| `data_processing/dexycb/assets/panda_description/urdf/panda.urdf` | Franka Panda description via [example-robot-data](https://github.com/Gepetto/example-robot-data/tree/78b6101a6387de721316b0db717351d899f97b4f/robots/panda_description) | [Panda Apache-2.0](licenses/Panda-Apache-2.0.txt); mesh assets downloaded separately |
| DROID composed Panda/Robotiq URDFs | Robot definitions referencing [robosuite](https://github.com/ARISE-Initiative/robosuite) assets | [robosuite MIT / MuJoCo notice](licenses/robosuite-MIT.txt); meshes not bundled |

RoFacto modifications integrate these components with robot-render conditioning
and this repository's data / inference interfaces. Source-file comments identify
upstream origins where available.

## External components

`setup_cosmos3.sh` installs a pinned NVIDIA cosmos-framework checkout externally;
its code is not vendored here. Cosmos3-Super weights/adapters use OpenMDW-1.1.
Wan base models and our Wan adapters use Apache-2.0; read their model cards.

`fetch_assets.sh` downloads dependencies and meshes with their upstream licenses.
Datasets (DROID, RoboCasa, DexYCB), MANO, SAM3, MiniMax-Remover, SPIDER, and
retargeting assets must be obtained separately under their respective terms.
