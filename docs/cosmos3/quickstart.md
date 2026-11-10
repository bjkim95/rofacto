# Cosmos3 inference quickstart

This guide starts from prepared scene and robot-render conditions. It generates
robot videos with the RoFacto RGB adapter. Human removal, motion retargeting and
robot rendering happen before this step. A target/GT video is not required.

## Get the code and environment

Clone the public repository:

```bash
git clone https://github.com/bjkim95/rofacto.git
cd rofacto
./setup_cosmos3.sh
source .deps/cosmos-framework/.venv/bin/activate
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=''
```

Run subsequent commands from this repository directory. Installation downloads
large CUDA packages; see [installation.md](installation.md) to place the package
cache and environment on a larger disk.

## Prepare weights and inputs

Obtain the three model components separately, as described in
[checkpoints.md](checkpoints.md):

- Cosmos3-Super base directory at revision
  `f543c56225b2e04d0ad141e29655be3a45d9c455`, including `text_tokenizer/`.
- Matching Wan2.2 `Wan2.2_VAE.pth`.
- Our Super RGB export directory, containing `adapter.safetensors`,
  `model_config.json`, `export.json`, and `COMPLETE` together.

Download the released adapter and its matching base/VAE:

```bash
hf download byungjun-kim/RoFacto-Cosmos3-Super --local-dir models/rofacto-super
hf download nvidia/Cosmos3-Super \
  --revision f543c56225b2e04d0ad141e29655be3a45d9c455 --local-dir models/cosmos3-super
hf download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth \
  --revision 921dbaf3f1674a56f47e83fb80a34bac8a8f203e --local-dir models/wan22-vae
```

See the [checkpoint guide](../checkpoints.md) for file verification. Prepared
human-demonstration media are not bundled; supply your own conditions below.

For your own sample, place these files together:

```text
my_inputs/
  static.png          # scene with the human removed
  mesh.mp4            # camera-aligned robot-render video
  prompt.json         # prompt describing this robot, scene and action
  input.json
```

Both visual inputs must be 832×480. The mesh must have constant FPS and `4n+1`
frames. Copy [the example manifest](../../examples/cosmos3/h2r/input.example.json)
and [prompt template](../../examples/cosmos3/h2r/prompt.example.json), then edit
them to match the actual sample. Example `input.json`:

```json
{
  "id": "my_clip",
  "static_image": "static.png",
  "robot_mesh_video": "mesh.mp4",
  "prompt_file": "prompt.json",
  "fps": 15
}
```

The FPS value must match the mesh video. Use a JSONL list for several clips;
the worker will load the model once and generate them sequentially.

## Generate on one A100 80GB

Set these paths to the downloaded weights and your prepared input:

```bash
BASE=models/cosmos3-super
VAE=models/wan22-vae/Wan2.2_VAE.pth
ADAPTER=models/rofacto-super
INPUT=my_inputs/input.json
RESULTS=outputs/my_h2r

CUDA_VISIBLE_DEVICES=0 python -m training.cosmos3.inference \
  --model super --base "$BASE" --vae "$VAE" --adapter "$ADAPTER" \
  --inputs "$INPUT" --output "$RESULTS" \
  --config exps_infer/cosmos3/h2r_super.yaml \
  --resident-layers 20 --pin-cpu-memory
```

Add `--validate-only` for a CPU-only input/configuration check before running.
The tested profile uses 35 steps, CFG6 and CPU offload. Provision at least
150 GiB host RAM; measured peak GPU allocations were about 54 GiB for 133 frames
and 72 GiB for 397 frames. Those clips took approximately 37 minutes and
3 hours 25 minutes on the tested A100 system. Begin with the shorter example.

Read the generated video at `$RESULTS/my_clip/generated.mp4`. `protocol.json`
records the input and sampler settings; `result.json` records completion and
timing. Re-running the same command skips verified completed outputs. When
changing the prompt, inputs or settings, select a new output directory.

For Nano, static videos, multiple-GPU list sharding and other options, see
[inference.md](inference.md). Keep the matching base and adapter together when
switching model size. Current verification coverage is in
[verification.md](verification.md); distributed training/resume validation is
deferred and is not needed to use these inference exports.
