# Controlled video generation

RoFacto supplies three vision streams in order: static scene RGB, robot mesh RGB,
and generated RGB. Static and mesh tokens are clean; the target is noisy.
All three streams share spatial and temporal RoPE positions and interact through
the model's joint attention. This implementation does not concatenate conditions
onto the input patch channels.

The static input can be a repeated image for a fixed camera, or a video for a
moving camera. In H2R, prepare an image with the human removed before calling this
entrypoint. Inpainting and motion retargeting are upstream preparation steps.
The human/GT video is not required for generation.

## Input manifest

A JSON file holds one sample; JSONL holds one sample per line. File paths are
relative to the manifest directory, or absolute. IDs must be unique and safe as
folder names. Use exactly one static input and one prompt source:

```json
{
  "id": "example",
  "static_image": "static.png",
  "robot_mesh_video": "mesh.mp4",
  "prompt_file": "prompt.json",
  "fps": 15
}
```

Replace `static_image` with `static_video` to use a scene video. Replace
`prompt_file` with `prompt` for inline text or a structured JSON object. Structured
prompts are serialized compactly without flattening their fields.

Prepare both conditions at **832×480**, with identical constant FPS. The mesh
length must be `4n+1` frames, at least 5. A static video must have the same frame
count as the mesh. The entrypoint does not silently crop, resize, resample or
truncate inputs. FPS is passed to the model and used to encode the output; it is
not an instruction to retime the source motion.

## Run

```bash
CUDA_VISIBLE_DEVICES=0 python -m training.cosmos3.inference \
  --model super \
  --base /models/Cosmos3-Super \
  --vae /models/Wan2.2_VAE.pth \
  --adapter /models/rofacto-super-rgb \
  --config exps_infer/cosmos3/h2r_super.yaml \
  --inputs /inputs/clips.jsonl \
  --output /results/h2r \
  --resident-layers 20 --pin-cpu-memory
```

The 20-resident-layer configuration passed on 1×A100 80GB for 133- and 397-frame
clips at 35 steps/CFG6. Reported peak GPU allocations were 53.76 and 72.34 GiB;
the process reported approximately 99 GiB peak host RSS. Provision at least
150 GiB of host RAM for this tested profile, including runtime/loading overhead.
Measured generation times were about 37 minutes and 3 hours 25 minutes,
respectively. The 40-resident-layer configuration comes from the prior 1×H200
research run. Full verification details are in
[verification.md](verification.md). Expose exactly one GPU. The worker loads the
model once and processes all listed videos sequentially. VAE encoding happens
before denoising; the VAE moves to CPU during denoising and returns for decoding.
`resident-layers` controls how many transformer layers remain on GPU; decreasing
it reduces GPU weight memory but increases CPU transfers. Sufficient host RAM is
also required. Nano has 36 layers and Super has 64.

For Nano, use `--model nano`, its matching base/adapter, and
`--resident-layers 36`. Do not load a Super adapter into Nano.

`h2r_super.yaml` records 35 denoising steps, CFG 6, shift 10 and the negative prompt.
This uses 70 denoiser calls. `evaluation.yaml` records the research validation
recipe: 12 steps, CFG 1, shift 5, 12 calls. These are different protocols; record
which was used when comparing models. They are RoFacto experiment settings, not a
claim that every Cosmos task should use them.

Add `--validate-only` to check files, metadata, prompts and export compatibility
without loading the GPU model. To distribute independent videos over GPUs, launch
workers with the same manifest/output and distinct `--shard-index 0 ... N-1`, each
with `--num-shards N`. This splits the list; it does not split one video's attention
across GPUs.

## Outputs and restart

Each sample gets `generated.mp4`, `protocol.json`, and `result.json` under
`<output>/<id>/`. A completed result is skipped only when its protocol hash,
encoded video hash, FPS and frame count all match. A changed input, prompt,
sampler, VAE or adapter requires a new output directory. Incomplete/corrupt outputs
are regenerated. A per-sample file lock prevents concurrent workers writing the
same item.

`result.json` includes sampler calls, generation time, peak GPU/host memory,
condition and initial-noise tensor hashes. `--save-latents` additionally saves
large tensor files for numerical comparisons; ordinary inference does not need it.

A long clip is generated in one denoising process. This entrypoint does not carry
previously generated context across separate clips.
