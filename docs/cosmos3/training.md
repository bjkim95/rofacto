# RGB LoRA training

**Experimental:** the refactored distributed training/resume path has not yet
passed its GPU acceptance suite. The released inference adapters were trained
with the original research implementation. See [verification.md](verification.md).

The port uses the native Cosmos3 flow-matching training step, FSDP and the same
static/mesh/target packing as inference. Only generator attention Q/K/V/output
LoRA parameters are optimized. The frozen pretrained network includes the text
reasoner; this is not the separate T5 encoder used by Wan. EMA and compile are off.
The RGB recipe uses rank 64, alpha 64, AdamW at `1e-4`, betas `(0.9, 0.95)`, weight
decay `0.01`, gradient clipping at 1, 10% caption dropout and a cosine schedule.

## Data

Provide a directory with `train.jsonl` and `test.jsonl`. Each row is:

```json
{"domain":"robocasa","episode":"episode_01","split":"train","target":"episode_01/target.mp4","mesh":"episode_01/mesh.mp4","static":"episode_01/static.mp4","static_mode":"video","caption":"A robot places the object in the box."}
```

Paths are relative to `--data-root`. RGB clips have 81 frames. The loader preserves
native FPS and all frames, applying the research pipeline's direct FFmpeg bilinear
resize to 832×480. Train/test episodes must be disjoint. Validation expects both
`droid` and `robocasa` domains and `split: test` for held-out rows.
For the original DROID fixed-camera recipe, use `static_mode: first_frame_repeat`
and `static: null`; the target's first frame is repeated as the static condition.
The inference equivalent is to export that image explicitly.

## Preencode

Preencoding stores deterministic native Wan2.2 BF16 x0 latents for all three
streams. These are not Wan2.1 posterior caches. The payload records source hashes,
VAE hash, preprocessing, shape/dtype and static policy. Stale caches fail explicitly.

```bash
CUDA_VISIBLE_DEVICES=0 python -m training.cosmos3.preencode \
  --manifest-dir /data/manifests --data-root /data/clips \
  --source-config /models/rofacto-super-rgb/model_config.json \
  --vae /models/Wan2.2_VAE.pth --cache /data/cosmos3-latents
```

For distributed preencoding, run the same command with
`torchrun --standalone --nproc_per_node=8 -m training.cosmos3.preencode ...`.
Rows are sharded independently; no distributed model is constructed.

## Train and resume

```bash
torchrun --standalone --nproc_per_node=8 -m training.cosmos3.train \
  --config exps/cosmos3/super_rgb.yaml \
  --checkpoint /models/Cosmos3-Super --vae /models/Wan2.2_VAE.pth \
  --manifest-dir /data/manifests --data-root /data/clips \
  --latent-cache /data/cosmos3-latents --verify-cache-parity \
  --output /results/super-rgb
```

The configuration's microbatch 1, accumulation 4, CP 1 on 8 GPUs gives effective
batch **32**. In general, distinct clips per update are
`world_size × microbatch × accumulation / CP`. For 4 GPUs, set `--accum 8` to keep
batch 32. The initial distributed trainer supports 4 or 8 GPUs. Microbatch >1
requires CP 1. Activation checkpointing is configurable; memory/speed depends on
model, length and batch. Consult the verification record before choosing a profile.

Use `nano_rgb.yaml` with the Nano base for Nano. Command-line options override YAML
values. The stop step can be shorter than the schedule horizon: this allows a
smoke run without changing the learning-rate trajectory.

Add `--resume /results/super-rgb/checkpoint_step000100` to restart. The trainer
verifies optimizer and adapter hashes, RNG state, sampler position and the
training signature. Changing dataset membership, base, VAE, world size or effective
batch does not count as an exact resume and is rejected. `--gradient-audit` checks
that every accumulated microstep contributes before the training loop.

`--verify-cache-parity` runs the online VAE and cached paths on the same input and
requires exact x0 equality. Cached training bypasses repeated video decoding/VAE
encoding and moves the VAE to CPU outside validation. Omit `--latent-cache` for
online encoding. Current cache-vs-online throughput verification is recorded in
[verification.md](verification.md); the option is not evidence of a measured speedup.

## Validation and retention

Validation loss uses fixed held-out clips and noise draws per domain; each record
has a protocol ID. Validation preserves training RNG/optimizer state. Generation
uses the 12-step CFG-1 shift-5 recipe. Metrics and generated clips are saved under
the run directory.

Full checkpoints contain LoRA, optimizer and per-rank RNG/sampler state. Only the
latest complete full checkpoint is retained. Milestones selected by
`keep_checkpoints_every`, `keep_checkpoint_steps`, or validation export lightweight
adapters under `evaluation_weights/stepNNNNNN`. `COMPLETE` is written only after
all ranks finish. Use these exports in the common inference entrypoint; no
optimizer files are required for inference.
