# RoFacto checkpoints

The release packages contain inference adapters and their matching configuration.
Download the upstream base model separately. Keep the Wan and Cosmos runtimes
separate; installation instructions are in the main README.

**Publication status:** Wan and Cosmos3-Super adapters are public as of
November 10, 2026. Remote file checksums and anonymous access were verified.

| Hugging Face repository | Checkpoint | Conditions |
| --- | --- | --- |
| [RoFacto-Wan2.1-14B / rgb-droid](https://huggingface.co/byungjun-kim/RoFacto-Wan2.1-14B/tree/main/rgb-droid) | DROID, step 8,000 | Static RGB + nominal robot mesh |
| [RoFacto-Wan2.1-14B / depth-droid-robocasa](https://huggingface.co/byungjun-kim/RoFacto-Wan2.1-14B/tree/main/depth-droid-robocasa) | DROID + RoboCasa, step 8,000 | RGB/mesh + scene depth + EEF depth |
| [RoFacto-Cosmos3-Super](https://huggingface.co/byungjun-kim/RoFacto-Cosmos3-Super) | DROID + RoboCasa, batch 32, step 5,000 | Static RGB + nominal robot mesh |

Both Wan variants are released at 8,000 optimizer updates. Their training datasets
differ, so they do not constitute a matched-data RGB/depth ablation. Super serves
the H2R experiment. Nano checkpoint publication is outside this release; its code
support remains available.

Download:

```bash
hf download byungjun-kim/RoFacto-Wan2.1-14B --local-dir models/rofacto-wan
hf download byungjun-kim/RoFacto-Cosmos3-Super --local-dir models/rofacto-super
```

Verified release revisions: Wan
[`381369d`](https://huggingface.co/byungjun-kim/RoFacto-Wan2.1-14B/tree/381369db5311a288dc5a1f267b0e97a5b48c55e9)
and Super
[`2ddd12c`](https://huggingface.co/byungjun-kim/RoFacto-Cosmos3-Super/tree/2ddd12c0ec96c4ad90287bf924ca31f6ef2bf482).

Run `sha256sum -c SHA256SUMS` inside a downloaded package. Each model card provides
the base model, pinned Cosmos revisions, VAE, configuration and inference command.
The code repository contains the same cards under [model_cards/](model_cards/).

Wan requires **both** `lora_diffusion_pytorch_model.safetensors` and
`non_lora_weights.safetensors`. Select the matching `inference.yaml` explicitly
with `--experiment_config`. The latter weight file is the extended patch embedding,
not an optional auxiliary asset. RGB and depth versions are not interchangeable.

Cosmos requires `adapter.safetensors`, `model_config.json`, `export.json` and
`COMPLETE` together. See the [Cosmos checkpoint contract](cosmos3/checkpoints.md)
and [quickstart](cosmos3/quickstart.md). The public package removes private mount
paths from metadata while retaining the trained tensor hash and signature.

## Maintainer release procedure

Use `python -m tools.release_checkpoints check --manifest /path/to/release.json`
to inspect inputs without loading a base model. The manifest supplies explicitly
chosen checkpoint directories, model cards, licenses and Wan configurations.
Never upload a training directory recursively.

On a CPU worker:

```bash
python -m tools.release_checkpoints prepare \
  --manifest /path/to/release.json --output /path/to/new-prepared-release
python -m tools.release_checkpoints publish \
  --prepared /path/to/new-prepared-release --namespace byungjun-kim
```

Preparation validates the Wan patch embedding dimensions and Cosmos tensor
hashes, copies only the inference files, and writes checksums. Publication first
verifies every local package, creates new repositories privately, commits each
package, compares remote hashes, then makes each verified repository public.
Existing repositories with different content are never overwritten.
`publication.json` records the public commit URLs and revisions. No GPU is needed.

The initial integration has verified Wan/Nano/Super inference profiles. Cosmos
distributed training/resume acceptance remains pending; refer to the
[verification record](cosmos3/verification.md) before starting a new training run.
