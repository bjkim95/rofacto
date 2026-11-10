# Checkpoint contract

Three separate inputs are needed:

1. **Base model directory** in NVIDIA's Cosmos3 layout, including model configuration,
   safetensors indexes/shards, and `text_tokenizer/`.
2. **Wan2.2 VAE** file matching the training run.
3. **RoFacto RGB adapter export** with `adapter.safetensors`, `model_config.json`,
   `export.json`, and `COMPLETE`.

The initial backend pins these base revisions:

| Variant | Base revision | Transformer layers | Rank-64 trainable LoRA parameters |
| --- | --- | ---: | ---: |
| Nano | `e59a53c25979a090fa8706c9acc0c254a6e89b92` | 36 | 61,341,696 |
| Super | `f543c56225b2e04d0ad141e29655be3a45d9c455` | 64 | 159,383,552 |

Obtain base weights through NVIDIA's distribution and applicable access/license
terms. The RoFacto code does not bundle or redistribute base weights, VAE weights,
robot meshes or private human demonstrations. Adapter publication status,
selected variants and download commands are in the
[checkpoint guide](../checkpoints.md). Private prepared examples remain separate.

Export a completed native training checkpoint on CPU:

```bash
CUDA_VISIBLE_DEVICES='' python -m training.cosmos3.checkpoint \
  --checkpoint /results/super-rgb/checkpoint_step001000 \
  --output /models/rofacto-super-rgb
```

Choose an unused export directory. Export files are staged in a temporary sibling
directory and published together only after verification; an existing export is
never overwritten.

The exporter checks each saved rank's adapter hash before assembling safetensors.
The inference loader checks the base family, LoRA rank/alpha, RGB stream signature,
parameter names and tensor hash. A checkpoint from the depth experiment requires
a different conditioning implementation and is rejected by this RGB release.

Resume state is tied to the original training topology/signature. An adapter
export is portable for inference; it does not claim to restore optimizer history.
