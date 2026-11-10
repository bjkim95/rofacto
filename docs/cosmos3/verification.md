# Integration verification

The public integration includes the paper's Wan runtime and an additional
Cosmos3 backend. Inference verification covers the profiles below. The released
adapters were trained with the original research code; the refactored Cosmos
distributed training/resume path remains experimental.

| Area | Evidence | Status |
| --- | --- | --- |
| Cosmos installation | Fresh Python 3.13.9 environment; pinned framework and dependency lock; native model import without experiment PYTHONPATH | Passed |
| Cosmos CPU suite | Input validation/sharding/restart, adapter compatibility, deterministic validation selection, RNG isolation, checkpoint retention, latent-cache contracts | 25 tests |
| Wan CPU regression suite | Configuration normalization and consumers, split-output guard, self-contained fixtures | 13 tests |
| Checkpoint release suite | Inference-only file allowlist, metadata sanitization, checksum verification, publication guards | 5 tests |
| Super inference | Single A100 80GB, 133/397 frames, full-step H2R, CPU offload, 20 GPU-resident layers | Passed |
| Nano inference | Fresh environment, single A100 80GB, DROID/RoboCasa, 832×480×81, 12 steps / CFG1 / shift5 | Passed |
| Nano reference equivalence | Repeated native-pixel reference and cache path: all three x0 streams and generated latents exactly equal on both domains | Passed |
| Wan GPU regression | Single A100 80GB, 14B RGB adapter + patch embedding, 832×480×81, LightX2V four-step CFG1 | Passed |
| Preencoding | Native VAE encoding of a DROID + RoboCasa RGB subset on H200 | Passed; training parity pending |
| Cosmos distributed training/resume | Optimizer updates, separate-process resume, validation, gradient audit and export inference | Not yet GPU-validated |
| Super native-reference equivalence | Comparison against the original research recipe | Pending |

## Inference timing and repeatability

Nano loaded once for two 81-frame clips: 56.70 s for DROID and 56.00 s for
RoboCasa, with 36.23 GiB peak allocated GPU memory and all 36 layers resident.
Each generation used 12 denoiser calls. Completed-output restart skipped both
videos without loading the model. A clean exported checkout resolved relative
input paths and reproduced a third output as a byte-identical MP4.

Super used 35 steps / CFG6 / shift10 (70 denoiser calls), with CPU offload on a
single A100 80GB. The 133-frame clip took 2,244 s with 53.76 GiB peak allocated
VRAM; the 397-frame clip took 12,300 s with 72.34 GiB. A repeated 133-frame
sample produced exactly equal generated latents and a byte-identical MP4.
Restart skipped all completed outputs without model loading. These timings
describe the tested A100 offload configuration, not H200 throughput.

## Scope

Input schema and prompt examples are included; users supply their own media.
Wan RGB/depth and Cosmos3-Super adapters are public, with remote checksums and
anonymous access verified. See the [checkpoint guide](../checkpoints.md).

No new full-data training run or metric re-evaluation is part of this code
integration. GPU acceptance of Cosmos distributed training/resume and Super
native-reference equivalence remain outstanding.
