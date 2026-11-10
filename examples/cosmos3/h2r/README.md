# Prepared H2R inputs

Follow the [H2R quickstart](../../../docs/cosmos3/quickstart.md) to install
the public runtime, download the Super adapter/base/VAE, and generate from your inputs.

Copy `input.example.json` and `prompt.example.json` to your input directory.
Supply your own `static.png` and `mesh.mp4` at 832×480. The image should show the
scene with the human removed. The mesh video should use the same camera and timing.
Edit the prompt to describe the actual robot, scene and motion; the example is a
format illustration, not a prompt from an evaluated clip.

Run the command in [the inference guide](../../../docs/cosmos3/inference.md).
No raw demonstration, GT video, retargeting database or cluster-specific directory
is needed once these two conditions are prepared. Existing DexYCB preprocessing
remains under `data_processing/dexycb`; prepared ALLEX/RB-Y1 conditions use this same
interface.

Input media are supplied by the user. Download pretrained adapters from
[Hugging Face](https://huggingface.co/byungjun-kim/RoFacto-Cosmos3-Super).
