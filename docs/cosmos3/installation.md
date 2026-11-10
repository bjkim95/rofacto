# Cosmos3 environment

Cosmos3 is an additional RoFacto backend. The paper's Wan implementation remains
under `training/wan`; its environment is installed with `setup_env.sh`.

Use a separate environment for Cosmos3. The pinned runtime uses Linux x86-64,
Python 3.13.9, PyTorch 2.10.0 with CUDA 12.8, and NVIDIA's framework commit
`96303bb0bdd14d9efa18f24d8ef98c7a0bfb8412`. Install Git, FFmpeg/ffprobe and
[uv](https://docs.astral.sh/uv/) before running:

```bash
./setup_cosmos3.sh
source .deps/cosmos-framework/.venv/bin/activate
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=''
python -m training.cosmos3.inference --help
```

The script verifies both the framework revision and its dependency-lock hash.
The `cu128-train` group selects CUDA wheels; the `train` extra supplies the
native framework's IO, training and configuration dependencies. The installer
never modifies the Wan environment or an existing framework at another revision.
The first installation downloads large CUDA packages; allow tens of GB of disk
space for both the environment and the package cache.

For a small home partition, select another cache and environment destination:

```bash
export UV_CACHE_DIR=/scratch/rofacto-uv-cache
export UV_PROJECT_ENVIRONMENT=/scratch/rofacto-cosmos3-env
./setup_cosmos3.sh
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
```

Run entrypoints from the RoFacto repository root. No experiment directory belongs
in `PYTHONPATH`. The framework is installed as an external editable dependency.
Read [checkpoints.md](checkpoints.md) for the separate base, VAE and adapter files.

Implementation validation is tracked in [verification.md](verification.md).
A source installation passing does not by itself establish that a particular GPU
or video length fits; use the tested profiles listed there.
