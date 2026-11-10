"""RoFacto's aligned RGB controls for Cosmos3 Nano and Super.

CUDA and upstream Cosmos imports are deliberately lazy so input validation and
CLI help work without installing the model runtime.
"""
FRAMEWORK_REVISION = "96303bb0bdd14d9efa18f24d8ef98c7a0bfb8412"
PROFILES = {
    "super": {"revision": "f543c56225b2e04d0ad141e29655be3a45d9c455", "layers": 64,
              "frozen": 63997928640, "tensors": 512, "rank16": 39845888},
    "nano": {"revision": "e59a53c25979a090fa8706c9acc0c254a6e89b92", "layers": 36,
             "frozen": 15156222144, "tensors": 288, "rank16": 15335424},
}
