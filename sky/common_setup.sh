#!/bin/bash
# Shared SkyPilot setup: a Python 3.12 venv (on local disk, not ~/persist) with the packages from environment.yml.
set -euo pipefail
command -v uv >/dev/null || pip install -q uv
export PATH=$HOME/.local/bin:$PATH
[ -x ~/evacc-venv/bin/python ] || uv venv -q -p 3.12 ~/evacc-venv
uv pip install -q --python ~/evacc-venv/bin/python \
    torch==2.11.0 torchvision==0.26.0 transformers==5.8.0 huggingface_hub \
    decord==0.6.0 h5py numpy pandas matplotlib pytest
