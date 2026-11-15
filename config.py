"""All paths and constants for the project, in one place.

Large files (data, model weights, features, results) live under ROOT, which
defaults to FarmShare scratch. On another cluster: `export EVACC_ROOT=/path`.
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("EVACC_ROOT", "/scratch/users/liubr/aa228"))

HF_HOME = ROOT / "hf_cache"                    # HuggingFace model cache
SSV2_DIR = ROOT / "data" / "ssv2"              # labels.json, train.json, validation.json
SSV2_VIDEOS = SSV2_DIR / "videos"              # <id>.webm files
FEATURES_DIR = ROOT / "features"               # features_<tag>_shard<i>.h5
RESULTS_DIR = ROOT / "results"                 # tables + figures

# Encoder: smallest V-JEPA 2 (ViT-L/16, 256px) with its frozen-backbone SSv2 attentive probe.
MODEL_ID = "facebook/vjepa2-vitl-fpc16-256-ssv2"
MODEL_TAG = "vitl16_ssv2"                      # used in feature file names

# Observation windows: o_t = K frames sampled every FRAME_STEP raw frames, starting at raw frame t*STRIDE.
# (The SSv2 probe was trained with K=16 frames at step 4 over ~45-frame, 12 fps videos.)
K = 16
FRAME_STEP = 2
STRIDE = 4
IMG_SIZE = 256
NUM_CLASSES = 174

os.environ.setdefault("HF_HOME", str(HF_HOME))
