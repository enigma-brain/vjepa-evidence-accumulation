"""All paths and constants for the project, in one place.

Large files (data, model weights, features, results) live under ROOT, which
defaults to FarmShare scratch. On another cluster: `export EVACC_ROOT=/path`.
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("EVACC_ROOT", "/scratch/users/liubr/aa228"))

HF_HOME = ROOT / "hf_cache"                    # HuggingFace model cache
FEATURES_DIR = ROOT / "features"               # features_<tag>_shard<i>.h5
RESULTS_DIR = ROOT / "results"                 # tables + figures

# Dataset: EVACC_DATASET=ssv2 (default) or diving48. Everything below follows from it.
# Common window protocol (identical for every encoder compared): each video is resampled to FPS=30 Hz by
# timestamp (frame i at time i/30 s takes native frame floor(i * native_fps / 30); SSv2's 12 fps repeats frames),
# then observation window o_t = K=16 consecutive 30 Hz frames (533 ms), starting at 30 Hz frame t*STRIDE (16: no
# overlap), plus one end-aligned last window so the end of the clip is always seen. 16 frames = the neural model's
# input (16 frames in, 16 firing-rate bins out).
#   full_segments   the "full clip" sanity check of Meta's probe averages it over this many evenly spaced clips at
#                   NATIVE fps, as Meta's eval does (SSv2: 1; Diving48: 4)
# EVACC_FPS=native restores the old native-fps windows (k, frame_step, stride below) to re-read old *_fs<step>_* files.
DATASETS = {
    # SSv2: ~45-frame, 12 fps videos (~3.7 s -> ~7 windows at 30 Hz); Meta's probe was trained on 16 frames at step 4.
    "ssv2": dict(num_classes=174, ext=".webm", eval_split="validation", train_split="train-50000",
                 native_fps=12, legacy=dict(k=16, frame_step=2, stride=4, end_align=False), full_segments=1,
                 model_id="facebook/vjepa2-vitl-fpc16-256-ssv2", model_tag="vitl16_ssv2"),
    # Diving48 V2: test clips median 103 frames (p10 61, p90 174); Meta's probe was trained on 32 frames at step 2.
    "diving48": dict(num_classes=48, ext=".mp4", eval_split="test", train_split="train",
                     native_fps=30, legacy=dict(k=16, frame_step=1, stride=16, end_align=True), full_segments=4,
                     model_id="facebook/vjepa2-vitl-fpc32-256-diving48", model_tag="vitl16_diving48"),
}
DATASET = os.environ.get("EVACC_DATASET", "ssv2")


def source_video(video_id):
    """Group key for held-out splits: clips from one source video share scene, camera and people.
    Diving48 clips "<youtube id>_<clip #>" come ~190 per competition broadcast; SSv2 clips are independent."""
    v = video_id.decode() if isinstance(video_id, bytes) else str(video_id)
    return v.rsplit("_", 1)[0] if DATASET == "diving48" else v

_d = DATASETS[DATASET]
DATA_DIR = ROOT / "data" / DATASET                 # labels.json, <split>.json
VIDEOS = Path(os.environ.get("EVACC_VIDEOS", DATA_DIR / "videos"))   # <id><ext> files (override: fast local copy)
VIDEO_EXT = _d["ext"]
NUM_CLASSES = _d["num_classes"]
EVAL_SPLIT, TRAIN_SPLIT = _d["eval_split"], _d["train_split"]
FULL_SEGMENTS = _d["full_segments"]
DEFAULT_NATIVE_FPS = _d["native_fps"]   # fallback if a video reports no fps
_fps = os.environ.get("EVACC_FPS", "30")
if _fps == "native":   # old protocol: windows in native frames
    FPS = None
    K, FRAME_STEP, STRIDE, END_ALIGN = (_d["legacy"][k] for k in ("k", "frame_step", "stride", "end_align"))
else:
    FPS = float(_fps)
    K, FRAME_STEP, STRIDE, END_ALIGN = 16, 1, 16, True


def window_tag(fps=FPS, k=K, frame_step=FRAME_STEP, stride=STRIDE):
    """The part of every feature / probe / results name that fixes the windows: '30hz_k16s16' (old: 'fs2')."""
    if fps is None:
        return f"fs{frame_step}"
    return f"{fps:g}hz_k{k}s{stride}" + (f"fs{frame_step}" if frame_step != 1 else "")


WINDOW = os.environ.get("EVACC_WINDOW", window_tag())   # override only to read files made with other windows

# Encoder (+ its frozen-backbone attentive probe, if the checkpoint has one).
# Override per run with EVACC_MODEL_ID / EVACC_MODEL_TAG (e.g. from a SkyPilot --env).
MODEL_ID = os.environ.get("EVACC_MODEL_ID", _d["model_id"])
MODEL_TAG = os.environ.get("EVACC_MODEL_TAG", _d["model_tag"])   # used in feature file names

# Encoder layers whose (final-LayerNorm'ed) tokens Meta's probe reads, concatenated along the token axis
# (configs/eval/*/diving48.yaml: out_layers, vit_encoder_multiclip_multilevel). HF's VJEPA2ForVideoClassification
# always feeds only the last layer, which silently breaks these probes. Default: last layer only (the SSv2 probes).
PROBE_OUT_LAYERS = {
    "facebook/vjepa2-vitl-fpc32-256-diving48": [17, 19, 21, 23],
    "facebook/vjepa2-vitg-fpc32-384-diving48": [24, 29, 34, 39],
}.get(MODEL_ID)

os.environ.setdefault("HF_HOME", str(HF_HOME))
