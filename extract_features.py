"""Run a V-JEPA 2 encoder (+ its attentive probe, if it has one) on sliding windows of every video of a
dataset (config.DATASET: SSv2 or Diving48).

For each video we cut observation windows o_0, o_1, ... :
    the video is resampled to config.FPS (30 Hz) by timestamp, then window t = K=16 consecutive 30 Hz frames
    starting at 30 Hz frame t * STRIDE (16: no overlap), plus one end-aligned last window
(frames past the end of a too-short video repeat the last frame). For each window we save the
probe logits p(y | o_t) and two embeddings. As a sanity check we also save "full-clip" logits per video, as
the probe was trained/evaluated: config.FULL_SEGMENTS evenly spaced clips of frames_per_clip frames spanning
~64 raw frames each (SSv2: 1 clip from frame 0; Diving48: 4), whose tokens go through the probe together
(Meta's ClipAggregation), one center-crop view.

Probes trained on several encoder layers (config.PROBE_OUT_LAYERS, e.g. Meta's Diving48 probes) get those
layers' tokens, each through the final LayerNorm, concatenated along the token axis, as in Meta's eval code.

Any HF V-JEPA 2 checkpoint works (config.MODEL_ID): the crop/resize follow the checkpoint's crop_size
(256 or 384), and the embedding width its hidden_size (D = 1024 ViT-L, 1280 ViT-H, 1408 ViT-g).
Checkpoints without a head for this dataset (e.g. the pretrained encoders) only get the embeddings.

Output: FEATURES_DIR/features_<tag>_<split>_shard<i>of<n>.h5 with
    emb_mean [N,D] f16, emb_std [N,D] f16, emb_ml [N,4D] f16, video_idx [N], t [N]   (one row per window)
        emb_std: per-channel std of last-layer tokens; emb_ml: token means of 4 upper layers (feature_layers),
        concatenated
    video_id [V] (bytes), label [V], n_frames [V] (native), n_frames_resampled [V], native_fps [V]   (per video)
    with a head also: logits [N,C] f32, emb_attn [N,D] f16, full_logits [V,C]

GPU utilization: CPU workers only decode (uint8, native res); resize/crop/normalize happen
on the GPU; windows from many videos are packed into fixed-size batches; bf16 + SDPA.

Usage:
    python extract_features.py --split validation --shard 0 --num-shards 4
    EVACC_DATASET=diving48 python extract_features.py --split test
    python extract_features.py --limit 50 --device cpu          # smoke test
"""
import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

import config  # sets HF_HOME before transformers is imported
from transformers import AutoConfig, VJEPA2ForVideoClassification, VJEPA2Model

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ---------------------------------------------------------------- data
def video_path(vid):
    return config.VIDEOS / f"{vid}{config.VIDEO_EXT}"


def load_split(split):
    """List of (video_id: str, label) for a split. Label indices match the HF checkpoints' id2label."""
    items = json.load(open(config.DATA_DIR / f"{split}.json"))
    if config.DATASET == "ssv2":  # label = index of the template in labels.json
        labels = json.load(open(config.DATA_DIR / "labels.json"))
        out = [(it["id"], int(labels[it["template"].replace("[", "").replace("]", "")])) for it in items]
    else:  # diving48 V2: label = index into labels.json (the vocab); clips are pre-trimmed (start_frame 0)
        out = [(it["vid_name"], int(it["label"])) for it in items]
    present = [v for v in out if video_path(v[0]).exists()]
    if len(present) < len(out):
        print(f"[warn] {len(out) - len(present)} of {len(out)} {split} videos missing on disk", flush=True)
    return present


def resample_index(n_native, native_fps, fps=config.FPS):
    """Native frame index of each frame of the video resampled to `fps` by timestamp: frame i (time i/fps)
    -> native frame floor(i * native_fps / fps). fps=None: identity (native frames). Duration is kept, so a
    12 fps video repeats each frame 2.5x on average and a 30 fps one maps to itself."""
    if fps is None or n_native == 0:
        return np.arange(n_native)
    n = max(1, int(np.floor(n_native * fps / native_fps + 1e-6)))
    return np.minimum(np.floor(np.arange(n) * native_fps / fps + 1e-6).astype(int), n_native - 1)


def window_indices(n_frames, k, frame_step, stride, end_align=False):
    """[n_windows, k] raw frame indices. Windows start every `stride` frames while the window fits;
    a video shorter than one window gets a single window padded with its last frame.
    end_align: if the last regular window stops before the last frame, add one window that ends on it."""
    span = (k - 1) * frame_step + 1
    starts = np.arange(0, max(n_frames - span, 0) + 1, stride)
    if end_align and n_frames > span and starts[-1] + span < n_frames:
        starts = np.append(starts, n_frames - span)
    idx = starts[:, None] + np.arange(k)[None, :] * frame_step
    return np.minimum(idx, n_frames - 1)


class VideoDataset(torch.utils.data.Dataset):
    """Returns (i, all decoded frames of one video (uint8, T x H x W x 3, native resolution), native fps)."""

    def __init__(self, videos):
        self.videos = videos

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, i):
        from decord import VideoReader, cpu  # import inside worker

        vid, label = self.videos[i]
        try:
            vr = VideoReader(str(video_path(vid)), ctx=cpu(0), num_threads=1)
            frames = torch.from_numpy(vr.get_batch(list(range(len(vr)))).asnumpy())
            fps = float(vr.get_avg_fps())
        except Exception as e:  # corrupt video: skip it
            print(f"[warn] failed to decode {vid}: {e}", flush=True)
            frames, fps = torch.zeros(0, 1, 1, 3, dtype=torch.uint8), 0.0
        if not np.isfinite(fps) or fps <= 0:
            fps = float(config.DEFAULT_NATIVE_FPS)
        return i, frames, fps


# ---------------------------------------------------------------- model
def preprocess_on_device(frames, device, dtype, crop):
    """uint8 [T,H,W,3] -> normalized [T,3,crop,crop] on device (resize short side to crop*256/224, center crop),
    as the HF video processor does (256 -> 292, 384 -> 438)."""
    x = frames.to(device, non_blocking=True).permute(0, 3, 1, 2).float() / 255.0
    h, w = x.shape[-2:]
    s = int(crop * 256 / 224) / min(h, w)
    x = F.interpolate(x, size=(round(h * s), round(w * s)), mode="bilinear", align_corners=False, antialias=True)
    h, w = x.shape[-2:]
    top, left = (h - crop) // 2, (w - crop) // 2
    x = x[:, :, top : top + crop, left : left + crop]
    x = (x - MEAN.to(device)) / STD.to(device)
    return x.to(dtype)


def feature_layers(num_layers):
    """Encoder layers for emb_ml: the probe's layers when known, else 4 spread over the top quarter
    (24 layers -> 17,19,21,23 as Meta's ViT-L probes; 40 -> 24,29,34,39 as Meta's ViT-g probes)."""
    if config.PROBE_OUT_LAYERS:
        return list(config.PROBE_OUT_LAYERS)
    return {24: [17, 19, 21, 23], 40: [24, 29, 34, 39]}.get(
        num_layers, [round(num_layers * f) - 1 for f in (0.75, 5 / 6, 11 / 12, 1.0)])


def encode(model, clips, layers=()):
    """clips [B,K,3,H,W] -> (last-layer tokens [B,N,D], {i: layer-i tokens through the final LayerNorm})."""
    enc = (model if isinstance(model, VJEPA2Model) else model.vjepa2).encoder
    assert not layers or max(layers) < len(enc.layer), f"layers {layers} vs {len(enc.layer)} layers"
    h, outs = enc.embeddings(clips), {}
    for i, layer in enumerate(enc.layer):
        h = layer(h, None)[0]
        if i in layers:
            outs[i] = enc.layernorm(h)
    return enc.layernorm(h), outs


@torch.inference_mode()
def run_model(model, clips, probe_layers=None, feat_layers=()):
    """clips [B,K,3,H,W] -> (dict of per-window outputs, probe tokens [B,M,D]).
        emb_mean [B,D]    mean of last-layer tokens
        emb_std  [B,D]    std over tokens of the last layer (per channel)
        emb_ml   [B,L*D]  per-layer token means of feat_layers, concatenated (if feat_layers)
        logits [B,C], emb_attn [B,D]   the attentive probe, fed probe_layers' tokens concatenated along
                                       tokens (or the last layer); only with a classification head"""
    tokens, hs = encode(model, clips, set(probe_layers or ()) | set(feat_layers))
    probe_tokens = torch.cat([hs[i] for i in probe_layers], 1) if probe_layers else tokens
    out = {"emb_mean": tokens.mean(1), "emb_std": tokens.float().std(1)}
    if feat_layers:
        out["emb_ml"] = torch.cat([hs[i].mean(1) for i in feat_layers], -1)
    if not isinstance(model, VJEPA2Model):
        out["emb_attn"] = model.pooler(probe_tokens)
        out["logits"] = model.classifier(out["emb_attn"]).float()
    return out, probe_tokens


@torch.inference_mode()
def probe_joint(model, probe_tokens):
    """Probe over the tokens of several clips of one video at once [S,M,D] -> logits [C]."""
    return model.classifier(model.pooler(probe_tokens.reshape(1, -1, probe_tokens.shape[-1])))[0].float()


# ---------------------------------------------------------------- main
def split_name(args):
    """e.g. 'validation', or 'train-50000' for a random subset."""
    return f"{args.split}-{args.max_videos}" if args.max_videos else args.split


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="validation")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--limit", type=int, default=None, help="only the first N videos of the shard (smoke test)")
    p.add_argument("--max-videos", type=int, default=None, help="fixed random subset (seed 0) of the split, before sharding")
    p.add_argument("--batch-size", type=int, default=64, help="windows per forward pass")
    p.add_argument("--num-workers", type=int, default=7)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=None, help="output .h5 path (default: under FEATURES_DIR)")
    p.add_argument("--overwrite", action="store_true", help="recompute even if the output exists")
    args = p.parse_args()

    config.FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else config.FEATURES_DIR / (
        f"features_{config.MODEL_TAG}_{split_name(args)}_{config.WINDOW}_shard{args.shard}of{args.num_shards}.h5"
    )
    if out.exists() and not args.overwrite:
        print(f"{out} exists, skipping.")
        return

    videos = load_split(args.split)
    if args.max_videos:
        keep = np.sort(np.random.default_rng(0).permutation(len(videos))[: args.max_videos])
        videos = [videos[i] for i in keep]
    videos = videos[args.shard :: args.num_shards][: args.limit]
    print(f"{len(videos)} videos -> {out}", flush=True)

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    cfg = AutoConfig.from_pretrained(config.MODEL_ID)
    has_head = "VJEPA2ForVideoClassification" in (cfg.architectures or []) and cfg.num_labels == config.NUM_CLASSES
    model_cls = VJEPA2ForVideoClassification if has_head else VJEPA2Model
    model = model_cls.from_pretrained(config.MODEL_ID, attn_implementation="sdpa", torch_dtype=dtype)
    model.to(device).eval()
    # full clip as the probe saw it: frames_per_clip frames over ~64 raw frames (16 -> step 4, 32 -> step 2)
    full_k, full_step = cfg.frames_per_clip, max(1, 64 // cfg.frames_per_clip)
    full_span = (full_k - 1) * full_step + 1
    probe_layers = config.PROBE_OUT_LAYERS if has_head else None
    print(f"{config.MODEL_ID}: {model_cls.__name__}, crop {cfg.crop_size}, D={cfg.hidden_size}"
          f"{'' if has_head else f' (no {config.DATASET} head: embeddings only)'}"
          f"{f', probe on layers {config.PROBE_OUT_LAYERS}' if has_head and config.PROBE_OUT_LAYERS else ''}", flush=True)

    loader = torch.utils.data.DataLoader(
        VideoDataset(videos),
        batch_size=None,  # one video per item; we pack windows ourselves
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        prefetch_factor=4 if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
    )

    V = len(videos)
    n_frames = np.zeros(V, dtype=np.int32)    # native frames
    n_frames_rs = np.zeros(V, dtype=np.int32)  # frames after resampling to config.FPS
    native_fps = np.zeros(V, dtype=np.float32)
    full_logits = np.full((V, config.NUM_CLASSES), np.nan, dtype=np.float32)
    feat_layers = feature_layers(cfg.num_hidden_layers)
    print(f"emb_ml: mean tokens of layers {feat_layers}", flush=True)
    rows = {"video_idx": [], "t": []}  # + every output of run_model

    # Buffer of preprocessed windows waiting for a full batch (+ their (video_idx, t) tags).
    buf_clips, buf_tags = [], []

    def flush(n):
        clips = torch.cat(buf_clips)
        tags = np.concatenate(buf_tags)
        outs, _ = run_model(model, clips[:n], probe_layers, feat_layers)
        for k, v in outs.items():
            rows.setdefault(k, []).append(v.cpu().numpy() if k == "logits" else v.half().cpu().numpy())
        rows["video_idx"].append(tags[:n, 0])
        rows["t"].append(tags[:n, 1])
        buf_clips[:], buf_tags[:] = ([clips[n:]], [tags[n:]]) if n < len(tags) else ([], [])

    t0, n_done, n_batches = time.time(), 0, 0
    for i, frames, fps in loader:
        n_frames[i], native_fps[i] = len(frames), fps
        if len(frames) == 0:
            continue
        x = preprocess_on_device(frames, device, dtype, cfg.crop_size)  # [T,3,crop,crop]

        # sanity check: how the probe was evaluated (FULL_SEGMENTS evenly spaced clips, probed jointly)
        if has_head:
            starts = np.linspace(0, max(len(frames) - full_span, 0), config.FULL_SEGMENTS).round().astype(int)
            full_idx = np.minimum(starts[:, None] + np.arange(full_k)[None] * full_step, len(frames) - 1)
            seg_tokens = run_model(model, x[torch.from_numpy(full_idx).to(device)], probe_layers)[1]
            full_logits[i] = probe_joint(model, seg_tokens).cpu().numpy()

        rs = resample_index(len(frames), fps)  # windows live on the 30 Hz timeline
        n_frames_rs[i] = len(rs)
        idx = rs[window_indices(len(rs), config.K, config.FRAME_STEP, config.STRIDE, config.END_ALIGN)]
        buf_clips.append(x[torch.from_numpy(idx).to(device)])  # [n_win,K,3,H,W]
        buf_tags.append(np.stack([np.full(len(idx), i), np.arange(len(idx))], 1))

        while sum(len(t) for t in buf_tags) >= args.batch_size:
            flush(args.batch_size)
            n_done += args.batch_size
            n_batches += 1
            if n_batches % 50 == 0:
                print(f"video {i+1}/{V}  {n_done / (time.time() - t0):.1f} windows/s", flush=True)
    if buf_tags:
        n_done += sum(len(t) for t in buf_tags)
        flush(sum(len(t) for t in buf_tags))
    print(f"done: {n_done} windows in {time.time() - t0:.0f}s ({n_done / (time.time() - t0):.1f} windows/s)")

    tmp = out.with_suffix(".tmp.h5")
    with h5py.File(tmp, "w") as f:
        for k, v in rows.items():
            f[k] = np.concatenate(v)
        f["video_id"] = np.array([str(v) for v, _ in videos], dtype="S")
        f["label"] = np.array([l for _, l in videos])
        f["n_frames"] = n_frames
        f["n_frames_resampled"] = n_frames_rs
        f["native_fps"] = native_fps
        if has_head:
            f["full_logits"] = full_logits
        f.attrs.update(model_id=config.MODEL_ID, dataset=config.DATASET, k=config.K,
                       probe_out_layers=str(probe_layers), feature_layers=str(feat_layers),
                       full_segments=config.FULL_SEGMENTS, fps=config.FPS or 0, window=config.WINDOW, frame_step=config.FRAME_STEP, stride=config.STRIDE, end_align=config.END_ALIGN,
                       crop_size=cfg.crop_size)
    tmp.rename(out)  # atomic: a shard file only exists once it is complete

    if not has_head:
        return
    labels = np.array([l for _, l in videos])
    ok = n_frames > 0
    print(f"full-clip probe top-1 acc: {(full_logits[ok].argmax(1) == labels[ok]).mean():.3f}")
    with h5py.File(out) as f:
        per_window = (f["logits"][:].argmax(1) == labels[f["video_idx"][:]]).mean()
    print(f"per-window probe top-1 acc: {per_window:.3f}")


if __name__ == "__main__":
    main()
