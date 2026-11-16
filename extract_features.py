"""Run V-JEPA 2 (+ its SSv2 attentive probe) on sliding windows of every SSv2 video.

For each video we cut observation windows o_0, o_1, ... :
    window t = K frames, taken every FRAME_STEP raw frames, starting at raw frame t * STRIDE
(frames past the end of the video repeat the last frame). For each window we save the
probe logits p(y | o_t) and two embeddings. We also save one "full" clip per video
(16 frames, step 4 from frame 0, i.e. how the probe was trained/evaluated) as a sanity check.

Output: FEATURES_DIR/features_<tag>_<split>_shard<i>of<n>.h5 with
    logits [N,174] f32, emb_attn [N,1024] f16, emb_mean [N,1024] f16,
    video_idx [N], t [N]                      (one row per window)
    video_id [V], label [V], n_frames [V], full_logits [V,174]   (one row per video)

GPU utilization: CPU workers only decode (uint8, native res); resize/crop/normalize happen
on the GPU; windows from many videos are packed into fixed-size batches; bf16 + SDPA.

Usage:
    python extract_features.py --split validation --shard 0 --num-shards 4
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
from transformers import VJEPA2ForVideoClassification

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
RESIZE_SHORT = 292  # = 256 * 256/224, from the HF video processor config


# ---------------------------------------------------------------- data
def load_split(split):
    """List of (video_id, label) for an SSv2 split. Label index follows labels.json (== model id2label)."""
    labels = json.load(open(config.SSV2_DIR / "labels.json"))
    items = json.load(open(config.SSV2_DIR / f"{split}.json"))
    out = []
    for it in items:
        name = it["template"].replace("[", "").replace("]", "")
        out.append((it["id"], int(labels[name])))
    present = [v for v in out if (config.SSV2_VIDEOS / f"{v[0]}.webm").exists()]
    if len(present) < len(out):
        print(f"[warn] {len(out) - len(present)} of {len(out)} {split} videos missing on disk", flush=True)
    return present


def window_indices(n_frames, k, frame_step, stride):
    """[n_windows, k] raw frame indices. Windows start every `stride` frames while the window fits;
    a video shorter than one window gets a single window padded with its last frame."""
    span = (k - 1) * frame_step + 1
    starts = np.arange(0, max(n_frames - span, 0) + 1, stride)
    idx = starts[:, None] + np.arange(k)[None, :] * frame_step
    return np.minimum(idx, n_frames - 1)


class VideoDataset(torch.utils.data.Dataset):
    """Returns all decoded frames of one video (uint8, T x H x W x 3, native resolution)."""

    def __init__(self, videos):
        self.videos = videos

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, i):
        from decord import VideoReader, cpu  # import inside worker

        vid, label = self.videos[i]
        try:
            vr = VideoReader(str(config.SSV2_VIDEOS / f"{vid}.webm"), ctx=cpu(0), num_threads=1)
            frames = torch.from_numpy(vr.get_batch(list(range(len(vr)))).asnumpy())
        except Exception as e:  # corrupt video: skip it
            print(f"[warn] failed to decode {vid}: {e}", flush=True)
            frames = torch.zeros(0, 1, 1, 3, dtype=torch.uint8)
        return i, frames


# ---------------------------------------------------------------- model
def preprocess_on_device(frames, device, dtype):
    """uint8 [T,H,W,3] -> normalized [T,3,256,256] on device (resize short side to 292, center crop)."""
    x = frames.to(device, non_blocking=True).permute(0, 3, 1, 2).float() / 255.0
    h, w = x.shape[-2:]
    s = RESIZE_SHORT / min(h, w)
    x = F.interpolate(x, size=(round(h * s), round(w * s)), mode="bilinear", align_corners=False, antialias=True)
    h, w = x.shape[-2:]
    top, left = (h - config.IMG_SIZE) // 2, (w - config.IMG_SIZE) // 2
    x = x[:, :, top : top + config.IMG_SIZE, left : left + config.IMG_SIZE]
    x = (x - MEAN.to(device)) / STD.to(device)
    return x.to(dtype)


@torch.inference_mode()
def run_model(model, clips):
    """clips [B,K,3,H,W] -> logits [B,174], attentive-pooled emb [B,D], mean-pooled emb [B,D]."""
    tokens = model.vjepa2(pixel_values_videos=clips, skip_predictor=True).last_hidden_state
    emb_attn = model.pooler(tokens)
    logits = model.classifier(emb_attn)
    return logits.float(), emb_attn, tokens.mean(1)


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
    p.add_argument("--frame-step", type=int, default=config.FRAME_STEP)
    p.add_argument("--stride", type=int, default=config.STRIDE)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=None, help="output .h5 path (default: under FEATURES_DIR)")
    args = p.parse_args()

    config.FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else config.FEATURES_DIR / (
        f"features_{config.MODEL_TAG}_{split_name(args)}_fs{args.frame_step}_shard{args.shard}of{args.num_shards}.h5"
    )
    if out.exists():
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
    model = VJEPA2ForVideoClassification.from_pretrained(config.MODEL_ID, attn_implementation="sdpa", torch_dtype=dtype)
    model.to(device).eval()

    loader = torch.utils.data.DataLoader(
        VideoDataset(videos),
        batch_size=None,  # one video per item; we pack windows ourselves
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        prefetch_factor=4 if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
    )

    V = len(videos)
    n_frames = np.zeros(V, dtype=np.int32)
    full_logits = np.full((V, config.NUM_CLASSES), np.nan, dtype=np.float32)
    rows = {"logits": [], "emb_attn": [], "emb_mean": [], "video_idx": [], "t": []}

    # Buffer of preprocessed windows waiting for a full batch (+ their (video_idx, t) tags).
    buf_clips, buf_tags = [], []

    def flush(n):
        clips = torch.cat(buf_clips)
        tags = np.concatenate(buf_tags)
        logits, ea, em = run_model(model, clips[:n])
        rows["logits"].append(logits.cpu().numpy())
        rows["emb_attn"].append(ea.half().cpu().numpy())
        rows["emb_mean"].append(em.half().cpu().numpy())
        rows["video_idx"].append(tags[:n, 0])
        rows["t"].append(tags[:n, 1])
        buf_clips[:], buf_tags[:] = ([clips[n:]], [tags[n:]]) if n < len(tags) else ([], [])

    t0, n_done, n_batches = time.time(), 0, 0
    for i, frames in loader:
        n_frames[i] = len(frames)
        if len(frames) == 0:
            continue
        x = preprocess_on_device(frames, device, dtype)  # [T,3,256,256]

        # sanity-check clip: how the probe was evaluated (16 frames, step 4, from frame 0)
        full_idx = torch.from_numpy(window_indices(len(frames), config.K, 4, 1)[0]).to(device)
        full_logits[i] = run_model(model, x[full_idx][None])[0][0].cpu().numpy()

        idx = window_indices(len(frames), config.K, args.frame_step, args.stride)
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
        f["video_id"] = np.array([int(v) for v, _ in videos])
        f["label"] = np.array([l for _, l in videos])
        f["n_frames"] = n_frames
        f["full_logits"] = full_logits
        f.attrs.update(model_id=config.MODEL_ID, k=config.K, frame_step=args.frame_step, stride=args.stride)
    tmp.rename(out)  # atomic: a shard file only exists once it is complete

    labels = np.array([l for _, l in videos])
    ok = n_frames > 0
    print(f"full-clip probe top-1 acc: {(full_logits[ok].argmax(1) == labels[ok]).mean():.3f}")
    with h5py.File(out) as f:
        per_window = (f["logits"][:].argmax(1) == labels[f["video_idx"][:]]).mean()
    print(f"per-window probe top-1 acc: {per_window:.3f}")


if __name__ == "__main__":
    main()
