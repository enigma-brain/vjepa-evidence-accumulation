"""Negative control: raw pixels instead of V-JEPA 2, on exactly the same observation windows.

Each window (same K frames / frame step / stride as extract_features.py, same resize + crop + normalize)
becomes its frames average-pooled to SxS RGB and flattened: K * 3 * S * S values (3072 for S=8).
No GPU needed: it is just decoding. The output has the same layout as a features file (emb_mean holds the
pixel vector; no logits), so train_probe.py and compare_models.py work unchanged:

    python extract_pixels.py --split validation
    python extract_pixels.py --split train --max-videos 50000
    python train_probe.py --tag pixels8
    python compare_models.py vitl16_ssv2:linear pixels8:linear

Output: FEATURES_DIR/features_pixels<S>_<split>_fs<step>_shard0of1.h5
"""
import argparse
import time

import h5py
import numpy as np
import torch
import torch.nn.functional as F

import config
from extract_features import VideoDataset, load_split, preprocess_on_device, resample_index, window_indices


class PixelDataset(VideoDataset):
    """Decode + preprocess in the worker; returns per-frame pooled pixels [T, 3, S, S]."""

    def __init__(self, videos, size):
        super().__init__(videos)
        self.size = size

    def __getitem__(self, i):
        i, frames, fps = super().__getitem__(i)
        if len(frames) == 0:
            return i, torch.zeros(0, 3, self.size, self.size), fps
        torch.set_num_threads(1)
        x = preprocess_on_device(frames, "cpu", torch.float32, 256)  # [T,3,256,256], same as the encoder sees
        return i, F.adaptive_avg_pool2d(x, self.size), fps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default=config.EVAL_SPLIT)
    p.add_argument("--max-videos", type=int, default=None, help="same seed-0 subset as extract_features.py")
    p.add_argument("--size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=31)
    args = p.parse_args()

    split = f"{args.split}-{args.max_videos}" if args.max_videos else args.split
    out = config.FEATURES_DIR / f"features_pixels{args.size}_{split}_{config.WINDOW}_shard0of1.h5"
    if out.exists():
        print(f"{out} exists, skipping.")
        return
    videos = load_split(args.split)
    if args.max_videos:
        keep = np.sort(np.random.default_rng(0).permutation(len(videos))[: args.max_videos])
        videos = [videos[i] for i in keep]
    print(f"{len(videos)} videos -> {out}", flush=True)

    loader = torch.utils.data.DataLoader(PixelDataset(videos, args.size), batch_size=None,
                                         num_workers=args.num_workers, prefetch_factor=4)
    n_frames = np.zeros(len(videos), dtype=np.int32)
    emb, vidx, ts = [], [], []
    t0 = time.time()
    for n, (i, x, fps) in enumerate(loader):
        n_frames[i] = len(x)
        if len(x) == 0:
            continue
        rs = resample_index(len(x), fps)
        idx = rs[window_indices(len(rs), config.K, config.FRAME_STEP, config.STRIDE, config.END_ALIGN)]
        emb.append(x[torch.from_numpy(idx)].flatten(1).half().numpy())  # [n_win, K*3*S*S]
        vidx.append(np.full(len(idx), i))
        ts.append(np.arange(len(idx)))
        if (n + 1) % 2000 == 0:
            print(f"video {n + 1}/{len(videos)}  {(n + 1) / (time.time() - t0):.0f} videos/s", flush=True)

    tmp = out.with_suffix(".tmp.h5")
    with h5py.File(tmp, "w") as f:
        f["emb_mean"] = np.concatenate(emb)
        f["video_idx"] = np.concatenate(vidx)
        f["t"] = np.concatenate(ts)
        f["video_id"] = np.array([str(v) for v, _ in videos], dtype="S")
        f["label"] = np.array([l for _, l in videos])
        f["n_frames"] = n_frames
        f.attrs.update(features=f"pixels {args.size}x{args.size}", k=config.K, fps=config.FPS or 0, window=config.WINDOW, frame_step=config.FRAME_STEP, stride=config.STRIDE, end_align=config.END_ALIGN)
    tmp.rename(out)
    print(f"done: {sum(len(e) for e in emb)} windows in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
