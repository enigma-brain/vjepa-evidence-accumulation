"""Train our own linear probe p(y | o_t) on frozen V-JEPA 2 window embeddings, then apply it to the eval split.

Deliberately simple and identical for every encoder, so differences between encoders are not
absorbed by a powerful readout:
    o_t  -> mean of the encoder's output tokens (emb_mean, 1024-d)  -> standardize -> Linear -> softmax
No pretrained head is used. Trained with cross-entropy on *train* windows, with 10% of train held out to
pick the epoch by NLL, so the eval split stays clean for fitting/testing the agents. The held-out set is
whole source videos (config.source_video): Diving48 cuts ~190 clips from each competition broadcast and its
test split uses different broadcasts, so held-out clips from the training broadcasts would reward
memorizing the competition. Test accuracy per epoch is printed as a diagnostic only (never used to choose).

Output:
    FEATURES_DIR/probe-linear_<tag>_<train split>_fs<step>.pt            (weights + standardization)
    FEATURES_DIR/probe-linear_<tag>_<eval split>_fs<step>_shard<i>of<n>.h5  (logits, same rows as the features shard)
    (with --key other than emb_mean, "probe-linear" becomes "probe-linear-<key>")

Usage: python train_probe.py [--train-split train-50000] [--epochs 30]
"""
import argparse
import glob

import h5py
import numpy as np
import torch
import torch.nn.functional as F

import config


def load(pattern, key="emb_mean"):
    X, y, vid = [], [], []
    for fn in sorted(glob.glob(pattern)):
        with h5py.File(fn) as f:
            X.append(np.concatenate([f[k][:].astype(np.float32) for k in key.split("+")], 1))
            vi = f["video_idx"][:]
            y.append(f["label"][:][vi])
            vid.append(f["video_id"][:][vi])
    assert X, f"no files match {pattern}"
    return np.concatenate(X), np.concatenate(y), np.concatenate(vid)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default=config.MODEL_TAG, help="features to read, e.g. pixels8 for the pixel control")
    p.add_argument("--train-split", default=config.TRAIN_SPLIT)
    p.add_argument("--eval-split", default=config.EVAL_SPLIT)
    p.add_argument("--window", default=config.WINDOW, help="window tag in file names (default: the current protocol)")
    p.add_argument("--train-window", default=None, help="window tag of the train features (default: --window)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--train-shards", default="shard*", help="train shard files to use, e.g. shard0of3 (one third)")
    p.add_argument("--suffix", default="", help="appended to the probe name, e.g. -third: obs linear-third / linear-emb_ml-third")
    p.add_argument("--key", default="emb_mean", help="embedding(s) in the features files, e.g. emb_ml or emb_mean+emb_std")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    tag = f"{args.tag}_{args.train_split}_{args.train_window or args.window}"
    X, y, vid = load(str(config.FEATURES_DIR / f"features_{tag}_{args.train_shards}.h5"), args.key)
    print(f"train windows: {len(y)} from {len(np.unique(vid))} videos")

    # hold out 10% of train *source videos* (not windows: windows of one video are near-duplicates;
    # not clips either, for Diving48: clips of one broadcast share scene and divers)
    rng = np.random.default_rng(0)
    src = np.array([config.source_video(v) for v in vid])
    srcs = np.unique(src)
    held = np.isin(src, rng.choice(srcs, size=max(1, len(srcs) // 10), replace=False))
    print(f"held out {held.mean():.1%} of windows from {max(1, len(srcs) // 10)} of {len(srcs)} source videos")

    # eval-split windows: test accuracy per epoch, a diagnostic only
    eval_pattern = str(config.FEATURES_DIR / f"features_{args.tag}_{args.eval_split}_{args.window}_shard*.h5")
    eval_files = sorted(glob.glob(eval_pattern))
    Xe, ye, _ = load(eval_pattern, args.key) if eval_files else (None, None, None)

    mu, sd = X[~held].mean(0), X[~held].std(0) + 1e-6
    Xt = torch.tensor((X - mu) / sd, device=device)
    yt = torch.tensor(y, device=device)
    tr, ho = torch.tensor(~held, device=device), torch.tensor(held, device=device)
    Xtr, ytr = Xt[tr], yt[tr]
    Xet = torch.tensor((Xe - mu) / sd, device=device) if Xe is not None else None

    lin = torch.nn.Linear(X.shape[1], config.NUM_CLASSES).to(device)
    opt = torch.optim.AdamW(lin.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = (np.inf, None, -1, 0.0)  # (nll, weights, epoch, acc): keep the best-calibrated epoch,
                                     # since the belief uses the probabilities, not just the argmax
    for ep in range(args.epochs):
        perm = torch.randperm(len(ytr), device=device)
        for i in range(0, len(perm), 1024):
            b = perm[i : i + 1024]
            loss = F.cross_entropy(lin(Xtr[b]), ytr[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            out = lin(Xt[ho])
            acc = (out.argmax(1) == yt[ho]).float().mean().item()
            nll = F.cross_entropy(out, yt[ho]).item()
            test_acc = (lin(Xet).argmax(1).cpu().numpy() == ye).mean() if Xet is not None else np.nan
        print(f"epoch {ep + 1:2d}  held-out acc {acc:.3f}  nll {nll:.3f}   ({args.eval_split} acc {test_acc:.3f}, diagnostic)", flush=True)
        if nll < best[0]:
            best = (nll, {k: v.clone() for k, v in lin.state_dict().items()}, ep + 1, acc)
    lin.load_state_dict(best[1])
    print(f"kept epoch {best[2]} (held-out nll {best[0]:.3f}, acc {best[3]:.3f})")

    prefix = ("probe-linear" if args.key == "emb_mean" else f"probe-linear-{args.key}") + args.suffix
    ckpt = config.FEATURES_DIR / f"{prefix}_{tag}.pt"
    torch.save({"state_dict": best[1], "mu": mu, "sd": sd, "epoch": best[2], "heldout_nll": best[0], "heldout_acc": best[3]}, ckpt)

    # apply to every evaluation shard, keeping row order identical to the features file
    accs = []
    for fn in eval_files:
        with h5py.File(fn) as f:
            E = np.concatenate([f[k][:].astype(np.float32) for k in args.key.split("+")], 1)
            labels = f["label"][:][f["video_idx"][:]]
        with torch.no_grad():
            logits = lin(torch.tensor((E - mu) / sd, device=device)).cpu().numpy()
        accs.append((logits.argmax(1) == labels).mean())
        out = fn.replace("/features_", f"/{prefix}_")
        with h5py.File(out, "w") as f:
            f["logits"] = logits.astype(np.float32)
            f.attrs.update(probe=str(ckpt))
        print(f"wrote {out}")
    print(f"{args.eval_split} per-window acc (linear probe): {np.mean(accs):.3f}")


if __name__ == "__main__":
    main()
