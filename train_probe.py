"""Train our own linear probe p(y | o_t) on frozen V-JEPA 2 window embeddings, then apply it to validation.

Deliberately simple and identical for every encoder, so differences between encoders are not
absorbed by a powerful readout:
    o_t  -> mean of the encoder's output tokens (emb_mean, 1024-d)  -> standardize -> Linear -> softmax
No pretrained head is used. Trained with cross-entropy on SSv2 *train* windows (10% of train videos
held out to pick the epoch by NLL), so all of SSv2 validation stays clean for fitting/testing the agents.

Output:
    FEATURES_DIR/probe-linear_<tag>_<train split>_fs<step>.pt            (weights + standardization)
    FEATURES_DIR/probe-linear_<tag>_validation_fs<step>_shard<i>of<n>.h5  (logits, same rows as the features shard)

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
            X.append(f[key][:].astype(np.float32))
            vi = f["video_idx"][:]
            y.append(f["label"][:][vi])
            vid.append(f["video_id"][:][vi])
    assert X, f"no files match {pattern}"
    return np.concatenate(X), np.concatenate(y), np.concatenate(vid)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default=config.MODEL_TAG, help="features to read, e.g. pixels8 for the pixel control")
    p.add_argument("--train-split", default="train-50000")
    p.add_argument("--eval-split", default="validation")
    p.add_argument("--frame-step", type=int, default=config.FRAME_STEP)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    tag = f"{args.tag}_{args.train_split}_fs{args.frame_step}"
    X, y, vid = load(str(config.FEATURES_DIR / f"features_{tag}_shard*.h5"))
    print(f"train windows: {len(y)} from {len(np.unique(vid))} videos")

    # hold out 10% of train *videos* (not windows: windows of one video are near-duplicates)
    rng = np.random.default_rng(0)
    vids = np.unique(vid)
    held = np.isin(vid, rng.choice(vids, size=len(vids) // 10, replace=False))

    mu, sd = X[~held].mean(0), X[~held].std(0) + 1e-6
    Xt = torch.tensor((X - mu) / sd, device=device)
    yt = torch.tensor(y, device=device)
    tr, ho = torch.tensor(~held, device=device), torch.tensor(held, device=device)
    Xtr, ytr = Xt[tr], yt[tr]

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
        print(f"epoch {ep + 1:2d}  held-out acc {acc:.3f}  nll {nll:.3f}", flush=True)
        if nll < best[0]:
            best = (nll, {k: v.clone() for k, v in lin.state_dict().items()}, ep + 1, acc)
    lin.load_state_dict(best[1])
    print(f"kept epoch {best[2]} (held-out nll {best[0]:.3f}, acc {best[3]:.3f})")

    ckpt = config.FEATURES_DIR / f"probe-linear_{tag}.pt"
    torch.save({"state_dict": best[1], "mu": mu, "sd": sd, "epoch": best[2], "heldout_nll": best[0], "heldout_acc": best[3]}, ckpt)

    # apply to every evaluation shard, keeping row order identical to the features file
    accs = []
    for fn in sorted(glob.glob(str(config.FEATURES_DIR / f"features_{args.tag}_{args.eval_split}_fs{args.frame_step}_shard*.h5"))):
        with h5py.File(fn) as f:
            E = f["emb_mean"][:].astype(np.float32)
            labels = f["label"][:][f["video_idx"][:]]
        with torch.no_grad():
            logits = lin(torch.tensor((E - mu) / sd, device=device)).cpu().numpy()
        accs.append((logits.argmax(1) == labels).mean())
        out = fn.replace("/features_", "/probe-linear_")
        with h5py.File(out, "w") as f:
            f["logits"] = logits.astype(np.float32)
            f.attrs.update(probe=str(ckpt))
        print(f"wrote {out}")
    print(f"{args.eval_split} per-window acc (linear probe): {np.mean(accs):.3f}")


if __name__ == "__main__":
    main()
