"""Label-free, backbone-level analyses of the window embeddings (B6-B9). CPU.

Per encoder (one features tag), per layer of emb_ml (the token means of 4 upper layers, e.g. 17/19/21/23):
  B6 temporal predictability ("world model"): ridge regression z_t -> z_{t+1} between consecutive regular
     (non-overlapping) windows of a video, fitted on half of the source videos, scored on the other half
     (2-fold, grouped by config.source_video). Reported as held-out R^2 of z_{t+1}, next to two references:
     "copy" (z_{t+1} ~ z_t) and the R^2 of the CHANGE z_{t+1} - z_t (how much of what changes is predictable;
     high R^2 of z_{t+1} alone can just mean a static representation).
  B7 trajectory curvature: mean angle (degrees) between successive steps z_{t+1} - z_t of a video's
     trajectory over its regular windows (videos with >= 3). Straighter = more predictable by extrapolation
     (Henaff et al. 2019). Per video, so two encoders can be paired.
  B8 effective dimensionality: participation ratio (sum l)^2 / sum l^2 of the eigenvalues l of the
     correlation matrix of all windows; fraction of variance within videos (vs between videos).
     95% CI by leave-one-group-out jackknife over 20 groups of videos.
Embeddings are z-scored per dimension (each encoder on its own statistics) before B6-B8, so the numbers are
unitless and comparable across encoders of different widths.

With two tags (--tags A B) it also computes, on the windows both have (matched by video_id, t):
  B9 linear CKA between the encoders, per layer pair (4x4), and the paired jackknife difference of B8.

Output: ROOT/comparison/<label>/<dataset>/<window>/representation/{B6_predictability.csv, B7_curvature.csv.gz,
        B8_dimensionality.csv, summary.csv}; with two tags ROOT/comparison/paired/<labelB>_vs_<labelA>/... B9_cka.csv
Usage: python representation_analysis.py --tags vitl16_diving48 --labels baseline
       python representation_analysis.py --tags vitl16_diving48 neural16_diving48 --labels baseline neural
"""
import argparse
import glob
import hashlib

import h5py
import numpy as np
import pandas as pd

import config

N_JACK = 20
LAMBDAS = [1e-3, 1e-2, 1e-1, 1.0, 10.0]   # ridge strength, relative to mean eigenvalue of X^T X / n


def load(tag, split, window):
    files = sorted(glob.glob(str(config.FEATURES_DIR / f"features_{tag}_{split}_{window}_shard*.h5")))
    assert files, f"no features for {tag} {split} {window}"
    E, vid, t, reg = [], [], [], []
    for fn in files:
        with h5py.File(fn) as f:
            vi, tt = f["video_idx"][:], f["t"][:]
            ids = np.array([v.decode() for v in f["video_id"][:]])
            n_rs = f["n_frames_resampled"][:] if "n_frames_resampled" in f else None
            E.append(f["emb_ml"][:])
            layers = eval(f.attrs.get("feature_layers", "[0, 1, 2, 3]"))
        vid.append(ids[vi])
        t.append(tt)
        if n_rs is None or config.FPS is None:
            reg.append(np.ones(len(tt), bool))
        else:  # regular windows start every STRIDE frames; an extra last one is end-aligned (overlapping)
            n_reg = np.maximum(n_rs[vi] - config.K, 0) // config.STRIDE + 1
            reg.append(tt < n_reg)
    E = np.concatenate(E)
    L = len(layers)
    D = E.shape[1] // L
    return dict(E=E, D=D, layers=layers, vid=np.concatenate(vid), t=np.concatenate(t), reg=np.concatenate(reg))


def layer(d, j):
    x = d["E"][:, j * d["D"] : (j + 1) * d["D"]].astype(np.float32)
    return (x - x.mean(0)) / (x.std(0) + 1e-6)


def group_of(vids, n, salt=""):
    """Deterministic group (0..n-1) per window from its source video (sha1), same for every encoder."""
    src = np.array([config.source_video(v) + salt for v in vids])
    u, inv = np.unique(src, return_inverse=True)
    g = np.array([int(hashlib.sha1(s.encode()).hexdigest(), 16) % n for s in u])
    return g[inv]


def consecutive_pairs(d):
    """Row indices (i, j) with j = window t+1 of the same video, both regular."""
    o = np.lexsort((d["t"], d["vid"]))
    v, t, r = d["vid"][o], d["t"][o], d["reg"][o]
    ok = (v[1:] == v[:-1]) & (t[1:] == t[:-1] + 1) & r[1:] & r[:-1]
    return o[:-1][ok], o[1:][ok]


def ridge_fit(X, Y, lam_rel):
    G = X.T @ X
    lam = lam_rel * np.trace(G) / G.shape[0]
    return np.linalg.solve(G + lam * np.eye(G.shape[0]), X.T @ Y)


def predictability(z, i, j, grp2, g5):
    """Held-out (2-fold by source video) R^2 for z_t -> z_{t+1}, with lambda picked on an inner split."""
    X, Y = z[i], z[j]
    pred = np.zeros_like(Y)
    lam_used = []
    for k in (0, 1):
        tr, te = grp2[i] != k, grp2[i] == k
        inner = g5[i[tr]] == 0      # inner validation: ~1/5 of the training source videos
        Xa, Ya, Xv, Yv = X[tr][~inner], Y[tr][~inner], X[tr][inner], Y[tr][inner]
        assert tr.any() and te.any(), "need source videos in both halves"
        if inner.all() or not inner.any():  # too few source videos for an inner split
            lam = 1e-1
        else:
            err = [((Xv @ ridge_fit(Xa, Ya, l) - Yv) ** 2).mean() for l in LAMBDAS]
            lam = LAMBDAS[int(np.argmin(err))]
        lam_used.append(lam)
        pred[te] = X[te] @ ridge_fit(X[tr], Y[tr], lam)
    sst = ((Y - Y.mean(0)) ** 2).sum()
    dY = Y - X
    return dict(r2_next=1 - ((Y - pred) ** 2).sum() / sst,
                r2_copy=1 - ((Y - X) ** 2).sum() / sst,
                r2_change=1 - ((dY - (pred - X)) ** 2).sum() / ((dY - dY.mean(0)) ** 2).sum(),
                lambdas=str(lam_used)), ((Y - pred) ** 2).mean(1)


def curvature(z, d):
    """Per video: mean angle between successive displacement vectors over regular windows."""
    o = np.lexsort((d["t"], d["vid"]))
    o = o[d["reg"][o]]
    v = d["vid"][o]
    starts = np.flatnonzero(np.r_[True, v[1:] != v[:-1]])
    ends = np.r_[starts[1:], len(o)]
    out = {}
    for s, e in zip(starts, ends):
        if e - s < 3:
            continue
        dz = np.diff(z[o[s:e]], axis=0)
        dz /= np.linalg.norm(dz, axis=1, keepdims=True) + 1e-12
        cos = np.clip((dz[1:] * dz[:-1]).sum(1), -1, 1)
        out[v[s]] = np.degrees(np.arccos(cos)).mean()
    return out


def group_stats(z, g, vids):
    """Per jackknife group: n, sum, sum of outer products, and within-video sum of squares."""
    S = []
    for k in range(N_JACK):
        m = g == k
        x = z[m].astype(np.float64)
        _, inv = np.unique(vids[m], return_inverse=True)
        cnt = np.bincount(inv)
        mu = np.zeros((len(cnt), x.shape[1]))
        np.add.at(mu, inv, x)
        within = (x ** 2).sum() - ((mu ** 2).sum(1) / cnt).sum()
        S.append(dict(n=len(x), s=x.sum(0), o=x.T @ x, w=within))
    return S


def dim_stats(S, drop=None):
    keep = [s for k, s in enumerate(S) if k != drop]
    n = sum(s["n"] for s in keep)
    mu = sum(s["s"] for s in keep) / n
    C = sum(s["o"] for s in keep) / n - np.outer(mu, mu)
    sd = np.sqrt(np.maximum(np.diag(C), 1e-12))
    R = C / np.outer(sd, sd)
    l = np.linalg.eigvalsh(R)
    return dict(participation_ratio=l.sum() ** 2 / (l ** 2).sum(),
                within_video_var_frac=sum(s["w"] for s in keep) / (n * np.trace(C)))


def jackknife(S, fn=dim_stats):
    full = fn(S)
    loo = [fn(S, k) for k in range(N_JACK)]
    out = {}
    for key, val in full.items():
        x = np.array([d[key] for d in loo])
        se = np.sqrt((N_JACK - 1) / N_JACK * ((x - x.mean()) ** 2).sum())
        out[key], out[key + "_lo"], out[key + "_hi"] = val, val - 1.96 * se, val + 1.96 * se
    return out


def linear_cka(X, Y):
    X, Y = X - X.mean(0), Y - Y.mean(0)
    return (np.linalg.norm(Y.T @ X) ** 2) / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y))


def single(tag, label, split, window):
    d = load(tag, split, window)
    out = config.ROOT / "comparison" / label / config.DATASET / window / "representation"
    out.mkdir(parents=True, exist_ok=True)
    i, j = consecutive_pairs(d)
    grp2 = group_of(d["vid"], 2)
    g5 = group_of(d["vid"], 5, salt="#inner")
    gj = group_of(d["vid"], N_JACK)
    print(f"{tag}: {len(d['vid'])} windows, {len(np.unique(d['vid']))} videos, {len(i)} consecutive regular pairs, "
          f"layers {d['layers']}, D={d['D']}", flush=True)
    b6, b7, b8 = [], [], []
    for jl, lay in enumerate(d["layers"]):
        z = layer(d, jl)
        r, err = predictability(z, i, j, grp2, g5)
        b6.append(dict(layer=lay, n_pairs=len(i), **r))
        b6_pv = pd.Series(err).groupby(d["vid"][i]).mean()
        cv = curvature(z, d)
        b7.append(pd.DataFrame(dict(video_id=list(cv), layer=lay, curvature_deg=list(cv.values())))
                  .merge(b6_pv.rename("pred_err").rename_axis("video_id").reset_index(), on="video_id", how="left"))
        b8.append(dict(layer=lay, **jackknife(group_stats(z, gj, d["vid"]))))
        print(f"  layer {lay}: R2 next {r['r2_next']:.3f} (copy {r['r2_copy']:.3f}, change {r['r2_change']:.3f})  "
              f"curvature {np.mean(list(cv.values())):.1f} deg  PR {b8[-1]['participation_ratio']:.1f}  "
              f"within-video var {b8[-1]['within_video_var_frac']:.3f}", flush=True)
    pd.DataFrame(b6).to_csv(out / "B6_predictability.csv", index=False)
    pd.concat(b7).to_csv(out / "B7_curvature.csv.gz", index=False)
    pd.DataFrame(b8).to_csv(out / "B8_dimensionality.csv", index=False)
    print(f"-> {out}")


def paired(tags, labels, split, window, max_windows=60000):
    a, b = (load(t, split, window) for t in tags)
    key = lambda d: np.char.add(np.char.add(d["vid"].astype(str), "#"), d["t"].astype(str))
    ka, kb = key(a), key(b)
    common, ia, ib = np.intersect1d(ka, kb, return_indices=True)
    rng = np.random.default_rng(0)
    sub = np.sort(rng.permutation(len(common))[:max_windows])
    out = config.ROOT / "comparison" / "paired" / f"{labels[1]}_vs_{labels[0]}" / config.DATASET / window / "representation"
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(common)} common windows; CKA on {len(sub)}", flush=True)
    rows, dims = [], []
    gj = group_of(a["vid"][ia], N_JACK)
    for ja, la in enumerate(a["layers"]):
        za = layer(a, ja)[ia]
        for jb, lb in enumerate(b["layers"]):
            zb = layer(b, jb)[ib]
            rows.append(dict(layer_a=la, layer_b=lb, cka=linear_cka(za[sub].astype(np.float64), zb[sub].astype(np.float64))))
        # paired jackknife of B8 at matching layer index
        zb = layer(b, ja)[ib] if ja < len(b["layers"]) else None
        if zb is not None:
            Sa, Sb = group_stats(za, gj, a["vid"][ia]), group_stats(zb, gj, b["vid"][ib])
            fn = lambda S, k=None: {key: dim_stats(S[1], k)[key] - dim_stats(S[0], k)[key]
                                    for key in ("participation_ratio", "within_video_var_frac")}
            dims.append(dict(layer_a=la, layer_b=b["layers"][ja], **{f"d_{k}": v for k, v in jackknife((Sa, Sb), fn).items()}))
    pd.DataFrame(rows).to_csv(out / "B9_cka.csv", index=False)
    pd.DataFrame(dims).to_csv(out / "B8_dimensionality_paired.csv", index=False)
    print(pd.DataFrame(rows).pivot(index="layer_a", columns="layer_b", values="cka").round(3))
    print(pd.DataFrame(dims).round(3).to_string(index=False))
    print(f"-> {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tags", nargs="+", default=[config.MODEL_TAG], help="one features tag, or two for B9 / paired B8")
    p.add_argument("--labels", nargs="+", default=["baseline"])
    p.add_argument("--split", default=config.EVAL_SPLIT)
    p.add_argument("--window", default=config.WINDOW)
    args = p.parse_args()
    assert len(args.tags) == len(args.labels) in (1, 2)
    if len(args.tags) == 1:
        single(args.tags[0], args.labels[0], args.split, args.window)
    else:
        paired(args.tags, args.labels, args.split, args.window)


if __name__ == "__main__":
    main()
