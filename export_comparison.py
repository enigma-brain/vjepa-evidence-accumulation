"""One encoder's side of a cross-model comparison: per-window / per-video tables for analyses A1-A4.

Everything that may differ between encoders is computed here by ONE fixed protocol, so a second encoder
(e.g. the neural-fine-tuned V-JEPA) exported the same way can be paired video by video
(paired_comparison.py):
  same windows (config.WINDOW), same 2-fold stratified cross-fit folds (folds_by_id: a function of
  (video_id, label) only, so independent of shard layout), same belief models / agents, every parameter fitted on the other fold.

Tables (under ROOT/comparison/<label>/<dataset>/<window>/<obs>/, gzipped CSV):
  A1_window.csv.gz     one row per window: video_id, fold, t, n_win, y, pred, conf, nll (raw probe) and
                       conf_cal / nll_cal (one temperature per fold, fit on the other fold).
                       Also the base for A5 (position profile, Diving48 attributes, SSv2 direction pairs).
  A2_accumulation.csv.gz   one row per video x t x belief model ("const": one alpha, naive Bayes;
                       "power": weight alpha * (t+1)^(gamma-1), belief_models.py): pred, conf, nll of b_t.
                       The forced-stop accumulation curve and the incremental information per window
                       dNLL_t = NLL(b_{t-1}) - NLL(b_t) both come from it.
  A3_params.json       fitted per-fold parameters (prior is implied by the folds): alpha, gamma, temperatures.
  A4_agents.csv.gz     one row per video x cost x agent: correct, steps, reward (const-alpha beliefs, as run_pomdp.py).
  A4_confidence.csv.gz one row per video: first window with calibrated max b~_t >= 0.5/0.7/0.9 (censored at T),
                       commitment window (compare_models.py definitions; per-window temperatures fit on the other fold).
  summary.csv          headline numbers with 95% bootstrap CIs over videos.
  ../manifest.json     model, window protocol, folds, feature files, code version.

Usage: python export_comparison.py --label baseline --obs linear linear-emb_ml linear-emb_ml+emb_std meta
"""
import argparse
import datetime
import glob
import json
import os
import subprocess

import h5py
import numpy as np
import pandas as pd

import config
import pomdp
from belief_models import beliefs as belief_log, fit as fit_belief
from compare_models import THETAS, calibrate, fit_step_temperatures, per_video_metrics
from run_pomdp import bootstrap_ci, load_features

TEMP_GRID = np.logspace(-1, 1, 41)


def folds_by_id(ids, y):
    """2-fold cross-fit folds that depend only on (video_id, label), never on file or shard order: within each
    class, videos sorted by sha1(video_id) alternate between the folds. Any encoder exported from the same videos
    gets exactly the same folds; a video missing for one encoder only shifts its own class."""
    import hashlib
    fold = np.zeros(len(y), int)
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        idx = idx[np.argsort([hashlib.sha1(ids[i].encode()).hexdigest() for i in idx])]
        fold[idx[1::2]] = 1
    a, b = np.where(fold == 0)[0], np.where(fold == 1)[0]
    return [(a, b), (b, a)]  # (fit, score): fold k is scored by parameters fitted on the other half


def fit_temperature(logq, y):
    """One temperature on the per-window log-posteriors (max likelihood)."""
    lq = np.concatenate(logq)
    yy = np.repeat(y, [len(q) for q in logq])
    nll = [-(lq / tau - pomdp.logsumexp(lq / tau))[np.arange(len(yy)), yy].mean() for tau in TEMP_GRID]
    return float(TEMP_GRID[int(np.argmin(nll))])


def code_version():
    if os.environ.get("EVACC_GIT"):
        return os.environ["EVACC_GIT"]
    try:
        h = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.call(["git", "diff", "--quiet"], stderr=subprocess.DEVNULL) != 0
        return h + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def export_obs(obs, pattern, out, costs, m):
    logq, y, full_correct, ids = load_features(pattern, obs)
    ids = np.array([v.decode() if isinstance(v, bytes) else str(v) for v in ids])
    T = np.array([len(q) for q in logq])
    folds = folds_by_id(ids, y)
    fold_of = np.zeros(len(y), int)
    for k, (_, sc) in enumerate(folds):
        fold_of[sc] = k

    params, A1, A2, A4, A4c = [], [], [], [], []
    for k, (fi, sc) in enumerate(folds):
        lq_fit = [logq[i] for i in fi]
        lp = np.log(pomdp.class_prior(y[fi], config.NUM_CLASSES))
        alpha, _ = pomdp.fit_alpha(lq_fit, y[fi], lp)
        pw = fit_belief(lq_fit, y[fi], lp, use_temp=False, use_gamma=True)
        tau = fit_temperature(lq_fit, y[fi])
        B_fit = [pomdp.beliefs(q, lp, alpha) for q in lq_fit]
        B_sc = [pomdp.beliefs(logq[i], lp, alpha) for i in sc]
        taus_t = fit_step_temperatures(B_fit, y[fi], int(np.percentile(T[fi], 95)))
        params.append(dict(fold=k, alpha=alpha, power_alpha=pw["alpha"], power_gamma=pw["gamma"],
                           window_temperature=tau, step_temperatures=taus_t.tolist()))

        for j, i in enumerate(sc):
            q, n, yy = logq[i], T[i], y[i]
            qc = q / tau - pomdp.logsumexp(q / tau)
            A1.append(pd.DataFrame(dict(video_id=ids[i], fold=k, t=np.arange(n), n_win=n, y=yy, pred=q.argmax(1),
                                        conf=np.exp(q.max(1)), nll=-q[:, yy], conf_cal=np.exp(qc.max(1)), nll_cal=-qc[:, yy])))
            for name, lb in [("const", np.log(np.maximum(B_sc[j], 1e-300))),
                             ("power", belief_log(q, lp, pw["alpha"], pw["gamma"]))]:
                A2.append(pd.DataFrame(dict(video_id=ids[i], belief=name, t=np.arange(n), n_win=n, y=yy,
                                            pred=lb.argmax(1), conf=np.exp(lb.max(1)), nll=-lb[:, yy])))
            A4c.append(dict(video_id=ids[i], fold=k, y=yy, n_win=n, **per_video_metrics(calibrate(B_sc[j], taus_t), yy)))

        for cost in costs:
            agents = {"FixedBudget": pomdp.FixedBudget(), "Threshold": pomdp.Threshold(),
                      "Lookahead": pomdp.Lookahead(lp, alpha, m=m), "FittedQ": pomdp.FittedQ(),
                      "LSM": pomdp.LSM()}
            for name, agent in agents.items():
                agent.fit(B_fit, lq_fit, y[fi], cost)
                r = pomdp.rollouts(agent, B_sc, y[sc], cost)
                A4.append(pd.DataFrame(dict(video_id=ids[sc], fold=k, cost=cost, agent=name, params=repr(agent),
                                            correct=r[:, 0], steps=r[:, 1], reward=r[:, 2])))

    out.mkdir(parents=True, exist_ok=True)
    A1, A2, A4, A4c = pd.concat(A1), pd.concat(A2), pd.concat(A4), pd.DataFrame(A4c)
    A1.to_csv(out / "A1_window.csv.gz", index=False)
    A2.to_csv(out / "A2_accumulation.csv.gz", index=False)
    A4.to_csv(out / "A4_agents.csv.gz", index=False)
    A4c.to_csv(out / "A4_confidence.csv.gz", index=False)
    json.dump(dict(obs=obs, n_videos=len(y), n_windows=int(T.sum()), full_clip_acc=float(np.nanmean(full_correct)),
                   folds=params), open(out / "A3_params.json", "w"), indent=1)

    # ---- headline summary (per-video means, 95% bootstrap CI over videos)
    rows = []

    def add(metric, per_video):
        v = np.asarray(per_video, float)
        lo, hi = bootstrap_ci(v)
        rows.append(dict(obs=obs, metric=metric, mean=v.mean(), lo=lo, hi=hi, n=len(v)))

    pv1 = A1.groupby("video_id")
    add("A1 window acc (video mean)", (A1.pred == A1.y).groupby(A1.video_id).mean())
    add("A1 window NLL, raw", pv1.nll.mean())
    add("A1 window NLL, calibrated", pv1.nll_cal.mean())
    for name, g in A2.groupby("belief"):
        last = g[g.t == g.n_win - 1]
        add(f"A2 final acc ({name})", last.pred == last.y)
        add(f"A2 final NLL ({name})", last.nll)
        first = g[g.t == 0].set_index("video_id").nll
        add(f"A2 NLL drop first->final ({name})", first.loc[last.video_id].values - last.nll.values)
        add(f"A2 mean NLL over steps ({name})", g.groupby("video_id").nll.mean())
    for (cost, agent), g in A4.groupby(["cost", "agent"]):
        add(f"A4 reward c={cost} {agent}", g.reward)
        add(f"A4 windows c={cost} {agent}", g.steps)
    for th in THETAS:
        add(f"A4 windows to calibrated conf {th} (censored)", A4c[f"win_{th}"])
    add("A4 commitment window", A4c.commit_win)
    s = pd.DataFrame(rows)
    s.to_csv(out / "summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.precision", 4, "display.max_rows", 500):
        print(s.to_string(index=False))
    print(f"alpha {[p['alpha'] for p in params]}  gamma {[p['power_gamma'] for p in params]}  -> {out}", flush=True)
    return dict(n_videos=len(y), n_windows=int(T.sum()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True, help="e.g. baseline, neural, vitg_control")
    p.add_argument("--obs", nargs="+", default=["linear", "linear-emb_ml", "linear-emb_ml+emb_std", "meta"])
    p.add_argument("--split", default=config.EVAL_SPLIT)
    p.add_argument("--window", default=config.WINDOW)
    p.add_argument("--costs", default="0.0,0.002,0.005,0.01,0.02,0.05,0.1")
    p.add_argument("--lookahead-samples", type=int, default=64)
    args = p.parse_args()

    name = f"{config.MODEL_TAG}_{args.split}_{args.window}"
    pattern = str(config.FEATURES_DIR / f"features_{name}_shard*.h5")
    root = config.ROOT / "comparison" / args.label / config.DATASET / args.window
    root.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(pattern))
    with h5py.File(files[0]) as f:
        attrs = {k: (v.item() if hasattr(v, "item") else str(v)) for k, v in f.attrs.items()}
    fps = np.concatenate([h5py.File(fn)["native_fps"][:] for fn in files if "native_fps" in h5py.File(fn)] or [np.zeros(0)])
    fps_summary = {}
    if len(fps):
        vals, cnt = np.unique(np.round(fps, 2), return_counts=True)
        top = np.argsort(-cnt)[:8]
        fps_summary = dict(quantiles=dict(zip(["min", "p10", "median", "p90", "max"], np.percentile(fps, [0, 10, 50, 90, 100]).round(3).tolist())),
                           most_common={f"{vals[i]:g}": int(cnt[i]) for i in top})
        print(f"native fps: {fps_summary}")
    costs = [float(c) for c in args.costs.split(",")]
    done = {}
    for obs in args.obs:
        print(f"===== {obs}", flush=True)
        done[obs] = export_obs(obs, pattern, root / obs, costs, args.lookahead_samples)
    manifest_fn = root / "manifest.json"
    manifest = json.load(open(manifest_fn)) if manifest_fn.exists() else {}
    manifest.update(label=args.label, dataset=config.DATASET, split=args.split, window=args.window,
                    model_tag=config.MODEL_TAG, feature_files=files, feature_attrs=attrs, native_fps=fps_summary,
                    window_protocol=dict(fps=config.FPS, k=config.K, stride=config.STRIDE, end_align=config.END_ALIGN),
                    folds="export_comparison.folds_by_id: per class, sha1(video_id) order, alternating",
                    costs=costs, code=code_version(), date=datetime.datetime.now().isoformat(timespec="seconds"))
    manifest.setdefault("obs", {}).update(done)
    json.dump(manifest, open(manifest_fn, "w"), indent=1, default=str)
    print(f"manifest -> {manifest_fn}")


if __name__ == "__main__":
    main()
