"""Fit and evaluate evidence-accumulation agents on cached V-JEPA 2 window features.

1. Load every features shard for one (model, split, window) -> per-video log p(y | o_t).
2. 2-fold cross-fitting: split videos 50/50 (stratified by class) into halves A and B; fit on A and
   score B, then fit on B and score A. Every video is scored once, by parameters it was not fitted on.
3. Per fold: fit the class prior and the naive-Bayes temperature alpha on the fit half.
4. For each step cost c: fit FixedBudget / Threshold / Lookahead / FittedQ per fold, score the other half.
5. Save summary.csv + figures under RESULTS_DIR/<run name>/. summary.csv has 95% bootstrap CIs over
   videos for the reward, and paired CIs for each agent's reward minus FixedBudget's on the same videos.

Usage: python run_pomdp.py [--obs linear|linear-<key>|meta] [--split validation]
"""
import argparse
import glob
import json

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import pomdp

# Reference categorical palette, slots in fixed order (identity follows the agent, never its rank).
COLORS = {"Threshold": "#2a78d6", "Lookahead": "#eb6834", "FixedBudget": "#1baf7a", "FittedQ": "#eda100",
          "LSM": "#e87ba4"}
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "lines.linewidth": 2, "lines.markersize": 6,
    "font.size": 10, "legend.frameon": False,
})


def load_features(pattern, obs="meta"):
    """obs="meta": Meta's pretrained SSv2 probe (logits stored in the features file).
    obs="linear": our linear probe from train_probe.py (logits in the matching probe-linear_ file).
    obs="linear-<key>": the same probe trained on another embedding (train_probe.py --key <key>)."""
    files = sorted(glob.glob(pattern))
    assert files, f"no feature files match {pattern}"
    logq, y, full_correct, ids = [], [], [], []
    for fn in files:
        with h5py.File(fn) as f:
            vi, t, labels, nfr, vid = f["video_idx"][:], f["t"][:], f["label"][:], f["n_frames"][:], f["video_id"][:]
            logits = f["logits"][:] if obs == "meta" else None
            # full-clip Meta-probe logits exist only for V-JEPA features (not e.g. the pixel control)
            full = f["full_logits"][:] if "full_logits" in f else np.full((len(labels), 1), np.nan)
        if obs.startswith("linear"):
            with h5py.File(fn.replace("/features_", f"/probe-{obs}_")) as g:
                logits = g["logits"][:]
        lq = logits - pomdp.logsumexp(logits)  # log-softmax
        order = np.lexsort((t, vi))
        lq, vi = lq[order], vi[order]
        bounds = np.searchsorted(vi, np.arange(len(labels) + 1))
        for v in range(len(labels)):
            if nfr[v] == 0 or bounds[v] == bounds[v + 1]:
                continue
            logq.append(lq[bounds[v] : bounds[v + 1]])
            y.append(labels[v])
            full_correct.append(full[v].argmax() == labels[v] if np.isfinite(full[v]).all() else np.nan)
            ids.append(vid[v])
    print(f"loaded {len(files)} shards, {len(y)} videos, {sum(len(q) for q in logq)} windows")
    return logq, np.array(y), np.array(full_correct), np.array(ids)


def stratified_split(y, seed=0):
    rng = np.random.default_rng(seed)
    fit = np.zeros(len(y), bool)
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        fit[idx[: len(idx) // 2]] = True
    return np.where(fit)[0], np.where(~fit)[0]


def crossfit_folds(y, seed=0):
    """[(fit, score), (score, fit)]: 2-fold stratified cross-fitting."""
    a, b = stratified_split(y, seed)
    return [(a, b), (b, a)]


def fit_fold(logq, y, fit_idx):
    """Class prior + alpha from the fit half. Returns (log_prior, alpha, alpha_scores)."""
    lp = np.log(pomdp.class_prior(y[fit_idx], config.NUM_CLASSES))
    alpha, scores = pomdp.fit_alpha([logq[i] for i in fit_idx], y[fit_idx], lp)
    return lp, alpha, scores


def bootstrap_ci(x, n_boot=2000, seed=0):
    """95% percentile CI of mean(x), resampling videos."""
    idx = np.random.default_rng(seed).integers(0, len(x), size=(n_boot, len(x)))
    return tuple(np.percentile(x[idx].mean(1), [2.5, 97.5]))


def acc_vs_t(Bs, y, T_max):
    """Accuracy if forced to stop after t+1 windows (or at the video's end, if shorter)."""
    return np.array([np.mean([B[min(t, len(B) - 1)].argmax() == yy for B, yy in zip(Bs, y)]) for t in range(T_max)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default=config.EVAL_SPLIT)
    p.add_argument("--window", default=config.WINDOW, help="window tag in file names (default: the current protocol)")
    p.add_argument("--obs", default="linear",
                   help="observation model: our linear probe (default; linear-<key> for one trained on "
                        "another embedding) or meta (Meta's pretrained probe)")
    p.add_argument("--costs", default="0.0,0.002,0.005,0.01,0.02,0.05,0.1")
    p.add_argument("--lookahead-samples", type=int, default=64)
    args = p.parse_args()

    name = f"{config.MODEL_TAG}_{args.split}_{args.window}"
    out = config.RESULTS_DIR / f"{name}_{args.obs}"
    out.mkdir(parents=True, exist_ok=True)
    logq, y, full_correct, _ = load_features(str(config.FEATURES_DIR / f"features_{name}_shard*.h5"), args.obs)

    T = np.array([len(q) for q in logq])
    print(f"windows per video: mean {T.mean():.1f}, median {np.median(T):.0f}, min {T.min()}, max {T.max()}")
    print(f"sanity: full-clip Meta-probe acc {full_correct.mean():.3f} | per-window acc ({args.obs} probe) "
          f"{np.mean(np.concatenate([q.argmax(1) == yy for q, yy in zip(logq, y)])):.3f}")

    folds = crossfit_folds(y)
    fitted = [fit_fold(logq, y, f) for f, _ in folds]  # (log_prior, alpha, scores) per fold
    alphas = [a for _, a, _ in fitted]
    print(f"fitted alpha per fold = {', '.join(f'{a:.3f}' for a in alphas)}")
    json.dump({"alpha": float(np.mean(alphas)), "alpha_folds": alphas, "scores_folds": [sc for *_, sc in fitted],
               "n_videos": len(y), "full_clip_acc": float(np.nanmean(full_correct))},
              open(out / "alpha.json", "w"), indent=1)

    def pooled(alpha=None, mode="nb"):
        """Beliefs of every video, each computed with its scoring fold's prior (and alpha, unless given)."""
        B = [None] * len(y)
        for (_, sc), (lp, a, _) in zip(folds, fitted):
            for i in sc:
                B[i] = pomdp.beliefs(logq[i], lp, a if alpha is None else alpha, mode)
        return B

    B_all = pooled()  # scoring beliefs (cross-fitted: every video is scored once)
    B_fit = [[pomdp.beliefs(logq[i], lp, a) for i in fi] for (fi, _), (lp, a, _) in zip(folds, fitted)]

    # ---- Figure 1: evidence accumulation curve (watch everything, stop at t)
    T_max = int(np.percentile(T, 95))
    ts = np.arange(1, T_max + 1)
    curves = {
        f"naive Bayes, fitted α={np.mean(alphas):.2f}": (acc_vs_t(B_all, y, T_max), COLORS["Threshold"]),
        "naive Bayes, α=1": (acc_vs_t(pooled(1.0), y, T_max), COLORS["Lookahead"]),
        "latest window only": (acc_vs_t(pooled(mode="latest"), y, T_max), COLORS["FixedBudget"]),
    }
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for label, (a, c) in curves.items():
        ax.plot(ts, a, color=c, marker="o", label=label)
    if args.obs == "meta":  # full-clip reference only exists for Meta's probe
        ax.axhline(full_correct.mean(), color=INK2, ls="--", lw=1)
        ax.annotate("full-clip probe", (ts[-1], full_correct.mean()), textcoords="offset points",
                    xytext=(0, 4), ha="right", color=INK2, fontsize=9)
    ax.set(xlabel="windows observed", ylabel="top-1 accuracy (cross-fitted)", title="Accuracy vs. evidence")
    ax.legend(loc="best")
    fig.tight_layout(); fig.savefig(out / "fig1_acc_vs_windows.png", dpi=200); plt.close(fig)

    # ---- Figure 2: accuracy / evidence trade-off (Pareto), no cost needed
    th = [pomdp.evaluate(pomdp.Threshold(t), B_all, y, 0) for t in np.linspace(0.05, 0.99, 30)]
    fx = [pomdp.evaluate(pomdp.FixedBudget(n), B_all, y, 0) for n in range(1, T_max + 1)]
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.plot([r["steps"] for r in th], [r["acc"] for r in th], color=COLORS["Threshold"], marker="o", label="Threshold (sweep θ)")
    ax.plot([r["steps"] for r in fx], [r["acc"] for r in fx], color=COLORS["FixedBudget"], marker="s", label="FixedBudget (sweep n)")
    ax.set(xlabel="mean windows observed", ylabel="top-1 accuracy (cross-fitted)", title="Adaptive vs. fixed stopping")
    ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(out / "fig2_pareto.png", dpi=200); plt.close(fig)

    # ---- Cost sweep: per fold, fit each agent on the fit half and score the other half; pool the scores
    rows = []
    for cost in [float(c) for c in args.costs.split(",")]:
        R, params = {}, {}
        for k, ((fi, sc), (lp, a, _)) in enumerate(zip(folds, fitted)):
            agents = {
                "FixedBudget": pomdp.FixedBudget(),
                "Threshold": pomdp.Threshold(),
                "Lookahead": pomdp.Lookahead(lp, a, m=args.lookahead_samples),
                "FittedQ": pomdp.FittedQ(),
            }
            for agent_name, agent in agents.items():
                agent.fit(B_fit[k], [logq[i] for i in fi], y[fi], cost)
                R.setdefault(agent_name, np.zeros((len(y), 3)))[sc] = pomdp.rollouts(agent, [B_all[i] for i in sc], y[sc], cost)
                params.setdefault(agent_name, []).append(repr(agent))
        base = R["FixedBudget"][:, 2]
        for agent_name, r in R.items():
            lo, hi = bootstrap_ci(r[:, 2])
            dlo, dhi = bootstrap_ci(r[:, 2] - base)
            rows.append(dict(cost=cost, agent=agent_name, params=" | ".join(params[agent_name]),
                             acc=r[:, 0].mean(), steps=r[:, 1].mean(), reward=r[:, 2].mean(),
                             reward_lo=lo, reward_hi=hi, d_reward_vs_fixed=(r[:, 2] - base).mean(),
                             d_lo=dlo, d_hi=dhi))
            print(f"c={cost:<6} {agent_name:<12} acc={r[:, 0].mean():.3f} steps={r[:, 1].mean():.2f} "
                  f"reward={r[:, 2].mean():.3f} [{lo:.3f},{hi:.3f}]  vs Fixed {(r[:, 2] - base).mean():+.4f} "
                  f"[{dlo:+.4f},{dhi:+.4f}]   {' | '.join(params[agent_name])}")
    df = pd.DataFrame(rows)
    df.to_csv(out / "summary.csv", index=False)

    # ---- Figure 3: reward and windows used vs cost (two panels, one axis each)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for agent_name, g in df.groupby("agent", sort=False):
        axes[0].plot(g.cost, g.reward, color=COLORS[agent_name], marker="o", label=agent_name)
        axes[0].fill_between(g.cost, g.reward_lo, g.reward_hi, color=COLORS[agent_name], alpha=0.15, lw=0)
        axes[1].plot(g.cost, g.steps, color=COLORS[agent_name], marker="o", label=agent_name)
    axes[0].set(xscale="symlog", xlabel="cost per extra window c", ylabel="mean reward (95% CI)", title="Reward")
    axes[1].set(xscale="symlog", xlabel="cost per extra window c", ylabel="mean windows observed", title="Evidence used")
    axes[0].set_xscale("symlog", linthresh=0.002); axes[1].set_xscale("symlog", linthresh=0.002)
    axes[1].legend()
    fig.tight_layout(); fig.savefig(out / "fig3_cost_sweep.png", dpi=200); plt.close(fig)
    print(f"results -> {out}")


if __name__ == "__main__":
    main()
