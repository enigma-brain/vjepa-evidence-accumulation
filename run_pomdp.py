"""Fit and evaluate evidence-accumulation agents on cached V-JEPA 2 window features.

1. Load every features shard for one (model, split, frame_step) -> per-video log p(y | o_t).
2. Split videos 50/50 (stratified by class): agent-fit / agent-test.
3. Fit the naive-Bayes temperature alpha on agent-fit.
4. For each step cost c: fit FixedBudget / Threshold / Lookahead on agent-fit, evaluate on agent-test.
5. Save summary.csv + figures under RESULTS_DIR/<run name>/.

Usage: python run_pomdp.py [--frame-step 2] [--split validation]
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
COLORS = {"Threshold": "#2a78d6", "Lookahead": "#eb6834", "FixedBudget": "#1baf7a", "FittedQ": "#eda100"}
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "lines.linewidth": 2, "lines.markersize": 6,
    "font.size": 10, "legend.frameon": False,
})


def load_features(pattern):
    files = sorted(glob.glob(pattern))
    assert files, f"no feature files match {pattern}"
    logq, y, full_correct, T_list = [], [], [], []
    for fn in files:
        with h5py.File(fn) as f:
            logits, vi, t = f["logits"][:], f["video_idx"][:], f["t"][:]
            labels, nfr, full = f["label"][:], f["n_frames"][:], f["full_logits"][:]
        lq = logits - pomdp.logsumexp(logits)  # log-softmax
        order = np.lexsort((t, vi))
        lq, vi = lq[order], vi[order]
        bounds = np.searchsorted(vi, np.arange(len(labels) + 1))
        for v in range(len(labels)):
            if nfr[v] == 0 or bounds[v] == bounds[v + 1]:
                continue
            logq.append(lq[bounds[v] : bounds[v + 1]])
            y.append(labels[v])
            full_correct.append(full[v].argmax() == labels[v])
    print(f"loaded {len(files)} shards, {len(y)} videos, {sum(len(q) for q in logq)} windows")
    return logq, np.array(y), np.array(full_correct)


def stratified_split(y, seed=0):
    rng = np.random.default_rng(seed)
    fit = np.zeros(len(y), bool)
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        fit[idx[: len(idx) // 2]] = True
    return np.where(fit)[0], np.where(~fit)[0]


def acc_vs_t(Bs, y, T_max):
    """Accuracy if forced to stop after t+1 windows (or at the video's end, if shorter)."""
    return np.array([np.mean([B[min(t, len(B) - 1)].argmax() == yy for B, yy in zip(Bs, y)]) for t in range(T_max)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="validation")
    p.add_argument("--frame-step", type=int, default=config.FRAME_STEP)
    p.add_argument("--costs", default="0.0,0.002,0.005,0.01,0.02,0.05,0.1")
    p.add_argument("--lookahead-samples", type=int, default=64)
    args = p.parse_args()

    name = f"{config.MODEL_TAG}_{args.split}_fs{args.frame_step}"
    out = config.RESULTS_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    logq, y, full_correct = load_features(str(config.FEATURES_DIR / f"features_{name}_shard*.h5"))

    T = np.array([len(q) for q in logq])
    print(f"windows per video: mean {T.mean():.1f}, median {np.median(T):.0f}, min {T.min()}, max {T.max()}")
    print(f"sanity: full-clip probe acc {full_correct.mean():.3f} | per-window acc "
          f"{np.mean(np.concatenate([q.argmax(1) == yy for q, yy in zip(logq, y)])):.3f}")

    fit_idx, test_idx = stratified_split(y)
    q_fit, y_fit = [logq[i] for i in fit_idx], y[fit_idx]
    q_test, y_test = [logq[i] for i in test_idx], y[test_idx]
    prior = pomdp.class_prior(y_fit, config.NUM_CLASSES)
    lp = np.log(prior)

    alpha, alpha_scores = pomdp.fit_alpha(q_fit, y_fit, lp)
    print(f"fitted alpha = {alpha:.3f}")
    json.dump({"alpha": alpha, "scores": alpha_scores, "n_fit": len(y_fit), "n_test": len(y_test),
               "full_clip_acc": float(full_correct.mean())}, open(out / "alpha.json", "w"), indent=1)

    B_fit = [pomdp.beliefs(q, lp, alpha) for q in q_fit]
    B_test = [pomdp.beliefs(q, lp, alpha) for q in q_test]

    # ---- Figure 1: evidence accumulation curve (watch everything, stop at t)
    T_max = int(np.percentile(T, 95))
    ts = np.arange(1, T_max + 1)
    curves = {
        f"naive Bayes, fitted α={alpha:.2f}": (acc_vs_t(B_test, y_test, T_max), COLORS["Threshold"]),
        "naive Bayes, α=1": (acc_vs_t([pomdp.beliefs(q, lp, 1.0) for q in q_test], y_test, T_max), COLORS["Lookahead"]),
        "latest window only": (acc_vs_t([pomdp.beliefs(q, lp, mode="latest") for q in q_test], y_test, T_max), COLORS["FixedBudget"]),
    }
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for label, (a, c) in curves.items():
        ax.plot(ts, a, color=c, marker="o", label=label)
    ax.axhline(full_correct[test_idx].mean(), color=INK2, ls="--", lw=1)
    ax.annotate("full-clip probe", (ts[-1], full_correct[test_idx].mean()), textcoords="offset points",
                xytext=(0, 4), ha="right", color=INK2, fontsize=9)
    ax.set(xlabel="windows observed", ylabel="top-1 accuracy (agent-test)", title="Accuracy vs. evidence")
    ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(out / "fig1_acc_vs_windows.png", dpi=200); plt.close(fig)

    # ---- Figure 2: accuracy / evidence trade-off (Pareto), no cost needed
    th = [pomdp.evaluate(pomdp.Threshold(t), B_test, y_test, 0) for t in np.linspace(0.05, 0.99, 30)]
    fx = [pomdp.evaluate(pomdp.FixedBudget(n), B_test, y_test, 0) for n in range(1, T_max + 1)]
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.plot([r["steps"] for r in th], [r["acc"] for r in th], color=COLORS["Threshold"], marker="o", label="Threshold (sweep θ)")
    ax.plot([r["steps"] for r in fx], [r["acc"] for r in fx], color=COLORS["FixedBudget"], marker="s", label="FixedBudget (sweep n)")
    ax.set(xlabel="mean windows observed", ylabel="top-1 accuracy (agent-test)", title="Adaptive vs. fixed stopping")
    ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(out / "fig2_pareto.png", dpi=200); plt.close(fig)

    # ---- Cost sweep: fit each agent per cost on agent-fit, evaluate on agent-test
    rows = []
    for cost in [float(c) for c in args.costs.split(",")]:
        agents = {
            "FixedBudget": pomdp.FixedBudget(),
            "Threshold": pomdp.Threshold(),
            "Lookahead": pomdp.Lookahead(lp, alpha, m=args.lookahead_samples),
            "FittedQ": pomdp.FittedQ(),
        }
        for agent_name, agent in agents.items():
            agent.fit(B_fit, q_fit, y_fit, cost)
            r = pomdp.evaluate(agent, B_test, y_test, cost)
            rows.append(dict(cost=cost, agent=agent_name, params=repr(agent), **r))
            print(f"c={cost:<6} {repr(agent):<32} acc={r['acc']:.3f} steps={r['steps']:.2f} reward={r['reward']:.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(out / "summary.csv", index=False)

    # ---- Figure 3: reward and windows used vs cost (two panels, one axis each)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for agent_name, g in df.groupby("agent", sort=False):
        axes[0].plot(g.cost, g.reward, color=COLORS[agent_name], marker="o", label=agent_name)
        axes[1].plot(g.cost, g.steps, color=COLORS[agent_name], marker="o", label=agent_name)
    axes[0].set(xscale="symlog", xlabel="cost per extra window c", ylabel="mean reward (agent-test)", title="Reward")
    axes[1].set(xscale="symlog", xlabel="cost per extra window c", ylabel="mean windows observed", title="Evidence used")
    axes[0].set_xscale("symlog", linthresh=0.002); axes[1].set_xscale("symlog", linthresh=0.002)
    axes[1].legend()
    fig.tight_layout(); fig.savefig(out / "fig3_cost_sweep.png", dpi=200); plt.close(fig)
    print(f"results -> {out}")


if __name__ == "__main__":
    main()
