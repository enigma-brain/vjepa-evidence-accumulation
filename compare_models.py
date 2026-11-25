"""Compare observation models (encoder + probe): which one becomes confident, and stops, earlier?

A "model" is <features tag>:<probe>, e.g.  vitl16_ssv2:linear  (our linear probe)  or  vitl16_ssv2:meta.
A neural-fine-tuned encoder is just another tag once its features (+ train_probe.py) exist.

Every model is evaluated on the SAME agent-test videos (paired), with its own belief temperature
alpha fit on agent-fit.

Calibration matters: an overconfident model "gets confident earlier" for free. One alpha cannot
calibrate every step (it flattens window 1 toward the prior and sharpens later windows), so the
confidence metrics (1-2) use beliefs recalibrated per window t with a temperature tau_t fit on
agent-fit:  b~_t = softmax(log b_t / tau_t). Then "max b~_t = 0.7" means ~70% correct for every model.
ECE before and after is reported. Task stopping (4) uses the raw beliefs, exactly as run_pomdp.py.

Metrics (agent-test):
  1. per window t:   accuracy, mean calibrated confidence max_y b~_t, mean b~_t(true y), ECE
  2. first passage:  first window where max b~_t >= theta (theta = 0.5, 0.7, 0.9):
                     fraction of videos that ever get there, windows needed, accuracy when they do
  3. commitment:     first window after which argmax b_t never changes again
  4. task stopping:  Threshold agent fit per cost c on agent-fit -> windows used, accuracy, reward
Every model after the first is compared with the first (reference) by the per-video paired difference,
with a 95% bootstrap CI over videos. Negative windows = the model stops / gets confident earlier.

Usage: python compare_models.py vitl16_ssv2:linear vitl16_ssv2:meta
Output: RESULTS_DIR/compare_<a>_vs_<b>/{per_window.csv, summary.csv, paired.csv, fig5_*.png}
"""
import argparse

import numpy as np
import pandas as pd

import config
import pomdp
from run_pomdp import load_features, plt, stratified_split

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]  # fixed order
THETAS = (0.5, 0.7, 0.9)


def load_model(spec, window):
    tag, obs = spec.split(":")
    logq, y, _, ids = load_features(str(config.FEATURES_DIR / f"features_{tag}_{config.EVAL_SPLIT}_{window}_shard*.h5"), obs)
    return dict(zip(ids.tolist(), zip(logq, y)))


def fit_step_temperatures(B_fit, y_fit, T_max, grid=np.logspace(-1, 1, 41)):
    """tau_t for t < T_max, each fit by max likelihood on the agent-fit videos that have a window t."""
    taus = []
    for t in range(T_max):
        keep = [i for i, B in enumerate(B_fit) if len(B) > t]
        lb = np.log(np.maximum(np.stack([B_fit[i][t] for i in keep]), 1e-300))
        yy = y_fit[keep]
        nll = [-(lb / tau - pomdp.logsumexp(lb / tau))[np.arange(len(yy)), yy].mean() for tau in grid]
        taus.append(grid[int(np.argmin(nll))])
    return np.array(taus)


def calibrate(B, taus):
    """b~_t = softmax(log b_t / tau_t); windows past T_max reuse the last tau."""
    tau = taus[np.minimum(np.arange(len(B)), len(taus) - 1)][:, None]
    lb = np.log(np.maximum(B, 1e-300)) / tau
    return np.exp(lb - pomdp.logsumexp(lb))


def ece(conf, correct, bins=15):
    edges = np.minimum((conf * bins).astype(int), bins - 1)
    return sum(abs(conf[edges == k].mean() - correct[edges == k].mean()) * (edges == k).mean()
               for k in range(bins) if (edges == k).any())


def per_video_metrics(B, y):
    """One video's beliefs [T, C] -> first passage per theta, commitment window, (1-based window counts)."""
    T, top, pred = len(B), B.max(1), B.argmax(1)
    out = {}
    for th in THETAS:
        hit = np.flatnonzero(top >= th)
        out[f"reached_{th}"] = len(hit) > 0
        out[f"win_{th}"] = hit[0] + 1 if len(hit) else T  # censored at the video's end (forced stop)
        out[f"correct_{th}"] = pred[hit[0]] == y if len(hit) else np.nan
    changed = np.flatnonzero(pred != pred[-1])
    out["commit_win"] = changed[-1] + 2 if len(changed) else 1
    out["commit_correct"] = pred[-1] == y
    return out


def paired_ci(d, n_boot=2000, seed=0):
    d = np.asarray(d, float)
    idx = np.random.default_rng(seed).integers(0, len(d), (n_boot, len(d)))
    boots = d[idx].mean(1)
    return d.mean(), *np.percentile(boots, [2.5, 97.5])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="+", help="<features tag>:<linear|meta>; the first one is the reference")
    p.add_argument("--window", default=config.WINDOW, help="window tag in file names (default: the current protocol)")
    p.add_argument("--costs", default="0.0,0.002,0.005,0.01,0.02,0.05,0.1")
    args = p.parse_args()
    costs = [float(c) for c in args.costs.split(",")]

    data = [load_model(m, args.window) for m in args.models]
    ids = [i for i in data[0] if all(i in d for d in data[1:])]  # videos every model has, in a fixed order
    y = np.array([data[0][i][1] for i in ids])
    for d in data[1:]:
        assert all(d[i][1] == data[0][i][1] for i in ids), "labels disagree across models"
    fit_idx, test_idx = stratified_split(y)
    y_fit, y_test = y[fit_idx], y[test_idx]
    lp = np.log(pomdp.class_prior(y_fit, config.NUM_CLASSES))
    print(f"{len(ids)} common videos: agent-fit {len(fit_idx)} / agent-test {len(test_idx)}")

    out = config.RESULTS_DIR / ("compare_" + "_vs_".join(m.replace(":", "-") for m in args.models))
    out.mkdir(parents=True, exist_ok=True)

    T_test = np.array([len(data[0][ids[i]][0]) for i in test_idx])
    T_max = int(np.percentile(T_test, 95))
    per_window, summary, per_video, stops = [], [], {}, {}
    for m, d in zip(args.models, data):
        q_fit = [d[ids[i]][0] for i in fit_idx]
        q_test = [d[ids[i]][0] for i in test_idx]
        alpha, _ = pomdp.fit_alpha(q_fit, y_fit, lp)
        B_fit = [pomdp.beliefs(q, lp, alpha) for q in q_fit]
        B_test = [pomdp.beliefs(q, lp, alpha) for q in q_test]
        taus = fit_step_temperatures(B_fit, y_fit, T_max)
        C_test = [calibrate(B, taus) for B in B_test]  # calibrated beliefs, for metrics 1-2

        # 1. per window (videos shorter than t stay at their last window, as in run_pomdp.acc_vs_t)
        for t in range(T_max):
            raw = np.stack([B[min(t, len(B) - 1)] for B in B_test])
            b = np.stack([B[min(t, len(B) - 1)] for B in C_test])
            correct = b.argmax(1) == y_test
            per_window.append(dict(model=m, window=t + 1, tau=taus[t], acc=correct.mean(), conf=b.max(1).mean(),
                                   p_true=b[np.arange(len(b)), y_test].mean(), ece=ece(b.max(1), correct),
                                   conf_raw=raw.max(1).mean(), ece_raw=ece(raw.max(1), correct)))

        # 2-3. first passage + commitment, per video
        pv = pd.DataFrame([per_video_metrics(B, yy) for B, yy in zip(C_test, y_test)])
        per_video[m] = pv
        row = dict(model=m, alpha=alpha, final_acc=np.mean([B[-1].argmax() == yy for B, yy in zip(B_test, y_test)]),
                   ece_window1_raw=per_window[-T_max]["ece_raw"], ece_window1=per_window[-T_max]["ece"],
                   commit_win=pv.commit_win.mean())
        for th in THETAS:
            r = pv[f"reached_{th}"]
            row.update({f"reached_{th}": r.mean(), f"win_{th}": pv.loc[r, f"win_{th}"].mean(),
                        f"acc_at_{th}": pv.loc[r, f"correct_{th}"].mean()})

        # 4. task stopping: Threshold fit on agent-fit, per-video stop windows on agent-test
        for c in costs:
            agent = pomdp.Threshold().fit(B_fit, q_fit, y_fit, c)
            r = np.array([pomdp.rollout(agent, B, yy, c) for B, yy in zip(B_test, y_test)])
            stops[m, c] = r  # columns: correct, windows, reward
            row[f"stop_win_c{c}"] = r[:, 1].mean()
        summary.append(row)
        print(f"{m:<24} alpha={alpha:.2f} final acc={row['final_acc']:.3f} ECE@1 raw/cal={row['ece_window1_raw']:.3f}/{row['ece_window1']:.3f} "
              + " ".join(f"reach{th}={row[f'reached_{th}']:.2f}@{row[f'win_{th}']:.2f}w" for th in THETAS)
              + f" commit={row['commit_win']:.2f}w", flush=True)
    per_window, summary = pd.DataFrame(per_window), pd.DataFrame(summary)
    per_window.to_csv(out / "per_window.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)

    # ---- paired differences vs the reference model (same agent-test videos)
    ref, paired = args.models[0], []
    for m in args.models[1:]:
        tests = {f"windows to max b >= {th} (censored at T)": per_video[m][f"win_{th}"] - per_video[ref][f"win_{th}"]
                 for th in THETAS}
        tests["commitment window"] = per_video[m].commit_win - per_video[ref].commit_win
        for c in costs:
            tests[f"Threshold stop windows, c={c}"] = stops[m, c][:, 1] - stops[ref, c][:, 1]
            tests[f"Threshold reward, c={c}"] = stops[m, c][:, 2] - stops[ref, c][:, 2]
        for name, d in tests.items():
            mean, lo, hi = paired_ci(d)
            paired.append(dict(model=m, reference=ref, metric=name, diff=mean, ci_lo=lo, ci_hi=hi))
            print(f"{m} - {ref}  {name:<38} {mean:+.3f}  [{lo:+.3f}, {hi:+.3f}]")
    pd.DataFrame(paired).to_csv(out / "paired.csv", index=False)

    # ---- Figure 5a: confidence vs evidence (one measure per panel, one axis each)
    colors = dict(zip(args.models, PALETTE))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for m, g in per_window.groupby("model", sort=False):
        for ax, col in zip(axes, ["acc", "conf", "ece"]):
            ax.plot(g.window, g[col], color=colors[m], marker="o", label=m)
    for ax, title in zip(axes, ["Accuracy", "Calibrated confidence (mean max b~_t)", "Calibration error (ECE, calibrated)"]):
        ax.set(xlabel="windows observed", title=title)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=len(args.models))
    fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(out / "fig5a_confidence.png", dpi=200); plt.close(fig)

    # ---- Figure 5b: P(confident by window t), one panel per theta
    fig, axes = plt.subplots(1, len(THETAS), figsize=(12, 3.6), sharey=True)
    ts = np.arange(1, T_max + 1)
    for ax, th in zip(axes, THETAS):
        for m in args.models:
            pv = per_video[m]
            frac = [(pv[f"reached_{th}"] & (pv[f"win_{th}"] <= t)).mean() for t in ts]
            ax.plot(ts, frac, color=colors[m], marker="o", label=m)
        ax.set(xlabel="windows observed", title=f"calibrated max b~_t ≥ {th}")
    axes[0].set_ylabel("fraction of videos")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=len(args.models))
    fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(out / "fig5b_first_passage.png", dpi=200); plt.close(fig)

    # ---- Figure 5c: task stopping time vs cost (Threshold agent), mean with 95% bootstrap CI
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for m in args.models:
        ci = np.array([paired_ci(stops[m, c][:, 1]) for c in costs])
        ax.errorbar(costs, ci[:, 0], yerr=[ci[:, 0] - ci[:, 1], ci[:, 2] - ci[:, 0]], color=colors[m],
                    marker="o", capsize=3, label=m)
    ax.set_xscale("symlog", linthresh=0.002)
    ax.set(xlabel="cost per extra window c", ylabel="mean windows before STOP", title="When does the agent stop?")
    ax.legend()
    fig.tight_layout(); fig.savefig(out / "fig5c_stop_time.png", dpi=200); plt.close(fig)
    print(f"results -> {out}")


if __name__ == "__main__":
    main()
