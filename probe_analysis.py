"""Where does each observation model get its evidence, and can its confidence be trusted?

For each obs model (our linear probe, Meta's probe), on the eval split:
  by_position.csv   per-window accuracy and confidence by window index t, and by relative position in the
                    video (t / (T-1), 5 bins; videos with T > 1). Answers "early or late evidence?"
  calibration.csv   ECE (15 bins, on the max probability) of the raw per-window posterior q_t and of the
                    cross-fitted belief b_t (all steps, and the final step). Low fitted alpha + high
                    window ECE = overconfident, correlated windows.
  attributes.csv    Diving48 only: accuracy of each dive attribute (takeoff, somersaults, twists, position)
                    under the argmax class, per window and for the final belief, with the majority-class rate.
                    Which parts of the dive does each probe miss?
  fig_position.png  accuracy vs relative position, both probes.

Beliefs use the per-fold prior and alpha that run_pomdp.py fitted (results/<name>_<obs>/alpha.json),
so run run_pomdp.py for each obs first.

Usage: python probe_analysis.py [--obs linear meta]
Output: RESULTS_DIR/<model>_<split>_fs<step>_probe_analysis/
"""
import argparse
import json

import numpy as np
import pandas as pd

import config
import pomdp
from run_pomdp import COLORS, crossfit_folds, load_features, plt

ATTRIBUTES = ["takeoff", "somersaults", "twists", "position"]  # Diving48 labels.json: one 4-list per class


def ece(conf, correct, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(conf, edges) - 1, 0, bins - 1)
    return sum(abs(correct[b == i].mean() - conf[b == i].mean()) * (b == i).mean() for i in range(bins) if (b == i).any())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--obs", nargs="+", default=["linear", "meta"])
    p.add_argument("--split", default=config.EVAL_SPLIT)
    p.add_argument("--window", default=config.WINDOW, help="window tag in file names (default: the current protocol)")
    args = p.parse_args()

    name = f"{config.MODEL_TAG}_{args.split}_{args.window}"
    out = config.RESULTS_DIR / f"{name}_probe_analysis"
    out.mkdir(parents=True, exist_ok=True)
    attr = None
    if config.DATASET == "diving48":
        attr = np.array(json.load(open(config.DATA_DIR / "labels.json")))  # [C, 4] strings

    pos_rows, cal_rows, attr_rows = [], [], []
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for obs in args.obs:
        logq, y, _, _ = load_features(str(config.FEATURES_DIR / f"features_{name}_shard*.h5"), obs)
        alphas = json.load(open(config.RESULTS_DIR / f"{name}_{obs}" / "alpha.json"))["alpha_folds"]
        B = [None] * len(y)
        for (fi, sc), a in zip(crossfit_folds(y), alphas):  # same folds as run_pomdp.py
            lp = np.log(pomdp.class_prior(y[fi], config.NUM_CLASSES))
            for i in sc:
                B[i] = pomdp.beliefs(logq[i], lp, a)

        T = np.array([len(q) for q in logq])
        Q = np.exp(np.concatenate(logq))
        Bw = np.concatenate(B)
        yw = np.repeat(y, T)
        t = np.concatenate([np.arange(n) for n in T])
        rel = np.concatenate([np.arange(n) / max(n - 1, 1) for n in T])
        multi = np.repeat(T > 1, T)
        qc, qok = Q.max(1), Q.argmax(1) == yw

        # ---- by position
        for k in range(11):
            m = (t == k) if k < 10 else (t >= 10)
            pos_rows.append(dict(obs=obs, axis="t", bin=str(k) if k < 10 else "10+", n=int(m.sum()),
                                 acc=qok[m].mean(), conf=qc[m].mean()))
        rb = np.minimum((rel * 5).astype(int), 4)
        rel_acc = []
        for k in range(5):
            m = multi & (rb == k)
            rel_acc.append(qok[m].mean())
            pos_rows.append(dict(obs=obs, axis="relative", bin=f"{k / 5:.1f}-{(k + 1) / 5:.1f}", n=int(m.sum()),
                                 acc=qok[m].mean(), conf=qc[m].mean()))
        ax.plot(np.arange(5) / 5 + 0.1, rel_acc, marker="o", color=list(COLORS.values())[args.obs.index(obs) % len(COLORS)],
                label=f"{obs} probe")

        # ---- calibration
        Bf = np.stack([b[-1] for b in B])
        for level, P, yy in [("window q_t", Q, yw), ("belief b_t, all steps", Bw, yw), ("belief, final", Bf, y)]:
            c, ok = P.max(1), P.argmax(1) == yy
            cal_rows.append(dict(obs=obs, level=level, n=len(yy), acc=ok.mean(), mean_conf=c.mean(), ece=ece(c, ok)))
        cal_rows[-1]["alpha_folds"] = str(alphas)

        # ---- dive attributes
        if attr is not None:
            for level, P, yy in [("window", Q, yw), ("final_belief", Bf, y)]:
                pred = P.argmax(1)
                for j, a in enumerate(ATTRIBUTES):
                    truth = attr[yy, j]
                    _, counts = np.unique(truth, return_counts=True)
                    attr_rows.append(dict(obs=obs, level=level, attribute=a, acc=(attr[pred, j] == truth).mean(),
                                          majority=counts.max() / len(truth), n_values=len(np.unique(attr[:, j]))))
                attr_rows.append(dict(obs=obs, level=level, attribute="all (class)", acc=(pred == yy).mean(),
                                      majority=np.bincount(yy).max() / len(yy), n_values=config.NUM_CLASSES))

    ax.set(xlabel="relative position in video  t / (T−1)", ylabel="per-window top-1 accuracy",
           title="Where the evidence is", xlim=(0, 1))
    ax.legend()
    fig.tight_layout(); fig.savefig(out / "fig_position.png", dpi=200); plt.close(fig)

    tables = {"by_position": pd.DataFrame(pos_rows), "calibration": pd.DataFrame(cal_rows)}
    if attr_rows:
        tables["attributes"] = pd.DataFrame(attr_rows)
    with pd.option_context("display.width", 200, "display.precision", 3, "display.max_rows", 200):
        for k, df in tables.items():
            df.to_csv(out / f"{k}.csv", index=False)
            print(f"== {k}\n{df.to_string(index=False)}")
    print(f"results -> {out}")


if __name__ == "__main__":
    main()
