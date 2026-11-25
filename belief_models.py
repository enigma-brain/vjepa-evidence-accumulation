"""Does a step-dependent evidence weight fix the belief calibration? (offline, CPU)

probe_analysis.py found that a single alpha is a compromise: single windows are overconfident, but the
accumulated final belief is underconfident. This compares cross-fitted belief models of the form

    log b_t = log p(y) + w(t) * sum_{tau<=t} [ tau_T * log q_tau(y) - log p(y) ]   (q_tau re-normalized)
    w(t)    = alpha * (t+1)^(gamma-1)

  const        tau_T=1, gamma=1     (current model: tempered naive Bayes, one alpha)
  temp         tau_T fit, gamma=1   (per-window temperature, then naive Bayes)
  power        tau_T=1, gamma fit   (gamma<1: later windows add less; gamma=0 averages log q)
  temp+power   both

Parameters are fit on one cross-fit half (max mean log b_t(y) over all steps) and scored on the other,
with the same folds as run_pomdp.py. Reported per obs: NLL and accuracy over all steps and at the final
step, and ECE (15 bins) of b_t at all steps and at the final step.

Usage: python belief_models.py [--obs linear meta]
Output: RESULTS_DIR/<model>_<split>_fs<step>_belief_models.csv
"""
import argparse
import itertools

import numpy as np
import pandas as pd

import config
import pomdp
from probe_analysis import ece
from run_pomdp import crossfit_folds, load_features

TEMPS = np.round(np.linspace(0.3, 1.5, 13), 2)
ALPHAS = np.round(np.linspace(0.05, 1.5, 30), 3)
GAMMAS = np.round(np.linspace(0.0, 1.0, 11), 2)
MODELS = {"const": (False, False), "temp": (True, False), "power": (False, True), "temp+power": (True, True)}


def beliefs(lq, lp, alpha, gamma=1.0, temp=1.0):
    if temp != 1.0:
        lq = temp * lq
        lq = lq - pomdp.logsumexp(lq)
    n = np.arange(1, len(lq) + 1)[:, None]
    lb = lp + alpha * n ** (gamma - 1) * np.cumsum(lq - lp, axis=0)
    return lb - pomdp.logsumexp(lb)  # log b


def mean_ll(logq, y, lp, **kw):
    return np.mean(np.concatenate([beliefs(q, lp, **kw)[:, yy] for q, yy in zip(logq, y)]))


def fit(logq, y, lp, use_temp, use_gamma, max_videos=600, seed=0):
    sub = np.random.default_rng(seed).permutation(len(y))[:max_videos]  # grid search on a subsample
    lq, yy = [logq[i] for i in sub], y[sub]
    grid = itertools.product(TEMPS if use_temp else [1.0], GAMMAS if use_gamma else [1.0], ALPHAS)
    best = max(grid, key=lambda g: mean_ll(lq, yy, lp, temp=g[0], gamma=g[1], alpha=g[2]))
    return dict(temp=float(best[0]), gamma=float(best[1]), alpha=float(best[2]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--obs", nargs="+", default=["linear", "meta"])
    p.add_argument("--split", default=config.EVAL_SPLIT)
    p.add_argument("--window", default=config.WINDOW, help="window tag in file names (default: the current protocol)")
    args = p.parse_args()
    name = f"{config.MODEL_TAG}_{args.split}_{args.window}"

    rows = []
    for obs in args.obs:
        logq, y, _, _ = load_features(str(config.FEATURES_DIR / f"features_{name}_shard*.h5"), obs)
        for model, (ut, ug) in MODELS.items():
            LB, params = [None] * len(y), []
            for fi, sc in crossfit_folds(y):
                lp = np.log(pomdp.class_prior(y[fi], config.NUM_CLASSES))
                kw = fit([logq[i] for i in fi], y[fi], lp, ut, ug)
                params.append(kw)
                for i in sc:
                    LB[i] = beliefs(logq[i], lp, **kw)
            allb = np.exp(np.concatenate(LB))
            ally = np.repeat(y, [len(b) for b in LB])
            fin = np.exp(np.stack([b[-1] for b in LB]))
            row = dict(obs=obs, model=model)
            for lvl, P, yy in [("all", allb, ally), ("final", fin, y)]:
                c, ok = P.max(1), P.argmax(1) == yy
                row |= {f"nll_{lvl}": -np.log(P[np.arange(len(yy)), yy] + 1e-12).mean(), f"acc_{lvl}": ok.mean(),
                        f"conf_{lvl}": c.mean(), f"ece_{lvl}": ece(c, ok)}
            row["params"] = " | ".join(f"T={k['temp']} g={k['gamma']} a={k['alpha']}" for k in params)
            rows.append(row)
            print(row, flush=True)

    df = pd.DataFrame(rows)
    out = config.RESULTS_DIR / f"{name}_belief_models.csv"
    df.to_csv(out, index=False)
    with pd.option_context("display.width", 250, "display.precision", 3):
        print(df.to_string(index=False))
    print(f"results -> {out}")


if __name__ == "__main__":
    main()
