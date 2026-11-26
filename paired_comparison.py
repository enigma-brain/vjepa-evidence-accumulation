"""Paired comparison of two encoders exported by export_comparison.py (+ representation_analysis.py).

Joins the two exports on video_id (same eval videos, same windows, same folds) and reports, for every
observation model both have, the per-video difference (B - A) with a 95% bootstrap CI over videos:
  A1  window accuracy / NLL / calibrated NLL (video means)
  A2  accumulation, per belief model: final accuracy and NLL, mean NLL over steps (area under the curve),
      and the curves vs t (forced stop at t; videos shorter than t keep their last belief) with the
      per-window information gain dNLL_t = NLL(b_{t-1}) - NLL(b_t)
  A3  fitted alpha / gamma / temperature side by side (not paired: one value per fold)
  A4  reward and windows per cost x agent; windows to calibrated confidence; commitment window
  B7  trajectory curvature per layer (if both have representation/B7_curvature.csv.gz); B6/B8 side by side
Sanity check: comparing an export with itself must give 0 everywhere.

Output: ROOT/comparison/paired/<B>_vs_<A>/<dataset>/<window>/{paired.csv, curves.csv, params.csv, fig_accumulation_<obs>.png}
Usage: python paired_comparison.py baseline neural [--obs linear linear-emb_ml]
"""
import argparse
import json

import numpy as np
import pandas as pd

import config
from run_pomdp import COLORS, bootstrap_ci, plt

CURVE_COLORS = [COLORS["Threshold"], COLORS["Lookahead"]]  # A, B (fixed order)
EXCLUDED_AGENTS = {"LSM"}  # older baseline exports still carry LSM rows; it is not part of the comparison


def ci_row(metric, d, **kw):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    lo, hi = bootstrap_ci(d)
    return dict(metric=metric, diff=d.mean(), lo=lo, hi=hi, n=len(d), **kw)


def video_join(a, b, cols, on="video_id"):
    m = a[[on] + cols].merge(b[[on] + cols], on=on, suffixes=("_a", "_b"))
    return m


def curve(A2, belief, t_max):
    """[videos x t_max] NLL and correctness of b_t, forward-filled past each video's end."""
    g = A2[A2.belief == belief]
    nll = g.pivot(index="video_id", columns="t", values="nll").reindex(columns=range(t_max)).ffill(axis=1)
    ok = (g.pred == g.y).astype(float)
    ok = g.assign(ok=ok).pivot(index="video_id", columns="t", values="ok").reindex(columns=range(t_max)).ffill(axis=1)
    return nll, ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("label_a", help="reference export, e.g. baseline")
    p.add_argument("label_b", help="compared export, e.g. neural")
    p.add_argument("--obs", nargs="+", default=None, help="default: every obs both exports have")
    p.add_argument("--window", default=config.WINDOW)
    args = p.parse_args()

    base = config.ROOT / "comparison"
    ra, rb = (base / l / config.DATASET / args.window for l in (args.label_a, args.label_b))
    ma, mb = json.load(open(ra / "manifest.json")), json.load(open(rb / "manifest.json"))
    assert ma["window"] == mb["window"] and ma["split"] == mb["split"], "exports use different windows / splits"
    obs_list = args.obs or sorted(set(ma["obs"]) & set(mb["obs"]))
    out = base / "paired" / f"{args.label_b}_vs_{args.label_a}" / config.DATASET / args.window
    out.mkdir(parents=True, exist_ok=True)
    print(f"{args.label_b} ({mb['model_tag']}) - {args.label_a} ({ma['model_tag']}), obs {obs_list}")

    rows, curves, params = [], [], []
    for obs in obs_list:
        da, db = ra / obs, rb / obs
        kw = dict(obs=obs)
        # A1
        w = {}
        for lab, d in (("a", da), ("b", db)):
            t = pd.read_csv(d / "A1_window.csv.gz")
            w[lab] = t.assign(ok=(t.pred == t.y).astype(float)).groupby("video_id")[["ok", "nll", "nll_cal"]].mean().reset_index()
        m = video_join(w["a"], w["b"], ["ok", "nll", "nll_cal"])
        print(f"[{obs}] {len(m)} paired videos (A {len(w['a'])}, B {len(w['b'])})")
        rows += [ci_row("A1 window acc", m.ok_b - m.ok_a, **kw), ci_row("A1 window NLL", m.nll_b - m.nll_a, **kw),
                 ci_row("A1 window NLL, calibrated", m.nll_cal_b - m.nll_cal_a, **kw)]
        # A2
        A2a, A2b = pd.read_csv(da / "A2_accumulation.csv.gz"), pd.read_csv(db / "A2_accumulation.csv.gz")
        t_max = int(np.percentile(A2a.groupby("video_id").n_win.first(), 95))
        common = np.intersect1d(A2a.video_id.unique(), A2b.video_id.unique())
        for belief in sorted(A2a.belief.unique()):
            (na, oka), (nb, okb) = curve(A2a, belief, t_max), curve(A2b, belief, t_max)
            na, oka, nb, okb = (x.loc[common] for x in (na, oka, nb, okb))
            lastA = A2a[(A2a.belief == belief) & (A2a.t == A2a.n_win - 1)].set_index("video_id").loc[common]
            lastB = A2b[(A2b.belief == belief) & (A2b.t == A2b.n_win - 1)].set_index("video_id").loc[common]
            rows += [ci_row(f"A2 final acc ({belief})", (lastB.pred == lastB.y).astype(float).values - (lastA.pred == lastA.y).astype(float).values, **kw),
                     ci_row(f"A2 final NLL ({belief})", lastB.nll.values - lastA.nll.values, **kw),
                     ci_row(f"A2 mean NLL over t<{t_max} ({belief})", nb.mean(axis=1).values - na.mean(axis=1).values, **kw)]
            gain_a, gain_b = -na.diff(axis=1), -nb.diff(axis=1)
            for t in range(t_max):
                r = dict(obs=obs, belief=belief, t=t + 1, nll_a=na[t].mean(), nll_b=nb[t].mean(),
                         acc_a=oka[t].mean(), acc_b=okb[t].mean())
                r.update({f"d_nll_{k}": v for k, v in zip(("mean", "lo", "hi"), (lambda d: (d.mean(), *bootstrap_ci(d)))((nb[t] - na[t]).values))})
                if t:
                    r.update(gain_a=gain_a[t].mean(), gain_b=gain_b[t].mean())
                curves.append(r)
        # A3
        for lab, d in ((args.label_a, da), (args.label_b, db)):
            for f in json.load(open(d / "A3_params.json"))["folds"]:
                params.append(dict(obs=obs, model=lab, **{k: v for k, v in f.items() if k != "step_temperatures"}))
        # A4
        A4a, A4b = pd.read_csv(da / "A4_agents.csv.gz"), pd.read_csv(db / "A4_agents.csv.gz")
        A4a, A4b = (d[~d.agent.isin(EXCLUDED_AGENTS)] for d in (A4a, A4b))
        m4 = A4a.merge(A4b, on=["video_id", "cost", "agent"], suffixes=("_a", "_b"))
        for (cost, agent), g in m4.groupby(["cost", "agent"]):
            rows += [ci_row(f"A4 reward c={cost} {agent}", g.reward_b - g.reward_a, **kw),
                     ci_row(f"A4 windows c={cost} {agent}", g.steps_b - g.steps_a, **kw)]
        c = video_join(pd.read_csv(da / "A4_confidence.csv.gz"), pd.read_csv(db / "A4_confidence.csv.gz"),
                       [col for col in ("win_0.5", "win_0.7", "win_0.9", "commit_win")])
        for col in ("win_0.5", "win_0.7", "win_0.9", "commit_win"):
            rows.append(ci_row(f"A4 {col} (calibrated)", c[f"{col}_b"] - c[f"{col}_a"], **kw))

        # figure: accumulation curves + paired difference (one measure per panel)
        cv = pd.DataFrame([r for r in curves if r["obs"] == obs and r["belief"] == "const"])
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
        for col, lab, color in (("a", args.label_a, CURVE_COLORS[0]), ("b", args.label_b, CURVE_COLORS[1])):
            axes[0].plot(cv.t, cv[f"acc_{col}"], marker="o", color=color, label=lab)
            axes[1].plot(cv.t, cv[f"nll_{col}"], marker="o", color=color, label=lab)
        axes[2].plot(cv.t, cv.d_nll_mean, marker="o", color=CURVE_COLORS[1])
        axes[2].fill_between(cv.t, cv.d_nll_lo, cv.d_nll_hi, color=CURVE_COLORS[1], alpha=0.15, lw=0)
        axes[2].axhline(0, color="#52514e", lw=1)
        axes[0].set(xlabel="windows observed", ylabel="top-1 accuracy", title="Accuracy of b_t")
        axes[1].set(xlabel="windows observed", ylabel="NLL of true class", title="NLL of b_t")
        axes[2].set(xlabel="windows observed", ylabel=f"NLL {args.label_b} − {args.label_a}", title="Paired difference (95% CI)")
        axes[0].legend()
        fig.suptitle(f"{config.DATASET}, obs {obs}, one-α belief")
        fig.tight_layout(); fig.savefig(out / f"fig_accumulation_{obs}.png", dpi=200); plt.close(fig)

    # B: representation (if both exported it)
    reps = [r / "representation" for r in (ra, rb)]
    if all((r / "B7_curvature.csv.gz").exists() for r in reps):
        ca, cb = (pd.read_csv(r / "B7_curvature.csv.gz") for r in reps)
        la, lb = sorted(ca.layer.unique()), sorted(cb.layer.unique())
        for i, (x, y) in enumerate(zip(la, lb)):  # pair layers by rank (ViT-L 17/19/21/23 vs same)
            m = ca[ca.layer == x].merge(cb[cb.layer == y], on="video_id", suffixes=("_a", "_b"))
            rows.append(ci_row(f"B7 curvature deg (layer {x} vs {y})", m.curvature_deg_b - m.curvature_deg_a, obs="representation"))
            rows.append(ci_row(f"B6 per-video prediction error (layer {x} vs {y})", m.pred_err_b - m.pred_err_a, obs="representation"))
        for fn in ("B6_predictability.csv", "B8_dimensionality.csv"):
            pd.concat([pd.read_csv(r / fn).assign(model=l) for r, l in zip(reps, (args.label_a, args.label_b))]).to_csv(out / fn, index=False)

    P = pd.DataFrame(rows)
    P.to_csv(out / "paired.csv", index=False)
    pd.DataFrame(curves).to_csv(out / "curves.csv", index=False)
    pd.DataFrame(params).to_csv(out / "params.csv", index=False)
    P["sig"] = np.where((P.lo > 0) | (P.hi < 0), "*", "")
    with pd.option_context("display.width", 200, "display.precision", 4, "display.max_rows", 1000):
        print(P.to_string(index=False))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
