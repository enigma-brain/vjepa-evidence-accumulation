"""Is direction / arrow of time the bottleneck? Error analysis on SSv2's reversed class pairs.

SSv2 has classes that differ only in direction ("Pushing something from left to right" vs
"... right to left") or in the arrow of time ("Opening" vs "Closing" = the same video reversed).
The naive-Bayes belief sums per-window evidence, so it is blind to window order: if one window
cannot tell a pair apart, accumulating windows cannot either. This script measures, on the
whole eval split, cross-fitted like run_pomdp.py (each video's belief uses the prior and alpha fitted
on the other half):

  pair_acc   P(score[y] > score[partner])       - 0.5 means the readout is blind to direction
  ctrl_acc   P(score[y] > score[z*])            - same, against y's most-confused NON-partner class z*
  to_partner share of errors that go to the partner
  acc / collapsed_acc  top-1 accuracy, and accuracy if y and its partner counted as one class

for each observation model (linear probe on emb_mean, Meta's attentive probe), both per window and
for the final belief after all windows.

Usage: python direction_check.py [--obs linear meta]
Output: RESULTS_DIR/<model>_validation_fs<step>_direction/{summary.csv,per_pair.csv}
"""
import argparse
import json

import numpy as np
import pandas as pd

import config
import pomdp
from run_pomdp import crossfit_folds, load_features

# (a, b) class indices from labels.json
PAIRS = {
    "direction": [  # mirror image or vertical flip
        (93, 94),    # Pushing something from left to right / right to left
        (86, 87),    # Pulling something from left to right / right to left
        (166, 167),  # Turning the camera left / right
        (168, 165),  # Turning the camera upwards / downwards
        (45, 43),    # Moving something up / down
    ],
    "time_reversal": [  # roughly the same video played backwards
        (46, 5),     # Opening / Closing something
        (67, 66),    # Pretending to open / close something
        (14, 172),   # Folding / Unfolding something
        (6, 171),    # Covering / Uncovering something
        (106, 148),  # Putting something into something / Taking something out of something
        (72, 82),    # Pretending to put something into / take something out of something
        (49, 89),    # Plugging something into / Pulling something out of something
        (96, 95),    # Pushing something onto / off of something
        (4, 7),      # Burying something in / Digging something out of something
        (59, 62),    # Pouring something into / out of something
        (47, 109),   # Picking something up / Putting something on a surface
        (68, 74),    # Pretending to pick something up / to put something on a surface
        (0, 32),     # Approaching / Moving away from something with your camera
        (42, 40),    # Moving something closer to / away from something
        (44, 41),    # Moving something towards / away from the camera
        (37, 36),    # Moving something and something closer to / away from each other
    ],
}


def pair_stats(S, y, partner):
    """S [N,C] scores (log-probs), y [N] true labels (all have a partner)."""
    p = partner[y]
    pred = S.argmax(1)
    wrong = pred != y
    # most-confused non-partner class for each true class (from these same predictions)
    zstar = np.full(config.NUM_CLASSES, -1)
    for c in np.unique(y):
        e = pred[(y == c) & wrong & (pred != partner[c])]
        if len(e):
            zstar[c] = np.bincount(e).argmax()
    z = zstar[y]
    has_z = z >= 0
    rows = np.arange(len(y))
    return dict(
        n=len(y),
        acc=np.mean(~wrong),
        collapsed_acc=np.mean((pred == y) | (pred == p)),
        to_partner=np.mean(pred[wrong] == p[wrong]) if wrong.any() else np.nan,
        pair_acc=np.mean(S[rows, y] > S[rows, p]),
        ctrl_acc=np.mean(S[rows[has_z], y[has_z]] > S[rows[has_z], z[has_z]]),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--obs", nargs="+", default=["linear", "meta"])
    p.add_argument("--split", default=config.EVAL_SPLIT)
    p.add_argument("--window", default=config.WINDOW, help="window tag in file names (default: the current protocol)")
    args = p.parse_args()

    name = f"{config.MODEL_TAG}_{args.split}_{args.window}"
    out = config.RESULTS_DIR / f"{name}_direction"
    out.mkdir(parents=True, exist_ok=True)
    partner = np.full(config.NUM_CLASSES, -1)
    group = {}
    for g, pairs in PAIRS.items():
        for a, b in pairs:
            partner[a], partner[b] = b, a
            group[a] = group[b] = g

    summary, per_pair = [], []
    for obs in args.obs:
        logq, y, _, _ = load_features(str(config.FEATURES_DIR / f"features_{name}_shard*.h5"), obs)
        alphas = json.load(open(config.RESULTS_DIR / f"{name}_{obs}" / "alpha.json"))["alpha_folds"]
        final = {}  # video -> final log-belief, from its scoring fold
        for (fi, sc), a in zip(crossfit_folds(y), alphas):  # same folds as run_pomdp.py
            lp = np.log(pomdp.class_prior(y[fi], config.NUM_CLASSES))
            final.update({i: np.log(pomdp.beliefs(logq[i], lp, a)[-1] + 1e-300) for i in sc})
        test_idx = np.where(partner[y] >= 0)[0]

        levels = {  # (scores, labels)
            "window": (np.concatenate([logq[i] for i in test_idx]),
                       np.concatenate([np.full(len(logq[i]), y[i]) for i in test_idx])),
            "final_belief": (np.stack([final[i] for i in test_idx]), y[test_idx]),
        }
        for level, (S, yy) in levels.items():
            for g in ["all", *PAIRS]:
                m = np.ones(len(yy), bool) if g == "all" else np.array([group[c] == g for c in yy])
                summary.append(dict(obs=obs, level=level, group=g, **pair_stats(S[m], yy[m], partner)))
            for g, pairs in PAIRS.items():
                for a, b in pairs:
                    m = (yy == a) | (yy == b)
                    per_pair.append(dict(obs=obs, level=level, group=g, a=a, b=b, **pair_stats(S[m], yy[m], partner)))

    summary, per_pair = pd.DataFrame(summary), pd.DataFrame(per_pair)
    summary.to_csv(out / "summary.csv", index=False)
    per_pair.to_csv(out / "per_pair.csv", index=False)
    with pd.option_context("display.width", 200, "display.precision", 3, "display.max_rows", 200):
        print(summary.to_string(index=False))
        print(per_pair[per_pair.level == "window"].to_string(index=False))
    print(f"results -> {out}")


if __name__ == "__main__":
    main()
