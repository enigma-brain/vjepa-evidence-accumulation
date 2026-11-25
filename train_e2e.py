"""End-to-end: learn the linear probe, the belief temperature alpha and the stopping rule together.

Model (the readout stays linear; the V-JEPA 2 encoder is frozen, we only read its cached embeddings):
    log q_t  = log_softmax(W x_t + b)                              linear probe on emb_mean (as train_probe.py)
    log b_t  = log p + alpha * sum_{tau<=t} [log q_tau - log p]    same tempered naive Bayes as pomdp.beliefs
    h_t      = sigmoid(w . phi(b_t, t/T) + w0)                     hazard = P(stop at t | not stopped yet)
               phi = same features as pomdp.FittedQ;  h_{T-1} = 1 (forced stop at the last window)

STOP/CONTINUE never changes what the agent sees, so the expected reward of this stochastic policy is
an exact, differentiable sum. No RL sampling is needed:
    P(stop at t) = h_t * prod_{s<t} (1 - h_s)
    loss = -sum_t P(stop at t) * (acc_t - c t)  +  lam * mean_t CE(b_t, y)
    acc_t = softmax(k log b_t)[y]   a sharpened soft-argmax, -> the real reward 1[argmax b_t = y] as k grows.
(b_t(y) itself is a bad stand-in: it keeps rising with more windows even when the argmax doesn't change,
so it pays to keep watching, and it pays to inflate alpha.)

Data: the same agent-fit / agent-test split of SSv2 validation as run_pomdp.py. The probe starts from
train_probe.py's weights (50k SSv2-train videos). Everything is then fit on agent-fit (20% of it held
out to pick the epoch) and evaluated on agent-test with pomdp.rollout, stopping when h_t >= 0.5.

Three variants per cost c (all learn the stop head from the expected reward):
    frozen : probe fixed (train-split weights); alpha learns from the full loss
    ce     : probe + alpha learn only from the cross-entropy, i.e. the Bayes pipeline's objective,
             but on agent-fit data. The reward gradient reaches only the stop head.
    full   : probe + alpha also learn from the expected reward (true end-to-end)
    full vs ce  = what reward-driven (end-to-end) training does to the probe;
    frozen vs Threshold (run_pomdp.py) = a learned stopping rule vs a tuned confidence threshold.

Usage: python train_e2e.py [--costs 0.0,0.01] [--lam 0.1]
Output: RESULTS_DIR/<model>_validation_fs<step>_linear/e2e_summary.csv + fig4_e2e.png
"""
import argparse
import glob

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import config
import pomdp
import run_pomdp  # load_features, stratified_split, plot style
from run_pomdp import COLORS, plt


# ---------------------------------------------------------------- data
def load_embeddings(pattern):
    """Per-video window embeddings, in exactly the video order of run_pomdp.load_features."""
    X, y = [], []
    for fn in sorted(glob.glob(pattern)):
        with h5py.File(fn) as f:
            E, vi, t = f["emb_mean"][:].astype(np.float32), f["video_idx"][:], f["t"][:]
            labels, nfr = f["label"][:], f["n_frames"][:]
        order = np.lexsort((t, vi))
        E, vi = E[order], vi[order]
        bounds = np.searchsorted(vi, np.arange(len(labels) + 1))
        for v in range(len(labels)):
            if nfr[v] == 0 or bounds[v] == bounds[v + 1]:
                continue
            X.append(E[bounds[v] : bounds[v + 1]])
            y.append(labels[v])
    return X, np.array(y)


class Videos:
    """All windows in one flat tensor; a batch of videos is gathered into a padded [B, T, D] block."""

    def __init__(self, X, y):
        self.flat = torch.tensor(np.concatenate(X))
        self.T = torch.tensor([len(x) for x in X])
        self.start = torch.cumsum(self.T, 0) - self.T
        self.y = torch.tensor(y)

    def batch(self, vids):
        T = self.T[vids]
        t = torch.arange(int(T.max()))
        mask = t[None] < T[:, None]                                         # [B, Tb]
        rows = (self.start[vids, None] + t[None]).clamp(max=len(self.flat) - 1)
        return self.flat[rows], mask, T, self.y[vids]


# ---------------------------------------------------------------- model
class E2E(torch.nn.Module):
    def __init__(self, probe_state, log_prior, alpha0=0.2):
        super().__init__()
        self.probe = torch.nn.Linear(*probe_state["weight"].T.shape)
        self.probe.load_state_dict(probe_state)
        self.alpha_raw = torch.nn.Parameter(torch.tensor(float(np.log(np.expm1(alpha0)))))  # softplus^-1
        self.stop = torch.nn.Linear(9, 1)  # 9 = pomdp.FittedQ.features without its constant 1
        torch.nn.init.zeros_(self.stop.weight)
        torch.nn.init.zeros_(self.stop.bias)  # start at h = 0.5
        self.register_buffer("lp", torch.tensor(log_prior, dtype=torch.float32))

    @property
    def alpha(self):
        return F.softplus(self.alpha_raw)

    def forward(self, x, mask, T, detach_belief=False):
        """x [B,Tb,D] -> log beliefs [B,Tb,C], hazards [B,Tb] (forced to 1 at each video's last window).
        detach_belief: the stop head sees the belief but sends no gradient back into probe / alpha."""
        logq = F.log_softmax(self.probe(x), -1)
        inc = self.alpha * (logq - self.lp) * mask[..., None]
        logb = F.log_softmax(self.lp + torch.cumsum(inc, 1), -1)
        b = logb.detach().exp() if detach_belief else logb.exp()
        h = torch.sigmoid(self.stop(self.phi(b, T))[..., 0])
        last = torch.arange(x.shape[1])[None] >= (T[:, None] - 1)
        return logb, torch.where(last, torch.ones_like(h), h)

    @staticmethod
    def phi(b, T):
        """Same features as pomdp.FittedQ.features: [1, top-5 b, entropy, t/T, top1^2, top1*t/T] minus the 1."""
        s = b.topk(5, -1).values
        H = -(b * b.clamp_min(1e-12).log()).sum(-1, keepdim=True)
        tt = (torch.arange(b.shape[1])[None] / T[:, None])[..., None].expand_as(H)
        return torch.cat([s, H, tt, s[..., :1] ** 2, s[..., :1] * tt], -1)


def loss_fn(model, data, vids, cost, lam, variant, k):
    x, mask, T, y = data.batch(vids)
    logb, h = model(x, mask, T, detach_belief=variant == "ce")
    surv = torch.cumprod(torch.cat([torch.ones_like(h[:, :1]), 1 - h[:, :-1]], 1), 1)  # prod_{s<t}(1-h_s)
    p_stop = h * surv * mask
    logb_y = logb.gather(-1, y[:, None, None].expand(-1, logb.shape[1], 1))[..., 0]
    t = torch.arange(h.shape[1])[None].float()
    lb = logb.detach() if variant == "ce" else logb
    acc = F.softmax(k * lb, -1).gather(-1, y[:, None, None].expand(-1, lb.shape[1], 1))[..., 0]
    exp_reward = (p_stop * (acc - cost * t)).sum(1).mean()
    ce = -(logb_y * mask).sum() / mask.sum()
    return -exp_reward + lam * ce, exp_reward.item()


@torch.no_grad()
def quick_reward(model, data, vids, cost):
    """Real reward (argmax correct - c t, deterministic stop at the first h_t >= 0.5), vectorized."""
    x, mask, T, y = data.batch(torch.as_tensor(vids))
    logb, h = model(x, mask, T)
    t_stop = (h >= 0.5).float().argmax(1)  # first True; h is forced to 1 at the last window
    correct = logb[torch.arange(len(vids)), t_stop].argmax(-1) == y
    return (correct.float() - cost * t_stop).mean().item()


@torch.no_grad()
def to_numpy_policy(model, data, vids):
    """Learned beliefs B (via pomdp.beliefs, the same formula) + a pomdp-style agent for the stop head."""
    lp, alpha = model.lp.numpy(), model.alpha.item()
    Bs = []
    for v in vids:
        x = data.flat[data.start[v] : data.start[v] + data.T[v]]
        Bs.append(pomdp.beliefs(F.log_softmax(model.probe(x), -1).numpy().astype(np.float64), lp, alpha))
    return Bs, StopHead(model)


class StopHead:
    """pomdp agent interface: act(belief, t, T) -> True = stop.  Stops when the learned hazard >= 0.5."""

    def __init__(self, model):
        self.w = model.stop.weight.detach().numpy()[0].astype(np.float64)
        self.w0 = float(model.stop.bias.item())

    def act(self, belief, t, T):
        phi = pomdp.FittedQ.features(belief[None], [t], T)[0][1:]  # drop the constant 1 (we have a bias)
        return self.w @ phi + self.w0 >= 0.0  # sigmoid(z) >= 0.5

    def __repr__(self):
        return "StopHead"


def train_one(data, fit_vids, val_vids, probe_state, lp, cost, lam, variant, epochs, k, seed=0):
    torch.manual_seed(seed)
    model = E2E(probe_state, lp)
    model.probe.requires_grad_(variant != "frozen")
    opt = torch.optim.AdamW([
        {"params": model.probe.parameters(), "lr": 1e-4, "weight_decay": 1e-2},
        {"params": [model.alpha_raw, *model.stop.parameters()], "lr": 3e-2, "weight_decay": 0.0},
    ])
    best = (-np.inf, None, -1)
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(fit_vids)
        for i in range(0, len(perm), 256):
            loss, _ = loss_fn(model, data, torch.tensor(perm[i : i + 256]), cost, lam, variant, k)
            opt.zero_grad()
            loss.backward()
            opt.step()
        # pick the epoch by the *real* (argmax, deterministic-stop) reward on held-out agent-fit videos
        model.eval()
        r = quick_reward(model, data, val_vids, cost)
        if r > best[0]:
            best = (r, {k: v.clone() for k, v in model.state_dict().items()}, ep + 1)
    model.load_state_dict(best[1])
    return model, best[2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--window", default=config.WINDOW, help="window tag in file names (default: the current protocol)")
    p.add_argument("--probe", default=None, help="default: probe-linear_<tag>_train-50000_fs<step>.pt")
    p.add_argument("--costs", default="0.0,0.002,0.005,0.01,0.02,0.05,0.1")
    p.add_argument("--lam", type=float, default=0.1, help="weight of the per-window cross-entropy term")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--sharpness", type=float, default=10.0, help="k in the soft-argmax reward")
    p.add_argument("--plot-only", action="store_true", help="redraw fig4 from e2e_summary.csv + summary.csv")
    args = p.parse_args()
    torch.set_num_threads(min(16, torch.get_num_threads()))

    name = f"{config.MODEL_TAG}_{config.EVAL_SPLIT}_{args.window}"
    out = config.RESULTS_DIR / f"{name}_linear"
    out.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        return plot(out)
    ckpt = torch.load(args.probe or config.FEATURES_DIR / f"probe-linear_{config.MODEL_TAG}_{config.TRAIN_SPLIT}_{args.window}.pt",
                      weights_only=False)

    X, y = load_embeddings(str(config.FEATURES_DIR / f"features_{name}_shard*.h5"))
    X = [(x - ckpt["mu"]) / ckpt["sd"] for x in X]  # same standardization the probe was trained with
    data = Videos(X, y)
    fit_idx, test_idx = run_pomdp.stratified_split(y)  # identical split to run_pomdp.py
    lp = np.log(pomdp.class_prior(y[fit_idx], config.NUM_CLASSES))
    rng = np.random.default_rng(0)
    held = rng.permutation(fit_idx)
    val_vids, fit_vids = np.sort(held[: len(held) // 5]), np.sort(held[len(held) // 5 :])
    print(f"{len(y)} videos: fit {len(fit_vids)} / val {len(val_vids)} (both agent-fit) / test {len(test_idx)}")

    rows = []
    for cost in [float(c) for c in args.costs.split(",")]:
        for variant in ["frozen", "ce", "full"]:
            model, ep = train_one(data, fit_vids, val_vids, ckpt["state_dict"], lp, cost, args.lam, variant, args.epochs, args.sharpness)
            Bs, agent = to_numpy_policy(model, data, test_idx)  # test through the standard pomdp interface
            r = pomdp.evaluate(agent, Bs, y[test_idx], cost)
            assert abs(r["reward"] - quick_reward(model, data, test_idx, cost)) < 1e-3  # torch == numpy path
            with torch.no_grad():  # per-window accuracy of the (possibly retrained) linear probe on agent-test
                rows_t = torch.cat([torch.arange(data.start[v], data.start[v] + data.T[v]) for v in test_idx])
                win_acc = (model.probe(data.flat[rows_t]).argmax(1) == data.y[np.repeat(test_idx, data.T[test_idx])]).float().mean().item()
            rows.append(dict(cost=cost, agent=f"E2E-{variant}", alpha=model.alpha.item(), epoch=ep, probe_acc=win_acc, **r))
            print(f"c={cost:<6} E2E-{variant:<7} alpha={model.alpha.item():.3f} epoch={ep:2d} probe={win_acc:.3f}  "
                  f"acc={r['acc']:.3f} steps={r['steps']:.2f} reward={r['reward']:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(out / "e2e_summary.csv", index=False)
    plot(out)


def plot(out):
    # ---- Figure 4: reward / windows vs cost, end-to-end vs the best Bayes agent (Threshold, run_pomdp.py)
    ref = pd.read_csv(out / "summary.csv")
    both = pd.concat([ref[ref.agent == "Threshold"], pd.read_csv(out / "e2e_summary.csv")])
    colors = {"Threshold": COLORS["Threshold"], "E2E-frozen": "#e87ba4", "E2E-ce": "#008300", "E2E-full": "#6250d6"}  # slots 1,5,6,7
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for agent_name, g in both.groupby("agent", sort=False):
        axes[0].plot(g.cost, g.reward, color=colors[agent_name], marker="o", label=agent_name)
        axes[1].plot(g.cost, g.steps, color=colors[agent_name], marker="o", label=agent_name)
    axes[0].set(xlabel="cost per extra window c", ylabel="mean reward (agent-test)", title="Reward")
    axes[1].set(xlabel="cost per extra window c", ylabel="mean windows observed", title="Evidence used")
    for ax in axes:
        ax.set_xscale("symlog", linthresh=0.002)
    axes[1].legend()
    fig.tight_layout(); fig.savefig(out / "fig4_e2e.png", dpi=200); plt.close(fig)
    print(f"results -> {out}")


if __name__ == "__main__":
    main()
