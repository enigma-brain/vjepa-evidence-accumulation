"""Evidence accumulation as a POMDP (proposal Fig. 1).

Hidden state: class y. Observation at step t: the probe posterior q_t(y) = p(y | o_t) for window t.
Belief: b_t(y) = p(y | o_0..o_t). At each step the agent STOPs (reward 1[argmax b_t == y])
or CONTINUEs (reward -c, observe the next window). The last window forces a stop.

Inputs everywhere: `logq` = list of per-video arrays [T_v, C] of log p(y | o_t); `y` = labels.

The only interface an agent needs:
    agent.act(belief, t, T) -> bool        # True = STOP. belief is b_t, t is 0-based, T = #windows
    agent.fit(Bs, logq, y, cost)           # optional, called on the agent-fit split
                                           # (Bs = precomputed beliefs per video, logq = raw observations)
"""
import numpy as np


# ---------------------------------------------------------------- beliefs
def logsumexp(x, axis=-1):
    m = x.max(axis=axis, keepdims=True)
    return m + np.log(np.exp(x - m).sum(axis=axis, keepdims=True))


def class_prior(y, num_classes, smoothing=1.0):
    counts = np.bincount(y, minlength=num_classes) + smoothing
    return counts / counts.sum()


def beliefs(logq, log_prior, alpha=1.0, mode="nb"):
    """All beliefs of one video, [T, C].

    mode="nb":     tempered naive Bayes. Using p(o|y) ∝ p(y|o) / p(y):
                   log b_t = log p(y) + alpha * sum_{tau<=t} [log q_tau(y) - log p(y)]
                   alpha=1 is exact if windows are independent given y; alpha<1 discounts correlation.
    mode="latest": b_t = q_t (no memory; only the newest window counts).
    """
    if mode == "latest":
        return np.exp(logq)
    lb = log_prior + alpha * np.cumsum(logq - log_prior, axis=0)
    return np.exp(lb - logsumexp(lb))


def update_belief(b, logq_next, log_prior, alpha):
    """One naive-Bayes step b_t -> b_{t+1}. Works on a batch: b [..., C], logq_next [..., C]."""
    lb = np.log(np.maximum(b, 1e-300)) + alpha * (logq_next - log_prior)
    return np.exp(lb - logsumexp(lb))


def fit_alpha(logq, y, log_prior, grid=np.linspace(0.05, 1.5, 30)):
    """Pick alpha maximizing the mean log-belief of the true class over all videos and steps."""
    def score(a):
        return np.mean(np.concatenate([np.log(beliefs(q, log_prior, a)[:, yy] + 1e-12) for q, yy in zip(logq, y)]))
    scores = [score(a) for a in grid]
    return float(grid[int(np.argmax(scores))]), dict(zip(grid.tolist(), scores))


# ---------------------------------------------------------------- rollouts
def rollout(agent, B, y, cost):
    """Run one episode on precomputed beliefs B [T, C] of one video."""
    T = len(B)
    for t in range(T):
        if t == T - 1 or agent.act(B[t], t, T):
            break
    correct = float(B[t].argmax() == y)
    return correct, t + 1, correct - cost * t  # (correct, windows observed, reward)


def evaluate(agent, Bs, y, cost):
    r = np.array([rollout(agent, B, yy, cost) for B, yy in zip(Bs, y)])
    return dict(acc=r[:, 0].mean(), steps=r[:, 1].mean(), reward=r[:, 2].mean())


# ---------------------------------------------------------------- agents
class FixedBudget:
    """Non-adaptive baseline: always watch n windows, then stop."""

    def __init__(self, n=1):
        self.n = n

    def act(self, b, t, T):
        return t + 1 >= self.n

    def fit(self, Bs, logq, y, cost, grid=range(1, 30)):
        self.n = max(grid, key=lambda n: evaluate(FixedBudget(n), Bs, y, cost)["reward"])
        return self

    def __repr__(self):
        return f"FixedBudget(n={self.n})"


class Threshold:
    """Stop as soon as the most likely class has belief >= theta (SPRT-like)."""

    def __init__(self, theta=0.9):
        self.theta = theta

    def act(self, b, t, T):
        return b.max() >= self.theta

    def fit(self, Bs, logq, y, cost, grid=np.linspace(0.0, 1.0, 41)):  # theta=0: always stop at once
        self.theta = float(max(grid, key=lambda th: evaluate(Threshold(th), Bs, y, cost)["reward"]))
        return self

    def __repr__(self):
        return f"Threshold(theta={self.theta:.3f})"


class Lookahead:
    """One-step expansion of the decision tree in Fig. 1:
        STOP  value = max_y b_t(y)
        CONT. value = E_{o_{t+1} ~ p(o | b_t)} [ max_y b_{t+1}(y) ] - c
    The expectation uses an empirical observation model p(o | y, t): sample y ~ b_t, then the
    window-(t+1) observation of a random agent-fit video of class y.
    """

    def __init__(self, log_prior, alpha, cost=0.0, m=64, seed=0):
        self.log_prior, self.alpha, self.cost, self.m = log_prior, alpha, cost, m
        self.rng = np.random.default_rng(seed)

    def fit(self, Bs, logq, y, cost):
        """Build bank[t][c] = array of log q_t from agent-fit videos of class c that have a window t."""
        self.cost = cost
        C = len(self.log_prior)
        T_max = max(len(q) for q in logq)
        self.bank = [[[] for _ in range(C)] for _ in range(T_max)]
        for q, yy in zip(logq, y):
            for t in range(len(q)):
                self.bank[t][yy].append(q[t])
        self.bank = [[np.array(lst) for lst in row] for row in self.bank]
        return self

    def act(self, b, t, T):
        if t + 1 >= T or t + 1 >= len(self.bank):
            return True
        ys = self.rng.choice(len(b), size=self.m, p=b)
        obs = []
        for c in ys:
            pool = self.bank[t + 1][c]
            if len(pool):
                obs.append(pool[self.rng.integers(len(pool))])
        if not obs:
            return True
        b_next = update_belief(b[None], np.array(obs), self.log_prior, self.alpha)
        return b.max() >= b_next.max(1).mean() - self.cost

    def __repr__(self):
        return f"Lookahead(m={self.m}, alpha={self.alpha:.2f})"


class FittedQ:
    """Model-free RL baseline: fitted Q-iteration with linear Q-functions over belief features.

    phi(b, t, T) = [1, top-5 sorted b, entropy(b), t/T, top1^2, top1*t/T]
    Q_stop(phi)  ~ regression of the realized terminal reward 1[argmax b == y]  (no calibration assumed)
    Q_cont(phi)  ~ -c + max(Q_stop(phi'), Q_cont(phi')), iterated on observed transitions (phi -> phi')
    """

    def __init__(self, iters=20, ridge=1e-3):
        self.iters, self.ridge = iters, ridge

    @staticmethod
    def features(B, ts, T):
        """B [n, C] beliefs at steps ts [n] of a video with T windows -> [n, F]."""
        s = -np.sort(-B, axis=1)[:, :5]
        s = np.pad(s, ((0, 0), (0, 5 - s.shape[1])))  # fewer than 5 classes
        H = -(B * np.log(np.maximum(B, 1e-12))).sum(1, keepdims=True)
        tt = (np.asarray(ts) / T)[:, None]
        return np.hstack([np.ones_like(tt), s, H, tt, s[:, :1] ** 2, s[:, :1] * tt])

    def _ridge(self, X, r):
        return np.linalg.solve(X.T @ X + self.ridge * np.eye(X.shape[1]), X.T @ r)

    def fit(self, Bs, logq, y, cost):
        X, X_next, correct, has_next = [], [], [], []
        for B, yy in zip(Bs, y):
            phi = self.features(B, np.arange(len(B)), len(B))
            X.append(phi)
            correct.append(B.argmax(1) == yy)
            X_next.append(np.vstack([phi[1:], phi[-1:]]))  # last row: dummy (no next state)
            has_next.append(np.arange(len(B)) < len(B) - 1)
        X, X_next = np.vstack(X), np.vstack(X_next)
        correct, has_next = np.concatenate(correct).astype(float), np.concatenate(has_next)
        self.w_stop = self._ridge(X, correct)
        q_stop_next = X_next @ self.w_stop
        is_last_next = np.concatenate([np.arange(len(B)) + 1 == len(B) - 1 for B in Bs])  # next state is terminal
        self.w_cont = np.zeros(X.shape[1])
        Xc = X[has_next]
        for _ in range(self.iters):
            v_next = np.where(is_last_next, q_stop_next, np.maximum(q_stop_next, X_next @ self.w_cont))
            self.w_cont = self._ridge(Xc, (-cost + v_next)[has_next])
        return self

    def act(self, b, t, T):
        phi = self.features(b[None], [t], T)[0]
        return float(phi @ self.w_stop) >= float(phi @ self.w_cont)

    def __repr__(self):
        return f"FittedQ(iters={self.iters})"
