"""Tests on a synthetic POMDP where the exact posterior is known.

Classes y in {0,1,2}; each window gives o_t ~ N(mu_y, sigma^2 I) independently.
The "probe" outputs the exact single-window posterior q_t(y) = p(y | o_t) under a known prior,
so tempered naive Bayes with alpha=1 must equal the exact multi-window posterior.
"""
import numpy as np

import pomdp

C, D, SIGMA = 3, 2, 2.0
MU = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
PRIOR = np.array([0.5, 0.3, 0.2])


def log_lik(o):  # [T, C] log p(o_t | y)
    return -((o[:, None, :] - MU[None]) ** 2).sum(-1) / (2 * SIGMA**2)


def make_data(n=600, T=10, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.choice(C, size=n, p=PRIOR)
    logq, exact = [], []
    for yy in y:
        o = MU[yy] + SIGMA * rng.standard_normal((T, D))
        ll = log_lik(o)
        lq = np.log(PRIOR) + ll
        logq.append(lq - pomdp.logsumexp(lq))  # single-window posterior
        le = np.log(PRIOR) + np.cumsum(ll, 0)
        exact.append(np.exp(le - pomdp.logsumexp(le)))  # exact posterior after t+1 windows
    return logq, y, exact


def test_naive_bayes_alpha1_is_exact_posterior():
    logq, y, exact = make_data(n=20)
    for q, e in zip(logq, exact):
        np.testing.assert_allclose(pomdp.beliefs(q, np.log(PRIOR), alpha=1.0), e, atol=1e-10)


def test_incremental_update_matches_batch():
    logq, _, _ = make_data(n=5)
    lp = np.log(PRIOR)
    B = pomdp.beliefs(logq[0], lp, alpha=0.7)
    b = B[0]
    for t in range(1, len(B)):
        b = pomdp.update_belief(b, logq[0][t], lp, 0.7)
        np.testing.assert_allclose(b, B[t], atol=1e-10)


def test_fit_alpha_recovers_one_for_independent_windows():
    logq, y, _ = make_data()
    alpha, _ = pomdp.fit_alpha(logq, y, np.log(PRIOR))
    assert 0.8 <= alpha <= 1.2


def test_reward_and_step_accounting():
    B = np.array([[0.6, 0.4], [0.2, 0.8], [0.1, 0.9]])
    assert pomdp.rollout(pomdp.FixedBudget(1), B, 0, cost=0.1) == (1.0, 1, 1.0)
    correct, steps, reward = pomdp.rollout(pomdp.FixedBudget(2), B, 1, cost=0.1)
    assert (correct, steps) == (1.0, 2) and np.isclose(reward, 0.9)
    assert pomdp.rollout(pomdp.FixedBudget(99), B, 0, cost=0.1)[1] == 3  # forced stop at the end
    assert pomdp.rollout(pomdp.Threshold(0.85), B, 1, cost=0.0)[1] == 3


def test_adaptive_agents_beat_fixed_budget():
    cost = 0.02
    logq_fit, y_fit, _ = make_data(seed=1)
    logq, y, _ = make_data(seed=2)
    lp = np.log(PRIOR)
    B_fit = [pomdp.beliefs(q, lp) for q in logq_fit]
    Bs = [pomdp.beliefs(q, lp) for q in logq]
    fixed = pomdp.evaluate(pomdp.FixedBudget().fit(B_fit, logq_fit, y_fit, cost), Bs, y, cost)
    thresh = pomdp.evaluate(pomdp.Threshold().fit(B_fit, logq_fit, y_fit, cost), Bs, y, cost)
    look = pomdp.evaluate(pomdp.Lookahead(lp, 1.0).fit(B_fit, logq_fit, y_fit, cost), Bs, y, cost)
    fq = pomdp.evaluate(pomdp.FittedQ().fit(B_fit, logq_fit, y_fit, cost), Bs, y, cost)
    lsm = pomdp.evaluate(pomdp.LSM().fit(B_fit, logq_fit, y_fit, cost), Bs, y, cost)
    assert lsm["reward"] > fixed["reward"] - 0.01
    assert fq["reward"] > fixed["reward"] - 0.01
    assert thresh["reward"] > fixed["reward"]
    assert look["reward"] > fixed["reward"] - 0.01  # myopic, but should be competitive


def test_lsm_exact_on_known_values():
    """Two-step toy: window 0 is uninformative for half the videos; LSM must learn to continue exactly there."""
    cost = 0.1
    Bs, y = [], []
    for i in range(400):
        easy = i % 2 == 0
        b0 = np.array([0.9, 0.05, 0.05]) if easy else np.array([0.34, 0.33, 0.33])
        Bs.append(np.stack([b0, np.array([0.98, 0.01, 0.01])]))
        y.append(0 if easy else [0, 1, 2][i % 3])
    y = np.array(y)
    for B, yy in zip(Bs, y):  # second window always reveals the class
        B[1] = np.eye(3)[yy] * 0.98 + 0.01
    agent = pomdp.LSM(min_n=10).fit(Bs, None, y, cost)
    steps = np.array([pomdp.rollout(agent, B, yy, cost)[1] for B, yy in zip(Bs, y)])
    assert (steps[::2] == 1).all() and (steps[1::2] == 2).all()
