"""Error metrics (original scale), persistence baseline and statistical tests."""
from __future__ import annotations

import itertools

import numpy as np
from scipy import stats


def metrics(y: np.ndarray, yhat: np.ndarray) -> dict:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    e = y - yhat
    denom = np.abs(y) + np.abs(yhat)
    ratio = np.where(denom > 0, 2 * np.abs(e) / np.where(denom > 0, denom, 1), 0.0)
    return dict(rmse=float(np.sqrt(np.mean(e ** 2))), mae=float(np.mean(np.abs(e))), smape=float(100 * np.mean(ratio)))


def persistence(y: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """ Naive one-step forecast"""
    return y[targets - 1]

""" scores: (nBlocks, k) lower-is-better -> average rank per method (1 = best)."""
def avg_ranks(scores: np.ndarray) -> np.ndarray:
    return np.mean(np.apply_along_axis(stats.rankdata, 1, scores), axis=0)

""" Critical difference"""
def nemenyi_cd(k: int, n: int, alpha: float = 0.05) -> float:  
    q = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    return float(q * np.sqrt(k * (k + 1) / (6.0 * n)))


def friedman_nemenyi(scores: np.ndarray, names, alpha: float = 0.05) -> dict:
    n, k = scores.shape
    chi2, p = stats.friedmanchisquare(*[scores[:, j] for j in range(k)])
    ranks = avg_ranks(scores)
    cd = nemenyi_cd(k, n, alpha)
    pairs = {}
    for i, j in itertools.combinations(range(k), 2):
        pairs[f"{names[i]}-{names[j]}"] = dict(rank_diff=float(abs(ranks[i] - ranks[j])), significant=bool(abs(ranks[i] - ranks[j]) > cd))
    return dict(chi2=float(chi2), p=float(p), ranks=dict(zip(names, map(float, ranks))), cd=cd, n=n, k=k, pairs=pairs)

"""Pairwise two-sided Wilcoxon signed-rank tests with Holm correction."""
def wilcoxon_holm(scores: np.ndarray, names) -> dict:
    k = scores.shape[1]
    raw = {}
    for i, j in itertools.combinations(range(k), 2):
        d = scores[:, i] - scores[:, j]
        p = 1.0 if np.allclose(d, 0) else stats.wilcoxon(scores[:, i], scores[:, j]).pvalue
        raw[f"{names[i]}-{names[j]}"] = float(p)
    order = sorted(raw, key=raw.get)
    m, adj, running = len(order), {}, 0.0
    for r, key in enumerate(order):
        running = max(running, min(1.0, (m - r) * raw[key]))
        adj[key] = running
    return {key: dict(p=raw[key], p_holm=adj[key]) for key in raw}
