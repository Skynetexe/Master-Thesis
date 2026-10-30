"""Statistics for the P13 audit and the model choice. Pure arrays, no
files. stage_p13a.py runs them on the real data, and test_gam.py checks
each one against a direct computation.

1. stratified_flag_auc(): does a "has data" flag separate positives from
   negatives beyond what popularity explains? The AUC of the flag within
   popularity strata, pooled over the strata with the Mann-Whitney pair
   counts as weights, and a percentile interval from a listener cluster
   bootstrap (Field & Welsh 2007: resample the listeners, not the rows,
   because the rows of one listener are not independent).
2. concurvity(): the mgcv-style indices (Wood, mgcv help page
   "concurvity") from the Gram matrix of the within-listener centered
   design. For a term f = X_J b and the other terms X_O:
   - worst: the largest squared canonical correlation between the column
     spaces of X_J and X_O, "the largest value that the square of
     ||g||/||f|| could take for any coefficient vector".
   - estimate: the mean squared canonical correlation, the squared
     F-norm of the projected orthonormal basis over the F-norm of the
     basis.
   - observed (after the fit): ||P_O f||^2 / ||f||^2 for the fitted
     coefficients.
   Within-listener centering removes what a per-listener softmax cannot
   see: a term that differs only between listeners.
3. one_se_choice(): among the configurations within one paired standard
   error of the best mean score, the simplest (the glmnet
   "lambda.1se" rule: "largest value of lambda such that error is within
   1 standard error of the minimum"). The standard error comes from the
   per-listener differences to the best configuration.
"""

import math

import numpy as np

EIG_RTOL = 1e-10


# Stratified flag AUC


def strata_of(values, cuts):
    """Stratum index of each value: 0 below cuts[0], len(cuts) at or
    above cuts[-1]."""
    return np.searchsorted(
        np.asarray(cuts, dtype=np.float64),
        np.asarray(values, dtype=np.float64),
        side="right",
    ).astype(np.int64)


def flag_counts(user, stratum, flag, n_users, n_strata):
    """(n_users, n_strata, 2) counts: rows, and rows with the flag set."""
    key = np.asarray(user, dtype=np.int64) * n_strata + np.asarray(
        stratum, dtype=np.int64
    )
    size = n_users * n_strata
    n = np.bincount(key, minlength=size).astype(np.float64)
    h = np.bincount(key, weights=np.asarray(flag, dtype=np.float64), minlength=size)
    return np.stack([n, h], axis=1).reshape(n_users, n_strata, 2)


def pooled_auc(pos, neg):
    """Pooled stratified AUC from aggregated counts.

    pos, neg: (..., n_strata, 2) arrays of (rows, flagged rows). Within a
    stratum the AUC of a binary flag is 0.5 + 0.5 (p1 - p0), with p1 and p0
    the flagged shares of positives and negatives (ties count half).
    Strata are weighted by their positive-negative pair count n1 * n0, so
    the result is the Mann-Whitney AUC over the pairs that share a
    stratum. NaN when no stratum holds both groups."""
    n1, h1 = pos[..., 0], pos[..., 1]
    n0, h0 = neg[..., 0], neg[..., 1]
    wgt = n1 * n0
    with np.errstate(invalid="ignore", divide="ignore"):
        auc_s = 0.5 + 0.5 * (h1 / n1 - h0 / n0)
        auc_s = np.where(wgt > 0, auc_s, 0.0)
        tot = wgt.sum(axis=-1)
        return np.where(tot > 0, (wgt * auc_s).sum(axis=-1) / tot, np.nan)


def stratum_aucs(pos, neg):
    """Per-stratum AUC (NaN where a group is empty) and pair counts."""
    n1, h1 = pos[..., 0], pos[..., 1]
    n0, h0 = neg[..., 0], neg[..., 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        auc = np.where((n1 > 0) & (n0 > 0), 0.5 + 0.5 * (h1 / n1 - h0 / n0), np.nan)
    return auc, n1 * n0


def bootstrap_weights(n_users, n_boot, seed, chunk=25):
    """Yield (b0, W) with W (k, n_users) float64 resampling counts: each
    replicate draws n_users listeners with replacement. The counts are
    integers, so every product with integer counts below is exact in
    float64 and the result does not depend on the BLAS thread count."""
    rng = np.random.default_rng(seed)
    for b0 in range(0, n_boot, chunk):
        k = min(chunk, n_boot - b0)
        W = np.empty((k, n_users), dtype=np.float64)
        for j in range(k):
            W[j] = np.bincount(rng.integers(0, n_users, n_users), minlength=n_users)
        yield b0, W


def stratified_flag_auc(pos_counts, neg_counts, n_boot, seed, level=0.95):
    """The pooled AUC with a percentile interval from a listener cluster
    bootstrap. pos_counts, neg_counts: (n_users, n_strata, 2) from
    flag_counts(). Returns {"auc", "lo", "hi", "per_stratum",
    "pairs_per_stratum", "replicates"}."""
    n_users, n_strata, _ = pos_counts.shape
    P = pos_counts.reshape(n_users, -1)
    N = neg_counts.reshape(n_users, -1)
    point = float(pooled_auc(pos_counts.sum(axis=0), neg_counts.sum(axis=0)))
    reps = np.empty(n_boot, dtype=np.float64)
    for b0, W in bootstrap_weights(n_users, n_boot, seed):
        pp = (W @ P).reshape(len(W), n_strata, 2)
        nn = (W @ N).reshape(len(W), n_strata, 2)
        reps[b0 : b0 + len(W)] = pooled_auc(pp, nn)
    alpha = (1.0 - level) / 2.0
    ok = reps[np.isfinite(reps)]
    lo, hi = (
        (float(np.quantile(ok, alpha)), float(np.quantile(ok, 1 - alpha)))
        if len(ok)
        else (math.nan, math.nan)
    )
    per, pairs = stratum_aucs(pos_counts.sum(axis=0), neg_counts.sum(axis=0))
    return {
        "auc": point,
        "lo": lo,
        "hi": hi,
        "per_stratum": [None if not np.isfinite(x) else float(x) for x in per],
        "pairs_per_stratum": [float(x) for x in pairs],
        "replicates": int(len(ok)),
    }


def no_signal(result_by_kind, margin):
    """True when every interval lies inside 0.5 +- margin: the flag
    carries no signal beyond popularity against any negative kind."""
    for r in result_by_kind.values():
        lo, hi = r["lo"], r["hi"]
        if not (np.isfinite(lo) and np.isfinite(hi)):
            return False
        if lo < 0.5 - margin or hi > 0.5 + margin:
            return False
    return True


# Concurvity


def _whiten(G):
    """W with W' G W = I on the range of G (eigenvalues above EIG_RTOL
    times the largest), or None when G is zero."""
    G = 0.5 * (G + G.T)
    ev, U = np.linalg.eigh(G)
    top = ev.max() if len(ev) else 0.0
    if top <= 0:
        return None
    keep = ev > EIG_RTOL * top
    return U[:, keep] / np.sqrt(ev[keep])


def canonical_correlations(G, J, O):
    """Canonical correlations between the column spaces of X_J and X_O,
    from G = X'X. Sorted in decreasing order, clipped to [0, 1]."""
    WJ = _whiten(G[np.ix_(J, J)])
    WO = _whiten(G[np.ix_(O, O)])
    if WJ is None or WO is None:
        return np.zeros(0)
    C = WJ.T @ G[np.ix_(J, O)] @ WO
    s = np.linalg.svd(C, compute_uv=False)
    return np.clip(s, 0.0, 1.0)


def concurvity(G, groups):
    """mgcv-style worst and estimate for each group of columns against
    all other groups, and the pairwise worst matrix.
    groups: {name: column indices}. Columns with a zero diagonal (no
    data) are left out."""
    diag = np.diag(G)
    live = {
        k: np.asarray([c for c in v if diag[c] > 0], dtype=np.int64)
        for k, v in groups.items()
    }
    names = list(groups)
    full = {}
    for k in names:
        J = live[k]
        O = (
            np.concatenate([live[o] for o in names if o != k])
            if len(names) > 1
            else np.zeros(0, dtype=np.int64)
        )
        rho = canonical_correlations(G, J, O) if len(J) and len(O) else np.zeros(0)
        rank = _rank(G[np.ix_(J, J)]) if len(J) else 0
        full[k] = {
            "worst": float(rho[0] ** 2) if len(rho) else 0.0,
            "estimate": float(np.sum(rho**2) / rank) if rank else 0.0,
            "rank": int(rank),
        }
    pair = np.zeros((len(names), len(names)))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i == j:
                pair[i, j] = 1.0
            elif j > i and len(live[a]) and len(live[b]):
                rho = canonical_correlations(G, live[a], live[b])
                pair[i, j] = pair[j, i] = float(rho[0] ** 2) if len(rho) else 0.0
    return full, names, pair


def _rank(G):
    G = 0.5 * (G + G.T)
    ev = np.linalg.eigvalsh(G)
    top = ev.max() if len(ev) else 0.0
    return int(np.sum(ev > EIG_RTOL * top)) if top > 0 else 0


def observed_concurvity(G, groups, coefs):
    """||P_O f||^2 / ||f||^2 for f = X_J coefs[name], per group."""
    diag = np.diag(G)
    names = list(groups)
    out = {}
    for k in names:
        J = np.asarray(groups[k], dtype=np.int64)
        O = np.concatenate(
            [np.asarray(groups[o], dtype=np.int64) for o in names if o != k]
        )
        O = O[diag[O] > 0]
        b = np.asarray(coefs[k], dtype=np.float64)
        den = float(b @ G[np.ix_(J, J)] @ b)
        if den <= 0 or len(O) == 0:
            out[k] = 0.0
            continue
        WO = _whiten(G[np.ix_(O, O)])
        v = WO.T @ (G[np.ix_(O, J)] @ b)
        out[k] = float(min(max((v @ v) / den, 0.0), 1.0))
    return out


# Model choice


def per_listener_scores(ranks):
    """ranks: (..., n_listeners, 2) under the P5 rules for val_last and
    val_first. Returns the listener's mean NDCG@100 over the two picks."""
    r = np.asarray(ranks, dtype=np.float64)
    dcg = np.where(r <= 100, 1.0 / np.log2(r + 1.0), 0.0)
    return dcg.mean(axis=-1)


def one_se_choice(scores, order):
    """scores: (C, n) per-listener scores of C configurations. order:
    configuration indices from the simplest to the most complex. Returns
    (chosen index, best index, rows) where rows hold, per configuration,
    the mean, the mean difference to the best and its standard error. The
    best is the highest mean, a tie going to the simpler configuration.
    The chosen one is the first in order whose mean lies within one
    standard error of the paired difference below the best."""
    scores = np.asarray(scores, dtype=np.float64)
    C, n = scores.shape
    means = scores.mean(axis=1)
    best = order[0]
    for c in order:
        if means[c] > means[best]:
            best = c
    rows = []
    chosen = None
    for c in range(C):
        d = scores[best] - scores[c]
        se = float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else math.nan
        rows.append({"mean": float(means[c]), "diff": float(d.mean()), "se": se})
    for c in order:
        if c == best or rows[c]["diff"] <= rows[c]["se"]:
            chosen = c
            break
    return chosen, best, rows


def paired_summary(a, b):
    """Mean of a - b over listeners, its standard error, a 95 percent
    normal interval and the paired t statistic."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    n = len(d)
    m = float(d.mean())
    se = float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else math.nan
    return {
        "mean": m,
        "se": se,
        "lo": m - 1.959963984540054 * se,
        "hi": m + 1.959963984540054 * se,
        "t": m / se if se and se > 0 else math.nan,
        "n": n,
    }
