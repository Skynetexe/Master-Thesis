"""The P13b additions to the transparent model: the BPR loss, the order
of the zero state for monotone terms, and fits with bounds. Pure arrays,
no files. pipeline/gam.py stays exactly as P13a ran it.

1. BPR (Rendle, Freudenthaler, Gantner & Schmidt-Thieme 2009) over the
   same blocks as the softmax. For positive p of listener u and each of
   u's m negatives j, with d = s_p - s_j:
       loss = sum_p w_p (1/m) sum_j softplus(-d),
   the weight w_p of gam.softmax_block (1/n_u per positive, divided by
   the listener count), so every listener counts once. The score is
   linear in the tables, so the loss is convex. BPR takes no offsets:
   its optimum depends on the sampler, which is the point of the arm.
2. The order of the zero state. The design tests an order constraint
   per concept: the zero state below the first positive knot. For a
   non-decreasing term
   a free zero constant gets the bound "at or below the height of the
   first knot", which is 0 in gam.Layout. "Data, but no match" then never
   scores above the weakest match. A tied zero state equals that height
   already.
3. fit_bounded(): projected Newton steps with the exact Hessian (the
   active-set rule of gam.newton_polish: a parameter at its bound whose
   gradient points outward stays fixed for the step), from a feasible
   start. The problem is convex, so the steps reach the optimum. If they
   stop above newton_accept, gam.fit() runs L-BFGS-B (Byrd, Lu, Nocedal &
   Zhu 1995) with its Newton polish.
4. theta_from_tables(): the least-squares parameters of a layout for
   given tables, clipped to the bounds. It carries a fit from one layout
   to another as a warm start.
"""

import hashlib
import json

import numpy as np

from . import gam

# BPR


def _sigmoid(x):
    """1 / (1 + exp(-x)) without overflow."""
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def _softplus(x):
    """log(1 + exp(x)) without overflow."""
    return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))


def bpr_block(s, pos_rows, pos_user, pos_weight, neg_rows):
    """(sum over positives of w_p (1/m) sum_j softplus(s_j - s_p),
    dLoss/ds per row) for one block. neg_rows: (U, m)."""
    m = neg_rows.shape[1]
    sp_ = s[pos_rows]  # (P,)
    sn = s[neg_rows][pos_user]  # (P, m)
    d = sp_[:, None] - sn  # s_p - s_j
    c = (pos_weight / m)[:, None]
    loss = float(np.sum(c * _softplus(-d)))
    q = c * _sigmoid(-d)  # dloss/ds_j
    g = np.zeros_like(s, dtype=np.float64)
    g[pos_rows] = -q.sum(axis=1)
    np.add.at(g, neg_rows[pos_user].ravel(), q.ravel())
    return loss, g


def bpr_block_loss_grad(T, D, blk, L):
    """(loss, gradient over the L flat table entries) of one block under
    BPR. The same accumulation as gam.block_loss_grad."""
    s = gam.block_scores(T, D, blk)
    loss, g = bpr_block(
        s, blk["pos_rows"], blk["pos_user"], blk["pos_weight"], blk["neg_rows"]
    )
    grad = np.zeros(L + 1, dtype=np.float64)
    w = blk["w"]
    mult = blk.get("mult") or {}
    for k in range(w.shape[0]):
        gk = g * mult[k] if k in mult else g
        ik = gam.block_index(blk, k)
        grad += np.bincount(ik, weights=gk * (1.0 - w[k]), minlength=L + 1)
        grad += np.bincount(ik + 1, weights=gk * w[k], minlength=L + 1)
    return loss, grad[:L]


def bpr_block_hessian(T, D, blk, L):
    """The (L, L) Hessian of one block's BPR loss over the flat tables.
    With c_pj = w_p (1/m) sigma(d) sigma(-d) and X the block design,
        H = sum over pairs c_pj (x_p - x_j)(x_p - x_j)'
          = X' diag(a) X - X_P' (C X) - (C X)' X_P,
    where a_r sums c over the pairs that hold row r, X_P holds the rows of
    the positives, and row p of C holds c_pj at u's negatives."""
    import scipy.sparse as sp

    s = gam.block_scores(T, D, blk)
    pos_rows, pos_user = blk["pos_rows"], blk["pos_user"]
    pos_weight, neg_rows = blk["pos_weight"], blk["neg_rows"]
    m = neg_rows.shape[1]
    d = s[pos_rows][:, None] - s[neg_rows][pos_user]
    sg = _sigmoid(d)
    c = (pos_weight / m)[:, None] * sg * (1.0 - sg)  # (P, m)
    n = len(s)
    a = np.zeros(n, dtype=np.float64)
    a[pos_rows] += c.sum(axis=1)
    np.add.at(a, neg_rows[pos_user].ravel(), c.ravel())
    H = gam.block_xax(blk, a, L)
    X = gam.block_design(blk, L)
    n_p = len(pos_rows)
    C = sp.csr_matrix(
        (c.ravel(), (np.repeat(np.arange(n_p), m), neg_rows[pos_user].ravel())),
        shape=(n_p, n),
    )
    CX = C @ X  # (P, L)
    XP = X[pos_rows]  # (P, L)
    cross = (XP.T @ CX).toarray()
    H -= cross + cross.T
    return 0.5 * (H + H.T)


# Monotone order


def order_zero_states(layout):
    """Bounds of the layout with the zero state of every monotone term
    ordered against the first knot: at or below its height (0) for a
    non-decreasing term, at or above for a non-increasing one. Only a free
    zero constant needs the bound. The first knot of such a term must be
    0, the knot that the zero state sits next to. Sets layout.bounds and
    returns the layout."""
    bounds = list(layout.bounds)
    index = {lab: k for k, lab in enumerate(layout.labels)}
    for t in layout.terms:
        if not (t.monotone and t.zero_state):
            continue
        if t.knots[0] != 0.0:
            raise ValueError(
                f"term {t.name}: a monotone term with a zero "
                f"state needs its first knot at 0"
            )
        k = index.get(f"{t.name}:zero")
        if k is None:
            continue  # tied to the first knot height
        bounds[k] = (None, 0.0) if t.monotone > 0 else (0.0, None)
    layout.bounds = tuple(bounds)
    layout.zero_order = True
    return layout


def layout_fingerprint(layout):
    """gam.Layout.fingerprint() plus the bounds, which it leaves out."""
    h = hashlib.sha256()
    h.update(layout.fingerprint().encode("utf-8"))
    h.update(
        json.dumps(
            [[None if b is None else float(b) for b in pair] for pair in layout.bounds]
        ).encode("utf-8")
    )
    return h.hexdigest()


def monotone_violation(T, layout):
    """The largest violation of the monotone order in the tables: a
    decrease between neighbouring knot heights of a non-decreasing term,
    or a zero constant above the first knot height (and the mirror image
    for a non-increasing term). 0 when every order holds."""
    worst = 0.0
    for t, off in zip(layout.terms, layout.offsets):
        if not t.monotone:
            continue
        h = np.asarray(T[off : off + t.K], dtype=np.float64) * t.monotone
        if t.K > 1:
            worst = max(worst, float(np.max(np.maximum(h[:-1] - h[1:], 0.0))))
        if t.zero_state:
            z = float(T[off + t.K + gam.ZERO]) * t.monotone
            worst = max(worst, z - h[0])
    return worst


# Fits


def theta_from_tables(layout, T):
    """Parameters theta with M theta closest to T in least squares,
    clipped to the bounds: a warm start from another layout's tables."""
    theta, *_ = np.linalg.lstsq(layout.M, np.asarray(T, dtype=np.float64), rcond=None)
    return gam._clip(theta, layout.bounds)


def fit_bounded(objective, theta0, settings):
    """Minimize a convex objective with bounds: projected Newton steps
    from the clipped start, then L-BFGS-B through gam.fit() only if the
    steps stop above newton_accept. Returns gam.fit()'s record."""
    bounds = objective.layout.bounds
    x = gam._clip(theta0, bounds)
    steps = int(settings.get("bounded_newton_steps", 200))
    x, rec = gam.newton_polish(objective, x, steps, float(settings["newton_target"]))
    loss, smooth, ridge, grad = objective.parts(x)
    pg = gam.projected_gradient(x, grad, bounds)
    pg_max = float(np.max(np.abs(pg))) if len(pg) else 0.0
    if pg_max <= float(settings.get("newton_accept", 1e-9)):
        return {
            "theta": x,
            "objective": loss
            + objective.lam_smooth * smooth
            + objective.lam_ridge * ridge,
            "loss": loss,
            "smooth": smooth,
            "ridge": ridge,
            "proj_grad_max": pg_max,
            "nit": 0,
            "nfev": 0,
            "restarts": 0,
            "newton_steps": rec["steps"],
            "proj_grad_before_newton": rec["before"],
            "method": "projected newton",
            "message": f"PROJECTED NEWTON: {rec['steps']} steps",
            "success": True,
        }
    res = gam.fit(objective, x, dict(settings, method="lbfgsb"))
    res["newton_steps"] += rec["steps"]
    return res


def active_share(theta, layout, tol=0.0):
    """{term: share of its increments held at the bound} for monotone
    terms: the flat segments of the shape."""
    out = {}
    for t in layout.terms:
        if not t.monotone:
            continue
        ks = [
            k for k, lab in enumerate(layout.labels) if lab.startswith(f"{t.name}:inc")
        ]
        if ks:
            out[t.name] = float(np.mean(np.abs(theta[ks]) <= tol))
    return out
