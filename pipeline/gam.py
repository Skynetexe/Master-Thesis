"""The transparent fusion model of P13: a convex additive ranker.

Chapter 4 of the thesis gives the design and the evidence for each
choice. This module holds the model itself: pure arrays in, arrays out,
no files and no processes. stage_p13a.py runs it on the real data.

The model. The score of a (listener u, track i) pair is a sum of terms,
and each term is a lookup table:

- a piecewise-linear (PWL) shape on fixed knots, constant beyond the
  outer knots (clamped ends, as the TensorFlow Lattice calibrators do),
- one constant for "exactly zero": the data exists and shows no match,
- one constant for "no data": the missing state. "No data" is never read
  as a mismatch.

Terms (stage_p13a.base_terms() builds them):

- one per concept of pipeline/concepts.py that enters the score,
- popularity, f_0(pop),
- the popularity tilt m_u * f_1(pop), where m_u is the listener's
  standardized mainstreaminess: a varying-coefficient term (Hastie &
  Tibshirani 1993),
- a deviation curve of audio fit for the fuzzy_dur tier, against the map
  tiers as the reference (the mgcv ordered-factor convention), with mean
  zero over the reference rows, so it adds no tier main effect.

Convexity and uniqueness. The score is linear in the table entries. The
loss is a log-sum-exp of an affine map of the parameters, so it is convex
(Boyd & Vandenberghe 2004, section 3.2.2). The second-difference penalty
(Eilers & Marx 1996) is a positive semidefinite quadratic form, and a
small ridge term makes the objective strictly convex. Each penalty setting
therefore has one optimum, and fit() reaches it from any start: damped
Newton steps with the exact Hessian (block_hessian), or L-BFGS-B (Byrd,
Lu, Nocedal & Zhu 1995) followed by Newton steps when bounds are present.
There is no seed. Two runs give the same tables to the solver tolerance.

Identifiability. A per-listener softmax ignores a constant added to all
scores of one listener: "the only parameters that can be estimated (that
is, are identified) are those that capture differences across
alternatives" (Train 2009). Every row holds exactly one state of each
concept term, so a constant added to a whole table changes nothing, and
the tilt term shifts all scores of one listener together. One knot height
per term is therefore fixed at 0 (Layout). After the fit, center()
subtracts each term's mean over a declared background, which gives the
constants their meaning and changes no ranking.

Evaluation, the one formula that the fit, the full-catalog scorer and the
explanation layer share (pwl_value): with T the table and
D[k] = T[k + 1] - T[k] inside the knots (0 elsewhere),

    value = T[slot] + w * D[slot]

where slot is the left knot (or a state slot) and w the position between
the two knots. The explanation cites exactly this number.

Table layout per term with K knots (size K + 4):
  0 .. K-1   knot heights
  K          exactly zero
  K + 1      no data
  K + 2      not applicable: the term does not apply to the row (always 0)
  K + 3      padding, so that slot + 1 is always a valid index (always 0)
"""

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np
import scipy.sparse as sp
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

ZERO, MISS, NA, PAD = 0, 1, 2, 3  # slot = K + one of these
MULTIPLIERS = ("mainstreaminess",)


# Terms


@dataclass(frozen=True)
class Term:
    """One additive term.

    source: the concept column that the term reads.
    knots: sorted, distinct floats.
    zero_state: a separate constant for a value of exactly 0 with data.
    missing_state: a separate constant for "no data".
    multiplier: None, or "mainstreaminess" (the row value multiplies the
        term, which then shifts every score of one listener together
        when the table shifts).
    tiers: None (every row), or the tier codes where the term applies.
        Elsewhere, and where the source has no data, the term is "not
        applicable" and adds 0.
    centered: the knot heights are taken relative to their mean over the
        reference rows where the term applies (Layout).
    monotone: +1 or -1 for a non-decreasing or non-increasing shape,
        through bounded increments. 0: free.
    """

    name: str
    source: str
    knots: tuple
    zero_state: bool = False
    missing_state: bool = False
    multiplier: str = None
    tiers: tuple = None
    centered: bool = False
    monotone: int = 0

    def __post_init__(self):
        k = np.asarray(self.knots, dtype=np.float64)
        if (
            k.ndim != 1
            or len(k) < 2
            or not np.all(np.isfinite(k))
            or not np.all(np.diff(k) > 0)
        ):
            raise ValueError(
                f"term {self.name}: the knots must be at least "
                f"two sorted, distinct, finite values"
            )
        if self.multiplier not in (None,) + MULTIPLIERS:
            raise ValueError(
                f"term {self.name}: unknown multiplier " f"{self.multiplier}"
            )
        if self.monotone not in (-1, 0, 1):
            raise ValueError(f"term {self.name}: monotone must be -1, 0 or 1")
        if self.centered and (self.zero_state or self.missing_state):
            raise ValueError(
                f"term {self.name}: a centered term has no " f"state constants"
            )

    @property
    def K(self):
        return len(self.knots)

    @property
    def size(self):
        return self.K + 4

    @property
    def knot_array(self):
        return np.asarray(self.knots, dtype=np.float64)

    def to_json(self):
        d = asdict(self)
        d["knots"] = [float(x) for x in self.knots]
        d["tiers"] = None if self.tiers is None else [int(t) for t in self.tiers]
        return d

    @classmethod
    def from_json(cls, d):
        d = dict(d)
        d["knots"] = tuple(float(x) for x in d["knots"])
        if d.get("tiers") is not None:
            d["tiers"] = tuple(int(t) for t in d["tiers"])
        return cls(**d)


def knot_rule(
    pos_values, neg_values, pos_levels, neg_levels, anchors=(0.0,), min_gap=1e-6
):
    """Knots at quantiles of the positives, merged with a few quantiles
    of the negatives and the anchors (a knot at 0).

    With 32 negatives per positive, pooled quantiles would follow the
    negatives, so the positives set most knots. The negatives' quantiles
    add support where only negatives lie (the low tail of audio fit).
    Knots closer than min_gap times the knot range to the knot before
    them are merged: a point mass gives one knot, not several.
    Quantiles: numpy's default (linear) method."""
    pos = np.asarray(pos_values, dtype=np.float64)
    neg = np.asarray(neg_values, dtype=np.float64)
    parts = [np.asarray(anchors, dtype=np.float64)]
    if len(pos):
        parts.append(np.quantile(pos, pos_levels))
    if len(neg):
        parts.append(np.quantile(neg, neg_levels))
    k = np.unique(np.concatenate(parts))
    k = k[np.isfinite(k)]
    if len(k) < 2:
        raise ValueError("fewer than two distinct knots")
    gap = min_gap * (k[-1] - k[0])
    keep = [k[0]]
    for x in k[1:]:
        if x - keep[-1] > gap:
            keep.append(x)
    if len(keep) < 2:
        raise ValueError("the knots merge into one point")
    return tuple(float(x) for x in keep)


# Mainstreaminess


def mainstreaminess_raw(items, weights, popularity):
    """The listener's mainstreaminess before standardization: the
    listen-weighted mean of log popularity over the training history,
    sum_h w_h pop_h / sum_h w_h, with the concept layer's weights
    w_h = log1p(listens) and pop_h = log1p(train listens of track h).
    Low values mark beyond-mainstream listeners, who get the worst
    recommendations on Last.fm data (Kowald, Schedl & Lex 2020)."""
    w = np.asarray(weights, dtype=np.float64)
    pop = np.asarray(popularity, dtype=np.float64)[np.asarray(items, dtype=np.int64)]
    return float(np.sum(w * pop) / np.sum(w))


def mainstreaminess(raw, consts):
    """Standardized over the training listeners and clamped at their
    percentiles: clip((raw - mean) / sd, lo, hi). consts holds "mean",
    "sd", "lo" and "hi", frozen with the model."""
    z = (np.asarray(raw, dtype=np.float64) - consts["mean"]) / consts["sd"]
    return np.clip(z, consts["lo"], consts["hi"])


# Encoding


def encode_continuous(knots, x):
    """(slot, w) of values x on the knots: slot is the left knot of the
    segment, w in [0, 1) the position in it. Values beyond the outer
    knots are clamped. The last knot gets slot K - 1 and w = 0."""
    knots = np.asarray(knots, dtype=np.float64)
    K = len(knots)
    x = np.clip(np.asarray(x, dtype=np.float64), knots[0], knots[-1])
    b = np.searchsorted(knots, x, side="right") - 1
    b = np.clip(b, 0, K - 1)
    inner = b < K - 1
    w = np.zeros(len(x), dtype=np.float64)
    bi = b[inner]
    w[inner] = (x[inner] - knots[bi]) / (knots[bi + 1] - knots[bi])
    return b.astype(np.int16), w


def encode(term, value, has, tier=None):
    """(slot int16, w float64) of every row for one term.

    value, has: the term's source column and its data flag. tier: the
    tier code of each row, needed when term.tiers is set. Rows are:
    - not applicable (slot K + NA, adds 0) where the term's tiers exclude
      the row, or where the source has no data and the term has no
      missing state.
    - missing (slot K + MISS) where the source has no data.
    - exactly zero (slot K + ZERO) where the value is 0 and the term has a
      zero state.
    - on the shape otherwise."""
    value = np.asarray(value, dtype=np.float64)
    has = np.asarray(has, dtype=bool)
    n, K = len(value), term.K
    slot = np.full(n, K + NA, dtype=np.int16)
    w = np.zeros(n, dtype=np.float64)
    if term.tiers is None:
        applies = np.ones(n, dtype=bool)
    else:
        if tier is None:
            raise ValueError(f"term {term.name} needs the tier of each row")
        applies = np.isin(np.asarray(tier), np.asarray(term.tiers))
    if term.missing_state:
        slot[applies & ~has] = K + MISS
    cont = applies & has
    if np.any(cont & ~np.isfinite(value)):
        raise ValueError(f"term {term.name}: a row with data has no finite " f"value")
    if term.zero_state:
        zero = cont & (value == 0.0)
        slot[zero] = K + ZERO
        cont &= ~zero
    idx = np.flatnonzero(cont)
    s, ww = encode_continuous(term.knot_array, value[idx])
    slot[idx] = s
    w[idx] = ww
    return slot, w


def diff_table(T, terms, offsets):
    """D of the flat tables: D[k] = T[k + 1] - T[k] between the knots of
    each term, 0 on the last knot and on every state slot."""
    D = np.zeros_like(T)
    for term, off in zip(terms, offsets):
        K = term.K
        D[..., off : off + K - 1] = (
            T[..., off + 1 : off + K] - T[..., off : off + K - 1]
        )
    return D


def pwl_value(T, D, idx, w):
    """The value of a term: T[idx] + w * D[idx], where idx is the flat
    table index (term offset + slot). The one formula for the fit, the
    catalog scorer and the explanations."""
    return T[idx] + w * D[idx]


def basis_mean(term, slot, w, mask=None):
    """The mean basis vector of a term over the rows in mask: entry k is
    the mean weight that the rows put on table slot k. For any table T,
    basis_mean @ T is the mean of the term's value over those rows."""
    slot = np.asarray(slot)
    w = np.asarray(w, dtype=np.float64)
    if mask is not None:
        slot, w = slot[mask], w[mask]
    n = len(slot)
    if n == 0:
        return None
    s = slot.astype(np.int64)
    v = np.bincount(s, weights=1.0 - w, minlength=term.size) + np.bincount(
        s + 1, weights=w, minlength=term.size
    )
    return v[: term.size] / n


# Parameterization


class Layout:
    """The map from the free parameter vector theta to the flat tables of
    all terms, T = M @ theta, and its bounds and penalty.

    Per term:
    - knot heights: height 0 is fixed at 0 (only differences between the
      candidates of one listener are identified). For a monotone term the
      free parameters are the increments h[k] - h[k-1], bounded at 0.
    - centered terms: the heights become h - (a . h), where a is the mean
      basis over the reference rows where the term applies (dev_means).
      Their mean over those rows is then 0, so the term adds no level
      shift, only a change of shape.
    - zero state: a free constant when the training rows hold the state
      (n_zero > 0). Otherwise it is tied to the height at the knot at 0
      (or the first knot): continuity at zero.
    - missing state: a free constant when the training rows hold the
      state and the term is not in neutral. Otherwise it is tied to the
      neutral level, the mean value of the term over the reference rows
      with data (data_means): "no data" then contributes what an average
      track with data contributes, so it adds nothing after centering.
    - "not applicable" and padding: always 0.

    All ties are linear, so the fit stays convex."""

    def __init__(
        self,
        terms,
        n_zero=None,
        n_miss=None,
        neutral=(),
        data_means=None,
        dev_means=None,
    ):
        self.terms = tuple(terms)
        names = [t.name for t in self.terms]
        if len(set(names)) != len(names):
            raise ValueError("term names must be unique")
        self.neutral = tuple(sorted(set(neutral)))
        unknown = set(self.neutral) - set(names)
        if unknown:
            raise ValueError(f"neutral names unknown terms {sorted(unknown)}")
        n_zero = n_zero or {}
        n_miss = n_miss or {}
        data_means = data_means or {}
        dev_means = dev_means or {}
        self.offsets = []
        off = 0
        for t in self.terms:
            self.offsets.append(off)
            off += t.size
        self.L = off
        cols, bounds, labels = [], [], []
        rows_of = {}
        # Pass 1: one column per free parameter.
        entries = []  # (row, col, value) for M
        self.zero_free, self.miss_free = {}, {}
        for t, off in zip(self.terms, self.offsets):
            K = t.K
            if t.monotone:
                for k in range(1, K):
                    c = len(labels)
                    labels.append(f"{t.name}:inc{k}")
                    bounds.append((0.0, None) if t.monotone > 0 else (None, 0.0))
                    for kk in range(k, K):
                        entries.append((off + kk, c, 1.0))
            else:
                for k in range(1, K):
                    c = len(labels)
                    labels.append(f"{t.name}:h{k}")
                    bounds.append((None, None))
                    entries.append((off + k, c, 1.0))
            zf = bool(t.zero_state and n_zero.get(t.name, 0) > 0)
            self.zero_free[t.name] = zf
            if zf:
                c = len(labels)
                labels.append(f"{t.name}:zero")
                bounds.append((None, None))
                entries.append((off + K + ZERO, c, 1.0))
            mf = bool(
                t.missing_state
                and n_miss.get(t.name, 0) > 0
                and t.name not in self.neutral
            )
            self.miss_free[t.name] = mf
            if mf:
                c = len(labels)
                labels.append(f"{t.name}:miss")
                bounds.append((None, None))
                entries.append((off + K + MISS, c, 1.0))
            rows_of[t.name] = off
        self.P = len(labels)
        self.labels = tuple(labels)
        self.bounds = tuple(bounds)
        M = np.zeros((self.L, self.P), dtype=np.float64)
        for r, c, v in entries:
            M[r, c] += v
        # Pass 2: the ties, in this order: centering, zero, missing.
        self.dev_means, self.data_means = {}, {}
        for t, off in zip(self.terms, self.offsets):
            K = t.K
            if t.centered:
                a = dev_means.get(t.name)
                if a is None:
                    raise ValueError(
                        f"term {t.name} is centered but has no " f"reference mean"
                    )
                a = np.asarray(a, dtype=np.float64)[:K]
                if not np.isclose(a.sum(), 1.0, atol=1e-9):
                    raise ValueError(
                        f"term {t.name}: the reference mean " f"does not sum to 1"
                    )
                self.dev_means[t.name] = a
                H = M[off : off + K].copy()
                M[off : off + K] = H - np.outer(np.ones(K), a @ H)
            if t.zero_state and not self.zero_free[t.name]:
                k0 = zero_knot(t)
                M[off + K + ZERO] = M[off + k0]
            if t.missing_state and not self.miss_free[t.name]:
                a = data_means.get(t.name)
                if a is None:
                    raise ValueError(
                        f"term {t.name}: a tied missing state "
                        f"needs the mean basis over the "
                        f"reference rows with data"
                    )
                a = np.asarray(a, dtype=np.float64)[: K + 1]
                if not np.isclose(a.sum(), 1.0, atol=1e-9):
                    raise ValueError(
                        f"term {t.name}: the mean basis over "
                        f"the rows with data does not sum to 1"
                    )
                self.data_means[t.name] = a
                M[off + K + MISS] = a @ M[off : off + K + 1]
        self.M = M
        # The second-difference penalty on the knot heights of each term,
        # in parameter space: theta' (M' P1 M) theta.
        P1 = np.zeros((self.L, self.L), dtype=np.float64)
        for t, off in zip(self.terms, self.offsets):
            K = t.K
            if K < 3:
                continue
            D2 = np.zeros((K - 2, K))
            for i in range(K - 2):
                D2[i, i : i + 3] = (1.0, -2.0, 1.0)
            P1[off : off + K, off : off + K] = D2.T @ D2
        self.Q = M.T @ P1 @ M
        self.Q = 0.5 * (self.Q + self.Q.T)

    def tables(self, theta):
        return self.M @ np.asarray(theta, dtype=np.float64)

    def term_slice(self, name):
        for t, off in zip(self.terms, self.offsets):
            if t.name == name:
                return slice(off, off + t.size)
        raise KeyError(name)

    def fingerprint(self):
        """SHA-256 of the terms, the ties and M."""
        h = hashlib.sha256()
        h.update(
            json.dumps([t.to_json() for t in self.terms], sort_keys=True).encode(
                "utf-8"
            )
        )
        h.update(json.dumps(self.neutral).encode("utf-8"))
        h.update(np.ascontiguousarray(self.M).tobytes())
        return h.hexdigest()


def zero_knot(term):
    """Index of the knot at 0, or of the knot nearest 0."""
    k = term.knot_array
    return int(np.argmin(np.abs(k)))


# Loss


def softmax_block(s, offset, pos_rows, pos_user, pos_weight, neg_rows):
    """The corrected sampled softmax of one block of listeners.

    s, offset: score and logit offset of every row of the block.
    pos_rows: row of each positive. pos_user: its listener (0..U-1 in the
    block). pos_weight: its weight. neg_rows: (U, m) rows of each
    listener's negatives.

    For positive p of listener u, with z = s + offset:
        loss_p = log(exp(z_p) + sum_j exp(z_j)) - z_p,   j over u's
    negatives. Form A of the correction (Blanc & Rendle 2018) keeps the
    positive's offset at 0 and gives each negative -log(m q(j)). The
    negatives of one listener serve all its positives. The log-sum-exp is
    shifted by its maximum (Blanchard, Higham & Higham 2021).

    Returns (sum of pos_weight * loss_p, dLoss/ds per row)."""
    z = s + offset
    zn = z[neg_rows]
    mx = zn.max(axis=1)
    lneg = mx + np.log(np.exp(zn - mx[:, None]).sum(axis=1))
    zp = z[pos_rows]
    ln = lneg[pos_user]
    lse = np.logaddexp(zp, ln)
    loss = float(np.sum(pos_weight * (lse - zp)))
    g = np.zeros_like(z)
    g[pos_rows] = pos_weight * (np.exp(zp - lse) - 1.0)
    a = np.bincount(
        pos_user, weights=pos_weight * np.exp(ln - lse), minlength=len(neg_rows)
    )
    g[neg_rows] = a[:, None] * np.exp(zn - lneg[:, None])
    return loss, g


def block_index(blk, k):
    """Flat table index of term k for every row of a block: the term's
    offset plus the stored slot (int16, to save memory)."""
    return int(blk["off"][k]) + blk["slot"][k].astype(np.int64)


def block_scores(T, D, blk):
    """Score of every row of a block: the sum over terms, in term order,
    of multiplier * pwl_value. blk["slot"]: (terms, rows) int16 slots,
    blk["off"]: the term offsets, blk["w"]: (terms, rows), blk["mult"]:
    {term position: row multipliers}."""
    w = blk["w"]
    mult = blk.get("mult") or {}
    s = np.zeros(w.shape[1], dtype=np.float64)
    for k in range(w.shape[0]):
        v = pwl_value(T, D, block_index(blk, k), w[k])
        if k in mult:
            v = mult[k] * v
        s += v
    return s


def block_loss_grad(T, D, blk, L):
    """(loss, gradient over the L flat table entries) of one block."""
    s = block_scores(T, D, blk)
    loss, g = softmax_block(
        s,
        blk["offset"],
        blk["pos_rows"],
        blk["pos_user"],
        blk["pos_weight"],
        blk["neg_rows"],
    )
    grad = np.zeros(L + 1, dtype=np.float64)
    w = blk["w"]
    mult = blk.get("mult") or {}
    for k in range(w.shape[0]):
        gk = g * mult[k] if k in mult else g
        ik = block_index(blk, k)
        grad += np.bincount(ik, weights=gk * (1.0 - w[k]), minlength=L + 1)
        grad += np.bincount(ik + 1, weights=gk * w[k], minlength=L + 1)
    return loss, grad[:L]


def _entries(blk):
    """Per term, the two design entries of every row: (column, value) at
    the row's slot and at the slot after it, times the term's multiplier."""
    w = blk["w"]
    mult = blk.get("mult") or {}
    out = []
    for k in range(w.shape[0]):
        idx = block_index(blk, k)
        m = mult.get(k)
        lo = (1.0 - w[k]) if m is None else m * (1.0 - w[k])
        hi = w[k] if m is None else m * w[k]
        out.append(((idx, lo), (idx + 1, hi)))
    return out


def block_design(blk, L):
    """The design of one block as a sparse (rows, L) matrix X: row r
    holds (1 - w) at the row's slot and w at the slot after it, times the
    term's multiplier, for every term. The score is s = X T, so X is the
    Jacobian of the scores with respect to the flat tables. Every row has
    the same number of entries, in column order, so the matrix is built
    without a sort."""
    ents = _entries(blk)
    n = blk["w"].shape[1]
    width = 2 * len(ents)
    cols = np.empty((n, width), dtype=np.int64)
    vals = np.empty((n, width), dtype=np.float64)
    for k, ((i0, v0), (i1, v1)) in enumerate(ents):
        cols[:, 2 * k], vals[:, 2 * k] = i0, v0
        cols[:, 2 * k + 1], vals[:, 2 * k + 1] = i1, v1
    indptr = np.arange(0, n * width + 1, width, dtype=np.int64)
    return sp.csr_matrix((vals.ravel(), cols.ravel(), indptr), shape=(n, L))


def block_xax(blk, a, L):
    """X' diag(a) X of one block, dense (L, L), from the term structure:
    each row has two entries per term, so every pair of terms adds four
    weighted counts (np.bincount). The terms own disjoint column ranges,
    so the blocks above the diagonal are summed once and mirrored."""
    ents = _entries(blk)
    size = (L + 1) * (L + 1)
    upper = np.zeros(size, dtype=np.float64)
    diag = np.zeros(size, dtype=np.float64)
    for k in range(len(ents)):
        for l in range(k, len(ents)):
            acc = diag if k == l else upper
            for ik, vk in ents[k]:
                avk = a * vk
                for il, vl in ents[l]:
                    acc += np.bincount(
                        ik * (L + 1) + il, weights=avk * vl, minlength=size
                    )
    up = upper.reshape(L + 1, L + 1)
    H = up + up.T + diag.reshape(L + 1, L + 1)
    return H[:L, :L]


def block_hessian(T, D, blk, L):
    """The (L, L) Hessian of one block's loss with respect to the flat
    tables. For a group g (a positive and its listener's negatives) with
    softmax probabilities pi_g over its rows, the Hessian with respect to
    the scores is diag(pi_g) - pi_g pi_g'. With X the block design:
        H = X' diag(a) X - V' diag(omega) V,
    where a_r sums omega_g pi_g,r over the groups that hold row r, and
    row g of V is pi_g' X_g. The second product is dense: each row of V
    touches most table entries."""
    s = block_scores(T, D, blk)
    z = s + blk["offset"]
    pos_rows, pos_user = blk["pos_rows"], blk["pos_user"]
    pos_weight, neg_rows = blk["pos_weight"], blk["neg_rows"]
    zn = z[neg_rows]
    mx = zn.max(axis=1)
    lneg = mx + np.log(np.exp(zn - mx[:, None]).sum(axis=1))
    zp = z[pos_rows]
    lse = np.logaddexp(zp, lneg[pos_user])
    pi_pos = np.exp(zp - lse)  # (P,)
    pi_neg = np.exp(zn[pos_user] - lse[:, None])  # (P, m)
    a = np.zeros(len(z), dtype=np.float64)
    a[pos_rows] = pos_weight * pi_pos
    a_u = np.bincount(
        pos_user,
        weights=pos_weight * np.exp(lneg[pos_user] - lse),
        minlength=len(neg_rows),
    )
    a[neg_rows] = a_u[:, None] * np.exp(zn - lneg[:, None])
    H = block_xax(blk, a, L)
    n_g, m = pi_neg.shape
    g_idx = np.repeat(np.arange(n_g), m + 1)
    r_idx = np.concatenate([pos_rows[:, None], neg_rows[pos_user]], axis=1).ravel()
    vals = np.concatenate([pi_pos[:, None], pi_neg], axis=1).ravel()
    B = sp.csr_matrix((vals, (g_idx, r_idx)), shape=(n_g, len(z)))
    V = (B @ block_design(blk, L)).toarray()
    H -= V.T @ (pos_weight[:, None] * V)
    return 0.5 * (H + H.T)


class Objective:
    """F(theta) = loss(M theta) + lam_smooth theta'Q theta
    + lam_ridge theta'theta, with its gradient and, when hess_fn is
    given, its Hessian.

    loss_grad(T, D) returns the loss and its gradient over the flat
    tables, and hess_fn(T, D) the loss Hessian over the flat tables, each
    summed over the blocks in a fixed order. The caller supplies them (a
    process pool on the real data, a loop in the tests), so the result
    cannot depend on the number of workers."""

    def __init__(self, layout, loss_grad, lam_smooth, lam_ridge, hess_fn=None):
        self.layout = layout
        self.loss_grad = loss_grad
        self.hess_fn = hess_fn
        self.lam_smooth = float(lam_smooth)
        self.lam_ridge = float(lam_ridge)
        self.n_eval = 0

    def hessian(self, theta):
        """M' H_tables M + 2 lam_smooth Q + 2 lam_ridge I. The loss is
        linear in the tables, so M' H M is its exact Hessian."""
        theta = np.asarray(theta, dtype=np.float64)
        T = self.layout.tables(theta)
        D = diff_table(T, self.layout.terms, self.layout.offsets)
        M = self.layout.M
        H = M.T @ self.hess_fn(T, D) @ M
        H += 2.0 * self.lam_smooth * self.layout.Q
        H[np.diag_indices_from(H)] += 2.0 * self.lam_ridge
        return 0.5 * (H + H.T)

    def parts(self, theta):
        theta = np.asarray(theta, dtype=np.float64)
        T = self.layout.tables(theta)
        D = diff_table(T, self.layout.terms, self.layout.offsets)
        loss, g_t = self.loss_grad(T, D)
        smooth = float(theta @ self.layout.Q @ theta)
        ridge = float(theta @ theta)
        grad = (
            self.layout.M.T @ g_t
            + 2.0 * self.lam_smooth * (self.layout.Q @ theta)
            + 2.0 * self.lam_ridge * theta
        )
        return loss, smooth, ridge, grad

    def __call__(self, theta):
        self.n_eval += 1
        loss, smooth, ridge, grad = self.parts(theta)
        return (loss + self.lam_smooth * smooth + self.lam_ridge * ridge, grad)


def projected_gradient(theta, grad, bounds):
    """The gradient with the components that push against an active bound
    set to 0: the first-order optimality measure of L-BFGS-B."""
    pg = np.array(grad, dtype=np.float64)
    for k, (lo, hi) in enumerate(bounds):
        if lo is not None and theta[k] <= lo and pg[k] > 0:
            pg[k] = 0.0
        if hi is not None and theta[k] >= hi and pg[k] < 0:
            pg[k] = 0.0
    return pg


def fit(objective, theta0, settings):
    """Minimize the objective from theta0.

    With settings["method"] == "newton", an exact Hessian and no bounds,
    damped Newton steps run from theta0 (newton_polish). The problem has a
    few hundred parameters, so the Hessian is cheap next to the thousands
    of L-BFGS-B iterations that a penalized, badly scaled problem needs.
    When Newton does not reach newton_accept, or with bounds, L-BFGS-B
    runs, followed by the Newton polish on the free parameters.

    L-BFGS-B settings: {"maxcor", "ftol", "gtol", "maxiter", "maxfun"}, and
    optionally "pg_target" and "restarts": while the largest projected
    gradient component stays above pg_target (default gtol), L-BFGS-B
    starts again from its last point with a fresh curvature memory, at
    most "restarts" times (default 3). A line search can stop early on
    rounding noise, and a restart usually carries on. Returns a dict with
    the solution and the stop record that the thesis reports: the last
    message, the iteration and evaluation counts over all starts, the
    number of restarts, and the largest projected gradient component at
    the stop."""
    bounds = objective.layout.bounds
    x = np.asarray(theta0, dtype=np.float64).copy()
    for k, (lo, hi) in enumerate(bounds):
        if lo is not None:
            x[k] = max(x[k], lo)
        if hi is not None:
            x[k] = min(x[k], hi)
    target = float(settings.get("pg_target", settings["gtol"]))
    restarts = int(settings.get("restarts", 3))
    nit = nfev = 0
    if (
        settings.get("method") == "newton"
        and objective.hess_fn is not None
        and not any(lo is not None or hi is not None for lo, hi in bounds)
    ):
        x, newton = newton_polish(
            objective,
            x,
            int(settings["newton_steps"]),
            float(settings["newton_target"]),
        )
        loss, smooth, ridge, grad = objective.parts(x)
        pg_max = float(np.max(np.abs(grad))) if len(grad) else 0.0
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
                "newton_steps": newton["steps"],
                "proj_grad_before_newton": newton["before"],
                "method": "newton",
                "message": f"NEWTON: {newton['steps']} steps",
                "success": True,
            }
    for attempt in range(restarts + 1):
        res = minimize(
            objective,
            x,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxcor": int(settings["maxcor"]),
                "ftol": float(settings["ftol"]),
                "gtol": float(settings["gtol"]),
                "maxiter": int(settings["maxiter"]),
                "maxfun": int(settings["maxfun"]),
            },
        )
        x = np.asarray(res.x, dtype=np.float64)
        nit += int(res.nit)
        nfev += int(res.nfev)
        loss, smooth, ridge, grad = objective.parts(x)
        pg = projected_gradient(x, grad, bounds)
        pg_max = float(np.max(np.abs(pg))) if len(pg) else 0.0
        if pg_max <= target:
            break
    newton = {"steps": 0, "before": pg_max}
    if objective.hess_fn is not None and settings.get("newton_steps", 0):
        x, newton = newton_polish(
            objective,
            x,
            int(settings["newton_steps"]),
            float(settings.get("newton_target", 0.0)),
        )
        loss, smooth, ridge, grad = objective.parts(x)
        pg = projected_gradient(x, grad, bounds)
        pg_max = float(np.max(np.abs(pg))) if len(pg) else 0.0
    return {
        "theta": x,
        "objective": loss + objective.lam_smooth * smooth + objective.lam_ridge * ridge,
        "loss": loss,
        "smooth": smooth,
        "ridge": ridge,
        "proj_grad_max": pg_max,
        "nit": nit,
        "nfev": nfev,
        "restarts": attempt,
        "newton_steps": newton["steps"],
        "proj_grad_before_newton": newton["before"],
        "method": "lbfgsb",
        "message": str(res.message),
        "success": bool(res.success),
    }


def _free(theta, grad, bounds):
    """Parameters not held at a bound: a parameter at its bound whose
    gradient pushes outward stays fixed during a Newton step."""
    free = np.ones(len(theta), dtype=bool)
    for k, (lo, hi) in enumerate(bounds):
        if lo is not None and theta[k] <= lo and grad[k] > 0:
            free[k] = False
        if hi is not None and theta[k] >= hi and grad[k] < 0:
            free[k] = False
    return free


def _clip(theta, bounds):
    out = np.array(theta, dtype=np.float64)
    for k, (lo, hi) in enumerate(bounds):
        if lo is not None and out[k] < lo:
            out[k] = lo
        if hi is not None and out[k] > hi:
            out[k] = hi
    return out


def newton_polish(objective, theta, steps, target):
    """Newton steps with the exact Hessian on the free parameters, after
    L-BFGS-B. Near the optimum of a smooth convex function Newton's
    method converges quadratically (Boyd & Vandenberghe 2004, section
    9.5), so a few steps take the projected gradient from the L-BFGS-B
    stop to rounding level, also along directions with little curvature,
    where a gradient-based stop leaves the tables least determined.

    A step is taken in full when it lowers the objective (Armijo test),
    and halved otherwise. Near the optimum the objective changes by less
    than its rounding, so a step that halves the largest projected
    gradient without raising the objective beyond rounding is also
    accepted. The polish stops at target, after steps, or when no step
    helps. Returns (theta, {"steps", "before"})."""
    bounds = objective.layout.bounds
    x = np.array(theta, dtype=np.float64)
    F, grad = objective(x)
    pg = float(np.max(np.abs(projected_gradient(x, grad, bounds))))
    rec = {"steps": 0, "before": pg}
    for _ in range(steps):
        if pg <= target:
            break
        free = _free(x, grad, bounds)
        H = objective.hessian(x)[np.ix_(free, free)]
        try:
            d_f = -cho_solve(cho_factor(H), grad[free])
        except np.linalg.LinAlgError:
            d_f = -np.linalg.lstsq(H, grad[free], rcond=None)[0]
        d = np.zeros_like(x)
        d[free] = d_f
        alpha, moved = 1.0, False
        while alpha >= 2.0**-30:
            xn = _clip(x + alpha * d, bounds)
            Fn, gn = objective(xn)
            pgn = float(np.max(np.abs(projected_gradient(xn, gn, bounds))))
            armijo = Fn <= F + 1e-4 * float(grad @ (xn - x))
            flat = pgn <= 0.5 * pg and Fn <= F + 1e-13 * max(1.0, abs(F))
            if armijo or flat:
                moved = True
                break
            alpha *= 0.5
        if not moved:
            break
        x, F, grad, pg = xn, Fn, gn, pgn
        rec["steps"] += 1
    return x, rec


# Centering


def background_means(T, layout, ref_means):
    """{term name: mean value of the term over the background rows}, from
    the mean basis vectors ref_means[name] (basis_mean over those rows).
    For a term with a multiplier, the mean of the table value itself."""
    out = {}
    for t, off in zip(layout.terms, layout.offsets):
        a = ref_means.get(t.name)
        if a is None:
            raise ValueError(f"no background mean for term {t.name}")
        out[t.name] = float(np.asarray(a, dtype=np.float64) @ T[off : off + t.size])
    return out


def center(T, layout, ref_means):
    """Center the tables on a background: subtract each term's mean value
    over the background rows from its knots and state slots.

    Every background row of such a term holds a knot or a state slot, so
    each row moves by the same constant, and for a term with a multiplier
    the scores of one listener move together. No ranking changes. A
    centered term (the deviation curve) keeps its tables: it is "not
    applicable", and adds exactly 0, on the rows outside its tiers, so a
    shift would move its tier against the others. Its level is set by its
    own construction (Layout). Returns (centered tables, {name: the
    constant subtracted}, 0.0 for a centered term)."""
    means = background_means(T, layout, ref_means)
    Tc = np.array(T, dtype=np.float64)
    consts = {}
    for t, off in zip(layout.terms, layout.offsets):
        if t.centered:
            consts[t.name] = 0.0
            continue
        a = np.asarray(ref_means[t.name], dtype=np.float64)
        if a[t.K + NA] > 0:
            raise ValueError(
                f"term {t.name} is not applicable on some "
                f"background rows, so a shift would change "
                f"rankings"
            )
        consts[t.name] = means[t.name]
        Tc[off : off + t.K + 2] -= means[t.name]
    return Tc, consts


def table_dict(T, layout):
    """{term name: {"knots", "heights", "zero", "missing"}} for export."""
    out = {}
    for t, off in zip(layout.terms, layout.offsets):
        K = t.K
        out[t.name] = {
            "knots": [float(x) for x in t.knots],
            "heights": [float(x) for x in T[off : off + K]],
            "zero": float(T[off + K + ZERO]),
            "missing": float(T[off + K + MISS]),
        }
    return out
