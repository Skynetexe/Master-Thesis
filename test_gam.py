"""Checks for pipeline/gam.py, pipeline/gam_audit.py and
pipeline/catalog.py on synthetic data. Nothing touches the model folder.

1. Terms, the knot rule and the encoding: states, clamping, and shape
   values equal to np.interp.
2. Layout: the fixed height, the continuity tie of the zero state, the
   neutral tie of the missing state, the centered deviation curve with
   mean 0, monotone increments, and a positive semidefinite penalty.
3. The loss: the corrected softmax equals a direct computation per
   (listener, positive) group, the gradient equals finite differences, the
   objective is convex along random segments, and the loss ignores a
   constant added to a whole term or to the tilt table.
4. The fit: the block Hessian equals finite differences, two starts
   reach the same tables after the Newton polish, a monotone term stays
   monotone, and centering changes no loss and puts every mean at 0.
5. Mainstreaminess against math.fsum.
6. The audit statistics against brute force: the stratified AUC against
   an enumeration of the pairs, the bootstrap against a fixed seed,
   concurvity against a QR computation on the dense design, and the
   one-standard-error choice on constructed scores.
7. The catalog: concept_catalog() equals concepts.full_catalog() bit for
   bit, CatalogScorer's scores equal the dense reference, and its ranks
   equal a direct count under the P5 rule, with planted ties, exclusions
   and every chunk size.

Usage, from the repository root:
    python3 test_gam.py
"""

import math

import numpy as np

from pipeline import catalog as ct
from pipeline import concept_data as cd
from pipeline import concepts as cn
from pipeline import gam
from pipeline import gam_audit as ga

failures = []


def check(cond, msg):
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        failures.append(msg)


def raises(fn, exc=ValueError, text=""):
    try:
        fn()
    except exc as e:
        return text in str(e)
    except Exception as e:  # reported, then False
        print(f"      unexpected {type(e).__name__}: {e}")
        return False
    return False


rng = np.random.default_rng(20260927)

# 1: terms and encoding
check(
    raises(lambda: gam.Term("a", "x", (1.0, 1.0)), text="knots"),
    "Term rejects repeated knots",
)
check(
    raises(lambda: gam.Term("a", "x", (0.0,)), text="knots"),
    "Term rejects a single knot",
)
check(
    raises(
        lambda: gam.Term("a", "x", (0.0, 1.0), centered=True, missing_state=True),
        text="state",
    ),
    "Term rejects a centered term with a state constant",
)
check(
    raises(
        lambda: gam.Term("a", "x", (0.0, 1.0), multiplier="other"), text="multiplier"
    ),
    "Term rejects an unknown multiplier",
)
t_json = gam.Term(
    "a",
    "x",
    (0.0, 0.5, 2.0),
    zero_state=True,
    missing_state=True,
    tiers=(3,),
    monotone=1,
)
check(gam.Term.from_json(t_json.to_json()) == t_json, "Term survives a JSON round trip")

pos = np.concatenate([np.full(500, 0.2), rng.uniform(0.3, 0.9, 500)])
neg = rng.uniform(0.0, 1.5, 3000)
kn = gam.knot_rule(pos, neg, np.linspace(0.005, 0.995, 30), (0.01, 0.5, 0.99))
k_arr = np.asarray(kn)
want = np.unique(
    np.concatenate(
        (
            [0.0],
            np.quantile(pos, np.linspace(0.005, 0.995, 30)),
            np.quantile(neg, (0.01, 0.5, 0.99)),
        )
    )
)
check(
    kn[0] == 0.0
    and np.all(np.diff(k_arr) > 1e-6 * (k_arr[-1] - k_arr[0]))
    and set(kn) <= set(want.tolist())
    and np.sum(np.isclose(k_arr, 0.2)) == 1,
    "knot_rule: anchors at 0, keeps quantiles of both groups, one knot "
    "at a point mass, no two knots closer than the gap",
)
n_raw = 1 + 30 + 3
check(
    len(kn) < n_raw and len(want) < n_raw,
    "knot_rule merges the repeated quantiles of a point mass",
)

x = rng.uniform(-0.5, 2.0, 400)
s_c, w_c = gam.encode_continuous(kn, x)
heights = rng.normal(size=len(kn))
T_try = np.concatenate([heights, [0.0] * 4])
D_try = np.zeros_like(T_try)
D_try[: len(kn) - 1] = np.diff(heights)
check(
    np.allclose(
        gam.pwl_value(T_try, D_try, s_c.astype(np.int64), w_c),
        np.interp(np.clip(x, kn[0], kn[-1]), kn, heights),
        rtol=0,
        atol=1e-13,
    ),
    "encode_continuous + pwl_value equal np.interp with clamped ends",
)
s_e, w_e = gam.encode_continuous(
    kn, np.array([kn[-1], kn[-1] + 5.0, kn[0] - 1.0, kn[3]])
)
check(
    s_e.tolist() == [len(kn) - 1, len(kn) - 1, 0, 3]
    and w_e.tolist() == [0.0, 0.0, 0.0, 0.0],
    "the last knot and values beyond it get slot K-1 and w 0, values "
    "below the first knot get slot 0, a value on a knot gets w 0",
)

term = gam.Term("c", "c", kn, zero_state=True, missing_state=True)
val = np.array([0.0, 0.5, np.nan, 0.7, 0.0])
has = np.array([True, True, False, True, False])
sl, ww = gam.encode(term, val, has)
K = term.K
check(
    sl[0] == K + gam.ZERO
    and sl[2] == K + gam.MISS
    and sl[4] == K + gam.MISS
    and sl[1] < K
    and sl[3] < K
    and ww[0] == ww[2] == ww[4] == 0.0,
    "encode: exactly zero, missing, and missing wins over a zero value " "without data",
)
dev = gam.Term("d", "c", kn, tiers=(3,), centered=True)
sl, ww = gam.encode(
    dev,
    np.array([0.5, 0.5, np.nan]),
    np.array([True, True, False]),
    tier=np.array([3, 1, 3]),
)
check(
    sl[0] < dev.K and sl[1] == dev.K + gam.NA and sl[2] == dev.K + gam.NA,
    "encode: a tier term is not applicable outside its tiers and where "
    "a term without a missing state has no data",
)
check(
    raises(lambda: gam.encode(dev, np.array([0.5]), np.array([True])), text="tier"),
    "encode refuses a tier term without tiers",
)
check(
    raises(
        lambda: gam.encode(term, np.array([np.inf]), np.array([True])), text="finite"
    ),
    "encode refuses a non-finite value with data",
)

# Synthetic training data
U, M_NEG = 260, 8
npos = rng.integers(1, 3, U)
lens = npos + M_NEG
N = int(lens.sum())
starts = np.concatenate(([0], np.cumsum(lens)[:-1]))
row_user = np.repeat(np.arange(U), lens)
pos_rows = np.concatenate([starts[u] + np.arange(npos[u]) for u in range(U)])
pos_user = row_user[pos_rows]
neg_rows = (starts + npos)[:, None] + np.arange(M_NEG)[None, :]
is_pos = np.zeros(N, dtype=bool)
is_pos[pos_rows] = True
tier = rng.integers(0, 6, N)
cols = {}
v = rng.normal(size=N)
h = rng.random(N) > 0.25
v[~h] = np.nan
cols["a"] = (v, h)
v = np.abs(rng.normal(size=N)) * (rng.random(N) > 0.3)
h = rng.random(N) > 0.1
v[~h] = np.nan
cols["b"] = (v, h)
v = rng.gamma(2.0, 2.0, N)
cols["pop"] = (v, np.ones(N, dtype=bool))
m_user = rng.normal(size=U)
m_row = m_user[row_user]


def mk_knots(src, zero=False):
    v, h = cols[src]
    m = h & ((v != 0) if zero else True)
    return gam.knot_rule(
        v[m & is_pos], v[m & ~is_pos], np.linspace(0.05, 0.95, 7), (0.1, 0.5, 0.9)
    )


terms = [
    gam.Term("a", "a", mk_knots("a"), missing_state=True),
    gam.Term("b", "b", mk_knots("b", True), zero_state=True, missing_state=True),
    gam.Term("pop", "pop", mk_knots("pop")),
    gam.Term("tilt", "pop", mk_knots("pop"), multiplier="mainstreaminess"),
    gam.Term("dev", "a", mk_knots("a"), tiers=(3,), centered=True),
    gam.Term("mono", "pop", mk_knots("pop"), monotone=1),
]
enc = [gam.encode(t, *cols[t.source], tier) for t in terms]
slot = np.stack([e[0] for e in enc])
wmat = np.stack([e[1] for e in enc])
ref = ~is_pos
means_all, data_means, dev_means = {}, {}, {}
for k, t in enumerate(terms):
    means_all[t.name] = gam.basis_mean(t, slot[k], wmat[k], ref)
    if t.missing_state:
        data_means[t.name] = gam.basis_mean(
            t, slot[k], wmat[k], ref & (slot[k] < t.K + 1)
        )
    if t.centered:
        dev_means[t.name] = gam.basis_mean(t, slot[k], wmat[k], ref & (slot[k] < t.K))
n_zero = {t.name: int((slot[k] == t.K).sum()) for k, t in enumerate(terms)}
n_miss = {t.name: int((slot[k] == t.K + 1).sum()) for k, t in enumerate(terms)}
n_zero["b"] = 0  # the zero state of b is tied by continuity
layout = gam.Layout(
    terms, n_zero, n_miss, neutral=("a",), data_means=data_means, dev_means=dev_means
)
offset = np.where(is_pos, 0.0, rng.normal(0.5, 1.0, N))
pos_weight = (1.0 / npos[pos_user]) / U
blk = {
    "slot": slot,
    "w": wmat,
    "off": np.asarray(layout.offsets),
    "mult": {3: m_row},
    "offset": offset,
    "pos_rows": pos_rows,
    "pos_user": pos_user,
    "pos_weight": pos_weight,
    "neg_rows": neg_rows,
}

# 2: layout
th = rng.normal(size=layout.P)
T = layout.tables(th)
offs = layout.offsets
ok_pin = all(T[off] == 0.0 for off, t in zip(offs, terms) if not t.centered)
check(ok_pin, "Layout fixes height 0 of every term that is not centered " "at 0")
off_b, tb = offs[1], terms[1]
check(
    T[off_b + tb.K + gam.ZERO] == T[off_b + gam.zero_knot(tb)]
    and not layout.zero_free["b"],
    "Layout ties a zero state without training rows to the knot at 0",
)
off_a, ta = offs[0], terms[0]
check(
    abs(
        T[off_a + ta.K + gam.MISS]
        - data_means["a"][: ta.K + 1] @ T[off_a : off_a + ta.K + 1]
    )
    < 1e-12
    and not layout.miss_free["a"]
    and layout.miss_free["b"],
    "Layout ties a neutral missing state to the mean over the reference "
    "rows with data",
)
off_d, td = offs[4], terms[4]
check(
    abs(dev_means["dev"][: td.K] @ T[off_d : off_d + td.K]) < 1e-12,
    "a centered term has mean 0 over its reference rows for any theta",
)
off_m, tm = offs[5], terms[5]
mono_cols = [i for i, lab in enumerate(layout.labels) if lab.startswith("mono:")]
check(
    all(layout.bounds[i] == (0.0, None) for i in mono_cols),
    "a monotone term's increments are bounded at 0",
)
ev_q = np.linalg.eigvalsh(layout.Q)
check(ev_q.min() > -1e-10, "the smoothness penalty is positive semidefinite")
check(
    np.linalg.matrix_rank(layout.M) == layout.P,
    "every free parameter moves the tables (M has full column rank)",
)
na_rows = [off + t.K + gam.NA for off, t in zip(offs, terms)] + [
    off + t.K + gam.PAD for off, t in zip(offs, terms)
]
check(not layout.M[na_rows].any(), "the not-applicable and padding slots stay 0")
check(
    raises(
        lambda: gam.Layout(
            terms,
            n_zero,
            n_miss,
            neutral=("zz",),
            data_means=data_means,
            dev_means=dev_means,
        ),
        text="unknown",
    ),
    "Layout rejects a neutral name without term",
)
check(
    raises(
        lambda: gam.Layout(
            terms, n_zero, n_miss, neutral=("a",), data_means={}, dev_means=dev_means
        ),
        text="mean basis",
    ),
    "Layout needs the reference mean for a tied missing state",
)


# 3: loss


def loss_grad(T_, D_):
    return gam.block_loss_grad(T_, D_, blk, layout.L)


def hess_fn(T_, D_):
    return gam.block_hessian(T_, D_, blk, layout.L)


obj = gam.Objective(layout, loss_grad, 0.03, 1e-6, hess_fn)
D = gam.diff_table(T, terms, offs)
s_fast = gam.block_scores(T, D, blk)


def brute_value(t, off, T_, r):
    v_, h_ = cols[t.source]
    tab = T_[off : off + t.size]
    if t.tiers is not None and tier[r] not in t.tiers:
        return 0.0
    if not h_[r]:
        return tab[t.K + gam.MISS] if t.missing_state else 0.0
    if t.zero_state and v_[r] == 0.0:
        return tab[t.K + gam.ZERO]
    kk = t.knot_array
    out = float(np.interp(np.clip(v_[r], kk[0], kk[-1]), kk, tab[: t.K]))
    return out * m_row[r] if t.multiplier else out


s_brute = np.array(
    [
        math.fsum(brute_value(t, off, T, r) for t, off in zip(terms, offs))
        for r in range(N)
    ]
)
check(
    np.max(np.abs(s_fast - s_brute)) < 1e-12,
    "block_scores equals a row-by-row np.interp evaluation",
)
z = s_brute + offset
L_brute = math.fsum(
    pos_weight[j]
    * (
        math.log(
            math.fsum(math.exp(q) for q in np.concatenate(([z[p]], z[neg_rows[u]])))
        )
        - z[p]
    )
    for j, (p, u) in enumerate(zip(pos_rows, pos_user))
)
L_fast, g_fast = gam.softmax_block(
    s_fast, offset, pos_rows, pos_user, pos_weight, neg_rows
)
check(
    abs(L_fast - L_brute) < 1e-12,
    "softmax_block equals a direct computation per (listener, positive) "
    "group, with the negatives shared by two positives",
)
eps = 1e-6
F0, g0 = obj(th)
fd = np.array(
    [
        (obj(th + eps * e)[0] - obj(th - eps * e)[0]) / (2 * eps)
        for e in np.eye(layout.P)
    ]
)
check(
    np.max(np.abs(fd - g0)) < 1e-7 * max(1.0, np.max(np.abs(g0))),
    "the objective's gradient equals central finite differences, with "
    "every tie, the tilt multiplier and the deviation curve",
)
X_blk = gam.block_design(blk, layout.L)
check(
    np.max(np.abs(X_blk @ T - s_fast)) < 1e-12,
    "block_design is the Jacobian of the scores: X T equals the scores",
)
a_rand = rng.random(N)
xax_sparse = X_blk.T @ (a_rand[:, None] * X_blk.toarray())
check(
    np.max(np.abs(gam.block_xax(blk, a_rand, layout.L) - xax_sparse))
    < 1e-12 * max(1.0, np.max(np.abs(xax_sparse))),
    "block_xax equals X' diag(a) X of the explicit design",
)
H_t = gam.block_hessian(T, D, blk, layout.L)
eps_t = 1e-6
fd_t = np.zeros_like(H_t)
for i in range(layout.L):
    e = np.zeros(layout.L)
    e[i] = eps_t
    gp = gam.block_loss_grad(T + e, gam.diff_table(T + e, terms, offs), blk, layout.L)[
        1
    ]
    gm = gam.block_loss_grad(T - e, gam.diff_table(T - e, terms, offs), blk, layout.L)[
        1
    ]
    fd_t[:, i] = (gp - gm) / (2 * eps_t)
check(
    np.max(np.abs(fd_t - H_t)) < 1e-7 * max(1.0, np.max(np.abs(H_t))),
    "block_hessian equals central finite differences of the table " "gradient",
)
H_th = obj.hessian(th)
fd_th = np.array(
    [
        (obj(th + eps * e)[1] - obj(th - eps * e)[1]) / (2 * eps)
        for e in np.eye(layout.P)
    ]
).T
check(
    np.max(np.abs(fd_th - H_th)) < 1e-7 * max(1.0, np.max(np.abs(H_th)))
    and np.linalg.eigvalsh(H_th).min() > 0,
    "the objective's Hessian equals finite differences of its gradient "
    "and is positive definite",
)
worst = 0.0
for _ in range(20):
    a, b = rng.normal(size=layout.P), rng.normal(size=layout.P)
    lam = rng.random()
    worst = max(
        worst,
        obj(lam * a + (1 - lam) * b)[0] - (lam * obj(a)[0] + (1 - lam) * obj(b)[0]),
    )
check(worst <= 1e-12, "the objective is convex along 20 random segments")
T_shift = T.copy()
T_shift[off_a : off_a + ta.K + 2] += 3.7
D_shift = gam.diff_table(T_shift, terms, offs)
L_shift = gam.block_loss_grad(T_shift, D_shift, blk, layout.L)[0]
off_t, tt = offs[3], terms[3]
T_tilt = T.copy()
T_tilt[off_t : off_t + tt.K] -= 1.9
L_tilt = gam.block_loss_grad(
    T_tilt, gam.diff_table(T_tilt, terms, offs), blk, layout.L
)[0]
L0 = gam.block_loss_grad(T, D, blk, layout.L)[0]
check(
    abs(L_shift - L0) < 1e-12 and abs(L_tilt - L0) < 1e-12,
    "the loss ignores a constant added to a whole concept table or to "
    "the tilt table (only within-listener differences are identified)",
)

# 4: fit
settings = {
    "maxcor": 20,
    "ftol": 1e-14,
    "gtol": 1e-10,
    "maxiter": 5000,
    "maxfun": 10000,
    "pg_target": 1e-9,
    "restarts": 3,
    "newton_steps": 8,
    "newton_target": 1e-12,
}
plain = dict(settings, newton_steps=0)
obj_plain = gam.Objective(layout, loss_grad, 0.03, 1e-6)
p1 = gam.fit(obj_plain, np.zeros(layout.P), plain)
p2 = gam.fit(obj_plain, rng.normal(size=layout.P), plain)
r1 = gam.fit(obj, np.zeros(layout.P), settings)
r2 = gam.fit(obj, rng.normal(size=layout.P), settings)
T1, T2 = layout.tables(r1["theta"]), layout.tables(r2["theta"])
d_plain = np.max(np.abs(layout.tables(p1["theta"]) - layout.tables(p2["theta"])))
d_newton = np.max(np.abs(T1 - T2))
check(
    d_newton < 1e-9 and r1["proj_grad_max"] < 1e-11 and r1["newton_steps"] >= 1,
    f"with the Newton polish two starts reach the same tables (largest "
    f"difference {d_newton:.1e}, {d_plain:.1e} without it, projected "
    f"gradient {r1['proj_grad_max']:.1e} after {r1['newton_steps']} "
    f"Newton steps)",
)
free_terms = [t for t in terms if not t.monotone]
k_free = [k for k, t in enumerate(terms) if not t.monotone]
lay_free = gam.Layout(
    free_terms,
    n_zero,
    n_miss,
    neutral=("a",),
    data_means=data_means,
    dev_means=dev_means,
)
blk_free = dict(
    blk,
    slot=slot[k_free],
    w=wmat[k_free],
    off=np.asarray(lay_free.offsets),
    mult={3: m_row},
)
obj_free = gam.Objective(
    lay_free,
    lambda T_, D_: gam.block_loss_grad(T_, D_, blk_free, lay_free.L),
    0.03,
    1e-6,
    lambda T_, D_: gam.block_hessian(T_, D_, blk_free, lay_free.L),
)
nw = dict(settings, method="newton", newton_steps=60, newton_accept=1e-9)
n1 = gam.fit(obj_free, np.zeros(lay_free.P), nw)
n2 = gam.fit(obj_free, rng.normal(0, 0.5, lay_free.P), nw)
d_nw = np.max(np.abs(lay_free.tables(n1["theta"]) - lay_free.tables(n2["theta"])))
check(
    n1["method"] == n2["method"] == "newton"
    and d_nw < 1e-9
    and n1["proj_grad_max"] < 1e-11,
    f"damped Newton from zero and from a random start reaches the same "
    f"tables without bounds ({n1['newton_steps']} and "
    f"{n2['newton_steps']} steps, largest difference {d_nw:.1e})",
)
# With a weak penalty, far from the optimum a full Newton step overshoots:
# the softmax saturates and its curvature is small. The line search must
# then shorten the step, so the objective never rises, and damped Newton
# still reaches the optimum.
obj_weak = gam.Objective(lay_free, obj_free.loss_grad, 1e-7, 1e-7, obj_free.hess_fn)
x_far = rng.normal(0.0, 6.0, lay_free.P)
F_far, g_far = obj_weak(x_far)
x_full = x_far - np.linalg.solve(obj_weak.hessian(x_far), g_far)
overshoot = obj_weak(x_full)[0] > F_far
x_one, rec_one = gam.newton_polish(obj_weak, x_far, 1, 0.0)
check(
    overshoot and rec_one["steps"] == 1 and obj_weak(x_one)[0] < F_far,
    "a full Newton step from a far start raises the objective, and "
    "newton_polish shortens it so the objective falls",
)
x_all, _ = gam.newton_polish(obj_weak, x_far, 200, 1e-12)
x_ref, _ = gam.newton_polish(obj_weak, np.zeros(lay_free.P), 200, 1e-12)
check(
    np.max(np.abs(lay_free.tables(x_all) - lay_free.tables(x_ref))) < 1e-8,
    "damped Newton from the far start and from zero reaches the same " "tables",
)
check(
    r1["method"] == "lbfgsb",
    "with bounds, fit() runs L-BFGS-B before the Newton polish",
)
check(
    np.all(np.diff(T1[off_m : off_m + tm.K]) >= -1e-15),
    "a monotone term's fitted heights never decrease",
)
check(
    r1["objective"] == obj(r1["theta"])[0],
    "fit() reports the objective at the returned parameters",
)
Tc, consts = gam.center(T1, layout, means_all)
Dc = gam.diff_table(Tc, terms, offs)
means_c = gam.background_means(Tc, layout, means_all)
check(
    max(abs(means_c[t.name]) for t in terms if not t.centered) < 1e-12
    and consts["dev"] == 0.0,
    "center() puts every term's background mean at 0 and leaves the "
    "deviation curve alone",
)
check(
    abs(
        gam.block_loss_grad(Tc, Dc, blk, layout.L)[0]
        - gam.block_loss_grad(T1, gam.diff_table(T1, terms, offs), blk, layout.L)[0]
    )
    < 1e-12,
    "centering changes no loss, so no ranking",
)
s1 = gam.block_scores(Tc, Dc, blk)
brute_mean = {
    t.name: np.mean(
        [
            brute_value(t, off, Tc, r) / (m_row[r] if t.multiplier else 1.0)
            for r in np.flatnonzero(ref)
        ]
    )
    for t, off in zip(terms, offs)
}
check(
    max(abs(brute_mean[t.name] - means_c[t.name]) for t in terms) < 1e-12,
    "background_means equals the mean of the term values over the rows",
)
bad_means = dict(means_all)
bad_means["pop"] = means_all["pop"].copy()
bad_means["pop"][terms[2].K + gam.NA] = 0.1
check(
    raises(lambda: gam.center(T1, layout, bad_means), text="rankings"),
    "center() refuses a background where a shifted term is not " "applicable",
)

# 5: mainstreaminess
popv = rng.gamma(2.0, 3.0, 50)
items = np.array([3, 9, 11, 40])
wts = np.array([0.7, 2.1, 1.3, 0.2])
want_m = math.fsum(a * popv[i] for a, i in zip(wts, items)) / math.fsum(wts)
check(
    abs(gam.mainstreaminess_raw(items, wts, popv) - want_m) < 1e-12,
    "mainstreaminess_raw is the weighted mean of log popularity",
)
mc = {"mean": 1.0, "sd": 2.0, "lo": -1.0, "hi": 0.5}
check(
    np.allclose(
        gam.mainstreaminess(np.array([-9.0, 1.0, 2.0, 9.0]), mc), [-1.0, 0.0, 0.5, 0.5]
    ),
    "mainstreaminess standardizes and clamps",
)

# 6: audit
Uu, S = 40, 4
user_r = rng.integers(0, Uu, 900)
strat = rng.integers(0, S, 900)
flag = rng.random(900) < 0.4
lab = rng.random(900) < 0.3
cp = ga.flag_counts(user_r[lab], strat[lab], flag[lab], Uu, S)
cn_ = ga.flag_counts(user_r[~lab], strat[~lab], flag[~lab], Uu, S)
num = den = 0.0
for s_ in range(S):
    fp_ = flag[lab & (strat == s_)].astype(float)
    fn_ = flag[~lab & (strat == s_)].astype(float)
    for a_ in fp_:
        for b_ in fn_:
            num += 1.0 if a_ > b_ else (0.5 if a_ == b_ else 0.0)
            den += 1.0
check(
    abs(float(ga.pooled_auc(cp.sum(axis=0), cn_.sum(axis=0))) - num / den) < 1e-12,
    "pooled_auc equals the Mann-Whitney AUC over the pairs that share a "
    "stratum, ties counted half",
)
st1 = ga.stratified_flag_auc(cp, cn_, 60, 7)
st2 = ga.stratified_flag_auc(cp, cn_, 60, 7)
check(
    st1 == st2 and st1["lo"] <= st1["auc"] <= st1["hi"] and st1["replicates"] == 60,
    "the bootstrap interval repeats for one seed and holds the estimate",
)
cut = np.array([0.5, 1.5])
check(
    ga.strata_of(np.array([0.2, 0.5, 1.0, 1.5, 3.0]), cut).tolist() == [0, 1, 1, 2, 2],
    "strata_of puts a cut value into the upper stratum",
)
check(
    ga.no_signal({"u": {"lo": 0.49, "hi": 0.515}}, 0.02)
    and not ga.no_signal({"u": {"lo": 0.49, "hi": 0.53}}, 0.02)
    and not ga.no_signal({"u": {"lo": math.nan, "hi": 0.5}}, 0.02),
    "no_signal needs every interval inside 0.5 +- the margin",
)

# concurvity on a dense design
n_obs = 400
x1 = rng.normal(size=n_obs)
x2 = 0.8 * x1 + 0.6 * rng.normal(size=n_obs)
x3 = rng.normal(size=n_obs)
B1 = np.stack([x1, x1**2], 1)
B2 = np.stack([x2, np.sin(x2)], 1)
B3 = np.stack([x3, x3**3], 1)
X = np.concatenate([B1, B2, B3], 1)
X = X - X.mean(axis=0)
G = X.T @ X
groups = {"g1": [0, 1], "g2": [2, 3], "g3": [4, 5]}
full, names, pair = ga.concurvity(G, groups)


def direct(J, O):
    QJ, _ = np.linalg.qr(X[:, J])
    QO, _ = np.linalg.qr(X[:, O])
    return np.clip(np.linalg.svd(QJ.T @ QO, compute_uv=False), 0, 1)


rho = direct([0, 1], [2, 3, 4, 5])
check(
    abs(full["g1"]["worst"] - rho[0] ** 2) < 1e-9
    and abs(full["g1"]["estimate"] - np.mean(rho**2)) < 1e-9,
    "concurvity worst and estimate equal the canonical correlations of a "
    "QR computation",
)
check(
    abs(pair[0, 1] - direct([0, 1], [2, 3])[0] ** 2) < 1e-9
    and pair[0, 1] == pair[1, 0],
    "the pairwise worst matrix is symmetric and matches QR",
)
coef = {
    "g1": np.array([1.0, -0.5]),
    "g2": np.array([0.3, 2.0]),
    "g3": np.array([1.0, 1.0]),
}
obs = ga.observed_concurvity(G, groups, coef)
f1 = X[:, [0, 1]] @ coef["g1"]
proj = X[:, 2:] @ np.linalg.lstsq(X[:, 2:], f1, rcond=None)[0]
check(
    abs(obs["g1"] - (proj @ proj) / (f1 @ f1)) < 1e-9,
    "observed concurvity equals the R^2 of the fitted term on the other "
    "terms' bases",
)

# one-standard-error choice
base_s = rng.random(3000) * 0.1
sc = np.stack(
    [
        base_s - 0.004,
        base_s + 0.0 + rng.normal(0, 0.02, 3000),
        base_s + 0.001,
        base_s + 0.0015,
    ]
)
chosen, best, rows_se = ga.one_se_choice(sc, [0, 1, 2, 3])
d_want = sc[best] - sc
se_want = d_want.std(axis=1, ddof=1) / math.sqrt(3000)
first_ok = next(c for c in [0, 1, 2, 3] if c == best or d_want[c].mean() <= se_want[c])
check(
    best == 3
    and chosen == first_ok
    and all(abs(rows_se[c]["se"] - se_want[c]) < 1e-15 for c in range(4)),
    f"one_se_choice picks the simplest configuration within one paired "
    f"standard error (chosen {chosen}, best {best})",
)
# A configuration 0.7 standard errors below the best is within one
# standard error, so the rule must choose it, and one at 3 is not.
n_l = 2000
e0 = rng.normal(size=n_l)
e0 -= e0.mean()
se0 = e0.std(ddof=1) / math.sqrt(n_l)
e1 = rng.normal(size=n_l)
e1 -= e1.mean()
se1 = e1.std(ddof=1) / math.sqrt(n_l)
top = rng.random(n_l)
sc2 = np.stack([top - (e0 + 0.7 * se0), top - (e1 + 3.0 * se1), top])
ch2, be2, _ = ga.one_se_choice(sc2, [0, 1, 2])
check(
    be2 == 2 and ch2 == 0,
    "one_se_choice takes a configuration 0.7 standard errors below the "
    "best and passes over one 3 standard errors below",
)
tie = np.stack([base_s, base_s])
check(
    ga.one_se_choice(tie, [1, 0])[1] == 1,
    "a tie for the best goes to the simpler configuration",
)
rk = np.array([[1, 150], [100, 101], [2, 3]])
want_s = [
    (1.0 / np.log2(2) + 0.0) / 2,
    (1 / np.log2(101) + 0.0) / 2,
    (1 / np.log2(3) + 1 / np.log2(4)) / 2,
]
check(
    np.allclose(ga.per_listener_scores(rk), want_s),
    "per_listener_scores averages NDCG@100 over the two picks",
)

# 7: catalog
NI = 500
DA = len(cn.AUDIO_DIMS)


def rows_csr(n_cols, p_empty, max_per_row):
    rows = [[]]
    for _ in range(1, NI):
        if rng.random() < p_empty:
            rows.append([])
            continue
        k = int(rng.integers(1, max_per_row + 1))
        rows.append(sorted(rng.choice(n_cols, size=k, replace=False).tolist()))
    return rows


def pack(rows, dtype=np.int32):
    indptr = np.concatenate(([0], np.cumsum([len(r) for r in rows])))
    return indptr.astype(np.int64), np.array([x for r in rows for x in r], dtype=dtype)


term_rows = {
    "tags_direct": rows_csr(25, 0.5, 3),
    "tags_artist": rows_csr(25, 0.2, 5),
    "genres": rows_csr(10, 0.4, 2),
}
credit_rows = {"artists": rows_csr(50, 0.05, 2), "works": rows_csr(30, 0.7, 1)}
col_rows = {}
for kind_, nnb in (("same_artist", 5), ("other_artists", 3)):
    rn, rc = [[]], [[]]
    for i in range(1, NI):
        if rng.random() < 0.3:
            rn.append([])
            rc.append([])
            continue
        k = int(rng.integers(1, nnb + 1))
        nb = rng.choice(np.setdiff1d(np.arange(1, NI), [i]), size=k, replace=False)
        rn.append(nb.tolist())
        rc.append(rng.uniform(0.01, 1.0, size=k).tolist())
    col_rows[kind_] = (rn, rc)
audio = rng.normal(size=(NI, DA))
audio[0] = np.nan
audio[rng.choice(np.arange(1, NI), 80, replace=False)] = np.nan
audio[rng.choice(np.arange(1, NI), 40, replace=False), 1] = np.nan
listens = rng.integers(0, 2000, NI)
tiers_i = rng.integers(0, 6, NI)
# Items 11 and 12 copy item 10 in every source, so every listener gives
# them the same concepts and the same score: a planted tie.
for rows in list(term_rows.values()) + list(credit_rows.values()):
    rows[11] = list(rows[10])
    rows[12] = list(rows[10])
for rn, rc in col_rows.values():
    rn[11], rc[11] = list(rn[10]), list(rc[10])
    rn[12], rc[12] = list(rn[10]), list(rc[10])
audio[11] = audio[12] = audio[10]
listens[11] = listens[12] = listens[10]
tiers_i[11] = tiers_i[12] = tiers_i[10]
terms_src = {k: (*pack(v), 25 if k != "genres" else 10) for k, v in term_rows.items()}
credits, credit_ids = {}, {}
for k, v in credit_rows.items():
    ip, raw = pack(v, np.int64)
    c = cd.compact(ip, raw)
    credits[k], credit_ids[k] = c[:3], c[3]
colisten = {}
for k, (rn, rc) in col_rows.items():
    ip, nb = pack(rn)
    colisten[k] = (ip, nb, np.array([x for r in rc for x in r]))
popularity = np.log1p(listens.astype(np.float64))
popularity[0] = 0.0
SRC = cd.Sources(
    NI, terms_src, credits, credit_ids, colisten, audio, popularity, tiers_i, {}, {}
)
space = cd.item_side(SRC, transposed=True)

# concept_catalog against full_catalog, bit for bit
hist_items = np.sort(rng.choice(np.arange(13, NI), 60, replace=False))
hist_w = np.log1p(rng.integers(1, 50, 60).astype(np.float64))
state = cn.build_user_state(hist_items, hist_w, space)
full = cn.full_catalog(state, space)
same = True
audio_rows = np.flatnonzero(space.audio_mask.any(axis=1))
for c in cn.CONTINUOUS:
    v1, h1 = full[c]
    for rows_arg in (None, audio_rows):
        v2, h2 = ct.concept_catalog(
            state, space, c, audio_chunk=37, audio_rows=rows_arg
        )
        same &= v1.tobytes() == v2.tobytes() and np.array_equal(h1, h2)
    same &= not np.any(h2 & ~ct.item_has(space, c))
check(
    same,
    "concept_catalog() equals full_catalog() bit for bit for every "
    "concept, with audio fit in chunks of 37 rows over all slots or "
    "over the tracks with audio only, and a listener never has data "
    "where the track has none",
)

# a model over the catalog's concepts
cat_terms = []
for c in (
    "tag_affinity",
    "artist_tag_affinity",
    "genre_affinity",
    "artist_affinity",
    "work_affinity",
    "colisten_same_artist",
    "colisten_other_artists",
    "audio_fit",
):
    v_, h_ = full[c]
    zs = c in (
        set(cn.TERM_CONCEPTS) | set(cn.CREDIT_CONCEPTS) | set(cn.COLISTEN_CONCEPTS)
    )
    m = h_ & ((v_ != 0) if zs else True)
    vals = v_[m] if m.any() else np.array([0.0, 1.0])
    cat_terms.append(
        gam.Term(
            c,
            c,
            gam.knot_rule(vals, vals, np.linspace(0.05, 0.95, 9), (0.5,)),
            zero_state=zs,
            missing_state=True,
        )
    )
pk = gam.knot_rule(popularity[1:], popularity[1:], np.linspace(0.05, 0.95, 9), (0.5,))
cat_terms += [
    gam.Term("popularity", "popularity", pk),
    gam.Term("popularity_tilt", "popularity", pk, multiplier="mainstreaminess"),
    gam.Term(
        "audio_fit_fuzzy_dur",
        "audio_fit",
        cat_terms[7].knots,
        tiers=(3,),
        centered=True,
    ),
]
Lc = sum(t.size for t in cat_terms)
C = 3
tabs = rng.normal(size=(C, Lc))
coffs = np.cumsum([0] + [t.size for t in cat_terms])[:-1]
for t, off in zip(cat_terms, coffs):
    tabs[:, off + t.K + gam.NA : off + t.size] = 0.0
tabs[2] = np.round(tabs[2] * 4) / 4  # coarse tables: more ties
scorer = ct.CatalogScorer(cat_terms, tabs, space, chunk=64)
scorer_big = ct.CatalogScorer(cat_terms, tabs, space, chunk=10_000)
worst_s, rank_ok, chunk_ok, tie_seen = 0.0, True, True, False
for trial in range(6):
    hi_ = np.sort(
        rng.choice(np.arange(1, NI), int(rng.choice([1, 20, 90])), replace=False)
    )
    if trial == 0:
        hi_ = np.sort(np.setdiff1d(hi_, [10, 11, 12]))
    hw_ = np.log1p(rng.integers(1, 60, len(hi_)).astype(np.float64))
    st_ = cn.build_user_state(hi_, hw_, space)
    m_u = float(rng.normal())
    lst = scorer.listener(st_, space, m_u)
    fast = scorer.scores(lst)
    refs = ct.reference_scores(
        cn.full_catalog(st_, space), cat_terms, tabs, m_u, space.tier
    )
    worst_s = max(worst_s, float(np.max(np.abs(fast[:, 1:] - refs[:, 1:]))))
    cand = np.setdiff1d(np.arange(1, NI), hi_)
    picks = rng.choice(cand, 2, replace=False)
    if trial == 0:
        picks = np.array([12, picks[0]])
    r_fast = scorer.ranks(lst, picks, hi_)
    r_big = scorer_big.ranks(scorer_big.listener(st_, space, m_u), picks, hi_)
    chunk_ok &= np.array_equal(r_fast, r_big)
    for c_ in range(C):
        s_ = fast[c_].copy()
        s_[0] = -np.inf
        s_[hi_] = -np.inf
        for p_, t_ in enumerate(picks):
            want_r = (
                1
                + int(np.count_nonzero(s_ > s_[t_]))
                + int(np.count_nonzero(s_[:t_] == s_[t_]))
            )
            rank_ok &= int(r_fast[c_, p_]) == want_r
            lo_, hi2 = ct.reference_rank(refs[c_], t_, hi_, 1e-9)
            rank_ok &= lo_ <= int(r_fast[c_, p_]) <= hi2
            if trial == 0 and p_ == 0:
                tie_seen |= s_[10] == s_[12] == s_[11]
# The corrections equal the slots of gam.encode() that differ from the
# item-level slots, for every listener term.
st_c = cn.build_user_state(hist_items, hist_w, space)
lst_c = scorer.listener(st_c, space, 0.1)
full_c = cn.full_catalog(st_c, space)
corr_ok = True
for k in scorer.user_terms:
    t = cat_terms[k]
    v_, h_ = full_c[t.source]
    s_full, w_full = gam.encode(t, v_, h_, space.tier)
    want_items = np.flatnonzero(s_full != scorer.base_slot[k])
    items, slot, w, bslot = lst_c["corr"][k]
    corr_ok &= (
        np.array_equal(items, want_items)
        and np.array_equal(slot, s_full[want_items])
        and np.array_equal(w, w_full[want_items])
        and np.array_equal(bslot, scorer.base_slot[k][want_items])
    )
check(
    corr_ok,
    "the listener corrections are exactly the slots of "
    "gam.encode() that differ from the item-level slots",
)
check(
    worst_s < 1e-12,
    f"CatalogScorer's scores equal the dense reference " f"(largest gap {worst_s:.1e})",
)
check(
    rank_ok and tie_seen,
    "CatalogScorer's ranks equal a direct count under the P5 rule, and a "
    "planted three-way tie ranks the pick after the equal tracks with "
    "smaller ids",
)
check(chunk_ok, "the ranks do not depend on the chunk size")
lst0 = scorer.listener(state, space, 0.3)
r_in = scorer.ranks(lst0, [20, 21], hist_items)
check(
    raises(
        lambda: scorer.ranks(lst0, [int(hist_items[0]), 21], hist_items),
        text="train item",
    ),
    "ranks() refuses a pick that is a train item",
)
check(
    ct.reference_rank(np.array([0.0, 1.0, 1.0 + 1e-12, 2.0, 1.0]), 2, [3], 1e-9)
    == (1, 3),
    "reference_rank widens the range for a near tie",
)
check(
    ct.reference_rank(np.array([0.0, 1.0, 1.0, 2.0, 0.5]), 2, [], 1e-9) == (3, 3),
    "reference_rank without near ties is the exact P5 rank",
)

# 7b: ranking by slot ranges
# concept_range() against concept_catalog(), bit for bit, on ranges that
# start at 0, end at the last slot, hold one slot or cut through rows.
ranges = [
    (0, NI),
    (0, 1),
    (0, 37),
    (37, 38),
    (100, 163),
    (163, 164),
    (NI - 1, NI),
    (NI - 64, NI),
    (250, 250 + 129),
]
pre = scorer.prepare(state, space)
same_r = set(pre) == set(cn.CONTINUOUS) - {"popularity"}
for c in cn.CONTINUOUS:
    v1, h1 = ct.concept_catalog(state, space, c)
    for a_, b_ in ranges:
        v2, h2 = scorer.concept_range(state, space, pre, c, a_, b_)
        same_r &= v1[a_:b_].tobytes() == v2.tobytes() and np.array_equal(h1[a_:b_], h2)
check(
    same_r,
    "concept_range() equals concept_catalog() bit for bit on "
    "every range, one-slot ranges included, for every concept",
)

# chunk_scores() against score_range() on listener(), bit for bit.
same_s = True
for trial in range(4):
    hi_ = np.sort(
        rng.choice(np.arange(1, NI), int(rng.choice([1, 30, 120])), replace=False)
    )
    st_ = cn.build_user_state(
        hi_, np.log1p(rng.integers(1, 60, len(hi_)).astype(np.float64)), space
    )
    m_u = float(rng.normal())
    lst = scorer.listener(st_, space, m_u)
    pre_ = scorer.prepare(st_, space)
    for a_, b_ in ranges:
        same_s &= (
            scorer.chunk_scores(st_, space, pre_, m_u, a_, b_).tobytes()
            == scorer.score_range(lst, a_, b_).tobytes()
        )
check(
    same_s,
    "chunk_scores() equals score_range() on listener() bit for " "bit on every range",
)

# ranks_cut() against ranks() for many listeners, cuts and range sizes.
cut_ok, early, full_seen, same_pick_ok = True, False, False, True
n_cases = 0
for trial in range(14):
    hi_ = np.sort(
        rng.choice(np.arange(1, NI), int(rng.choice([1, 20, 90])), replace=False)
    )
    if trial == 0:
        hi_ = np.sort(np.setdiff1d(hi_, [10, 11, 12]))
    st_ = cn.build_user_state(
        hi_, np.log1p(rng.integers(1, 60, len(hi_)).astype(np.float64)), space
    )
    m_u = float(rng.normal())
    lst = scorer.listener(st_, space, m_u)
    cand = np.setdiff1d(np.arange(1, NI), hi_)
    picks = rng.choice(cand, 2, replace=False)
    if trial == 0:
        picks = np.array([12, picks[0]])  # the planted tie
    if trial == 1:
        picks = np.array([picks[0], picks[0]])  # val_last == val_first
    if trial == 2:
        picks = np.array([cand[0], cand[-1]])  # the first and last ranges
    full_r = scorer.ranks(lst, picks, hi_)
    for cut_ in (1, 3, 25, 100, NI + 5):
        for ch_ in (1, 7, 64, 100, NI):
            got, scored = scorer.ranks_cut(st_, space, m_u, picks, hi_, cut_, chunk=ch_)
            want = np.minimum(full_r, cut_ + 1)
            cut_ok &= np.array_equal(got, want)
            n_cases += 1
            early |= scored < NI and ch_ < NI
            full_seen |= scored == NI
            if cut_ > NI:
                cut_ok &= scored == NI  # no stop without a cut
    same_pick_ok &= trial != 1 or np.array_equal(full_r[:, 0], full_r[:, 1])
check(
    cut_ok and same_pick_ok,
    f"ranks_cut() equals ranks() censored at the cut in {n_cases} cases: "
    f"cuts 1 to beyond the catalog, ranges of 1 to {NI} slots, a planted "
    f"tie, two equal picks, picks in the first and last ranges",
)
check(
    early and full_seen,
    "ranks_cut() stops early when every pick ranks beyond the cut and "
    "scores the whole catalog when a pick ranks within it",
)
check(
    raises(
        lambda: scorer.ranks_cut(
            state, space, 0.3, [int(hist_items[0]), 21], hist_items, 100
        ),
        text="train item",
    )
    and raises(
        lambda: scorer.ranks_cut(state, space, 0.3, [20, 21], hist_items, 0),
        text="positive",
    ),
    "ranks_cut() refuses a train-item pick and a cut below 1",
)

print()
print(f"{len(failures)} failure(s)")
raise SystemExit(1 if failures else 0)
