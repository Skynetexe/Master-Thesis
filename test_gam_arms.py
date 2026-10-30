"""Checks for pipeline/gam_arms.py on synthetic data. Nothing touches the
model folder.

1. BPR: the block loss and its score gradient equal a direct sum over
   the (positive, negative) pairs, the table gradient and the table
   Hessian equal central finite differences, the Hessian equals the
   explicit sum of c (x_p - x_j)(x_p - x_j)' and is positive
   semidefinite, the objective is convex along random segments, and the
   loss ignores a constant added to a whole term.
2. The order of the zero state: the bound of each monotone term's free
   zero constant, no bound for a tied or a free-shape term, and the
   refusal of a monotone zero-state term whose first knot is not 0.
3. monotone_violation(): planted breaks of the order, measured exactly.
4. theta_from_tables(): exact recovery without bounds, clipping with them.
5. fit_bounded(): the projected Newton steps reach the KKT point that
   L-BFGS-B with the Newton polish reaches from another start, the
   bounds bind, and with too few steps the fit falls back to L-BFGS-B
   and still reaches the same point. active_share() counts the flat
   segments. The fingerprint changes with the bounds.

Usage, from the repository root:
    python3 test_gam_arms.py
"""

import math
import os

import numpy as np

from pipeline import gam
from pipeline import gam_arms as ga_

failures = []


def check(cond, msg):
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        failures.append(msg)
        if os.environ.get("FAIL_FAST"):  # FAIL_FAST=1: stop at the first failed check
            raise SystemExit(1)


def raises(fn, exc=ValueError, text=""):
    try:
        fn()
    except exc as e:
        return text in str(e)
    except Exception as e:  # reported, then False
        print(f"      unexpected {type(e).__name__}: {e}")
        return False
    return False


rng = np.random.default_rng(20260928)

# Synthetic training data
U, M_NEG = 180, 6
npos = rng.integers(1, 4, U)
lens = npos + M_NEG
N = int(lens.sum())
starts = np.concatenate(([0], np.cumsum(lens)[:-1]))
row_user = np.repeat(np.arange(U), lens)
pos_rows = np.concatenate([starts[u] + np.arange(npos[u]) for u in range(U)])
pos_user = row_user[pos_rows]
neg_rows = (starts + npos)[:, None] + np.arange(M_NEG)[None, :]
is_pos = np.zeros(N, dtype=bool)
is_pos[pos_rows] = True
pos_weight = (1.0 / npos[pos_user]) / U

cols = {}
v = np.abs(rng.normal(size=N)) * (rng.random(N) > 0.3)  # zeros with data
h = rng.random(N) > 0.1
v[~h] = np.nan
cols["up"] = (v, h)
v = np.abs(rng.normal(size=N)) * (rng.random(N) > 0.4)
h = rng.random(N) > 0.2
v[~h] = np.nan
cols["down"] = (v, h)
cols["pop"] = (rng.gamma(2.0, 2.0, N), np.ones(N, dtype=bool))
m_row = rng.normal(size=U)[row_user]
# A planted preference that falls with "up" beyond 1.5 and rises with
# "down": the free fit breaks both orders, so the bounds bind.
x_up = np.nan_to_num(cols["up"][0])
x_down = np.nan_to_num(cols["down"][0])
pref = np.where(x_up < 1.5, 1.2 * x_up, 1.8 - 2.0 * (x_up - 1.5))
pref += 0.8 * x_down
# Positives: move each positive row's values toward high preference by
# swapping in the best of a few candidate rows.
for p in pos_rows:
    cand = rng.integers(0, N, 6)
    best = cand[np.argmax(pref[cand])]
    for key in ("up", "down", "pop"):
        vv, hh = cols[key]
        vv[p], hh[p] = vv[best], hh[best]
    x_up[p], x_down[p] = x_up[best], x_down[best]


def mk_knots(src, zero=False, levels=7):
    vv, hh = cols[src]
    m = hh & ((vv != 0) if zero else True)
    return gam.knot_rule(
        vv[m & is_pos],
        vv[m & ~is_pos],
        np.linspace(0.05, 0.95, levels),
        (0.1, 0.5, 0.9),
    )


terms = [
    gam.Term(
        "up",
        "up",
        mk_knots("up", True),
        zero_state=True,
        missing_state=True,
        monotone=1,
    ),
    gam.Term(
        "down",
        "down",
        mk_knots("down", True),
        zero_state=True,
        missing_state=True,
        monotone=-1,
    ),
    gam.Term("pop", "pop", mk_knots("pop")),
    gam.Term("tilt", "pop", mk_knots("pop"), multiplier="mainstreaminess"),
]
enc = [gam.encode(t, *cols[t.source]) for t in terms]
slot = np.stack([e[0] for e in enc])
wmat = np.stack([e[1] for e in enc])
ref = ~is_pos
data_means = {}
for k, t in enumerate(terms):
    if t.missing_state:
        data_means[t.name] = gam.basis_mean(
            t, slot[k], wmat[k], ref & (slot[k] < t.K + 1)
        )
n_zero = {t.name: int((slot[k] == t.K).sum()) for k, t in enumerate(terms)}
n_miss = {t.name: int((slot[k] == t.K + 1).sum()) for k, t in enumerate(terms)}
check(
    n_zero["up"] > 0
    and n_zero["down"] > 0
    and terms[0].knots[0] == 0.0
    and terms[1].knots[0] == 0.0,
    "the synthetic data hold exact zeros, and both zero-state terms have "
    "their first knot at 0",
)
layout = gam.Layout(terms, n_zero, n_miss, data_means=data_means)
L = layout.L
offset = np.where(is_pos, 0.0, rng.normal(0.5, 1.0, N))
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
offs = layout.offsets

# 1: BPR
th = rng.normal(size=layout.P)
T = layout.tables(th)
D = gam.diff_table(T, terms, offs)
s = gam.block_scores(T, D, blk)
loss_f, g_f = ga_.bpr_block(s, pos_rows, pos_user, pos_weight, neg_rows)
terms_b = []
for j, (p, u) in enumerate(zip(pos_rows, pos_user)):
    for r in neg_rows[u]:
        terms_b.append(pos_weight[j] / M_NEG * math.log1p(math.exp(s[r] - s[p])))
loss_b = math.fsum(terms_b)
check(
    abs(loss_f - loss_b) < 1e-13 * max(1.0, loss_b),
    f"bpr_block equals a direct sum over the (positive, negative) pairs "
    f"({loss_f:.12f} against {loss_b:.12f})",
)
g_b = np.zeros(N)
for j, (p, u) in enumerate(zip(pos_rows, pos_user)):
    for r in neg_rows[u]:
        sg = 1.0 / (1.0 + math.exp(-(s[r] - s[p])))
        g_b[r] += pos_weight[j] / M_NEG * sg
        g_b[p] -= pos_weight[j] / M_NEG * sg
check(
    np.max(np.abs(g_f - g_b)) < 1e-15,
    "bpr_block's score gradient equals the direct sum of the sigmoid "
    "terms, the shared negatives included",
)
big = np.array([800.0, -800.0, 0.0])
loss_big, g_big = ga_.bpr_block(
    big, np.array([0, 1]), np.array([0, 1]), np.array([1.0, 1.0]), np.array([[2], [2]])
)
check(
    np.isfinite(loss_big)
    and np.all(np.isfinite(g_big))
    and abs(loss_big - 800.0) < 1e-9,
    "bpr_block stays finite for score gaps of 800 (softplus and sigmoid "
    "without overflow)",
)

loss_t, grad_t = ga_.bpr_block_loss_grad(T, D, blk, L)
eps = 1e-6
fd = np.zeros(L)
for i in range(L):
    e = np.zeros(L)
    e[i] = eps
    lp = ga_.bpr_block_loss_grad(T + e, gam.diff_table(T + e, terms, offs), blk, L)[0]
    lm = ga_.bpr_block_loss_grad(T - e, gam.diff_table(T - e, terms, offs), blk, L)[0]
    fd[i] = (lp - lm) / (2 * eps)
check(
    abs(loss_t - loss_f) < 1e-15
    and np.max(np.abs(fd - grad_t)) < 1e-8 * max(1.0, np.max(np.abs(grad_t))),
    "bpr_block_loss_grad: the loss equals bpr_block's and the table "
    "gradient equals central finite differences, with the tilt "
    "multiplier",
)
H_t = ga_.bpr_block_hessian(T, D, blk, L)
fd_h = np.zeros((L, L))
for i in range(L):
    e = np.zeros(L)
    e[i] = eps
    gp = ga_.bpr_block_loss_grad(T + e, gam.diff_table(T + e, terms, offs), blk, L)[1]
    gm = ga_.bpr_block_loss_grad(T - e, gam.diff_table(T - e, terms, offs), blk, L)[1]
    fd_h[:, i] = (gp - gm) / (2 * eps)
check(
    np.max(np.abs(fd_h - H_t)) < 1e-7 * max(1.0, np.max(np.abs(H_t))),
    "bpr_block_hessian equals central finite differences of the table " "gradient",
)
X = gam.block_design(blk, L).toarray()
H_ex = np.zeros((L, L))
for j, (p, u) in enumerate(zip(pos_rows, pos_user)):
    for r in neg_rows[u]:
        d_ = s[p] - s[r]
        sg = 1.0 / (1.0 + math.exp(-d_))
        c_ = pos_weight[j] / M_NEG * sg * (1.0 - sg)
        dx = X[p] - X[r]
        H_ex += c_ * np.outer(dx, dx)
ev = np.linalg.eigvalsh(H_t)
check(
    np.max(np.abs(H_ex - H_t)) < 1e-13 * max(1.0, np.max(np.abs(H_ex)))
    and np.array_equal(H_t, H_t.T)
    and ev.min() > -1e-12,
    f"bpr_block_hessian equals the explicit sum of c (x_p - x_j)(x_p - "
    f"x_j)' over the pairs, is symmetric and positive semidefinite "
    f"(smallest eigenvalue {ev.min():.1e})",
)


def bpr_lg(T_, D_):
    return ga_.bpr_block_loss_grad(T_, D_, blk, L)


def bpr_h(T_, D_):
    return ga_.bpr_block_hessian(T_, D_, blk, L)


free_terms = [
    gam.Term(t.name, t.source, t.knots, t.zero_state, t.missing_state, t.multiplier)
    for t in terms
]
lay_free = gam.Layout(free_terms, n_zero, n_miss, data_means=data_means)
obj_bpr = gam.Objective(lay_free, bpr_lg, 0.03, 1e-6, bpr_h)
worst = 0.0
for _ in range(20):
    a, b = rng.normal(size=lay_free.P), rng.normal(size=lay_free.P)
    lam = rng.random()
    worst = max(
        worst,
        obj_bpr(lam * a + (1 - lam) * b)[0]
        - (lam * obj_bpr(a)[0] + (1 - lam) * obj_bpr(b)[0]),
    )
th_f = rng.normal(size=lay_free.P)
H_obj = obj_bpr.hessian(th_f)
fd_obj = np.array(
    [
        (obj_bpr(th_f + eps * e)[1] - obj_bpr(th_f - eps * e)[1]) / (2 * eps)
        for e in np.eye(lay_free.P)
    ]
).T
check(
    worst <= 1e-12
    and np.max(np.abs(fd_obj - H_obj)) < 1e-7 * max(1.0, np.max(np.abs(H_obj))),
    "the BPR objective is convex along 20 random segments, and its "
    "parameter Hessian equals finite differences of its gradient",
)
T_shift = T.copy()
off_p, tp = offs[2], terms[2]
T_shift[off_p : off_p + tp.K + 2] += 2.3
L_shift = ga_.bpr_block_loss_grad(
    T_shift, gam.diff_table(T_shift, terms, offs), blk, L
)[0]
check(
    abs(L_shift - loss_t) < 1e-13,
    "the BPR loss ignores a constant added to a whole term (only "
    "differences between a listener's rows count)",
)

# 2: zero-state order
lay_o = gam.Layout(terms, n_zero, n_miss, data_means=data_means)
before = lay_o.bounds
fp_before = ga_.layout_fingerprint(lay_o)
ga_.order_zero_states(lay_o)
lab = {x: k for k, x in enumerate(lay_o.labels)}
changed = [k for k in range(lay_o.P) if lay_o.bounds[k] != before[k]]
check(
    lay_o.bounds[lab["up:zero"]] == (None, 0.0)
    and lay_o.bounds[lab["down:zero"]] == (0.0, None)
    and sorted(changed) == sorted([lab["up:zero"], lab["down:zero"]])
    and lay_o.zero_order is True,
    "order_zero_states bounds the free zero constant of a non-decreasing "
    "term at or below 0 and of a non-increasing one at or above 0, and "
    "changes no other bound",
)
check(
    ga_.layout_fingerprint(lay_o) != fp_before
    and lay_o.fingerprint()
    == gam.Layout(terms, n_zero, n_miss, data_means=data_means).fingerprint(),
    "layout_fingerprint() changes with the bounds, which "
    "Layout.fingerprint() leaves out",
)
n_zero_tied = dict(n_zero, up=0)
lay_t = ga_.order_zero_states(
    gam.Layout(terms, n_zero_tied, n_miss, data_means=data_means)
)
check(
    "up:zero" not in lay_t.labels
    and lay_t.bounds[list(lay_t.labels).index("down:zero")] == (0.0, None),
    "a tied zero state gets no bound (it equals the first knot height " "already)",
)
t_bad = gam.Term("bad", "up", (0.1, 0.5, 1.0), zero_state=True, monotone=1)
lay_bad = gam.Layout([t_bad], {"bad": 5}, {})
check(
    raises(lambda: ga_.order_zero_states(lay_bad), text="first knot"),
    "order_zero_states refuses a monotone zero-state term whose first " "knot is not 0",
)
lay_fr = gam.Layout(free_terms, n_zero, n_miss, data_means=data_means)
check(
    ga_.order_zero_states(lay_fr).bounds
    == tuple((None, None) for _ in range(lay_fr.P)),
    "order_zero_states leaves the layout without monotone terms free",
)

# 3: monotone_violation
T_ok = lay_o.tables(
    np.abs(rng.normal(size=lay_o.P))
    * np.array(
        [1.0 if b[0] == 0.0 else -1.0 if b[1] == 0.0 else 1.0 for b in lay_o.bounds]
    )
)
check(
    ga_.monotone_violation(T_ok, lay_o) == 0.0,
    "monotone_violation is 0 for tables from parameters within the bounds",
)
T_b = T_ok.copy()
o_up, t_up = offs[0], terms[0]
T_b[o_up + 3] = T_b[o_up + 4] + 0.25  # a drop of 0.25 at knot 3
T_b2 = T_ok.copy()
T_b2[o_up + t_up.K + gam.ZERO] = 0.4  # zero above h0 = 0
o_dn, t_dn = offs[1], terms[1]
T_b3 = T_ok.copy()
T_b3[o_dn + 2] = T_b3[o_dn + 1] + 0.125  # a rise in a falling term
T_b4 = T_ok.copy()
T_b4[o_dn + t_dn.K + gam.ZERO] = -0.3  # zero below h0 = 0
vals = [ga_.monotone_violation(x_, lay_o) for x_ in (T_b, T_b2, T_b3, T_b4)]
check(
    abs(vals[0] - 0.25) < 1e-15
    and abs(vals[1] - 0.4) < 1e-15
    and abs(vals[2] - 0.125) < 1e-15
    and abs(vals[3] - 0.3) < 1e-15,
    f"monotone_violation measures each planted break exactly: a drop, a "
    f"zero state above the first knot, a rise in a falling term, a zero "
    f"state below it ({', '.join(f'{x_:.3f}' for x_ in vals)})",
)

# 4: theta_from_tables
th_r = rng.normal(size=lay_fr.P)
check(
    np.max(np.abs(ga_.theta_from_tables(lay_fr, lay_fr.tables(th_r)) - th_r)) < 1e-12,
    "theta_from_tables recovers the parameters of tables from the layout",
)
th_o = ga_.theta_from_tables(lay_o, lay_fr.tables(th_r))
check(
    all(
        (lo is None or x_ >= lo) and (hi is None or x_ <= hi)
        for x_, (lo, hi) in zip(th_o, lay_o.bounds)
    )
    and ga_.monotone_violation(lay_o.tables(th_o), lay_o) == 0.0,
    "theta_from_tables clips to the bounds, so a free fit gives a "
    "feasible monotone start",
)

# 5: fit_bounded


def sm_lg(T_, D_):
    return gam.block_loss_grad(T_, D_, blk, L)


def sm_h(T_, D_):
    return gam.block_hessian(T_, D_, blk, L)


settings = {
    "method": "newton",
    "newton_steps": 60,
    "newton_target": 1e-12,
    "newton_accept": 1e-9,
    "maxcor": 20,
    "ftol": 1e-14,
    "gtol": 1e-10,
    "maxiter": 5000,
    "maxfun": 10000,
    "pg_target": 1e-9,
    "restarts": 3,
    "bounded_newton_steps": 200,
}
obj_o = gam.Objective(lay_o, sm_lg, 0.01, 1e-6, sm_h)
obj_fr = gam.Objective(lay_fr, sm_lg, 0.01, 1e-6, sm_h)
free_fit = gam.fit(obj_fr, np.zeros(lay_fr.P), settings)
T_free = lay_fr.tables(free_fit["theta"])
viol_free = ga_.monotone_violation(T_free, lay_o)
res = ga_.fit_bounded(obj_o, ga_.theta_from_tables(lay_o, T_free), settings)
res2 = gam.fit(obj_o, rng.normal(0.0, 0.3, lay_o.P), dict(settings, method="lbfgsb"))
T1, T2 = lay_o.tables(res["theta"]), lay_o.tables(res2["theta"])
F1 = obj_o(res["theta"])[0]
F2 = obj_o(res2["theta"])[0]
print(
    f"      free fit breaks the orders by {viol_free:.3f}. Projected "
    f"Newton: {res['message']}, projected gradient "
    f"{res['proj_grad_max']:.1e}. L-BFGS-B: {res2['nit']} iterations, "
    f"{res2['newton_steps']} Newton steps, projected gradient "
    f"{res2['proj_grad_max']:.1e}"
)
check(
    viol_free > 0.05
    and res["method"] == "projected newton"
    and res["proj_grad_max"] <= 1e-9,
    "the free fit breaks the orders, and the projected Newton steps "
    "reach a stationary point of the bounded problem",
)
check(
    np.max(np.abs(T1 - T2)) < 1e-7
    and F1 <= F2 + 1e-12 * abs(F2)
    and res2["proj_grad_max"] <= 1e-9,
    f"projected Newton and L-BFGS-B with the Newton polish, from "
    f"another start, reach the same tables (largest difference "
    f"{np.max(np.abs(T1 - T2)):.1e}) and projected Newton's objective is "
    f"not higher",
)
grad1 = obj_o(res["theta"])[1]
kkt_ok = True
for x_, g_, (lo, hi) in zip(res["theta"], grad1, lay_o.bounds):
    if lo is not None and x_ <= lo:
        kkt_ok &= g_ >= -1e-9  # at the lower bound: no pull down
    elif hi is not None and x_ >= hi:
        kkt_ok &= g_ <= 1e-9
    else:
        kkt_ok &= abs(g_) <= 1e-9
check(
    kkt_ok and ga_.monotone_violation(T1, lay_o) == 0.0,
    "the bounded fit meets the KKT conditions parameter by parameter and "
    "keeps both orders and both zero-state bounds",
)
flat = ga_.active_share(res["theta"], lay_o)
n_inc = {t.name: t.K - 1 for t in terms if t.monotone}
at0 = {
    t.name: sum(
        1
        for k, lb in enumerate(lay_o.labels)
        if lb.startswith(f"{t.name}:inc") and res["theta"][k] == 0.0
    )
    for t in terms
    if t.monotone
}
check(
    set(flat) == {"up", "down"}
    and all(abs(flat[n] - at0[n] / n_inc[n]) < 1e-15 for n in flat)
    and max(flat.values()) > 0.0,
    f"active_share counts the increments held at the bound "
    f"({ {k: round(v, 3) for k, v in flat.items()} })",
)
res3 = ga_.fit_bounded(
    obj_o, ga_.theta_from_tables(lay_o, T_free), dict(settings, bounded_newton_steps=1)
)
x_1, rec_1 = gam.newton_polish(
    obj_o, ga_.theta_from_tables(lay_o, T_free), 1, settings["newton_target"]
)
ref3 = gam.fit(obj_o, x_1, dict(settings, method="lbfgsb"))
check(
    res3["method"] == "lbfgsb"
    and rec_1["steps"] == 1
    and res3["newton_steps"] == rec_1["steps"] + ref3["newton_steps"]
    and np.array_equal(res3["theta"], ref3["theta"])
    and np.max(np.abs(lay_o.tables(res3["theta"]) - T1)) < 1e-7,
    "with one projected Newton step the fit falls back to L-BFGS-B from "
    "that step, counts both kinds of Newton steps, and reaches the same "
    "tables",
)
x_out = ga_.theta_from_tables(lay_o, T_free)
x_out[[k for k, b in enumerate(lay_o.bounds) if b[0] == 0.0][:3]] = -5.0
res4 = ga_.fit_bounded(obj_o, x_out, settings)
check(
    np.max(np.abs(lay_o.tables(res4["theta"]) - T1)) < 1e-7,
    "fit_bounded clips an infeasible start and reaches the same tables",
)
# The hardest infeasible start: the optimum of the same parameters without
# bounds, where the gradient is near 0, so no step from it lowers the
# objective. Only the clip of the start leads to the bounded optimum.
lay_u = gam.Layout(terms, n_zero, n_miss, data_means=data_means)
lay_u.bounds = tuple((None, None) for _ in lay_u.bounds)
obj_u = gam.Objective(lay_u, sm_lg, 0.01, 1e-6, sm_h)
x_unc = gam.fit(obj_u, np.zeros(lay_u.P), settings)["theta"]
n_out = sum(
    1
    for x_, (lo, hi) in zip(x_unc, lay_o.bounds)
    if (lo is not None and x_ < lo) or (hi is not None and x_ > hi)
)
res5 = ga_.fit_bounded(obj_o, x_unc, settings)
feasible5 = all(
    (lo is None or x_ >= lo) and (hi is None or x_ <= hi)
    for x_, (lo, hi) in zip(res5["theta"], lay_o.bounds)
)
check(
    n_out > 0 and feasible5 and np.max(np.abs(lay_o.tables(res5["theta"]) - T1)) < 1e-7,
    f"from the optimum without bounds ({n_out} parameters outside their "
    f"bounds, gradient near 0), fit_bounded still reaches the bounded "
    f"optimum",
)

# Summary
print()
if failures:
    print(f"{len(failures)} check(s) FAILED:")
    for f in failures:
        print("  - " + f)
    raise SystemExit(1)
print("all checks passed")
