"""Smoke test of pipeline/tree_audit.py on
the P13 chain of p13_chain.py. Every path points into a temporary folder,
and nothing touches the real model folder.

1. Units without the chain: the diagonal Hessian against central
   differences of the gradient, the depth-1 tree against a brute-force
   search (free and tied missing state), the step layout at the edges, the
   steps in chunks against one pass per block, and the pilot's power-law
   fit on a known curve.
2. The chain with the runner's markers, as test_runner_p13.py makes them:
   gam_refit, p13c_stability, a forced bounded p13d_bounds (as
   test_p13d.py forces it) and p13c_final.
3. The audit with small settings: the pilot (saved, no outputs), the full
   run after it, a run with 1 worker from scratch, and a run with 2
   workers stopped inside a bag and resumed. All three give the same bags.
4. T1 to T4 computed again here from the stored outputs and P13c's files,
   the early-stop rule replayed from the stored curves, the read-back
   against a corrupted output, the report, and check 6 against a stale
   marker.

Usage, from the repository root:
    python3 test_tree_audit.py
With P13B2_BASE_CACHE=<folder> and P13_CHAIN_CACHE=<folder> the chain's
outputs are reused while their code stays the same.
With KEEP_TMP=1 the temporary folder stays, for a look at the outputs.
"""

import json
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import p13_chain as C
import p13_world as W
from pipeline import common
from pipeline import config
from pipeline import gam
from pipeline import gam_knn_diag as G
from pipeline import gam_refit as GR
from pipeline import itemknn_select as IK
from pipeline import p13c as PC
from pipeline import p13d as D
from pipeline import runner as RN
from pipeline import stage_p13b2 as s13b2
from pipeline import tree_audit as TA

failures = []
lines = []
T0 = time.time()
PARAMS = dict(config.DEFAULT_PARAMS)
UPSTREAM = ("p6_als", "p9b_colisten", "p13b2_loss")
SIX = [
    "itemknn_select",
    "gam_knn_slices",
    "gam_refit",
    "p13c_stability",
    "p13d_bounds",
    "p13c_final",
]


def check(cond, msg):
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        failures.append(msg)


def quiet(msg):
    pass


def keep(msg):
    lines.append(msg)


def clock():
    return f"{time.time() - T0:.0f} s"


# 1: units without the chain
rng = np.random.default_rng(7)
U, m = 5, 7
n_pos = np.array([1, 2, 1, 2, 1])
pu = np.repeat(np.arange(U), n_pos)
w = (1.0 / n_pos)[pu] * np.array([1, 1, 0, 1, 1, 1, 1], dtype=float)
zp0, zn0 = rng.normal(size=len(pu)), rng.normal(size=(U, m))


def loss_of(zp, zn):
    _, _, _, lse = TA._softmax(zp, zn, pu)
    return float(np.dot(w, lse - zp))


_, gp, hp, g, h = TA._grad(zp0.copy(), zn0.copy(), pu, w)
eps, worst = 1e-5, 0.0
for i in range(len(pu)):
    a, b = zp0.copy(), zp0.copy()
    a[i] += eps
    b[i] -= eps
    ga_ = TA._grad(a, zn0.copy(), pu, w)[1][i]
    gb_ = TA._grad(b, zn0.copy(), pu, w)[1][i]
    worst = max(worst, abs((ga_ - gb_) / (2 * eps) - hp[i]))
    num = (loss_of(a, zn0) - loss_of(b, zn0)) / (2 * eps)
    worst = max(worst, abs(num - gp[i]))
for u in range(U):
    for k in range(m):
        a, b = zn0.copy(), zn0.copy()
        a[u, k] += eps
        b[u, k] -= eps
        ga_ = TA._grad(zp0.copy(), a, pu, w)[3][u, k]
        gb_ = TA._grad(zp0.copy(), b, pu, w)[3][u, k]
        worst = max(worst, abs((ga_ - gb_) / (2 * eps) - h[u, k]))
        num = (loss_of(zp0, a) - loss_of(zp0, b)) / (2 * eps)
        worst = max(worst, abs(num - g[u, k]))
check(
    worst < 1e-8,
    f"1: the gradient and the diagonal Hessian equal central "
    f"differences of the loss and the gradient (largest gap "
    f"{worst:.1e})",
)


def brute(G, H, C, n, zero_free, miss_free, a, lr):
    """An independent depth-1 tree: every split by its own sums."""
    tied = a is not None
    Gm, Hm = G[n + 1], H[n + 1]
    best, delta = None, np.zeros(n + 2)
    for s in range(1, n):
        L, R = range(0, s), range(s, n)
        gl, hl = sum(G[i] for i in L), sum(H[i] for i in L)
        gr_, hr = sum(G[i] for i in R), sum(H[i] for i in R)
        if tied:
            al, ar = sum(a[i] for i in L), sum(a[i] for i in R)
            gl, hl = gl + al * Gm, hl + al * al * Hm
            gr_, hr = gr_ + ar * Gm, hr + ar * ar * Hm
        if (
            sum(C[i] for i in L) < TA.MIN_SAMPLES_LEAF
            or sum(C[i] for i in R) < TA.MIN_SAMPLES_LEAF
            or hl < TA.MIN_HESSIAN
            or hr < TA.MIN_HESSIAN
        ):
            continue
        gain = gl * gl / hl + gr_ * gr_ / hr
        if best is None or gain > best[0] + 1e-12 * abs(best[0]):
            best = (gain, s, -lr * gl / hl, -lr * gr_ / hr)
    if best is not None:
        delta[: best[1]] = best[2]
        delta[best[1] : n] = best[3]
    if zero_free and C[n] >= TA.MIN_SAMPLES_LEAF:
        gz, hz = G[n], H[n]
        if tied:
            gz, hz = gz + a[n] * Gm, hz + a[n] * a[n] * Hm
        if hz >= TA.MIN_HESSIAN:
            delta[n] = -lr * gz / hz
    if tied:
        delta[n + 1] = sum(a[i] * delta[i] for i in range(n + 1))
    elif miss_free and C[n + 1] >= TA.MIN_SAMPLES_LEAF and Hm >= TA.MIN_HESSIAN:
        delta[n + 1] = -lr * Gm / Hm
    return delta, (None if best is None else best[1])


ok = True
for trial in range(40):
    n = int(rng.integers(1, 12))
    G_ = rng.normal(size=n + 2)
    H_ = rng.uniform(0.01, 2.0, size=n + 2)
    C_ = rng.integers(0, 12, size=n + 2)
    tied = trial % 2 == 1
    a = None
    if tied:
        a = rng.uniform(size=n + 2)
        a[n + 1] = 0.0
        a /= a.sum()
    d1, s1 = TA.tree_step(G_, H_, C_, n, True, not tied, a, lr=0.1)
    d2, s2 = brute(G_, H_, C_, n, True, not tied, a, 0.1)
    ok &= s1 == s2 and np.allclose(d1, d2, rtol=1e-12, atol=1e-15)
check(
    ok,
    "1: the depth-1 tree equals a brute-force search in 40 random "
    "cases (free and tied missing states, the leaf-size and Hessian "
    "limits)",
)

t = gam.Term("x", "tag_affinity", (0.0, 1.0), zero_state=True, missing_state=True)
e = np.array([0.1, 0.25, 0.5])
lay1 = TA.StepLayout([t], [e])
vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
x = np.array([0.05, 0.1, np.nextafter(0.1, 1.0), 0.2, 0.25, 0.3, 0.5, 0.9, 0.0, 0.7])
hx = np.array([True] * 9 + [False])
slot, ww = gam.encode(lay1.terms[0], x, hx)
T1 = lay1.term_table(0, vals)
got = gam.pwl_value(
    T1, gam.diff_table(T1, lay1.terms, lay1.offsets), slot.astype(np.int64), ww
)
want = vals[TA._bins(t, e, x, hx).astype(np.int64)]
check(
    np.array_equal(got, want)
    and list(TA._bins(t, e, x, hx)) == [0, 0, 1, 1, 1, 2, 2, 3, 4, 5],
    "1: the step layout gives each value its bin's value: an edge belongs "
    "to the bin below it, the next float to the bin above, 0 to the zero "
    "state and no data to the missing state",
)

# chunks: steps in chunks of 3 listeners (a short last chunk) against one
# chunk per block, and both against one _grad pass over each block's scores
# computed again from the changed tables
rng2 = np.random.default_rng(11)
Uc, mc, Jc, NBc = 23, 5, 2, 9
npc = 1 + rng2.integers(0, 3, Uc)
Pc = int(npc.sum())
psc = np.concatenate(([0], np.cumsum(npc))).astype(int)
Wc = {
    "ranges": [(0, 11, 0, int(psc[11])), (11, Uc, int(psc[11]), Pc)],
    "n_pos": npc.astype(np.int64),
    "pos_user": np.repeat(np.arange(Uc), npc).astype(np.int64),
    "ok_pos": rng2.random(Pc) > 0.2,
    "off_pos": rng2.normal(size=Pc),
    "off_neg": rng2.normal(size=(Uc, mc)),
    "bins_pos": rng2.integers(0, NBc, (Jc, Pc)).astype(np.uint16),
    "bins_neg": rng2.integers(0, NBc, (Jc, Uc, mc)).astype(np.uint16),
    "nb": [NBc] * Jc,
    "J": Jc,
}
TA._W.clear()
TA._W.update(Wc)
valc = rng2.random(Uc) < 0.3
tabc = [rng2.normal(size=NBc) for _ in range(Jc)]
dc = rng2.normal(scale=0.1, size=NBc)


def chunk_steps(chunk):
    TA.CHUNK = chunk
    S_ = TA._new_state()
    for b_ in range(2):
        TA._bag_block(S_, b_, valc)
        TA._set_block(S_, b_, tabc)
    return [TA._step_block(S_, b_, 0, dc, 1) for b_ in range(2)], S_


out3, S3 = chunk_steps(3)
out_one, S_one = chunk_steps(10**6)
gap = 0.0
for b_, (l0, l1, p0, p1) in enumerate(Wc["ranges"]):
    tab2 = [tabc[0] + dc, tabc[1]]
    zp = Wc["off_pos"][p0:p1] + sum(
        tab2[j][Wc["bins_pos"][j, p0:p1]] for j in range(Jc)
    )
    zn = Wc["off_neg"][l0:l1] + sum(
        tab2[j][Wc["bins_neg"][j, l0:l1]] for j in range(Jc)
    )
    pu_ = Wc["pos_user"][p0:p1] - l0
    w_ = (1.0 / npc[l0:l1])[pu_] * Wc["ok_pos"][p0:p1]
    vb_ = valc[l0:l1][pu_]
    wt_, wv_ = np.where(vb_, 0.0, w_), np.where(vb_, w_, 0.0)
    lse_, gp_, hp_, g_, h_ = TA._grad(zp, zn.copy(), pu_, wt_)
    Gr = np.bincount(Wc["bins_neg"][1, l0:l1].ravel(), g_.ravel(), NBc) + np.bincount(
        Wc["bins_pos"][1, p0:p1], gp_, NBc
    )
    Hr = np.bincount(Wc["bins_neg"][1, l0:l1].ravel(), h_.ravel(), NBc) + np.bincount(
        Wc["bins_pos"][1, p0:p1], hp_, NBc
    )
    ref_ = (Gr, Hr, float(np.dot(wt_, lse_ - zp)), float(np.dot(wv_, lse_ - zp)))
    for got_ in (out3[b_], out_one[b_]):
        for x_, y_ in zip(got_, ref_):
            gap = max(
                gap,
                float(np.max(np.abs(np.asarray(x_) - y_)))
                / max(1e-300, float(np.max(np.abs(y_)))),
            )
    gap = max(gap, float(np.max(np.abs(S3["zn"][b_] - zn))))
same_z = all(np.array_equal(S3["zn"][b_], S_one["zn"][b_]) for b_ in range(2))
check(
    gap < 1e-12 and same_z,
    f"1: steps in chunks of 3 listeners, one chunk per block, and one pass "
    f"over scores computed again agree (largest relative gap {gap:.1e})",
)

L_inf, A_, a_ = 1.9, 2.0, 0.6
curve = [L_inf + A_ * (r + 1.0) ** -a_ for r in range(1000)]
fit = TA.decay_fit(curve)
g_stop = TA.TOLERANCE * curve[-1] / TA.PATIENCE
r_true = (g_stop / (A_ * a_)) ** (-1.0 / (1.0 + a_)) - 1.0 + TA.PATIENCE
check(
    fit is not None
    and abs(fit["slope"] + 1.0 + a_) < 0.01
    and abs(fit["stop"] / min(r_true, TA.MAX_ROUNDS) - 1.0) < 0.02
    and abs(fit["limit"] - L_inf) < 1e-3
    and abs(fit["loss_at"](5000) - (L_inf + A_ * 5001**-a_)) < 1e-4,
    f"1: the pilot's power-law fit recovers a known curve (slope "
    f"{fit['slope']:.3f}, stop {fit['stop']:,}, limit {fit['limit']:.4f})",
)

# 2: the chain and its markers
print(W.summary())
TMP = Path(tempfile.mkdtemp(prefix="test_tree_audit_"))
ROOT = TMP / "chain"
CHAIN = C.chain_into(ROOT)
SCR = ROOT / "scratch"
common.MODEL_DIR = CHAIN
NBOOT = {"listener": 6, "twoway": 3}


def stage():
    return C.p13_loaders(s13b2.StageP13b2())


def fake_marker(name):
    st = RN.STAGE_BY_NAME[name]
    common.write_marker(
        name,
        RN.stage_params_digest(st, PARAMS),
        {"test": name},
        {},
        source_sha=RN.stage_source_sha(st),
    )


def stages(model, scratch):
    ik = IK.StageItemKNN(model, scratch)
    ik._load = lambda: (
        W.NI,
        C.U_ARRAYS,
        W.SEL.copy(),
        W.SEL_PICKS.copy(),
        C.POP,
        C.LISTS,
    )
    sl = G.StageGamKnnSlices(model, scratch)
    sl._load = lambda: (W.NI, C.U_ARRAYS, C.LISTS, C.pop_centiles())
    out = [
        ik,
        sl,
        GR.StageGamRefit(model, scratch),
        PC.StageP13c("gam_refit", model, scratch),
        D.StageP13d(model, scratch),
        PC.StageP13c("p13d", model, scratch),
    ]
    for s in out[2:]:
        s._stage = stage
    out[3].n_boot, out[5].n_boot = NBOOT, {"listener": 4, "twoway": 2}
    out[3].n_verify = out[5].n_verify = 6
    return out


def execute(sts, **kw):
    by = dict(RN.STAGE_BY_NAME)
    by.update({s.name: s for s in sts})
    return RN.execute(sts, PARAMS, [], by_name=by, **kw)


for name in UPSTREAM:
    fake_marker(name)
IK.run(
    CHAIN,
    W.NI,
    C.U_ARRAYS,
    W.SEL.copy(),
    W.SEL_PICKS.copy(),
    C.POP,
    C.LISTS,
    workers=1,
    say=quiet,
    n_check=12,
    inputs=IK.provenance(CHAIN),
)
G.run_slices(
    CHAIN,
    W.NI,
    C.U_ARRAYS,
    C.LISTS,
    C.pop_centiles(),
    workers=1,
    say=quiet,
    n_check=5,
    inputs=G.provenance(CHAIN),
)
GR.run(
    stage(),
    CHAIN,
    SCR / "gam_refit",
    say=quiet,
    n_verify=6,
    inputs=GR.provenance(CHAIN),
)
PC.run(
    stage(),
    CHAIN,
    SCR / "p13c",
    say=quiet,
    n_verify=6,
    n_boot=NBOOT,
    inputs=PC.provenance(CHAIN),
)
STS = stages(CHAIN, SCR)
real_decide = D.decide


def forced(scores, names_, model_arm):
    res = real_decide(scores, names_, model_arm)
    res["s2"] = dict(res["s2"], chosen=D.PILOT)
    return res


D.decide = forced
try:
    D.run(
        stage(), CHAIN, SCR / "p13d", say=quiet, n_verify=6, inputs=D.provenance(CHAIN)
    )
    rc_a = execute(STS, adopt=True)
finally:
    D.decide = real_decide
rc_f = execute(STS)
ptr = json.loads((CHAIN / PC.FINAL_FILE).read_text())
check(
    rc_a == 0
    and rc_f == 0
    and ptr["model"]["changed"]
    and all(common.marker_path(n).exists() for n in SIX),
    f"2: the chain has current markers for the six stages and a bounded "
    f"final model ({clock()})",
)

# 3: the audit
TA.BAGS, TA.MAX_ROUNDS, TA.PATIENCE, TA.TOLERANCE = 2, 40, 6, 1e-9
TA.CHECKPOINT_ROUNDS, TA.PILOT_ROUNDS, TA.LEARNING_RATE = 5, 10, 0.05
TA.POS_LEVELS = tuple((i + 1) / 64 for i in range(63))
TA.NEG_LEVELS = tuple((i + 1) / 16 for i in range(15))
TA.N_BLOCKS, TA.WORKERS, TA.CHUNK = 4, 2, 7
INPUTS = TA.provenance(CHAIN)
WA = SCR / "tree_a"
pil = TA.run(
    stage(), CHAIN, WA, say=keep, n_verify=6, inputs=INPUTS, pilot=True, stages=STS
)
ck = json.loads((WA / "checkpoint_00.json").read_text())
check(
    pil["pilot_rounds"] == 10
    and ck["round"] == 10
    and len(ck["curve_val"]) == 10
    and not (CHAIN / TA.OUT_FILES[0]).exists()
    and any("pilot done" in ln for ln in lines),
    "3: the pilot saves 10 rounds of bag 1, prints its time and "
    "projection, and writes no output",
)
lines.clear()
out = TA.run(stage(), CHAIN, WA, say=keep, n_verify=6, inputs=INPUTS, stages=STS)
doc = json.loads((CHAIN / TA.OUT_FILES[0]).read_text())
z = dict(np.load(CHAIN / TA.OUT_FILES[1]))
check(
    any("continues at round 10" in ln for ln in lines)
    and len(doc["bags"]) == 2
    and doc["check"]["dense"]["listeners"] == 6
    and doc["check"]["markers_current"] is True,
    f"3: the full run continues bag 1 at round 10, fits 2 bags, and passes "
    f"the dense check on 6 listeners ({clock()})",
)
print(
    "      bags: "
    + ", ".join(
        f"{b['stop']} at {b['rounds']} (best " f"{b['best_round']})"
        for b in doc["bags"]
    )
)

TA.WORKERS = 1
TA.run(stage(), CHAIN, SCR / "tree_b", say=quiet, n_verify=6, inputs=INPUTS, stages=STS)
z_b = dict(np.load(CHAIN / TA.OUT_FILES[1]))
TA.WORKERS = 2
real_step = TA.tree_step
calls = {"n": 0}
R_STOP = min(int(doc["bags"][0]["rounds"]) - 1, 13)
J_TERMS = len(doc["model"]["kept"])


def stopping(*a, **k):
    calls["n"] += 1
    if calls["n"] == J_TERMS * R_STOP + 3:
        raise KeyboardInterrupt
    return real_step(*a, **k)


TA.tree_step = stopping
stopped = False
try:
    TA.run(
        stage(), CHAIN, SCR / "tree_c", say=quiet, n_verify=6, inputs=INPUTS, stages=STS
    )
except KeyboardInterrupt:
    stopped = True
finally:
    TA.tree_step = real_step
ck_path = SCR / "tree_c" / "checkpoint_00.json"
ck_round = json.loads(ck_path.read_text())["round"] if ck_path.exists() else 0
want_ck = (R_STOP // TA.CHECKPOINT_ROUNDS) * TA.CHECKPOINT_ROUNDS
TA.run(stage(), CHAIN, SCR / "tree_c", say=quiet, n_verify=6, inputs=INPUTS, stages=STS)
z_c = dict(np.load(CHAIN / TA.OUT_FILES[1]))
same = all(
    np.array_equal(z[k], z_b[k], equal_nan=True)
    and np.array_equal(z[k], z_c[k], equal_nan=True)
    for k in ("bag_values", "mean_values", "ranks", "curve_val", "tree_reason_value")
)
print(f"      stopped in round {R_STOP} of bag 1, resumed from round " f"{ck_round}")
check(
    stopped and ck_round == want_ck and same,
    f"3: the pilot then the full run (2 workers), a run from scratch (1 "
    f"worker) and a run stopped inside bag 1 and resumed (2 workers) give "
    f"the same bags, ranks and reasons ({clock()})",
)
z = z_c
doc = json.loads((CHAIN / TA.OUT_FILES[0]).read_text())

# 4: results computed again
base = stage()._base_context()
nb = [int(x) for x in z["nb"]]
n_edges = [int(x) for x in z["n_edges"]]
edges = np.split(z["edges"], np.cumsum(n_edges)[:-1])


def split(flat):
    return np.split(np.asarray(flat), np.cumsum(nb)[:-1])


mean = split(z["mean_values"])
bags = [split(r) for r in z["bag_values"]]
bgs = split(z["background"])
check(
    all(
        np.allclose(mean[j], np.mean([b[j] for b in bags], axis=0), rtol=0, atol=0)
        for j in range(len(nb))
    )
    and all(abs(float(x.sum()) - 1.0) < 1e-12 for x in bgs),
    "4: the model is the mean of the bags, and the background shares sum " "to 1",
)
names = doc["model"]["kept"]
M_ = PC.load_model(CHAIN, base)[0]
ok = True
pts_here = {}
for j, name in enumerate(names):
    xs = base["shape_q"][name]
    c = mean[j] - float(np.dot(bgs[j], mean[j]))
    vals = [float(c[np.searchsorted(edges[j], x, side="left")]) for x in xs[:5]]
    n_c = n_edges[j] + 1
    tj = next(t for t in M_["lay"].terms if t.name == name)
    vals.append(float(c[n_c]) if tj.zero_state else None)
    vals.append(float(c[n_c + 1]) if tj.missing_state else None)
    pts_here[name] = vals
    ok &= vals == doc["shapes"][name]["tree"]
check(
    ok,
    "4: the tree-GAM's centered points follow from the stored mean "
    "tables, edges and background",
)
res_final = json.loads((CHAIN / ptr["files"]["results"]).read_text())
ok, n_rows = True, 0
for row in doc["points"]["rows"]:
    ref = next(
        (
            r_
            for r_ in res_final["bootstrap"]["bands"][row["term"]]
            if r_["point"] == row["point"]
        ),
        None,
    )
    if ref is None:
        ok &= row["final"] is None
        continue
    n_rows += 1
    tv = row["tree"]
    lo, hi = ref["listener"]["lo"], ref["listener"]["hi"]
    sep = abs(tv - ref["value"]) > TA.TAU and (tv < lo or tv > hi)
    ok &= row["final"]["separates"] == sep and row["final"]["diff"] == tv - ref["value"]
check(
    ok and n_rows > 0,
    f"4: T1 against the final model's P13c bands "
    f"follows the rule on all {n_rows} points",
)
ok = True
for j, name in enumerate(names):
    xs = base["shape_q"][name]
    i5, i95 = np.searchsorted(edges[j], [xs[0], xs[4]], side="left")
    nets = [float(b[j][i95] - b[j][i5]) for b in bags]
    r_ = doc["falls"][name]
    ok &= (
        nets == r_["bag_net"]
        and r_["falls"] == all(x < 0 for x in nets)
        and r_["rises"] == all(x > 0 for x in nets)
    )
check(ok, "4: T2 follows from the stored bags")
ranks = z["ranks"].astype(float)
dcg = np.where(ranks <= 100, 1.0 / np.log2(ranks + 1.0), 0.0).mean(axis=-1)
fr = np.load(CHAIN / ptr["files"]["arrays"])["ranks"][0].astype(float)
dfin = np.where(fr <= 100, 1.0 / np.log2(fr + 1.0), 0.0).mean(axis=-1)
d_ = dcg[0] - dfin
sesoi = 0.02 * dfin.mean()
s4 = doc["scores"]
check(
    abs(s4["tree"] - dcg[0].mean()) < 1e-15
    and abs(s4["tree_minus_final"]["mean"] - d_.mean()) < 1e-15
    and abs(s4["sesoi"] - sesoi) < 1e-15
    and len(s4["bags"]) == 2,
    f"4: T4 follows from the stored ranks (tree-GAM {s4['tree']:.5f}, "
    f"final {s4['final']:.5f}, {s4['tree_minus_final']['decision']})",
)


def top3(val, cand):
    """Signed top 3 per pick, by value among the candidates, ties by the
    reason order (written here, not from p13c)."""
    G_, n = val.shape
    out = []
    for i in range(n):
        idx = [g for g in range(G_) if cand[g, i]]
        idx.sort(key=lambda g: (-val[g, i], g))
        out.append(frozenset((g, bool(val[g, i] > 0)) for g in idx[:3]))
    return out


S = PC.reason_sample(CHAIN, base)
M_fin = PC.load_p13d_model(CHAIN, base)[0]
R_fin = PC.Reasons(base, M_fin, S)
val_fin = R_fin.values(M_fin["tables"])[0]
a_t = top3(z["tree_reason_value"], R_fin.cand)
a_f = top3(val_fin, R_fin.cand)
agree = float(np.mean([x == y for x, y in zip(a_t, a_f)]))
check(
    abs(agree - doc["reasons"]["agreement"]["final"]) < 1e-15,
    f"4: T3's agreement with the final model ({agree:.1%}) equals an "
    f"independent count of equal signed top-3 sets",
)
ok = True
for b, rec in enumerate(doc["bags"]):
    c = z["curve_val"][b][~np.isnan(z["curve_val"][b])]
    best, br, since, stop_r = math.inf, -1, 0, None
    for r_, v in enumerate(c):
        if v < best - TA.TOLERANCE * abs(best) or br < 0:
            best, br, since = v, r_, 0
        else:
            since += 1
        if since >= TA.PATIENCE or r_ >= TA.MAX_ROUNDS:
            stop_r = r_
            break
    ok &= (
        stop_r == rec["rounds"]
        and br == rec["best_round"]
        and best == rec["best_loss"]
        and rec["stop"] == ("early" if since >= TA.PATIENCE else "cap")
    )
check(
    ok,
    "4: the early-stop rule, replayed on the stored curves, gives "
    "each bag's stop round, best round and best loss",
)

layout = TA.StepLayout(M_["lay"].terms, edges)
ok_steps = True
for j, t_ in enumerate(layout.terms):
    Tj = layout.term_table(j, mean[j])
    ok_steps &= np.array_equal(Tj[1 : t_.K - 1 : 2], Tj[2 : t_.K : 2])
check(
    ok_steps,
    "4: every exported table is a step function (equal heights "
    "on the two knots of each bin)",
)

doc_bad = json.loads((CHAIN / TA.OUT_FILES[0]).read_text())
saved = (CHAIN / TA.OUT_FILES[0]).read_bytes()
doc_bad["scores"]["tree"] += 1e-6
(CHAIN / TA.OUT_FILES[0]).write_text(json.dumps(doc_bad))
S_u, reasons = TA.model_reasons(CHAIN, base, M_, M_fin)
reasons["tree"] = TA.tree_reasons(CHAIN, base, layout, names, S_u)
bands_ = {
    "free": TA.band_table(
        json.loads((CHAIN / PC.OUT_FILES[0]).read_text())["bootstrap"]["bands"]
    ),
    "final": TA.band_table(res_final["bootstrap"]["bands"]),
}
mr = {
    "free": np.load(CHAIN / PC.OUT_FILES[1])["ranks"][0],
    "final": np.load(CHAIN / ptr["files"]["arrays"])["ranks"][0],
}
try:
    TA.check_outputs(
        CHAIN, layout, bgs, base["shape_q"], bands_, reasons, mr, TA.OUT_FILES
    )
    refused = False
except ValueError as exc:
    refused = "'scores'" in str(exc)
(CHAIN / TA.OUT_FILES[0]).write_bytes(saved)
TA.check_outputs(CHAIN, layout, bgs, base["shape_q"], bands_, reasons, mr, TA.OUT_FILES)
check(
    refused,
    "4: the read-back refuses a changed score and accepts the " "stored outputs",
)

rep = (CHAIN / "reports" / TA.REPORT_NAME).read_text()
rows = [ln for ln in rep.splitlines() if ln.startswith("|")]
cols_ok = True
block = []
for ln in rep.splitlines() + [""]:
    if ln.startswith("|"):
        block.append(ln.count("|"))
    elif block:
        cols_ok &= len(set(block)) == 1
        block = []
check(
    ";" not in rep
    and cols_ok
    and "## T4: accuracy" in rep
    and "## T3: reasons" in rep
    and len(rows) > 20,
    "4: the report has every section, tables with even rows and no " "semicolon",
)

mk = common.marker_path("p13c_final")
held = mk.read_bytes()
mk.unlink()
try:
    TA.run(
        stage(), CHAIN, SCR / "tree_d", say=quiet, n_verify=6, inputs=INPUTS, stages=STS
    )
    stale = False
except ValueError as exc:
    stale = "p13c_final is not current" in str(exc)
mk.write_bytes(held)
check(
    stale and not (SCR / "tree_d").exists(),
    "4: check 6 stops the audit before any work when a marker is missing",
)

TA.BAGS, TA.TOLERANCE = 1, 0.5  # no round gains half the loss
out_e = TA.run(
    stage(), CHAIN, SCR / "tree_e", say=quiet, n_verify=6, inputs=INPUTS, stages=STS
)
b0 = out_e["bags"][0]
check(
    b0["stop"] == "early" and b0["rounds"] == TA.PATIENCE and b0["best_round"] == 0,
    "4: with a tolerance no gain reaches, the bag stops early after "
    "PATIENCE rounds and keeps round 0",
)

print(f"\n{len(failures)} failures, {clock()}")
if os.environ.get("KEEP_TMP"):
    print(f"kept {TMP}")
else:
    shutil.rmtree(TMP, ignore_errors=True)
sys.exit(1 if failures else 0)
