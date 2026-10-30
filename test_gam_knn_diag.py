"""Checks for gam_knn_diag.py on the P13 chain of p13_chain.py: P11 to
P13b part 2 and ItemKNN on the synthetic world of p13_world.py. Every path
points into a temporary folder, and nothing touches the real model folder.

1. The helpers: DCG, the cluster standard error (against a direct formula
   and against gam_audit.paired_summary), the evidence classes, the flat
   top, the bands, the variants and their tables, the task runner (order,
   progress, a failing task, Ctrl+C in the workers) and the log filter.
2. Stage 1 with the model's own top knots and with knots at one eighth,
   so that the clamp acts. Recomputed here without the script's code, from
   concepts.full_catalog(): every pick's values, classes and flags, every
   listener's counts, the raw and clamped ItemKNN ranks by a full sort,
   and every number of the results. One worker and three workers give the
   same outputs.
3. Stage 1 refuses inconsistent inputs, and its check fails for planted
   faults in its outputs.
4. Stage 2: the full variant equals P13b part 2's ranks, every variant's
   ranks equal a dense count written here from the stored terms and
   tables, the results follow from the ranks, the pass log names this
   script, a finished run scores nothing, a lost block is scored again,
   and three workers give the same ranks.
5. Stage 2 refuses inconsistent inputs, its dense check fails for a
   planted scorer fault, and its output check fails for planted faults.

Usage, from the repository root:
    python3 test_gam_knn_diag.py
With P13B2_BASE_CACHE=<folder> and P13_CHAIN_CACHE=<folder> the chain's
outputs are kept and reused while their code stays the same.
"""

import json
import logging
import math
import os
import shutil
import signal
import tempfile
import time
from pathlib import Path

import numpy as np

import p13_chain as C
import p13_world as W
from pipeline import catalog as ct
from pipeline import concept_data as cd
from pipeline import concepts as cn
from pipeline import gam
from pipeline import gam_audit as ga
from pipeline import itemknn as knn
from pipeline import stage_p13b2 as s13b2
from pipeline.als import Progress

from pipeline import gam_knn_diag as G

failures = []


def check(cond, msg):
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        failures.append(msg)
        if os.environ.get("FAIL_FAST"):  # FAIL_FAST=1: stop at the first failed check
            raise SystemExit(1)


def raises(fn, text, exc=ValueError):
    try:
        fn()
    except exc as e:
        if text in str(e):
            return True
        print(f"      other message: {e}")
        return False
    except Exception as e:  # reported, then False
        print(f"      unexpected {type(e).__name__}: {e}")
        return False
    return False


def load_npz(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


def quiet(msg):
    pass


print(W.summary())
TMP = Path(tempfile.mkdtemp(prefix="test_gam_knn_diag_"))
CHAIN = C.chain_into(TMP / "chain")
NI = W.NI
USERS = np.asarray(W.SEL, dtype=np.int64)
PICKS = np.asarray(W.SEL_PICKS, dtype=np.int64)
POPC = C.pop_centiles()
DOC2 = json.loads((CHAIN / "p13b2_model.json").read_text())
ARR2 = load_npz(CHAIN / "p13b2_arrays.npz")
FIN = int(DOC2["final"]["index"])
FIT = DOC2["wide_fits"][FIN]
G_RANK = ARR2["ranks"][FIN]
TERMS = [gam.Term.from_json(t) for t in FIT["terms"]]
TOP = {
    "same_artist": float(
        next(t for t in TERMS if t.name == "colisten_same_artist").knots[-1]
    ),
    "other_artists": float(
        next(t for t in TERMS if t.name == "colisten_other_artists").knots[-1]
    ),
}
KNN = load_npz(CHAIN / "itemknn_selection_ranks.npz")
G.USERS_PER_TASK = 7
print(
    f"      final model {FIT['cell']} {FIT['arm']} level {FIT['level']}, "
    f"top knots {TOP}"
)


def fresh(name):
    """A copy of the chain's model folder, with the paths of the stages
    pointed at it."""
    root = TMP / name
    if root.exists():
        shutil.rmtree(root)
    shutil.copytree(CHAIN.parent, root)
    C.configure(root)
    return root / "model"


# 1: helpers
r = np.array([1, 2, 100, 101, 5000])
check(
    np.allclose(
        G.pick_dcg(r),
        [1.0, 1 / math.log2(3), 1 / math.log2(101), 0.0, 0.0],
        rtol=0,
        atol=1e-15,
    ),
    "pick_dcg(): 1 / log2(rank + 1) up to rank 100, 0 beyond",
)

rng = np.random.default_rng(5)
x = rng.normal(size=(40, 2))
mask = rng.random((40, 2)) < 0.6
got = G.cluster_mean(x, mask)
n_u = mask.sum(axis=1)
tot = (x * mask).sum(axis=1)
m = tot.sum() / n_u.sum()
g_ = int((n_u > 0).sum())
se = (
    math.sqrt(
        g_ / (g_ - 1) * sum((tot[i] - m * n_u[i]) ** 2 for i in range(40) if n_u[i] > 0)
    )
    / n_u.sum()
)
check(
    abs(got["mean"] - m) < 1e-15
    and abs(got["se"] - se) < 1e-15
    and got["picks"] == int(mask.sum())
    and got["listeners"] == g_
    and abs(got["lo"] - (m - G.Z95 * se)) < 1e-15,
    "cluster_mean(): the ratio estimator and its cluster standard error",
)
full_mask = np.ones((40, 2), dtype=bool)
ps = ga.paired_summary(x.mean(axis=1), np.zeros(40))
got = G.cluster_mean(x, full_mask)
check(
    abs(got["mean"] - ps["mean"]) < 1e-15 and abs(got["se"] - ps["se"]) < 1e-15,
    "cluster_mean() over every pick equals gam_audit.paired_summary()",
)
empty = G.cluster_mean(x, np.zeros((40, 2), dtype=bool))
mask1 = np.zeros((40, 2), dtype=bool)
mask1[3] = True
one = G.cluster_mean(x, mask1)
mask2 = np.zeros((40, 2), dtype=bool)
mask2[[3, 8], 0] = True
two = G.cluster_mean(x, mask2)
d_ = x[[3, 8], 0]
check(
    two["listeners"] == 2 and abs(two["se"] - abs(d_[0] - d_[1]) / 2) < 1e-15,
    "cluster_mean(): two listeners give a standard error",
)
check(
    empty["picks"] == 0
    and empty["mean"] is None
    and one["listeners"] == 1
    and one["se"] is None
    and abs(one["mean"] - x[3].mean()) < 1e-15,
    "cluster_mean(): no pick gives None, one listener gives no standard " "error",
)

vs = np.array([np.nan, 0.0, 0.3, np.nan, 0.0, np.nan, 0.3])
vo = np.array([np.nan, 0.0, 0.0, 0.2, 0.4, 0.0, 0.2])
check(
    G.evidence_codes(vs, vo).tolist() == [3, 2, 0, 1, 1, 2, 0],
    "evidence_codes(): same-artist first, then other-artist only, then "
    "lists without overlap, then no lists",
)
check(
    G.on_top(np.array([np.nan, 0.5, 1.0, 1.5]), 1.0).tolist()
    == [False, False, True, True],
    "on_top(): at or above the top knot, never without data",
)
check(
    G.pop_bands(np.array([0, 1, 10, 11, 50, 51, 90, 91, 100])).tolist()
    == [0, 0, 0, 1, 1, 2, 2, 3, 3],
    "pop_bands(): the bands of P5, 0 in the lowest band as P5 has it",
)
check(
    G.has_rows(np.array([0, 2, 2, 3])).tolist() == [False, False, True],
    "has_rows(): rows with a neighbour, never slot 0",
)

names = [t.name for t in TERMS]
specs = G.variant_specs(names)
want_names = (
    ["full"]
    + [f"without {t}" for t in names]
    + [f"only {t}" for t in names]
    + ["only co-listening", "only artist", "without co-listening"]
)
check(
    [s[0] for s in specs] == want_names
    and len(specs) == 20
    and specs[0][1] == tuple(names)
    and dict(specs)["only co-listening"]
    == ("colisten_same_artist", "colisten_other_artists")
    and dict(specs)["only artist"] == ("artist_affinity", "colisten_same_artist")
    and dict(specs)["without co-listening"]
    == tuple(t for t in names if not t.startswith("colisten"))
    and all(
        dict(specs)[f"without {t}"] == tuple(x for x in names if x != t) for t in names
    ),
    f"variant_specs(): {len(specs)} variants in the declared order",
)
check(
    raises(
        lambda: G.variant_specs([t for t in names if t != "artist_affinity"]),
        "lacks the terms ['artist_affinity'] of the group artist",
    )
    and raises(lambda: G.variant_specs(names + names[:1]), "names a term twice"),
    "variant_specs() refuses a model without a group's term, or with a " "term twice",
)
G.check_every_candidate(TERMS)
ok = True
for change in (
    {"tiers": (3,)},
    {"multiplier": "mainstreaminess"},
    {"missing_state": False},
):
    bad_terms = [
        gam.Term(**{**t_.__dict__, **change}) if t_.name == "audio_fit" else t_
        for t_ in TERMS
    ]
    ok &= raises(
        lambda: G.check_every_candidate(bad_terms),
        "the term audio_fit does not apply to every candidate",
    )
pop_term = next(t_ for t_ in TERMS if t_.name == "popularity")
check(
    ok and not pop_term.missing_state,
    "check_every_candidate() accepts the final model, whose popularity "
    "term has no missing state, and refuses a term with tiers, a "
    "multiplier or no missing state",
)
lay = gam.Layout(TERMS, {n: 1 for n in names}, {n: 1 for n in names})
T_full = rng.normal(size=lay.L)
V = G.variant_tables(T_full, lay, specs)
ok = True
for c, (name, kept) in enumerate(specs):
    for t, off in zip(lay.terms, lay.offsets):
        seg = V[c, off : off + t.size]
        want = T_full[off : off + t.size] if t.name in kept else 0.0
        ok &= bool(np.all(seg == want))
check(
    ok and V.shape == (20, lay.L),
    "variant_tables(): each row keeps its terms' tables bit for bit and "
    "sets the others to 0",
)
check(
    raises(lambda: G.variant_tables(T_full, lay, [("x", ("nope",))]), "unknown terms")
    and raises(lambda: G.variant_tables(T_full[:-1], lay, specs), "the layout needs"),
    "variant_tables() refuses an unknown term and tables of another " "length",
)


def square(task):
    lo, hi = task
    time.sleep(0.01 * ((lo * 7) % 5))
    return [i * i for i in range(lo, hi)]


def fail_first(task):
    if task[0] == 0:
        raise RuntimeError("planted")
    time.sleep(30)
    return []


def ignores_sigint(task):
    os.kill(os.getpid(), signal.SIGINT)
    return [signal.getsignal(signal.SIGINT) is signal.SIG_IGN]


tasks = [(lo, min(lo + 3, 40)) for lo in range(0, 40, 3)]
logs = []
for wk in (1, 3):
    prog = Progress("t", 40, logs.append)
    parts = G.run_tasks(square, tasks, wk, prog, "t", logs.append)
    check(
        [v for p in parts for v in p] == [i * i for i in range(40)] and prog.done == 40,
        f"run_tasks() with {wk} worker(s): results in task order, every "
        f"task counted once",
    )
t0 = time.time()
ok = raises(
    lambda: G.run_tasks(fail_first, tasks, 3, Progress("t", 40, quiet), "t", quiet),
    "planted",
    RuntimeError,
)
check(
    ok and time.time() - t0 < 2.5,
    f"run_tasks(): a failing task stops the run at once "
    f"({time.time() - t0:.2f} s), the other workers end",
)
parts = G.run_tasks(
    ignores_sigint, [(0, 1), (1, 2)], 2, Progress("t", 2, quiet), "t", quiet
)
check(
    all(p[0] for p in parts),
    "run_tasks(): the workers ignore Ctrl+C, the main process handles it",
)

rec = logging.LogRecord(
    "pipeline", logging.INFO, "", 0, "p13a: probe mean %.2f s", (1.0,), None
)
rec2 = logging.LogRecord(
    "pipeline",
    logging.INFO,
    "",
    0,
    "p13a: swap. Start again with P13A_WORKERS=8 "
    "python3 -m pipeline.runner --only p13a_fit.",
    (),
    None,
)
recs = [
    logging.LogRecord("pipeline", logging.INFO, "", 0, m_, (), None)
    for m_ in (
        "p13b: probe listener 1 of 3",
        "p13b2: a line",
        "gam_knn_diag: own line",
        "p13ab: not a stage",
        "loading p13a: not at the start",
    )
]
with G._relabel_logs():
    for r_ in [rec, rec2] + recs:
        G.log.filter(r_)
rec_after = logging.LogRecord("pipeline", logging.INFO, "", 0, "p13a: x", (), None)
G.log.filter(rec_after)
check(
    rec.getMessage() == "gam_knn_diag: probe mean 1.00 s"
    and rec2.msg == "gam_knn_diag: swap. Start again with P13A_WORKERS=8 "
    "python3 -m pipeline.gam_knn_diag variants."
    and [r_.msg for r_ in recs]
    == [
        "gam_knn_diag: probe listener 1 of 3",
        "gam_knn_diag: a line",
        "gam_knn_diag: own line",
        "p13ab: not a stage",
        "loading p13a: not at the start",
    ]
    and rec_after.msg == "p13a: x"
    and not G.log.filters,
    "the log filter renames the lines of P13a, P13b and P13b2, points the "
    "advice at this script, leaves other lines and goes away after the "
    "pass",
)


# 2: stage 1, independent


def independent_slices(top):
    """Every per-pick and per-listener array of stage 1, from
    concepts.full_catalog() and full sorts, without the script's code."""
    n = len(USERS)
    out = {
        k: np.empty((n, 2), dtype=np.int64)
        for k in ("knn_both", "knn_same", "clamp_both", "clamp_same")
    }
    out.update(value_same=np.empty((n, 2)), value_other=np.empty((n, 2)))
    for k in ("n_flat_same", "n_flat_other", "n_pos_same", "n_pos_both"):
        out[k] = np.empty(n, dtype=np.int64)
    for j, u in enumerate(USERS):
        items, weights = cd.history_of(W.u_indptr, W.u_indices, W.u_data, int(u))
        state = cn.build_user_state(items, weights, W.SPACE)
        full = cn.full_catalog(state, W.SPACE)
        vs, hs = full["colisten_same_artist"]
        vo, ho = full["colisten_other_artists"]
        vs0, vo0 = np.nan_to_num(vs), np.nan_to_num(vo)
        cand = np.ones(NI, dtype=bool)
        cand[0] = False
        cand[items] = False
        ids = np.flatnonzero(cand)
        numer_s = vs0 * state.total
        scores = {
            "knn_same": np.asarray(
                W.SPACE.colisten_t["same_artist"][state.items].T @ state.weights
            ).ravel(),
            "clamp_same": np.minimum(vs0, top["same_artist"]),
            "clamp_both": np.minimum(vs0, top["same_artist"])
            + np.minimum(vo0, top["other_artists"]),
        }
        scores["knn_both"] = (
            scores["knn_same"]
            + np.asarray(
                W.SPACE.colisten_t["other_artists"][state.items].T @ state.weights
            ).ravel()
        )
        for key, s in scores.items():
            order = ids[np.lexsort((ids, -s[ids]))]
            pos = {int(i): p + 1 for p, i in enumerate(order)}
            out[key][j] = [pos[int(p)] for p in PICKS[j]]
        out["value_same"][j] = vs[PICKS[j]]
        out["value_other"][j] = vo[PICKS[j]]
        out["n_flat_same"][j] = int(np.sum(cand & hs & (vs0 >= top["same_artist"])))
        out["n_flat_other"][j] = int(np.sum(cand & ho & (vo0 >= top["other_artists"])))
        out["n_pos_same"][j] = int(np.sum(cand & (numer_s > 0)))
        out["n_pos_both"][j] = int(np.sum(cand & ((vs0 > 0) | (vo0 > 0))))
    return out


def classes(v_s, v_o):
    code = np.empty(v_s.shape, dtype=np.int64)
    for idx in np.ndindex(v_s.shape):
        a, b = v_s[idx], v_o[idx]
        if not math.isnan(a) and a > 0:
            code[idx] = 0
        elif not math.isnan(b) and b > 0:
            code[idx] = 1
        elif not (math.isnan(a) and math.isnan(b)):
            code[idx] = 2
        else:
            code[idx] = 3
    return code


def dcg_list(rk):
    return [1 / math.log2(r + 1) if r <= 100 else 0.0 for r in rk]


def slow_row(mask, g, k):
    """A slice row by loops over listeners."""
    per = []
    for j in range(len(g)):
        d = [
            (dcg_list([g[j, p]])[0], dcg_list([k[j, p]])[0])
            for p in range(2)
            if mask[j, p]
        ]
        if d:
            per.append(d)
    n_p = sum(len(d) for d in per)
    diff = sum(a - b for d in per for a, b in d) / n_p
    num = sum((sum(a - b for a, b in d) - diff * len(d)) ** 2 for d in per)
    se = math.sqrt(len(per) / (len(per) - 1) * num) / n_p if len(per) > 1 else math.nan
    return {
        "picks": n_p,
        "listeners": len(per),
        "gam": sum(a for d in per for a, _ in d) / n_p,
        "knn": sum(b for d in per for _, b in d) / n_p,
        "diff": diff,
        "se": se,
        "better": sum(max(a, b) for d in per for a, b in d) / n_p,
        "gam_only": sum(
            1
            for j in range(len(g))
            for p in range(2)
            if mask[j, p] and g[j, p] <= 100 and k[j, p] > 100
        ),
        "knn_only": sum(
            1
            for j in range(len(g))
            for p in range(2)
            if mask[j, p] and g[j, p] > 100 and k[j, p] <= 100
        ),
        "both": sum(
            1
            for j in range(len(g))
            for p in range(2)
            if mask[j, p] and g[j, p] <= 100 and k[j, p] <= 100
        ),
        "gam_r100": sum(
            1 for j in range(len(g)) for p in range(2) if mask[j, p] and g[j, p] <= 100
        )
        / n_p,
        "knn_r100": sum(
            1 for j in range(len(g)) for p in range(2) if mask[j, p] and k[j, p] <= 100
        )
        / n_p,
    }


def close(a, b, tol=1e-12):
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= tol * max(1.0, abs(b))


def check_stage1(model, top, label):
    doc = json.loads((model / "gam_knn_slices.json").read_text())
    z = load_npz(model / "gam_knn_slices.npz")
    ind = independent_slices(top)
    check(
        np.array_equal(z["user_id"], USERS)
        and np.array_equal(z["picks"], PICKS)
        and np.array_equal(z["gam_rank"], G_RANK)
        and np.array_equal(z["knn_both"], KNN["both_id"])
        and np.array_equal(z["knn_same"], KNN["same_artist_id"])
        and np.array_equal(z["knn_both"], ind["knn_both"])
        and np.array_equal(z["knn_same"], ind["knn_same"]),
        f"{label}: the listeners, picks and GAM ranks are P13's, and the "
        f"ItemKNN ranks equal the stored ones and a full sort",
    )
    check(
        np.array_equal(z["clamp_same"], ind["clamp_same"])
        and np.array_equal(z["clamp_both"], ind["clamp_both"]),
        f"{label}: the clamped ranks equal a full sort of the clamped "
        f"concept values of full_catalog()",
    )
    same_bits = all(
        np.array_equal(np.isnan(z[k]), np.isnan(ind[k]))
        and np.array_equal(
            np.nan_to_num(z[k]).view(np.uint64), np.nan_to_num(ind[k]).view(np.uint64)
        )
        for k in ("value_same", "value_other")
    )
    check(
        same_bits,
        f"{label}: every pick's two values equal the GAM's "
        f"concept values bit for bit, NaN where no data",
    )
    check(
        np.array_equal(z["evidence"], classes(ind["value_same"], ind["value_other"]))
        and np.array_equal(
            z["flat_same"],
            np.nan_to_num(ind["value_same"], nan=-1) >= top["same_artist"],
        )
        and np.array_equal(
            z["flat_other"],
            np.nan_to_num(ind["value_other"], nan=-1) >= top["other_artists"],
        ),
        f"{label}: the evidence classes and flat-top flags follow from " f"the values",
    )
    check(
        all(
            np.array_equal(z[k], ind[k])
            for k in ("n_flat_same", "n_flat_other", "n_pos_same", "n_pos_both")
        )
        and np.array_equal(z["n_pos_both"], KNN["positive"])
        and np.array_equal(z["pop_centile"], POPC[PICKS])
        and np.array_equal(z["history"], np.diff(W.u_indptr)[USERS]),
        f"{label}: the counts, the centiles and the history lengths "
        f"follow from the world",
    )
    res = doc["results"]
    ok = True
    for key, name in G.MODELS:
        s = float(np.mean([np.mean(dcg_list(z[key][j])) for j in range(len(USERS))]))
        ok &= close(res["models"][name]["score"], s)
        for p, pt in enumerate(G.PICK_TYPES):
            ok &= close(
                res["models"][name][pt]["recall_at_100"],
                float(np.mean(z[key][:, p] <= 100)),
            )
            ok &= close(
                res["models"][name][pt]["recall_at_10"],
                float(np.mean(z[key][:, p] <= 10)),
            )
            ok &= close(
                res["models"][name][pt]["ndcg_at_100"],
                float(np.mean(dcg_list(z[key][:, p]))),
            )
    check(
        ok
        and close(res["models"]["GAM, final model"]["score"], FIT["selection"]["mean"]),
        f"{label}: every model's score and recall follow from its ranks, "
        f"and the GAM's equals P13b part 2's",
    )
    ev_ = z["evidence"]
    band = np.digitize(z["pop_centile"], [10.5, 50.5, 90.5])
    hist = np.diff(W.u_indptr)[USERS]
    q1, q2 = np.quantile(hist, [1 / 3, 2 / 3])
    hc = np.repeat(np.digitize(hist, [q1, q2], right=True)[:, None], 2, 1)
    want_masks = {
        ("all picks", "all"): np.ones((len(USERS), 2), dtype=bool),
        ("pick type", "val_last"): np.array([[True, False]] * len(USERS)),
        ("pick type", "val_first"): np.array([[False, True]] * len(USERS)),
        ("same-artist value", "on the flat top"): (ev_ == 0) & z["flat_same"],
        ("same-artist value", "below the top knot"): (ev_ == 0) & ~z["flat_same"],
    }
    for c, nm in enumerate(G.EVIDENCE):
        want_masks[("co-listening evidence", nm)] = ev_ == c
    for c, nm in enumerate(G.POP_BANDS):
        want_masks[("popularity band", nm)] = band == c
    for c, nm in enumerate(G.HISTORY_LABELS):
        want_masks[("history length", nm)] = hc == c
    ok, n_rows = True, 0
    for (dim, nm), msk in want_masks.items():
        row = res["slices"][dim][nm]
        n_rows += 1
        if not msk.any():
            ok &= row["picks"] == 0
            continue
        w = slow_row(msk, z["gam_rank"], z["knn_both"])
        ok &= (
            row["picks"] == w["picks"]
            and row["listeners"] == w["listeners"]
            and close(row["gam_ndcg"], w["gam"])
            and close(row["knn_ndcg"], w["knn"])
            and close(row["diff"]["mean"], w["diff"])
            and close(row["diff"]["se"], None if math.isnan(w["se"]) else w["se"])
            and close(row["better"], w["better"])
            and row["gam_only"] == w["gam_only"]
            and row["knn_only"] == w["knn_only"]
            and row["both"] == w["both"]
            and close(row["gam_r100"], w["gam_r100"])
            and close(row["knn_r100"], w["knn_r100"])
            and row["both"] + row["gam_only"] + row["knn_only"] + row["neither"]
            == w["picks"]
        )
    check(
        ok and n_rows == sum(len(v) for v in res["slices"].values()),
        f"{label}: all {n_rows} slice rows (means, cluster intervals, "
        f"top-100 overlaps, better of two) follow from loops over the "
        f"listeners",
    )
    better = np.mean(
        [
            np.mean(
                [
                    max(a, b)
                    for a, b in zip(
                        dcg_list(z["gam_rank"][j]), dcg_list(z["knn_both"][j])
                    )
                ]
            )
            for j in range(len(USERS))
        ]
    )
    d_s = ga.paired_summary(
        ga.per_listener_scores(z["knn_same"]), ga.per_listener_scores(z["clamp_same"])
    )
    base = float(ga.per_listener_scores(z["knn_same"]).mean())
    s1 = res["readings"]["S1"]["same-artist list"]
    s2_want = sorted(
        f"{dim}: {nm}"
        for (dim, nm), msk in want_masks.items()
        if msk.any()
        and res["slices"][dim][nm]["diff"]["hi"] is not None
        and res["slices"][dim][nm]["diff"]["hi"] < 0
    )
    s1b = res["readings"]["S1"]["both lists"]
    d_b = ga.paired_summary(
        ga.per_listener_scores(z["knn_both"]), ga.per_listener_scores(z["clamp_both"])
    )
    base_b = float(ga.per_listener_scores(z["knn_both"]).mean())
    pl = ga.per_listener_scores
    want_d = {
        "GAM minus ItemKNN, both lists": (z["gam_rank"], z["knn_both"]),
        "GAM minus ItemKNN, same-artist list": (z["gam_rank"], z["knn_same"]),
        "ItemKNN same-artist list minus its clamped form": (
            z["knn_same"],
            z["clamp_same"],
        ),
        "ItemKNN both lists minus their clamped form": (z["knn_both"], z["clamp_both"]),
    }
    diffs_ok = set(res["differences"]) == set(want_d) and all(
        close(res["differences"][nm_]["mean"], float((pl(a) - pl(b)).mean()))
        and close(
            res["differences"][nm_]["se"],
            float((pl(a) - pl(b)).std(ddof=1) / math.sqrt(len(a))),
        )
        for nm_, (a, b) in want_d.items()
    )
    check(
        close(res["better_of_two"], better)
        and diffs_ok
        and close(s1["difference"]["mean"], d_s["mean"])
        and s1["holds"]
        == bool(d_s["mean"] > 0.02 * base and d_s["mean"] - G.Z95 * d_s["se"] > 0)
        and close(s1b["unclamped_score"], base_b)
        and s1b["holds"]
        == bool(d_b["mean"] > 0.02 * base_b and d_b["mean"] - G.Z95 * d_b["se"] > 0)
        and sorted(res["readings"]["S2"]) == s2_want,
        f"{label}: the better-of-two bound and the readings S1 and S2 "
        f"follow from the ranks (S2: {res['readings']['S2']})",
    )
    flat = z["flat_same"]
    nf = np.repeat(z["n_flat_same"][:, None], 2, 1)[flat]
    f = res["flat_top"]
    check(
        f["picks_on_flat_top"] == int(flat.sum())
        and f["picks_with_same_artist_evidence"] == int((ev_ == 0).sum())
        and f["listeners_median_flat_same"] == float(np.median(ind["n_flat_same"]))
        and f["listeners_median_flat_other"] == float(np.median(ind["n_flat_other"]))
        and f["listeners_median_positive_same"] == float(np.median(ind["n_pos_same"]))
        and f["listeners_median_positive_both"] == float(np.median(ind["n_pos_both"]))
        and f["listeners_with_any_on_top"] == int(np.sum(ind["n_flat_same"] > 0))
        and (
            not flat.any()
            or (
                f["n_flat_of_their_listener"]
                == {
                    f"q{int(100 * q_)}": float(np.quantile(nf, q_))
                    for q_ in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)
                }
                and f["share_with_100_or_more_on_top"] == float(np.mean(nf >= 100))
            )
        ),
        f"{label}: the flat-top statistics follow from the counts "
        f"({int(flat.sum())} picks on the flat top)",
    )
    rep = (model / "reports" / "gam_knn_slices.md").read_text()
    gam_score = res["models"]["GAM, final model"]["score"]
    gam_row = f"| GAM, final model | {gam_score:.5f} |"
    check(
        ";" not in rep
        and gam_row in rep
        and all(
            h in rep
            for h in (
                "## 1. Models",
                "## 2. The GAM's " "shapes",
                "## 3. Slices",
                "## 4. The flat top",
                "## 5. Reading",
                "- S1, same-artist list",
                "- S2,",
            )
        )
        and doc["check"]["listeners"] == 90
        and doc["check"]["itemknn_ranks_equal"] is True,
        f"{label}: the report has every section and the stored score, "
        f"without a semicolon",
    )
    return doc, z, ind


m1 = fresh("s1_one")
out1 = G.run_slices(
    m1,
    NI,
    C.U_ARRAYS,
    C.LISTS,
    POPC,
    workers=1,
    say=quiet,
    n_check=90,
    inputs={"x": "y"},
)
doc1, z1, ind1 = check_stage1(m1, TOP, "stage 1")
check(
    doc1["inputs"] == {"x": "y"}
    and doc1["final"]["cell"] == FIT["cell"]
    and doc1["results"]["top_knots"] == TOP,
    "stage 1 stores its inputs, the final model and the top knots",
)
T_own = np.asarray(FIT["tables"])
offs = np.cumsum([0] + [t.size for t in TERMS])[:-1]
ok = True
for t, off in zip(TERMS, offs):
    if t.name in G.SHAPE_TERMS:
        s = doc1["shapes"][t.name]
        h = T_own[off : off + t.K]
        ok &= (
            s["heights"] == [float(v) for v in h[-G.SHAPE_TAIL :]]
            and s["max_height"] == float(h.max())
            and s["top_minus_max"] == float(h[-1] - h.max())
            and s["missing"] == float(T_own[off + t.K + 1])
            and s["zero"] == float(T_own[off + t.K])
        )
check(
    ok and set(doc1["shapes"]) == set(G.SHAPE_TERMS),
    "the shape summary reads the final fit's own tables: heights, "
    "largest height, zero and missing constants",
)

# The clamp in action: knots at one eighth.
real_top = G.top_knots
# Knots at pick values, below the model's top knots: the clamp acts, and
# a pick at the knot itself lies on the flat top.
pos_s = np.sort(z1["value_same"][z1["evidence"] == 0])
pos_o = np.sort(
    z1["value_other"][~np.isnan(z1["value_other"]) & (z1["value_other"] > 0)]
)
LOW = {
    "same_artist": float(pos_s[len(pos_s) // 3]),
    "other_artists": float(pos_o[len(pos_o) // 3]),
}
assert LOW["same_artist"] < TOP["same_artist"]
assert LOW["other_artists"] < TOP["other_artists"]
G.top_knots = lambda terms: dict(LOW)
try:
    m_low = fresh("s1_low")
    G.run_slices(m_low, NI, C.U_ARRAYS, C.LISTS, POPC, workers=3, say=quiet, n_check=90)
    doc_l, z_l, ind_l = check_stage1(m_low, LOW, "stage 1, low knots")
    m_low1 = fresh("s1_low1")
    G.USERS_PER_TASK = 89  # two tasks, the last one of 1 listener
    G.run_slices(
        m_low1, NI, C.U_ARRAYS, C.LISTS, POPC, workers=1, say=quiet, n_check=200
    )
finally:
    G.top_knots = real_top
    G.USERS_PER_TASK = 7
z_l1 = load_npz(m_low1 / "gam_knn_slices.npz")
doc_l1 = json.loads((m_low1 / "gam_knn_slices.json").read_text())
check(
    int(z_l["flat_same"].sum()) > 0
    and bool(np.any(z_l["value_same"] == LOW["same_artist"]))
    and bool(np.any(z_l["value_other"] == LOW["other_artists"]))
    and int((z_l["clamp_same"] != z_l["knn_same"]).sum()) > 0
    and int((z_l["clamp_both"] != z_l["knn_both"]).sum()) > 0,
    f"with low knots the clamp changes ranks "
    f"({int((z_l['clamp_same'] != z_l['knn_same']).sum())} same-artist "
    f"picks, {int(z_l['flat_same'].sum())} picks on the flat top)",
)
check(
    set(z_l) == set(z_l1)
    and all(
        np.array_equal(z_l[k], z_l1[k], equal_nan=k.startswith("value")) for k in z_l
    )
    and doc_l["results"] == doc_l1["results"]
    and doc_l1["check"]["listeners"] == 90,
    "one worker and three workers, and other task sizes, give the same "
    "stage 1 outputs",
)

# Constructed inputs for the boundaries of summarize_slices().
zc = {k: v[:7].copy() for k, v in z1.items()}
zc["history"] = np.array([5, 5, 5, 5, 5, 5, 9])
zc["evidence"] = np.zeros((7, 2), dtype=np.int8)
zc["flat_same"] = np.ones((7, 2), dtype=bool)
zc["n_flat_same"] = np.full(7, 100)
zc["knn_same"] = np.ones((7, 2), dtype=np.int64)
zc["clamp_same"] = np.full((7, 2), 101)
zc["knn_both"] = np.ones((7, 2), dtype=np.int64)
zc["clamp_both"] = np.ones((7, 2), dtype=np.int64)
zc["gam_rank"] = np.full((7, 2), 101)
rc = G.summarize_slices(zc, TOP)
zc2 = dict(zc, knn_same=np.full((7, 2), 90), clamp_same=np.full((7, 2), 100))
s1_2 = G.summarize_slices(zc2, TOP)["readings"]["S1"]["same-artist list"]
check(
    s1_2["holds"] is True
    and 0.02 * s1_2["unclamped_score"] < s1_2["difference"]["mean"] < 0.02,
    "S1 compares the loss with SESOI times the unclamped score",
)
hl = rc["slices"]["history length"]
check(
    rc["history_cuts"] == [5.0, 5.0]
    and hl["short"]["picks"] == 12
    and hl["middle"]["picks"] == 0
    and hl["long"]["picks"] == 2,
    "history tertiles: a listener at a cut belongs to the lower group",
)
check(
    rc["flat_top"]["share_with_100_or_more_on_top"] == 1.0
    and rc["readings"]["S1"]["same-artist list"]["holds"] is True
    and rc["readings"]["S1"]["both lists"]["holds"] is False
    and "all picks: all" in rc["readings"]["S2"]
    and "same-artist value: on the flat top" in rc["readings"]["S2"]
    and "pick type: val_last" in rc["readings"]["S2"],
    "constructed ranks: 100 candidates on the top count, S1 holds only for "
    "the clamped list that loses, S2 names the slices where the GAM is "
    "behind",
)

# 3: stage 1, refusals and faults


def with_change(name, change, text, top=None):
    m = fresh("fault")
    path = m / name
    if name.endswith(".json"):
        d = json.loads(path.read_text())
        change(d)
        path.write_text(json.dumps(d))
    else:
        d = load_npz(path)
        change(d)
        np.savez(path, **d)
    return raises(
        lambda: G.run_slices(
            m, NI, C.U_ARRAYS, C.LISTS, POPC, workers=1, say=quiet, n_check=5
        ),
        text,
    )


def bump(key, j=0, p=0, by=1):
    def f(d):
        d[key] = d[key].copy()
        d[key][j, p] += by

    return f


def edit_score(d):
    d["variants"]["both/id"]["score"] += 1e-9


def edit_mean(d):
    d["wide_fits"][FIN]["selection"]["mean"] += 1e-9


def other_final(d):
    d["final"]["index"] = (FIN + 1) % len(d["wide_fits"])


def shuffle_users(key):
    def f(d):
        d[key] = d[key][::-1].copy()

    return f


def bad_gam_rank(d):
    d["ranks"] = d["ranks"].copy()
    d["ranks"][FIN, 0, 0] = 0


def drop_term(d):
    d["wide_fits"][FIN]["terms"] = [
        t for t in d["wide_fits"][FIN]["terms"] if t["name"] != "colisten_other_artists"
    ]


def no_zero_state(d):
    for t in d["wide_fits"][FIN]["terms"]:
        if t["name"] == "colisten_same_artist":
            t["zero_state"] = False


def other_source(d):
    for t in d["wide_fits"][FIN]["terms"]:
        if t["name"] == "colisten_other_artists":
            t["source"] = "colisten_same_artist"


def extra_table(d):
    d["wide_fits"][FIN]["tables"] = d["wide_fits"][FIN]["tables"] + [0.0]


def index_out(d):
    d["final"]["index"] = len(d["wide_fits"])


def same_cell_other(what):
    """The final index moved to a fit that differs from the final model in
    one of cell, arm and level only."""
    keys = [k for k in ("cell", "arm", "level") if k != what]

    def f(d):
        fits = d["wide_fits"]
        j = next(
            i
            for i, x in enumerate(fits)
            if i != FIN and all(x[k] == fits[FIN][k] for k in keys)
        )
        d["final"]["index"] = j

    return f


def fewer_fits(d):
    d["ranks"] = d["ranks"][:-1].copy()


def bump_pos(d):
    d["positive"] = d["positive"].copy()
    d["positive"][4] += 1


j_rk = next(j for j in range(len(USERS)) if KNN["both_id"][j, 0] > 101)
j_ks = next(j for j in range(len(USERS)) if KNN["same_artist_id"][j, 1] <= 99)
j_kf = next(j for j in range(len(USERS)) if KNN["same_artist_id"][j, 1] > 101)
checks = [
    (
        "itemknn_selection_ranks.npz",
        bump("both_id", j_rk, 0, -1),
        "the recomputed ItemKNN ranks differ",
    ),
    (
        "itemknn_selection_ranks.npz",
        bump("same_artist_id", j_ks, 1, 1),
        "does not follow from its stored ranks",
    ),
    (
        "itemknn_selection_ranks.npz",
        bump("same_artist_id", j_kf, 1, 1),
        "the recomputed ItemKNN ranks differ",
    ),
    ("itemknn_selection_ranks.npz", bump_pos, "the counts of positive"),
    (
        "itemknn_selection_ranks.npz",
        shuffle_users("user_id"),
        "ItemKNN's stored listeners differ",
    ),
    ("itemknn_selection.json", edit_score, "does not follow from its stored " "ranks"),
    (
        "p13b2_arrays.npz",
        shuffle_users("sel_users"),
        "p13b2_arrays.npz holds other listeners",
    ),
    ("p13b2_arrays.npz", bad_gam_rank, "a stored GAM rank lies outside"),
    ("p13b2_model.json", edit_mean, "stored selection score does not " "follow"),
    ("p13b2_model.json", other_final, "differs from the final model's"),
    ("p13b2_model.json", drop_term, "has no term colisten_other_artists"),
    (
        "p13b2_model.json",
        no_zero_state,
        "does not read its concept with a " "zero and a missing state",
    ),
    ("p13b2_model.json", other_source, "does not read its concept"),
    ("p13b2_model.json", extra_table, "tables do not match its terms"),
    ("p13b2_model.json", index_out, "lies outside the"),
    (
        "p13b2_model.json",
        same_cell_other("arm"),
        "arm differs from the " "final model's",
    ),
    (
        "p13b2_model.json",
        same_cell_other("level"),
        "level differs from the " "final model's",
    ),
    ("p13b2_arrays.npz", fewer_fits, "p13b2_arrays.npz holds ranks of " "shape"),
]
ok = True
for name, fn, text in checks:
    one_ok = with_change(name, fn, text)
    if not one_ok:
        print(f"      not refused: {name}, {text}")
    ok &= one_ok
check(ok, f"stage 1 refuses {len(checks)} kinds of inconsistent input")


def out_fault(change, text, npz=True):
    m = fresh("out_fault")
    for name in G.SLICES_FILES:
        shutil.copy(m1 / name, m / name)
    if npz:
        d = load_npz(m / G.SLICES_FILES[1])
        change(d)
        np.savez(m / G.SLICES_FILES[1], **d)
    else:
        d = json.loads((m / G.SLICES_FILES[0]).read_text())
        change(d)
        (m / G.SLICES_FILES[0]).write_text(json.dumps(d))
    t_mats = knn.transposed(C.LISTS, NI)
    return raises(
        lambda: G.check_slices(
            m, t_mats, C.LISTS, C.U_ARRAYS, NI, POPC, quiet, n_check=90
        ),
        text,
    )


j_s = next(
    j
    for j in range(len(USERS))
    if not np.isnan(z1["value_same"][j, 0]) and z1["value_same"][j, 0] > 0
)
j_n = next(j for j in range(len(USERS)) if np.isnan(z1["value_other"][j, 1]))


def scale_value(d):
    d["value_same"] = d["value_same"].copy()
    d["value_same"][j_s, 0] *= 1 + 1e-9


def nan_value(d):
    d["value_same"] = d["value_same"].copy()
    d["value_same"][j_s, 0] = np.nan
    d["evidence"] = G.evidence_codes(d["value_same"], d["value_other"])
    d["flat_same"] = G.on_top(d["value_same"], TOP["same_artist"])


def value_for_nan(d):
    d["value_other"] = d["value_other"].copy()
    d["value_other"][j_n, 1] = 0.0
    d["evidence"] = G.evidence_codes(d["value_same"], d["value_other"])


def flip(key, j=0, p=0):
    def f(d):
        d[key] = d[key].copy()
        d[key][j, p] = not d[key][j, p]

    return f


def bump1(key, j=0, by=1):
    def f(d):
        d[key] = d[key].copy()
        d[key][j] += by

    return f


def set_class(d):
    d["evidence"] = d["evidence"].copy()
    d["evidence"][j_s, 0] = 2


def edit_slice(d):
    d["results"]["slices"]["pick type"]["val_last"]["gam_ndcg"] += 1e-12


def edit_top(d):
    d["results"]["top_knots"]["same_artist"] *= 1.5


j_c = next(j for j in range(len(USERS)) if z1["clamp_same"][j, 0] > 1)


def scale1(key, j, f):
    def g(d):
        d[key] = d[key].copy()
        d[key][j] *= f

    return g


faults = [
    (scale_value, "differs from the second route"),
    (nan_value, "marks other picks as without data"),
    (value_for_nan, "marks other picks as without data"),
    (set_class, "evidence classes do not follow"),
    (flip("flat_same", j_s, 0), "flat-top flags do not follow"),
    (bump1("n_flat_same", 5, 1), "lies outside the second route's range"),
    (bump1("n_pos_same", 6, 1), "differs from the second route"),
    (bump1("n_pos_both", 7, -1), "ItemKNN ranks or counts differ"),
    (bump("clamp_same", j_c, 0, -1), "a full sort gives"),
    (bump("clamp_both", 2, 1, 1), "a full sort gives"),
    (bump("gam_rank", 1, 0, 1), "the stored GAM ranks differ"),
    (bump("knn_both", 1, 0, 1), "ItemKNN ranks or counts differ"),
    (bump("knn_same", 1, 1, 1), "ItemKNN ranks or counts differ"),
    (bump("picks", 0, 0, 1), "listeners or picks differ"),
    (bump1("user_id", 0, 1), "listeners or picks differ"),
    (bump("pop_centile", 0, 1, 1), "centiles or history lengths differ"),
    (bump1("history", 3, 1), "centiles or history lengths differ"),
    (scale1("weight_total", 0, 1 + 1e-12), "total weight differs"),
    (bump1("n_flat_other", 8, 1), "lies outside the second route's range"),
]
ok = True
for fn, text in faults:
    one_ok = out_fault(fn, text)
    if not one_ok:
        print(f"      fault not caught: {text}")
    ok &= one_ok
ok &= out_fault(edit_slice, "do not follow from the stored per-pick", False)
ok &= out_fault(edit_top, "the stored top knots differ", False)
check(
    ok, f"stage 1's check fails for {len(faults) + 2} planted faults in " f"its outputs"
)
real_savez = G._atomic_savez


def bad_savez(key, idx, change):
    """_atomic_savez() that changes one entry of one array on its way to
    the disk: a fault after the computation."""

    def f(path, **arrays):
        if key in arrays:
            a = np.array(arrays[key], copy=True)
            a[idx] = change(a[idx])
            arrays[key] = a
        real_savez(path, **arrays)

    return f


G._atomic_savez = bad_savez("clamp_same", (j_c, 0), lambda r_: r_ - 1)
try:
    ok = raises(
        lambda: G.run_slices(
            fresh("s1_write"),
            NI,
            C.U_ARRAYS,
            C.LISTS,
            POPC,
            workers=1,
            say=quiet,
            n_check=90,
        ),
        "a full sort gives",
    )
finally:
    G._atomic_savez = real_savez
check(
    ok,
    "stage 1 checks what it wrote: a rank changed on the way to the "
    "disk stops the run",
)
check(
    not raises(
        lambda: G.check_slices(
            m1, knn.transposed(C.LISTS, NI), C.LISTS, C.U_ARRAYS, NI, POPC, quiet, 90
        ),
        "no such text",
    ),
    "stage 1's check passes on the untouched outputs",
)

# 4: stage 2, independent


class Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


def run2(model, workers=1, n_verify=90, st=None):
    C.configure(model.parent, workers)
    cap = Capture()
    G.log.addHandler(cap)
    G.log.setLevel(logging.INFO)
    try:
        st = st or C.p13_loaders(s13b2.StageP13b2())
        G.run_variants(
            st,
            model,
            model.parent / "scratch" / G.WORK_NAME,
            say=G.log.info,
            n_verify=n_verify,
        )
    finally:
        G.log.removeHandler(cap)
    return cap.lines


def dense_scores(kept):
    """(listeners, NI) scores of a variant: the stored terms and tables of
    the final fit, summed over the kept terms with np.interp on the concept
    values of full_catalog()."""
    out = np.zeros((len(USERS), NI))
    for j, u in enumerate(USERS):
        items, weights = cd.history_of(W.u_indptr, W.u_indices, W.u_data, int(u))
        full = cn.full_catalog(cn.build_user_state(items, weights, W.SPACE), W.SPACE)
        for t, off in zip(TERMS, offs):
            if t.name not in kept:
                continue
            assert t.tiers is None and t.multiplier is None
            v, h = full[t.source]
            tab = T_own[off : off + t.size]
            kn = t.knot_array
            x = np.clip(np.where(h, v, kn[0]), kn[0], kn[-1])
            val = np.interp(x, kn, tab[: t.K])
            if t.zero_state:
                val = np.where(h & (v == 0.0), tab[t.K], val)
            val = np.where(h, val, tab[t.K + 1] if t.missing_state else 0.0)
            out[j] += val
    return out


def rank_range(s, pick, excl, tol=1e-9):
    s = s.copy()
    s[0] = -np.inf
    s[excl] = -np.inf
    sp = s[pick]
    s[pick] = -np.inf
    near = np.abs(s - sp) <= tol
    if near.sum() == (s == sp).sum():
        r = 1 + int((s > sp).sum()) + int((s[:pick] == sp).sum())
        return r, r
    g = int((s > sp + tol).sum())
    return 1 + g, 1 + g + int(near.sum())


# Constructed ranks for the reading rules of summarize_variants().
nV = 50
idx_c = {n_: i for i, (n_, _) in enumerate(specs)}


def s1_like(clamp_same):
    return {
        "knn_same": np.ones((nV, 2), dtype=np.int64),
        "clamp_same": np.asarray(clamp_same) * np.ones((nV, 2), dtype=np.int64),
        "knn_both": np.ones((nV, 2), dtype=np.int64),
        "clamp_both": np.ones((nV, 2), dtype=np.int64),
        "evidence": np.zeros((nV, 2), dtype=np.int8),
        "flat_same": np.zeros((nV, 2), dtype=bool),
        "pop_centile": np.full((nV, 2), 50),
    }


def v1_case(clamp_same, only_same, base=5):
    R = np.full((20, nV, 2), base, dtype=np.int64)
    R[idx_c["only colisten_same_artist"]] = (
        np.asarray(only_same)[:, None] if np.ndim(only_same) == 1 else only_same
    )
    return G.summarize_variants(specs, R, s1_like(clamp_same))


mix_clamp = np.where(np.arange(nV) < 25, 1, 101)[:, None]
mix_only = np.where(np.arange(nV) < 24, 101, 1)
outcomes = [
    v1_case(5, np.full(nV, 5))["readings"]["V1"]["outcome"],
    v1_case(1, np.full(nV, 101))["readings"]["V1"]["outcome"],
    v1_case(101, np.full(nV, 1))["readings"]["V1"]["outcome"],
    v1_case(mix_clamp, mix_only)["readings"]["V1"]["outcome"],
]
check(
    outcomes == ["within", "below", "above", "unclear"],
    f"V1 on constructed ranks: {outcomes}",
)
R = np.full((20, nV, 2), 100, dtype=np.int64)
R[idx_c["without audio_fit"]] = 1
R[idx_c["without popularity"]] = 90
R[idx_c["without tag_affinity"]] = 99
R[idx_c["without co-listening"]] = 1
R[idx_c["only co-listening"]] = 99
rv = G.summarize_variants(specs, R, s1_like(1))
R2 = np.full((20, nV, 2), 5, dtype=np.int64)
R2[idx_c["only co-listening"]] = 1
rv2 = G.summarize_variants(specs, R2, s1_like(1))
gain = {
    t_: rv["variants"][f"without {t_}"]["diff_to_full"]["mean"]
    for t_ in ("tag_affinity", "audio_fit", "popularity")
}
check(
    0.02 * rv["variants"]["full"]["score"] < gain["popularity"] < 0.02
    and gain["tag_affinity"] < 0.02 * rv["variants"]["full"]["score"]
    and rv["readings"]["V3"]
    == [
        {"term": "tag_affinity", "gain": gain["tag_affinity"], "above_sesoi": False},
        {"term": "audio_fit", "gain": gain["audio_fit"], "above_sesoi": True},
        {"term": "popularity", "gain": gain["popularity"], "above_sesoi": True},
    ]
    and rv["readings"]["V2"]["holds"] is False
    and rv2["readings"]["V2"]["holds"] is True
    and rv2["readings"]["V3"] == [],
    "V2 and V3 on constructed ranks: a term hurts when its removal gains "
    "with an interval above 0, above or below SESOI, and the group "
    "variant is no single term",
)

m2 = fresh("s2")
for name in G.SLICES_FILES:
    shutil.copy(m1 / name, m2 / name)
# The layouts of als_tune.py's and als_tune_d.py's results.
(m2 / "als_tuning.json").write_text(
    json.dumps(
        {
            "reference": {
                "score": 0.003,
                "val_last": {"recall_at_100": 0.02},
                "val_first": {"recall_at_100": 0.007},
            },
            "winner": {"alpha": 1.0, "reg": 100.0, "factors": 64},
            "configs": {
                "x": {
                    "finished": True,
                    "alpha": 1.0,
                    "reg": 100.0,
                    "factors": 64,
                    "score": 0.017,
                    "best_val_last": {"recall_at_100": 0.085},
                    "best_val_first": {"recall_at_100": 0.02},
                }
            },
        }
    )
)
(m2 / "als_tuning_d.json").write_text(
    json.dumps(
        {
            "winner": {
                "config": "D3 reg 100",
                "cell": "D3",
                "factors": 256,
                "nu": 0.0,
                "alpha": 1.0,
                "reg": 100.0,
                "iterations": 2,
                "score": 0.02674,
                "val_last": {"recall_at_100": 0.1},
                "val_first": {"recall_at_100": 0.03},
            }
        }
    )
)
lines2 = run2(m2)
doc2v = json.loads((m2 / "gam_knn_variants.json").read_text())
zv = load_npz(m2 / "gam_knn_variants.npz")
RV = zv["ranks"]
check(
    RV.shape == (20, len(USERS), 2)
    and np.array_equal(RV[0], G_RANK)
    and np.array_equal(zv["user_id"], USERS)
    and doc2v["variants"] == [[n, list(k)] for n, k in specs],
    "stage 2 stores 20 variants in the declared order, and the full "
    "variant's ranks equal P13b part 2's stored ranks",
)
ok, near_n, exact_n = True, 0, 0
for c, (name, kept) in enumerate(specs):
    S = dense_scores(kept)
    for j, u in enumerate(USERS):
        excl = W.u_indices[W.u_indptr[u] : W.u_indptr[u + 1]]
        for p in range(2):
            lo, hi = rank_range(S[j], int(PICKS[j, p]), excl)
            near_n += int(hi > lo)
            exact_n += int(hi == lo)
            got = int(RV[c, j, p])
            if not min(lo, 101) <= got <= min(hi, 101):
                ok = False
                print(
                    f"      {name}, listener {u}, pick {p}: {got} not in "
                    f"[{lo}, {hi}]"
                )
check(
    ok and exact_n > 0.9 * RV.size,
    f"every variant's rank of every pick equals a dense count from the "
    f"stored terms and tables ({exact_n} exact, {near_n} with a near tie, "
    f"censored at 101)",
)
res2 = doc2v["results"]
sc = ga.per_listener_scores(RV)
ok = all(
    close(res2["variants"][n]["score"], float(sc[c].mean()))
    and close(
        res2["variants"][n]["diff_to_full"]["mean"], float((sc[c] - sc[0]).mean())
    )
    for c, (n, _) in enumerate(specs)
)
idx = {n: c for c, (n, _) in enumerate(specs)}
comp = res2["comparisons"]
ok &= close(
    comp["only co-listening minus ItemKNN, both lists"]["mean"],
    float(
        (sc[idx["only co-listening"]] - ga.per_listener_scores(z1["knn_both"])).mean()
    ),
)
ok &= close(
    comp["only colisten_same_artist minus ItemKNN, same-artist " "list, clamped"][
        "mean"
    ],
    float(
        (
            sc[idx["only colisten_same_artist"]]
            - ga.per_listener_scores(z1["clamp_same"])
        ).mean()
    ),
)
ok &= close(
    comp["full minus only co-listening"]["mean"],
    float((sc[0] - sc[idx["only co-listening"]]).mean()),
)
pl1 = ga.per_listener_scores
want_c = {
    "only colisten_same_artist minus ItemKNN, same-artist list": sc[
        idx["only colisten_same_artist"]
    ]
    - pl1(z1["knn_same"]),
    "only colisten_same_artist minus ItemKNN, same-artist list, clamped": sc[
        idx["only colisten_same_artist"]
    ]
    - pl1(z1["clamp_same"]),
    "only co-listening minus ItemKNN, both lists": sc[idx["only co-listening"]]
    - pl1(z1["knn_both"]),
    "only co-listening minus ItemKNN, both lists, clamped": sc[idx["only co-listening"]]
    - pl1(z1["clamp_both"]),
    "full minus only co-listening": sc[0] - sc[idx["only co-listening"]],
    "full minus ItemKNN, both lists": sc[0] - pl1(z1["knn_both"]),
}
ok &= set(comp) == set(want_c) and all(
    close(comp[k]["mean"], float(v.mean()))
    and close(comp[k]["se"], float(v.std(ddof=1) / math.sqrt(len(v))))
    for k, v in want_c.items()
)
slice_ok = True
band1 = np.digitize(z1["pop_centile"], [10.5, 50.5, 90.5])
masks1 = {
    f"co-listening evidence: {nm}": z1["evidence"] == c
    for c, nm in enumerate(G.EVIDENCE)
}
masks1["same-artist value: below the top knot"] = (z1["evidence"] == 0) & ~z1[
    "flat_same"
]
masks1["same-artist value: on the flat top"] = (z1["evidence"] == 0) & z1["flat_same"]
masks1.update(
    {f"popularity band: {nm}": band1 == c for c, nm in enumerate(G.POP_BANDS)}
)
slice_ok &= res2["slice_picks"] == {k: int(v.sum()) for k, v in masks1.items()}
for c, (n, _) in enumerate(specs):
    slice_ok &= set(res2["slices"][n]) == set(masks1)
    for key, msk in masks1.items():
        vals = [
            dcg_list([RV[c, j, p]])[0]
            for j in range(len(USERS))
            for p in range(2)
            if msk[j, p]
        ]
        want = sum(vals) / len(vals) if vals else None
        slice_ok &= close(res2["slices"][n][key], want)
res_low = G.summarize_variants(specs, RV, z_l)
pl1 = ga.per_listener_scores
low_ok = all(
    close(res_low["comparisons"][k]["mean"], float(v.mean()))
    for k, v in {
        "only colisten_same_artist minus ItemKNN, same-artist list, clamped": sc[
            idx["only colisten_same_artist"]
        ]
        - pl1(z_l["clamp_same"]),
        "only co-listening minus ItemKNN, both lists, clamped": sc[
            idx["only co-listening"]
        ]
        - pl1(z_l["clamp_both"]),
        "only co-listening minus ItemKNN, both lists": sc[idx["only co-listening"]]
        - pl1(z_l["knn_both"]),
    }.items()
)
low_ok &= all(
    close(
        res_low["slices"][n_][k_],
        None if not m_.any() else float(np.mean(G.pick_dcg(RV[c_])[m_])),
    )
    for c_, (n_, _) in enumerate(specs)
    for k_, m_ in (
        (
            "same-artist value: on the flat top",
            (z_l["evidence"] == 0) & z_l["flat_same"],
        ),
        (
            "same-artist value: below the top knot",
            (z_l["evidence"] == 0) & ~z_l["flat_same"],
        ),
    )
)
check(
    ok and slice_ok and low_ok,
    "the variant scores, the differences to the full model, the "
    "comparisons with ItemKNN and the slice means follow from the ranks, "
    "also with stage 1's low-knot outputs",
)
full_s = float(sc[0].mean())
d1 = comp["only colisten_same_artist minus ItemKNN, same-artist list, " "clamped"]
s_ = 0.02 * float(ga.per_listener_scores(z1["clamp_same"]).mean())
want_v1 = (
    "within"
    if abs(d1["mean"]) <= s_
    else (
        "below"
        if d1["mean"] < -s_ and d1["hi"] < 0
        else "above" if d1["mean"] > s_ and d1["lo"] > 0 else "unclear"
    )
)
hurt_want = [
    n[len("without ") :]
    for n, _ in specs
    if n.startswith("without ")
    and n != "without co-listening"
    and res2["variants"][n]["diff_to_full"]["lo"] > 0
]
d2 = ga.paired_summary(sc[idx["only co-listening"]], sc[0])
check(
    res2["readings"]["V1"]["outcome"] == want_v1
    and res2["readings"]["V2"]["holds"]
    == bool(d2["mean"] > 0.02 * full_s and d2["lo"] > 0)
    and [h["term"] for h in res2["readings"]["V3"]] == hurt_want,
    f"the readings V1 ({want_v1}), V2 and V3 ({hurt_want}) follow from " f"the ranks",
)
rep2 = (m2 / "reports" / "gam_knn_variants.md").read_text()
check(
    ";" not in rep2
    and all(
        h in rep2
        for h in (
            "## 1. Variants",
            "## 2. Other models",
            "## 3. Paired comparisons",
            "## 4. Variants by slice",
            "## 5. Reading",
            "- V1,",
            "- V2,",
            "- V3",
        )
    )
    and f"| full | {res2['variants']['full']['score']:.5f} |" in rep2
    and "| ItemKNN, both lists |" in rep2
    and [r_[0] for r_ in doc2v["references"]]
    == [nm for key, nm in G.MODELS if key != "gam_rank"]
    + [
        "popularity",
        "ALS, 64 factors (phases A to C)",
        "ALS, phase D winner (256 factors, nu 0)",
    ]
    and all(
        close(r_[1], doc1["results"]["models"][r_[0]]["score"])
        for r_ in doc2v["references"][:4]
    )
    and [r_[1] for r_ in doc2v["references"][4:]] == [0.003, 0.017, 0.02674]
    and "| ALS, phase D winner (256 factors, nu 0) | 0.02674 |" in rep2,
    "the stage 2 report has every section, the scores and the ItemKNN "
    "references, without a semicolon",
)
check(
    doc2v["check"]["full_equals_stored"] is True
    and doc2v["check"]["dense"]["listeners"] == 90
    and not any(x.startswith(("p13a", "p13b")) for x in lines2)
    and any(x.startswith("gam_knn_diag: probe mean") for x in lines2)
    and any(
        x.startswith("gam_knn_diag: selection pass, 20 " "configurations")
        for x in lines2
    ),
    "stage 2 checks every listener against the dense reference, and "
    "every pass line names this script",
)

lines_again = run2(m2)
check(
    any("selection pass resume, 13 of 13 blocks done" in x for x in lines_again)
    and not any("probe listener" in x for x in lines_again)
    and np.array_equal(load_npz(m2 / "gam_knn_variants.npz")["ranks"], RV),
    "a finished pass scores nothing when it runs again, with the same " "ranks",
)
part = sorted((m2.parent / "scratch" / G.WORK_NAME / "selection").glob("part_*.npz"))
part[4].unlink()
lines_lost = run2(m2)
check(
    any("selection pass resume, 12 of 13 blocks done" in x for x in lines_lost)
    and np.array_equal(load_npz(m2 / "gam_knn_variants.npz")["ranks"], RV),
    "a lost block is scored again, with the same ranks",
)
m3 = fresh("s2_three")
for name in G.SLICES_FILES:
    shutil.copy(m1 / name, m3 / name)
run2(m3, workers=3, n_verify=4)
check(
    np.array_equal(load_npz(m3 / "gam_knn_variants.npz")["ranks"], RV)
    and json.loads((m3 / "gam_knn_variants.json").read_text())["results"] == res2,
    "three workers give the same ranks and results as one",
)

# 5: stage 2, refusals and faults


def s2_fault(prepare, text, n_verify=4, st=None):
    m = fresh("s2_fault")
    for name in G.SLICES_FILES:
        shutil.copy(m1 / name, m / name)
    prepare(m)
    return raises(lambda: run2(m, n_verify=n_verify, st=st), text)


def no_slices(m):
    (m / G.SLICES_FILES[1]).unlink()


def edit_npz(name, change):
    def f(m):
        d = load_npz(m / name)
        change(d)
        np.savez(m / name, **d)

    return f


def edit_json(name, change):
    def f(m):
        d = json.loads((m / name).read_text())
        change(d)
        (m / name).write_text(json.dumps(d))

    return f


def both_ranks(m):
    edit_npz("p13b2_arrays.npz", bump_final)(m)
    edit_npz(G.SLICES_FILES[1], bump("gam_rank", 2, 0, 1))(m)


def bump_final(d):
    d["ranks"] = d["ranks"].copy()
    d["ranks"][FIN, 2, 0] += 1


def edit_params(d):
    d["params"]["tau"] = 0.4


def rename_term(d):
    for t_ in d["wide_fits"][FIN]["terms"]:
        if t_["name"] == "audio_fit":
            t_["name"] = "audio_fit_x"


def tier_term(d):
    for t_ in d["wide_fits"][FIN]["terms"]:
        if t_["name"] == "audio_fit":
            t_["tiers"] = [3]


def other_picks(d):
    d["picks"] = d["picks"].copy()
    d["picks"][0, 0] += 1


ok = True
for prepare, text in (
    (no_slices, "is missing: run python3 -m pipeline.runner --only " "gam_knn_slices"),
    (
        edit_npz(G.SLICES_FILES[1], bump("gam_rank", 2, 0, 1)),
        "belongs to other listeners, picks or another final model",
    ),
    (
        edit_npz(G.SLICES_FILES[1], other_picks),
        "belongs to other listeners, picks or another final model",
    ),
    (
        both_ranks,
        "full variant's ranks differ from P13b part 2's stored "
        "ranks of the final model in 1 picks",
    ),
    (edit_json("p13b2_model.json", edit_params), "other P13b part 2 parameters"),
    (
        edit_json("p13b2_model.json", rename_term),
        "terms outside the union: ['audio_fit_x']",
    ),
    (
        edit_json("p13b2_model.json", tier_term),
        "the term audio_fit does not apply to every candidate",
    ),
):
    one_ok = s2_fault(prepare, text)
    if not one_ok:
        print(f"      not refused: {text}")
    ok &= one_ok
check(
    ok,
    "stage 2 refuses a missing or foreign stage 1 output, a changed "
    "final model, other parameters and a term outside the union",
)

real_cut = ct.CatalogScorer.ranks_cut
C_POP = idx["only popularity"]
J_POP = next(j for j in range(len(USERS)) if RV[C_POP, j, 0] <= 99)
PLANT = {}


def planted_cut(self, state, space, m_u, picks, exclude, cut, chunk=None):
    rk, sc_ = real_cut(self, state, space, m_u, picks, exclude, cut, chunk)
    if self.C == 20 and np.array_equal(picks, PICKS[PLANT["j"]]):
        rk = rk.copy()
        rk[C_POP, 0] = PLANT["rank"](int(rk[C_POP, 0]))
    return rk, sc_


ct.CatalogScorer.ranks_cut = planted_cut
ok = True
try:
    for j_, fn_ in (
        (J_POP, lambda r_: r_ + 1),
        (J_POP, lambda r_: 101),
        (next(j for j in range(len(USERS)) if RV[C_POP, j, 0] == 101), lambda r_: 1),
    ):
        PLANT.update(j=j_, rank=fn_)
        ok &= s2_fault(
            lambda m: None, "lies outside the dense reference's " "range", n_verify=90
        )
finally:
    ct.CatalogScorer.ranks_cut = real_cut
check(
    ok,
    f"the dense check of stage 2 fails when the scorer gives the "
    f"variant 'only popularity' (index {C_POP}) a rank one too low, "
    f"101 for a hit or 1 for a miss",
)
s1_arrays = load_npz(m1 / G.SLICES_FILES[1])
for fault, text in (
    (
        lambda d: d["results"]["variants"]["only popularity"].update(score=0.5),
        "results do not follow from the stored ranks",
    ),
    (lambda d: d.update(variants=d["variants"][::-1]), "stored variants differ"),
):
    m = fresh("s2_out")
    for name in G.VARIANTS_FILES:
        shutil.copy(m2 / name, m / name)
    d = json.loads((m / G.VARIANTS_FILES[0]).read_text())
    fault(d)
    (m / G.VARIANTS_FILES[0]).write_text(json.dumps(d))
    check(
        raises(lambda: G.check_variants(m, specs, s1_arrays, G_RANK), text),
        f"stage 2's output check fails: {text}",
    )
for c_, text in (
    (0, "stored full variant differs"),
    (5, "results do not follow from the stored ranks"),
):
    m = fresh("s2_out")
    for name in G.VARIANTS_FILES:
        shutil.copy(m2 / name, m / name)
    d = load_npz(m / G.VARIANTS_FILES[1])
    d["ranks"] = d["ranks"].copy()
    d["ranks"][c_, 3, 1] = 1 if d["ranks"][c_, 3, 1] > 1 else 101
    np.savez(m / G.VARIANTS_FILES[1], **d)
    check(
        raises(lambda: G.check_variants(m, specs, s1_arrays, G_RANK), text),
        f"stage 2's output check fails for a changed rank of variant " f"{c_}",
    )
G._atomic_savez = bad_savez("ranks", (5, 3, 1), lambda r_: 1 if r_ > 1 else 101)
try:
    ok = s2_fault(lambda m: None, "results do not follow from the stored " "ranks")
finally:
    G._atomic_savez = real_savez
check(
    ok,
    "stage 2 checks what it wrote: a rank changed on the way to the "
    "disk stops the run",
)
check(
    not raises(lambda: G.check_variants(m2, specs, s1_arrays, G_RANK), "no such text"),
    "stage 2's output check passes on the untouched outputs",
)

print()
if failures:
    print(f"{len(failures)} check(s) failed:")
    for f_ in failures:
        print(f"  - {f_}")
    raise SystemExit(1)
print("all checks passed")
