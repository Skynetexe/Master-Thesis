"""Checks for pipeline/itemknn.py and itemknn_select.py on the synthetic
world of p13_world.py (the world of the P13 tests: every concept source,
listeners with histories, selection listeners with val picks).

1. The transposed lists are concepts.ItemSide's own matrices, and every
   user's scores equal the numerators of the GAM's two co-listening
   concepts bit for bit (and the concept values times the user's total
   weight to rounding).
2. A second route (brute_scores) gives the same scores to rounding for
   every user, and a slow loop over the raw lists gives the same values
   for sampled (user, item) pairs.
3. Ranks under both tie rules equal a full sort of the candidates, for
   the held-out picks and for random candidates, and a train item as pick
   gets the rank n_slots.
4. One worker and three workers give the same ranks.
5. The loader refuses damaged list files.
6. itemknn_select.run() writes consistent outputs, its check passes, the
   report names the other models, and the check fails for planted faults
   in the stored outputs and in the lists.

Usage, from the repository root:
    python3 test_itemknn.py
"""

import copy
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

import p13_world as W
from pipeline import concept_data as cd
from pipeline import concepts as cn
from pipeline import itemknn as knn

from pipeline import itemknn_select as sel

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


print(W.summary())
NI = W.NI
U_ARRAYS = (W.u_indptr, W.u_indices, W.u_data)
USERS = np.arange(1, len(W.u_indptr) - 1)
LISTS = copy.deepcopy(W.SRC.colisten)
LISTS = {
    k: (
        np.asarray(v[0], dtype=np.int64),
        np.asarray(v[1], np.int32),
        np.asarray(v[2], dtype=np.float64),
    )
    for k, v in LISTS.items()
}
POP = np.asarray(W.listens, dtype=np.int64).copy()
POP[0] = -1

# 1: the GAM's numerators
T = knn.transposed(LISTS, NI)
same_t = all(
    np.array_equal(T[k].indptr, W.SPACE.colisten_t[k].indptr)
    and np.array_equal(T[k].indices, W.SPACE.colisten_t[k].indices)
    and np.array_equal(
        T[k].data.view(np.uint64), W.SPACE.colisten_t[k].data.view(np.uint64)
    )
    for k in knn.KINDS
)
check(
    same_t,
    "the transposed lists equal concepts.ItemSide.colisten_t, " "array for array",
)
check(
    all(
        np.array_equal(LISTS[k][1], W.SRC.colisten[k][1])
        and np.array_equal(LISTS[k][2], W.SRC.colisten[k][2])
        for k in knn.KINDS
    ),
    "transposed() leaves the caller's lists as they were",
)

bits_ok, value_ok, mask_ok = True, True, True
for u in USERS:
    items, weights = knn.history(*U_ARRAYS, int(u))
    got = knn.user_scores(T, items, weights)
    ih, wh = cd.history_of(*U_ARRAYS, int(u))
    state = cn.build_user_state(ih, wh, W.SPACE)
    full = cn.full_catalog(state, W.SPACE)
    for concept, kind in cn.COLISTEN_CONCEPTS.items():
        numer = np.asarray(
            W.SPACE.colisten_t[kind][state.items].T @ state.weights, dtype=np.float64
        ).ravel()
        bits_ok &= np.array_equal(got[kind].view(np.uint64), numer.view(np.uint64))
        val, has = full[concept]
        value_ok &= np.allclose(
            got[kind][has], val[has] * state.total, rtol=1e-12, atol=0
        )
        mask_ok &= np.array_equal(has, W.SPACE.has_colisten[kind]) and not np.any(
            got[kind][~has]
        )
    both = got["same_artist"] + got["other_artists"]
    bits_ok &= np.array_equal(got["both"].view(np.uint64), both.view(np.uint64))
check(
    bits_ok,
    f"for all {len(USERS)} users, each list's score equals the "
    f"GAM's co-listening numerator bit for bit, and 'both' is "
    f"their sum",
)
check(
    value_ok and mask_ok,
    "the scores equal the concept values times the user's total weight "
    "(to 1e-12), and a track without neighbours of a kind scores 0",
)

# 2: a second route
worst = 0.0
for u in USERS:
    items, weights = knn.history(*U_ARRAYS, int(u))
    fast = knn.user_scores(T, items, weights)
    slow = knn.brute_scores(LISTS, items, weights, NI)
    for k in ("same_artist", "other_artists", "both"):
        worst = max(worst, float(np.max(np.abs(fast[k] - slow[k]))))
check(
    worst <= 1e-12,
    f"brute_scores() equals user_scores() for every user " f"(largest gap {worst:.1e})",
)
rng = np.random.default_rng(29)
loop_ok = True
for u in rng.choice(USERS, 25, replace=False):
    items, weights = knn.history(*U_ARRAYS, int(u))
    hist = dict(zip(items.tolist(), weights.tolist()))
    fast = knn.user_scores(T, items, weights)
    for i in rng.choice(np.arange(1, NI), 40, replace=False):
        total = 0.0
        for k in knn.KINDS:
            ip, nb, cs = W.SRC.colisten[k]
            s, e = int(ip[i]), int(ip[i + 1])
            val = math.fsum(
                float(c) * hist.get(int(h), 0.0) for h, c in zip(nb[s:e], cs[s:e])
            )
            loop_ok &= abs(val - fast[k][i]) <= 1e-12 * max(1.0, val)
            total += val
        loop_ok &= abs(total - fast["both"][i]) <= 1e-12 * max(1.0, total)
check(
    loop_ok,
    "a slow loop over the raw lists gives the same score for "
    "1,000 sampled (user, track) pairs",
)

# 3: ranks
rank_ok, zero_seen, pos_seen, tie_pop_diff = True, 0, 0, 0
for u in USERS:
    items, weights = knn.history(*U_ARRAYS, int(u))
    sc = knn.user_scores(T, items, weights)
    cands = np.setdiff1d(np.arange(1, NI), items)
    picks = np.concatenate([rng.choice(cands, 6, replace=False)])
    for lists_, tie in knn.VARIANTS:
        s = knn.candidate_scores(sc[lists_], items)
        got = knn.pick_ranks(s, picks, tie, POP)
        want = [knn.sorted_rank(s, int(p), tie, POP) for p in picks]
        rank_ok &= got.tolist() == want
        if lists_ == "both":
            zero_seen += int(np.count_nonzero(s[picks] == 0.0))
            pos_seen += int(np.count_nonzero(s[picks] > 0.0))
    s = knn.candidate_scores(sc["both"], items)
    tie_pop_diff += int(
        np.count_nonzero(
            knn.pick_ranks(s, picks, "id", POP) != knn.pick_ranks(s, picks, "pop", POP)
        )
    )
check(
    rank_ok and zero_seen > 50 and pos_seen > 50,
    f"ranks under both tie rules equal a full sort of the candidates "
    f"({zero_seen} picks with score 0, {pos_seen} with a positive score)",
)
check(
    tie_pop_diff > 50,
    f"the popularity tie rule changes {tie_pop_diff} "
    f"ranks, all among picks with tied scores",
)
items, weights = knn.history(*U_ARRAYS, 5)
s = knn.candidate_scores(knn.user_scores(T, items, weights)["both"], items)
check(
    knn.pick_ranks(s, [int(items[0])], "id", POP)[0] == NI
    and knn.pick_ranks(s, [int(items[0])], "pop", POP)[0] == NI
    and raises(lambda: knn.pick_ranks(s, [1], "other", POP), "unknown tie rule"),
    "a train item as pick ranks n_slots (the rule of als.rank_items), and "
    "an unknown tie rule stops",
)
# A hand-made case: three candidates with a positive score, the rest 0.
s_hand = np.zeros(10)
s_hand[[2, 5, 7]] = [0.5, 0.25, 0.5]
s_hand[[0, 3]] = -np.inf
pop_hand = np.array([-1, 5, 9, 0, 9, 1, 2, 7, 9, 3])
check(
    knn.pick_ranks(s_hand, [7, 2, 5, 4, 1, 9], "id", pop_hand).tolist()
    == [2, 1, 3, 5, 4, 8]
    and knn.pick_ranks(s_hand, [7, 2, 5, 4, 1, 9], "pop", pop_hand).tolist()
    == [2, 1, 3, 4, 6, 7],
    "a hand-made catalog ranks as worked out on paper under both rules",
)
check(
    knn.rank_bounds(s_hand, 7, 0.0) == (1, 2)
    and knn.rank_bounds(s_hand, 4, 0.0) == (4, 8)
    and knn.rank_bounds(s_hand, 5, 0.3) == (1, 8)
    and knn.rank_bounds(s_hand, 5, 0.1) == (3, 3),
    "rank_bounds() gives the rank range of a pick when scores within the "
    "tolerance may tie, worked out on paper",
)
check(
    knn.sorted_rank(s_hand, 3, "id") == 10
    and knn.sorted_rank(s_hand, 0, "pop", pop_hand) == 10
    and knn.sorted_rank(s_hand, 9, "id") == 8,
    "sorted_rank() gives n_slots to a non-candidate and the paper rank to "
    "a candidate",
)

# 4: workers
users_s = W.SEL
picks_s = W.SEL_PICKS
r1, p1 = knn.rank_all(T, U_ARRAYS, users_s, picks_s, POP, workers=1)
r3, p3 = knn.rank_all(T, U_ARRAYS, users_s, picks_s, POP, workers=3)
per_user = True
for k, u in enumerate(users_s):
    items, weights = knn.history(*U_ARRAYS, int(u))
    r, pos = knn.user_ranks(T, items, weights, picks_s[k], POP)
    per_user &= all(np.array_equal(r[v], r1[v][k]) for v in knn.VARIANTS)
    per_user &= pos == p1[k]
check(
    all(np.array_equal(r1[v], r3[v]) for v in knn.VARIANTS)
    and np.array_equal(p1, p3)
    and per_user,
    "one worker and three workers give the same ranks, equal to the "
    "per-user computation",
)
indep_ok, n_zero = True, 0
for k, u in enumerate(users_s):
    items, weights = knn.history(*U_ARRAYS, int(u))
    sc = knn.user_scores(T, items, weights)
    cands = set(range(1, NI)) - set(items.tolist())
    for v in knn.VARIANTS:
        s_v = sc[v[0]]
        key = (
            (lambda i: (-s_v[i], i))
            if v[1] == "id"
            else (lambda i: (-s_v[i], -POP[i], i))
        )
        pos = {i: r + 1 for r, i in enumerate(sorted(cands, key=key))}
        want = [pos.get(int(p), NI) for p in picks_s[k]]
        indep_ok &= r1[v][k].tolist() == want
    n_zero += int(np.count_nonzero(sc["both"][picks_s[k]] == 0.0))
check(
    indep_ok and n_zero > 10,
    f"rank_all() equals an independent Python sort over the slots 1 to "
    f"n_slots - 1 without the train items, for every selection user and "
    f"variant ({n_zero} picks with score 0)",
)
check(
    raises(
        lambda: knn.rank_all(T, U_ARRAYS, users_s, picks_s[:-1], POP),
        "one row per user",
    ),
    "rank_all() refuses picks of another shape",
)
check(
    W.u_indptr[1] == W.u_indptr[0]
    and raises(
        lambda: knn.rank_all(T, U_ARRAYS, np.array([0]), picks_s[:1], POP, workers=1),
        "has no train items",
    ),
    "rank_all() refuses a user without train items (slot 0 here)",
)

# 5: the loader
TMP = Path(tempfile.mkdtemp(prefix="test_itemknn_"))


def write_lists(d, change=None):
    d.mkdir(parents=True, exist_ok=True)
    for kind in knn.KINDS:
        ip, nb, cs = LISTS[kind]
        arrs = {"indptr": ip.copy(), "neighbor": nb.copy(), "cosine": cs.copy()}
        if change and change[0] == kind:
            change[1](arrs)
        np.savez(d / knn.LIST_FILES[kind], **arrs)


good_dir = TMP / "lists_good"
write_lists(good_dir)
loaded = knn.load_lists(good_dir, NI)
check(
    all(np.array_equal(loaded[k][i], LISTS[k][i]) for k in knn.KINDS for i in range(3)),
    "load_lists() reads the lists as written",
)
faults = [
    ("rows", lambda a: a.update(indptr=a["indptr"][:-1]), "rows, the catalog"),
    ("length", lambda a: a.update(cosine=a["cosine"][:-1]), "do not match indptr"),
    ("range", lambda a: a["neighbor"].__setitem__(0, NI), "outside the item slots"),
    ("zero", lambda a: a["neighbor"].__setitem__(1, 0), "outside the item slots"),
    ("negative", lambda a: a["cosine"].__setitem__(2, -0.1), "not a finite positive"),
    ("zero cosine", lambda a: a["cosine"].__setitem__(2, 0.0), "not a finite positive"),
    ("nan", lambda a: a["cosine"].__setitem__(3, np.nan), "not a finite positive"),
    ("inf", lambda a: a["cosine"].__setitem__(3, np.inf), "not a finite positive"),
]
bad_ok = True
for name, fn, text in faults:
    d = TMP / f"lists_{name}"
    write_lists(d, ("other_artists", fn))
    ok = raises(lambda: knn.load_lists(d, NI), text)
    if not ok:
        print(f"      fault {name} was not refused")
    bad_ok &= ok
check(bad_ok, f"load_lists() refuses {len(faults)} kinds of damage")

# 6: the script
model = TMP / "model"
(model / "reports").mkdir(parents=True)
json.dump(
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
    },
    open(model / "als_tuning.json", "w"),
)
json.dump(
    {
        "final": {"index": 1},
        "wide_fits": [
            {"cell": "a", "arm": "b", "level": 1, "selection": {"mean": 0.1}},
            {
                "cell": "m0w0t0p0",
                "arm": "p_pop-B",
                "level": 64,
                "selection": {"mean": 0.04117},
            },
        ],
    },
    open(model / "p13b2_model.json", "w"),
)
lines = []
res = sel.run(
    model,
    NI,
    U_ARRAYS,
    users_s,
    picks_s,
    POP,
    LISTS,
    workers=2,
    say=lines.append,
    n_check=12,
    inputs={"marker:p6_als": "x"},
)
stored = json.loads((model / "itemknn_selection.json").read_text())
z = np.load(model / "itemknn_selection_ranks.npz")
want = sel.results_from_ranks(r1, p1, NI)
check(
    all(
        json.dumps(stored[k], sort_keys=True) == json.dumps(want[k], sort_keys=True)
        for k in want
    )
    and stored["inputs"] == {"marker:p6_als": "x"}
    and stored["check"]["users"] == 12
    and all(np.array_equal(z[sel.npz_key(v)], r1[v]) for v in knn.VARIANTS)
    and np.array_equal(z["user_id"], users_s),
    "run() stores the ranks and metrics of rank_all(), the inputs and the " "check",
)
s_id = stored["variants"]["both/id"]["score"]
s_pop = stored["variants"]["both/pop"]["score"]
check(
    stored["chosen"] == ("both/pop" if s_pop > s_id else "both/id")
    and sel.results_from_ranks({**r1, ("both", "pop"): r1[("both", "id")]}, p1, NI)[
        "chosen"
    ]
    == "both/id",
    f"the chosen variant follows the rule (id {s_id:.5f}, pop "
    f"{s_pop:.5f}), and a tie goes to id",
)
nd = [(r1[v][:, 0], r1[v][:, 1]) for v in knn.VARIANTS]
ndcg = [
    float(
        np.mean(np.where(a <= 100, 1 / np.log2(a + 1.0), 0)) / 2
        + np.mean(np.where(b <= 100, 1 / np.log2(b + 1.0), 0)) / 2
    )
    for a, b in nd
]
check(
    all(
        abs(stored["variants"][knn.variant_name(*v)]["score"] - s) < 1e-15
        for v, s in zip(knn.VARIANTS, ndcg)
    ),
    "each score is the mean NDCG@100 over the two picks, recomputed here",
)
pos_hand = [
    int(
        np.count_nonzero(
            knn.candidate_scores(
                knn.user_scores(T, *knn.history(*U_ARRAYS, int(u)))["both"],
                knn.history(*U_ARRAYS, int(u))[0],
            )
            > 0
        )
    )
    for u in users_s
]
check(
    stored["counts"]
    == {
        "users": len(users_s),
        "users_below_100_positive": sum(1 for x in pos_hand if x < 100),
        "median_positive_candidates": float(np.median(pos_hand)),
        "n_slots": NI,
    }
    and 0 < sum(1 for x in pos_hand if x < 100) < len(users_s),
    "the stored counts of positive candidates follow from the scores, "
    "with users on both sides of 100",
)
rep = (model / "reports" / "itemknn_selection.md").read_text()
check(
    "| both/id |" in rep
    and "| ALS, 64 factors (phases A to C) | " "0.01700 |" in rep
    and "GAM, final model of P13b part 2 "
    "(m0w0t0p0, p_pop-B, level 64) | 0.04117 |" in rep
    and "| popularity | 0.00300 |" in rep
    and ";" not in rep,
    "the report holds every variant and the other models on the same "
    "picks, without a semicolon",
)
check(
    any("check passed on 12 users" in x for x in lines)
    and any(x.startswith("itemknn: chosen for P14:") for x in lines),
    "run() logs its check and its choice",
)


def with_npz(change, text, lists=LISTS, t=T):
    path = model / "itemknn_selection_ranks.npz"
    good = dict(np.load(path))
    bad = {k: v.copy() for k, v in good.items()}
    change(bad)
    np.savez_compressed(path, **bad)
    try:
        return raises(
            lambda: sel.verify(
                model,
                t,
                lists,
                U_ARRAYS,
                users_s,
                picks_s,
                POP,
                NI,
                say=lambda m: None,
                n_check=12,
            ),
            text,
        )
    finally:
        np.savez_compressed(path, **good)


def with_json(change, text):
    path = model / "itemknn_selection.json"
    good = path.read_text()
    d = json.loads(good)
    change(d)
    path.write_text(json.dumps(d))
    try:
        return raises(
            lambda: sel.verify(
                model,
                T,
                LISTS,
                U_ARRAYS,
                users_s,
                picks_s,
                POP,
                NI,
                say=lambda m: None,
                n_check=12,
            ),
            text,
        )
    finally:
        path.write_text(good)


checked = sel.check_users(len(users_s), 12)
unchecked = next(k for k in range(len(users_s)) if k not in set(checked))
k0 = int(checked[0])


def bump(key, k, j, by=1):
    def f(d):
        d[key][k, j] += by

    return f


check(
    not raises(
        lambda: sel.verify(
            model,
            T,
            LISTS,
            U_ARRAYS,
            users_s,
            picks_s,
            POP,
            NI,
            say=lambda m: None,
            n_check=12,
        ),
        "no such text",
    ),
    "verify() passes on the stored outputs",
)
check(
    with_npz(bump("both_id", k0, 0), "a full sort gives"),
    "verify() fails when a checked user's stored rank changes",
)
check(
    with_npz(
        bump("same_artist_id", unchecked, 1), "do not follow from the stored ranks"
    ),
    "verify() fails when an unchecked user's stored rank changes the " "metrics",
)
check(
    with_npz(
        lambda d: d["user_id"].__setitem__(0, d["user_id"][0] + 1000),
        "stored users differ",
    ),
    "verify() fails when the stored users differ",
)
check(
    with_npz(
        lambda d: d["positive"].__setitem__(k0, d["positive"][k0] + 1),
        "positive candidates differs",
    ),
    "verify() fails when a stored count of positive candidates changes",
)
check(
    with_npz(lambda d: d["both_pop"].__setitem__((unchecked, 0), 0), "outside 1 to")
    and with_npz(
        lambda d: d["other_artists_id"].__setitem__((unchecked, 1), NI + 1),
        "outside 1 to",
    ),
    "verify() fails for a stored rank below 1 or above n_slots",
)
check(
    with_json(
        lambda d: d["variants"]["both/pop"].__setitem__("score", 0.5),
        "stored variants do not follow",
    ),
    "verify() fails when a stored score changes",
)
check(
    with_json(
        lambda d: d.__setitem__(
            "chosen", "both/id" if d["chosen"] == "both/pop" else "both/pop"
        ),
        "stored chosen do not follow",
    ),
    "verify() fails when the stored choice changes",
)
# The lists changed after the run: a checked user's scores move.
items0, _ = knn.history(*U_ARRAYS, int(users_s[k0]))
kind0 = "same_artist"
ip0, nb0, cs0 = LISTS[kind0]
row = next(i for i in range(1, NI) if np.isin(nb0[ip0[i] : ip0[i + 1]], items0).any())
lists_bad = {k: (v[0], v[1], v[2].copy()) for k, v in LISTS.items()}
lists_bad[kind0][2][ip0[row] : ip0[row + 1]] *= 1.5
check(
    with_npz(lambda d: None, "differ from the second route", lists=lists_bad),
    "verify() fails when the lists no longer give the stored scores",
)
# A checked user whose pick is one of the user's train items.
model_t = TMP / "model_train_pick"
(model_t / "reports").mkdir(parents=True)
picks_t = picks_s.copy()
k_t = int(checked[1])
picks_t[k_t, 0] = int(knn.history(*U_ARRAYS, int(users_s[k_t]))[0][0])
sel.run(
    model_t,
    NI,
    U_ARRAYS,
    users_s,
    picks_t,
    POP,
    LISTS,
    workers=1,
    say=lambda m: None,
    n_check=12,
)
z_t = dict(np.load(model_t / "itemknn_selection_ranks.npz"))
stored_nslots = all(int(z_t[sel.npz_key(v)][k_t, 0]) == NI for v in knn.VARIANTS)
bad_t = {k: v.copy() for k, v in z_t.items()}
bad_t["both_id"][k_t, 0] = NI - 1
np.savez_compressed(model_t / "itemknn_selection_ranks.npz", **bad_t)
check(
    stored_nslots
    and raises(
        lambda: sel.verify(
            model_t,
            T,
            LISTS,
            U_ARRAYS,
            users_s,
            picks_t,
            POP,
            NI,
            say=lambda m: None,
            n_check=12,
        ),
        "is a train item but ranks",
    ),
    "a pick that is a train item is stored with rank n_slots, and verify() "
    "fails when that rank changes",
)
shutil.rmtree(TMP, ignore_errors=True)

print()
if failures:
    print(f"{len(failures)} check(s) FAILED:")
    for f in failures:
        print("  - " + f)
    raise SystemExit(1)
print("all checks passed")
