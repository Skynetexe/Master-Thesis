"""Checks for pipeline/fusion_pairs.py and pipeline/stage_p11.py on
synthetic data. Every path points into a temporary folder, and the
DuckDB loaders are replaced by synthetic stand-ins, so nothing touches
the real model folder.

1. pick_table() reads the four roles and rejects a missing or doubled
   role.
2. positives() applies rules 3 and 4 and counts them.
3. The samplers map uniform doubles to the right items, including the
   interval edges, and their frequencies match the probabilities.
4. Each (seed, user, kind) has its own repeatable stream.
   draw_allowed() equals the one-draw-at-a-time reference for many
   exclusion sets, sizes and both samplers, and its accepted draws follow
   the proposal restricted to the allowed items.
5. excluded_items() equals np.union1d, and user_rows() returns the right
   layout, n_allowed and pop_allowed.
6. The stage: run(), verify() and report() on a synthetic catalog with
   every overlap pattern of the held-out picks. The eval users never
   appear, the selection users carry their flag, and the positives equal
   an independent construction.
7. The worker count and the block count do not change a single byte.
8. verify() fails for each planted fault.

Usage, from the repository root:
    python3 test_p11.py
"""

import json
import logging
import math
import tempfile
from pathlib import Path

import numpy as np

from pipeline import fusion_pairs as fp
from pipeline import selection as sel
from pipeline import stage_p11

failures = []


def check(cond, msg):
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        failures.append(msg)


def raises(fn, text):
    try:
        fn()
    except Exception as exc:  # the real check() raises
        return text in str(exc)
    return False


rng = np.random.default_rng(11)

# 1: pick_table.
U = [5, 5, 5, 5, 3, 3, 3]
I = [10, 11, 12, 13, 20, 21, 22]
TL = [1, 0, 0, 0, 1, 0, 0]
VL = [0, 1, 0, 0, 0, 1, 0]
TF = [0, 0, 1, 0, 1, 0, 0]  # user 3: test_last == test_first
VF = [0, 0, 0, 1, 0, 0, 1]
users, picks = fp.pick_table(U, I, TL, VL, TF, VF)
check(
    users.tolist() == [3, 5] and picks.tolist() == [[20, 21, 20, 22], [10, 11, 12, 13]],
    "pick_table() reads the four roles per user",
)
check(
    raises(lambda: fp.pick_table(U[:6], I[:6], TL[:6], VL[:6], TF[:6], VF[:6]), "lacks")
    and raises(lambda: fp.pick_table(U, I, TL, VL, TF, [0, 0, 1, 1, 0, 0, 1]), "twice"),
    "pick_table() rejects a missing role and a doubled role",
)

# 2: positives.
P = np.array(
    [
        [1, 2, 3, 4],  # plain
        [1, 2, 2, 5],  # val_last == test_first
        [1, 2, 3, 1],  # val_first == test_last
        [1, 2, 2, 1],  # both: no positive
        [1, 7, 3, 7],  # val_last == val_first
        [9, 8, 3, 4],
    ]
)  # sorted by item
ip, it, fl, counts = fp.positives(P)
got = [
    list(zip(it[ip[k] : ip[k + 1]].tolist(), fl[ip[k] : ip[k + 1]].tolist()))
    for k in range(len(P))
]
check(
    got == [[(2, 1), (4, 2)], [(5, 2)], [(2, 1)], [], [(7, 3)], [(4, 2), (8, 1)]]
    and counts
    == {
        "val_last_is_test_first": 2,
        "val_first_is_test_last": 2,
        "val_last_is_val_first": 1,
        "no_positive": 1,
    },
    f"positives() applies rules 3 and 4 and counts them ({got}, {counts})",
)

# 3: samplers.
us = fp.uniform_sampler(1, 11)
check(
    us(np.array([0.0, 0.0999, 0.1, 0.5, 1 - 1e-16])).tolist() == [1, 1, 2, 6, 10],
    "uniform_sampler() maps the interval edges",
)
ps = fp.pop_sampler([3, 5, 9], np.cumsum([0.2, 0.3, 0.5]))
check(
    ps(np.array([0.0, 0.1999, 0.2, 0.4999, 0.5, 0.9999])).tolist()
    == [3, 3, 5, 5, 9, 9],
    "pop_sampler() uses half-open intervals",
)
n = 400_000
w = rng.random(40) ** 3 + 1e-3
items = np.sort(rng.choice(np.arange(1, 500), size=40, replace=False))
draws = fp.pop_sampler(items, np.cumsum(w))(rng.random(n))
freq = np.array([(draws == x).sum() for x in items])
exp = n * w / w.sum()
chi2 = float(((freq - exp) ** 2 / exp).sum())
draws_u = fp.uniform_sampler(3, 23)(rng.random(n))
freq_u = np.bincount(draws_u - 3, minlength=20)
chi2_u = float(((freq_u - n / 20) ** 2 / (n / 20)).sum())
# 39 and 19 degrees of freedom: the 99.9 percent points are 72.1 and 43.8.
check(
    freq.sum() == n
    and chi2 < 72.1
    and chi2_u < 43.8
    and draws_u.min() == 3
    and draws_u.max() == 22,
    f"sampler frequencies match the probabilities (chi2 {chi2:.1f} on 39 "
    f"df, {chi2_u:.1f} on 19 df)",
)

# 4: streams, and draw_allowed against the reference.
a0 = fp.user_rng(7, 5, 1).random(8)
check(
    np.array_equal(a0, fp.user_rng(7, 5, 1).random(8))
    and not np.array_equal(a0, fp.user_rng(7, 5, 2).random(8))
    and not np.array_equal(a0, fp.user_rng(7, 6, 1).random(8))
    and not np.array_equal(a0, fp.user_rng(8, 5, 1).random(8)),
    "each (seed, user, kind) has its own repeatable stream",
)
cat_items = np.arange(1, 200)
cat_w = rng.random(len(cat_items)) ** 4 + 1e-4
samplers = {
    1: fp.uniform_sampler(1, 200),
    2: fp.pop_sampler(cat_items, np.cumsum(cat_w)),
}
same = True
for trial in range(120):
    size = int(rng.choice([0, 3, 40, 150, 190]))
    excl = np.sort(rng.choice(cat_items, size=size, replace=False))
    kd = int(rng.choice([1, 2]))
    k = int(rng.choice([1, 5, 16, 33]))
    user = int(rng.integers(0, 10**6))
    got_i, got_c = fp.draw_allowed(fp.user_rng(7, user, kd), k, samplers[kd], excl)
    want_i, want_c = fp.reference_draws(7, user, kd, k, samplers[kd], excl)
    same &= (
        np.array_equal(got_i, want_i)
        and got_c == want_c
        and not np.isin(got_i, excl).any()
    )
check(
    same,
    "draw_allowed() equals the one-draw reference in 120 trials, "
    "including 190 of 199 items excluded",
)
heavy = cat_items[np.argsort(-cat_w)][:20]  # the most probable 20
mass = cat_w[np.isin(cat_items, heavy)].sum() / cat_w.sum()
big, _ = fp.draw_allowed(fp.user_rng(7, 1, 2), 200_000, samplers[2], np.sort(heavy))
allowed = ~np.isin(cat_items, heavy)
f_obs = np.array([(big == x).sum() for x in cat_items[allowed]])
f_exp = 200_000 * cat_w[allowed] / cat_w[allowed].sum()
chi2_c = float(((f_obs - f_exp) ** 2 / f_exp).sum())
# 178 degrees of freedom: the 99.9 percent point is 245.
check(
    not np.isin(big, heavy).any() and chi2_c < 245,
    f"accepted draws follow p_pop restricted to the allowed items, with "
    f"{mass:.0%} of the mass excluded (chi2 {chi2_c:.1f} on 178 df)",
)

# 5: excluded_items and user_rows.
ok = True
for trial in range(200):
    row = np.sort(
        rng.choice(np.arange(1, 90), size=int(rng.integers(0, 40)), replace=False)
    ).astype(np.int32)
    held = rng.integers(1, 90, size=4)
    ok &= np.array_equal(fp.excluded_items(row, held), np.union1d(row, held))
check(
    ok,
    "excluded_items() equals np.union1d, with held picks inside and "
    "outside the row and repeated",
)
p_slot = np.zeros(200)
p_slot[cat_items] = cat_w
row = np.array([2, 5, 9, 40], dtype=np.int32)
held = np.array([7, 5, 11, 7])
out = fp.user_rows(
    3, 42, [7, 11], [1, 2], row, held, 4, samplers, 199, p_slot, float(p_slot.sum())
)
excl = [2, 5, 7, 9, 11, 40]
check(
    out[1].tolist() == [0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    and out[2].tolist() == [1, 2] + [0] * 8
    and out[3].tolist() == [0, 0, 0, 1, 2, 3, 0, 1, 2, 3]
    and out[4] == 199 - len(excl)
    and math.isclose(
        out[5], math.fsum(cat_w) - math.fsum(p_slot[excl]), rel_tol=0, abs_tol=1e-12
    )
    and not np.isin(out[0][2:], excl).any(),
    "user_rows() lays out positives, uniform and p_pop draws, and "
    "counts n_allowed and pop_allowed exactly",
)

# 6: the stage on synthetic data.
tmp = Path(tempfile.mkdtemp(prefix="test_p11_"))
stage_p11.MODEL_DIR = tmp / "model"
(tmp / "model" / "reports").mkdir(parents=True)
stage_p11.NEG_PER_KIND = 6
stage_p11.VERIFY_REDRAW_USERS = 10**6
stage_p11.VERIFY_SPOT_PAIRS = 50
stage_p11.WORKERS = 1
sel.SIZE = 10
lines = []


class Grab(logging.Handler):
    def emit(self, record):
        lines.append(record.getMessage())


logging.getLogger("pipeline").addHandler(Grab())
logging.getLogger("pipeline").setLevel(logging.INFO)

NU, NI = 90, 300
hist = [np.zeros(0, dtype=np.int32)]
for u in range(1, NU):
    size = int(rng.choice([3, 10, 40, 120, 250]))
    hist.append(
        np.sort(rng.choice(np.arange(1, NI), size=size, replace=False)).astype(np.int32)
    )
u_indptr = np.concatenate(([0], np.cumsum([len(h) for h in hist])))
u_indices = np.concatenate(hist).astype(np.int32)
pairs = np.bincount(u_indices, minlength=NI)

# Held-out picks with every overlap pattern. User NU-1 is not eligible.
rows, want_pos = [], {}
for u in range(1, NU - 1):
    free = np.setdiff1d(np.arange(1, NI), hist[u])
    a, b, c, d = rng.choice(free, size=4, replace=False).tolist()
    pattern = u % 6
    if pattern == 1:
        c = a  # test_first == test_last
    elif pattern == 2:
        d = b  # val_first == val_last
    elif pattern == 3:
        c = b  # val_last == test_first
    elif pattern == 4:
        d = a  # val_first == test_last
    elif pattern == 5:
        c, d = b, a  # both val picks are test picks
    roles = {}
    for col, x in enumerate((a, b, c, d)):
        roles.setdefault(x, [0, 0, 0, 0])[col] = 1
    for x, f in roles.items():
        rows.append((u, x, *f))
    pos = {}
    if b != c:
        pos[b] = pos.get(b, 0) | 1
    if d != a:
        pos[d] = pos.get(d, 0) | 2
    want_pos[u] = sorted(pos.items())
sp_cols = tuple(np.array([r[j] for r in rows]) for j in range(6))
eligible = np.unique(sp_cols[0])
evals = np.concatenate((eligible[:12], [NU - 1]))  # one eval user is
selection = sel.select(eligible, evals, size=10)  # not eligible
core = np.flatnonzero(pairs >= np.percentile(pairs[1:], 30))
core = core[core >= 1]  # about 30 percent outside
p_pop = pairs[core] ** 0.75 / (pairs[core] ** 0.75).sum()
train_set = {(u, int(x)) for u in range(1, NU) for x in hist[u]}
tier = rng.integers(1, 7, size=NI)
flags = {key: rng.random(NI) < 0.5 for _, key in stage_p11.COVERAGE}
cov = dict(
    pop_centile=np.clip(np.argsort(np.argsort(pairs)) * 100 // NI + 1, 1, 100),
    train_pairs=pairs,
    tier=tier,
    **flags,
)
spot_calls = []

stage = stage_p11.StageP11()
stage._splits = lambda: sp_cols
stage._eval_users = lambda: evals
stage._selection_users = lambda: sel.select(eligible, evals, size=10)
stage._proposals = lambda: (core, p_pop, np.cumsum(p_pop))
stage._csr = lambda: (u_indptr, u_indices)
stage._id_slots = lambda: (NI, 1)
stage._covariates = lambda n: cov


def train_hits(users, items):
    spot_calls.append(len(users))
    return sum((int(u), int(x)) in train_set for u, x in zip(users, items))


stage._train_hits = train_hits
out = stage.outputs()
stage.run({}, [])
check(not raises(lambda: stage.verify({}), ""), "verify() passes the result")
stage.report({})
z = np.load(out["pairs"])
d = {k: z[k] for k in z.files}
K = stage_p11.NEG_PER_KIND
users_out = d["user_id"]
check(
    not np.isin(users_out, evals).any()
    and np.array_equal(users_out[d["split"] == 1], np.intersect1d(selection, users_out))
    and len(np.intersect1d(selection, users_out)) >= 7
    and not np.isin(users_out[d["split"] == 0], selection).any(),
    "no eval user appears, and the selection users carry split 1",
)
got_pos, ok_neg = {}, True
for j, u in enumerate(users_out.tolist()):
    s, e = int(d["indptr"][j]), int(d["indptr"][j + 1])
    kinds = d["kind"][s:e]
    it_u = d["item"][s:e].astype(np.int64)
    m = kinds == 0
    got_pos[u] = list(zip(it_u[m].tolist(), d["pick"][s:e][m].tolist()))
    held_u = {r[1] for r in rows if r[0] == u}
    ok_neg &= (
        int((kinds == 1).sum()) == K
        and int((kinds == 2).sum()) == K
        and not any((u, int(x)) in train_set or int(x) in held_u for x in it_u[~m])
    )
want_users = sorted(
    u for u, pz in want_pos.items() if pz and u not in set(evals.tolist())
)
check(
    sorted(got_pos) == want_users
    and all(got_pos[u] == want_pos[u] for u in want_users),
    "the positives equal an independent construction of rules 3 and 4, "
    "and users without a positive are left out",
)
check(
    ok_neg,
    "every user holds K draws of each kind, none of them a train "
    "item or a held-out pick",
)
text = out["report"].read_text()
check(
    "## Sampling diagnostics" in text
    and "## Sampler audit" in text
    and "| p_pop negatives |" in text
    and "| test_first picks, train users |" in text
    and f"users with pairs: {len(users_out):,}" in text
    and spot_calls == [100],
    "report() writes the counts, the diagnostics and the audit, and "
    "verify() spot-checks both kinds in the train files",
)

# 7: worker and block counts.
first = {k: v.copy() for k, v in d.items()}
stage_p11.WORKERS, stage_p11.TASKS_PER_WORKER = 3, 5
stage.run({}, [])
z2 = np.load(out["pairs"])
check(
    all(np.array_equal(first[k], z2[k]) for k in first) and set(first) == set(z2.files),
    "three workers and 15 blocks write the same arrays as one worker",
)
stage_p11.WORKERS, stage_p11.TASKS_PER_WORKER = 1, 8
stage.run({}, [])

# 8: verify() against planted faults.
good = {k: v.copy() for k, v in first.items()}


def with_damage(fn, text):
    dd = {k: v.copy() for k, v in good.items()}
    fn(dd)
    np.savez(out["pairs"], **dd)
    result = raises(lambda: stage.verify({}), text)
    np.savez(out["pairs"], **good)
    return result


def user_row(dd, j, kind, n=0):
    s, e = int(dd["indptr"][j]), int(dd["indptr"][j + 1])
    return s + int(np.flatnonzero(dd["kind"][s:e] == kind)[n])


def put(dd, j, kind, value, n=0):
    dd["item"][user_row(dd, j, kind, n)] = value


def free_item(j, core_only=False, outside_core=False):
    u = int(good["user_id"][j])
    held_u = {r[1] for r in rows if r[0] == u}
    s, e = int(good["indptr"][j]), int(good["indptr"][j + 1])
    for x in range(1, NI):
        if (u, x) in train_set or x in held_u or x in set(good["item"][s:e].tolist()):
            continue
        if core_only and x not in set(core.tolist()):
            continue
        if outside_core and x in set(core.tolist()):
            continue
        return x
    raise AssertionError("no free item")


def set_params(dd):
    p = json.loads(str(dd["params"]))
    p["neg_per_kind"] = 17
    dd["params"] = np.array(json.dumps(p))


def swap_kinds(dd):
    a, b = user_row(dd, 0, 1), user_row(dd, 0, 2)
    dd["kind"][a], dd["kind"][b] = dd["kind"][b], dd["kind"][a]


def drop_user(dd):
    s, e = int(dd["indptr"][1]), int(dd["indptr"][2])
    keep = np.ones(len(dd["item"]), dtype=bool)
    keep[s:e] = False
    for key in ("item", "kind", "pick", "draw"):
        dd[key] = dd[key][keep]
    lens = np.delete(np.diff(dd["indptr"]), 1)
    dd["indptr"] = np.concatenate(([0], np.cumsum(lens)))
    for key in ("user_id", "split", "n_allowed", "pop_allowed", "consumed"):
        dd[key] = np.delete(dd[key], 1, axis=0)


u0 = int(good["user_id"][0])
light = int(np.argmin([len(hist[int(u)]) for u in good["user_id"]]))
held0 = [r[1] for r in rows if r[0] == u0 and r[2] == 1][0]  # test_last
faults = [
    (
        lambda dd: put(dd, 0, 1, int(hist[u0][0])),
        "train items of their",
        "a uniform negative that is a train item",
    ),
    (
        lambda dd: put(dd, 0, 2, held0),
        "held-out picks",
        "a p_pop negative that is the user's test_last pick",
    ),
    (
        lambda dd: dd["user_id"].__setitem__(0, int(evals[11])),
        "an eval user appears",
        "an eval user's id",
    ),
    (
        lambda dd: put(dd, 0, 0, free_item(0)),
        "positives differ",
        "a positive that is not a val pick",
    ),
    (
        lambda dd: dd["split"].__setitem__(0, 1 - dd["split"][0]),
        "selection flag",
        "a flipped selection flag",
    ),
    (
        lambda dd: put(dd, 2, 2, free_item(2, core_only=True), n=3),
        "differ from a fresh recomputation",
        "a p_pop negative replaced by another allowed core track",
    ),
    (set_params, "was built with", "other stored parameters"),
    (swap_kinds, "wrong kind or draw", "two rows of a user swapped"),
    (
        lambda dd: dd["pop_allowed"].__setitem__(3, dd["pop_allowed"][3] + 1e-6),
        "pop_allowed is wrong",
        "a changed pop_allowed",
    ),
    (
        lambda dd: dd["n_allowed"].__setitem__(3, dd["n_allowed"][3] + 1),
        "n_allowed is wrong",
        "a changed n_allowed",
    ),
    (
        lambda dd: dd["consumed"].__setitem__((3, 0), dd["consumed"][3, 0] + 1),
        "differ from a fresh recomputation",
        "a changed raw draw count",
    ),
    (
        lambda dd: put(dd, light, 2, free_item(light, outside_core=True)),
        "not a P3 proposal item",
        "a p_pop negative outside P3's core",
    ),
    (drop_user, "stored users differ", "a user left out"),
    (
        lambda dd: put(dd, 1, 1, 0),
        "outside the candidate slots",
        "the phantom slot as a negative",
    ),
    (
        lambda dd: dd["draw"].__setitem__(user_row(dd, 0, 1, 2), 5),
        "wrong kind or draw",
        "a wrong draw number",
    ),
]
for fn, text, label in faults:
    check(with_damage(fn, text), f"verify() catches {label}")
stage._train_hits = lambda users, items: 1
check(
    raises(lambda: stage.verify({}), "exist in the train parquet files"),
    "verify() stops when the train files hold a stored pair",
)
stage._train_hits = train_hits

print()
if failures:
    print(f"FAILED: {len(failures)} check(s)")
    raise SystemExit(1)
print("ALL P11 CHECKS PASSED")
