"""Checks for pipeline/stage_p9b.py on synthetic data.

Every path points into a temporary folder, so nothing touches the real
model folder. The DuckDB loaders are replaced by synthetic stand-ins, and
the stage constants are shrunk to fit the small catalog.

1. run() writes both lists equal to the colisten.py functions called
   directly: the same-artist lists and the other-artist chain.
2. verify() passes the result.
3. A second run() resumes all three cached steps and writes the same
   files.
4. A run stopped inside the exact counts resumes the first two steps and
   finishes with the same result.
5. A changed parameter, a changed colisten.py and changed artist
   credits each compute every step again. A changed artists.py that
   leaves the credits as they were resumes all three steps.
6. verify() fails for a changed co count, a row out of order, a self
   loop, a stored parameter set that differs from the stage's, an
   other-artist pair that shares an artist, and a same-artist list with
   a neighbour missing.
7. report() writes the coverage table, the validation table, the
   examples and the special-purpose artist table. Without sampling, both
   kinds reach recall 1.000. The credits line counts the special
   artists and does not call their printed names special artists.

Usage, from the repository root:
    python3 test_p9b.py
"""

import json
import logging
import tempfile
from pathlib import Path

import numpy as np

from pipeline import artists as ar
from pipeline import colisten as cl
from pipeline import stage_p9b

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


tmp = Path(tempfile.mkdtemp(prefix="test_p9b_"))
stage_p9b.MODEL_DIR = tmp / "model"
stage_p9b.SCRATCH_DIR = tmp / "scratch"
# No features table in the temporary folder: current_inputs() records the
# input item_artists as missing.
stage_p9b.features_table = lambda name: str(tmp / "features" / f"{name}.parquet")
M, S = stage_p9b.MODEL_DIR, stage_p9b.SCRATCH_DIR
(M / "reports").mkdir(parents=True)
S.mkdir(parents=True)
stage_p9b.SAMPLE_K, stage_p9b.MIN_SUPPORT = 6, 2
stage_p9b.CAND_PER_ITEM, stage_p9b.TOP_N = 4, 3
stage_p9b.VALIDATION_ITEMS, stage_p9b.VERIFY_EDGES = 20, 200
stage_p9b.VERIFY_EXACT_TRACKS = 10_000  # every track, so any fault shows

lines = []


class Keep(logging.Handler):
    def emit(self, record):
        lines.append(record.getMessage())


stage_p9b.log.addHandler(Keep())
stage_p9b.log.setLevel(logging.INFO)

rng = np.random.default_rng(9)
NU, NI = 240, 80
P = rng.random((NU, NI)) < rng.uniform(0.03, 0.5, size=(NU, 1))
P[0, :] = P[:, 0] = False
P[:, 75:] = False  # rare tracks: 1 listener, below the
for x in range(75, NI):  # support floor, so no sample slot
    P[int(rng.integers(1, NU)), x] = True
L = np.where(P, rng.integers(1, 60, size=(NU, NI)), 0)
D = np.where(P, np.round(1000 * np.log1p(L)), 0).astype(np.uint16)


def csr(Mx, Dx=None):
    r, c = np.nonzero(Mx)
    ip = np.concatenate(([0], np.cumsum(np.bincount(r, minlength=Mx.shape[0])))).astype(
        np.int64
    )
    return ip, c.astype(np.int32), (None if Dx is None else Dx[r, c])


u = csr(P, D)
i = csr(P.T)
np.save(M / "csr_user_indptr.npy", u[0])
np.save(M / "csr_user_indices.npy", u[1])
np.save(M / "csr_user_data.npy", u[2])
np.save(M / "csr_item_indptr.npy", i[0])
np.save(M / "csr_item_indices.npy", i[1])
pop = np.diff(i[0]).astype(np.int64)

# Artist credits: 0 to 2 of 7 artists per track.
item_art = [set()] + [
    set(rng.choice(7, size=int(rng.choice([0, 1, 1, 2])), replace=False).tolist())
    for _ in range(1, NI)
]
ai = np.array([x for x in range(NI) for _ in item_art[x]], dtype=np.int64)
aa = np.array([a for x in range(NI) for a in sorted(item_art[x])], dtype=np.int64)
art = ar.credit_csr(ai, aa, [], NI)
STATS = {
    "credit_rows": len(ai),
    "special_artists": 1,
    "special_rows": 4,
    "special": [
        {
            "artist_id": 99,
            "matched_by": ['name "[unknown]"'],
            "rows": 4,
            "distinct_names": 3,
            "names": [["A fisherman", 2], ["Pipe|Name", 1]],
        }
    ],
    "tracks_with_credit": int(np.count_nonzero(np.diff(art[0]))),
}

stage = stage_p9b.StageP9b()
stage._credits = lambda n: (art[0], art[1], STATS)
out = stage.outputs()


def load(kind):
    z = np.load(out[kind])
    return {k: z[k] for k in z.files}


def load_all():
    return {kind: load(kind) for kind in ("same", "other")}


def same_files(a, b):
    return all(np.array_equal(a[kd][k], b[kd][k]) for kd in a for k in a[kd])


# 1: run() equals the direct chain.
stage.run({}, [])
elig = pop >= 2
elig[0] = False
want_same = cl.same_artist_lists(art[0], art[1], i[0], i[1], pop, 2, 3, workers=1)
s_ip, s_m, s_l = cl.sample_users(u[0], u[1], u[2], pop, 6, eligible=elig)
cm = cl.candidate_pairs(s_ip, s_m, NI, 2, 4, workers=1, art=art)
cn = cl.candidate_pairs(s_ip, s_l, NI, 2, 4, workers=1, art=art)
a, b, sm, sl = cl.union_candidates(cm, cn, NI)
co = cl.exact_co_listeners(a, b, i[0], i[1], NU, workers=1)
want_other = cl.directed_topn(a, b, co, sm, sl, pop, NI, 3, 2)
got = load_all()
check(
    not elig[75:].any()
    and all(np.array_equal(got["same"][k], want_same[k]) for k in want_same)
    and all(np.array_equal(got["other"][k], want_other[k]) for k in want_other)
    and json.loads(str(got["other"]["params"]))["top_n"] == 3
    and len(got["same"]["neighbor"]) > 20
    and len(got["other"]["neighbor"]) > 20,
    "run() writes both lists equal to the direct chain, with parameters",
)

# 2: verify.
check(not raises(lambda: stage.verify({}), ""), "verify() passes the result")

# 3: resume.
lines.clear()
first = load_all()
stage.run({}, [])
check(
    all(
        any(f"{step} resume" in m for m in lines)
        for step in ("same-artist lists", "candidates", "exact counts")
    )
    and same_files(first, load_all()),
    "a second run() resumes all three cached steps and writes the same " "files",
)

# 4: a stop inside the exact counts.
(S / "p9b_state.json").unlink()
real = cl.exact_co_listeners


def stop(*args, **kwargs):
    raise KeyboardInterrupt


cl.exact_co_listeners = stop
try:
    stage.run({}, [])
    check(False, "the injected stop did not happen")
except KeyboardInterrupt:
    pass
cl.exact_co_listeners = real
lines.clear()
stage.run({}, [])
check(
    any("same-artist lists resume" in m for m in lines)
    and any("candidates resume" in m for m in lines)
    and not any("exact counts resume" in m for m in lines)
    and same_files(first, load_all()),
    "a run stopped inside the exact counts resumes the first two steps "
    "and finishes with the same result",
)


# 5: changes that must compute every step again.
def recomputes(change, undo):
    change()
    lines.clear()
    stage.run({}, [])
    result = not any("resume" in m for m in lines)
    undo()
    stage.run({}, [])
    return result


check(
    recomputes(
        lambda: setattr(stage_p9b, "CAND_PER_ITEM", 5),
        lambda: setattr(stage_p9b, "CAND_PER_ITEM", 4),
    ),
    "a changed parameter computes every step again",
)


def changed_module(mod, name):
    """Point mod.__file__ at a changed copy. Returns the undo."""
    changed = tmp / f"changed_{name}"
    changed.write_text(Path(mod.__file__).read_text() + "\n# a change\n")
    saved = mod.__file__
    return (
        lambda: setattr(mod, "__file__", str(changed)),
        lambda: setattr(mod, "__file__", saved),
    )


check(
    recomputes(*changed_module(cl, "colisten.py")),
    "a changed colisten.py computes every step again",
)
do, undo = changed_module(ar, "artists.py")
do()
lines.clear()
stage.run({}, [])
undo()
check(
    all(
        any(f"{step} resume" in m for m in lines)
        for step in ("same-artist lists", "candidates", "exact counts")
    ),
    "a changed artists.py with the same credits resumes all three steps",
)
before = stage.current_inputs([])["artists_module"]
do()
after = stage.current_inputs([])["artists_module"]
undo()
check(before != after, "a changed artists.py still changes the stage marker's inputs")
art2 = (art[0], np.where(art[1] == 0, 6, art[1]))  # new credits
check(
    recomputes(
        lambda: setattr(stage, "_credits", lambda n: (art2[0], art2[1], STATS)),
        lambda: setattr(stage, "_credits", lambda n: (art[0], art[1], STATS)),
    ),
    "changed artist credits compute every step again",
)

# 6: verify() against planted damage.
good = load_all()


def with_damage(kind, fn, text):
    d = {k: v.copy() for k, v in good[kind].items()}
    fn(d)
    np.savez(out[kind], **d)
    result = raises(lambda: stage.verify({}), text)
    np.savez(out[kind], **good[kind])
    return result


def long_row(d):
    r = int(np.flatnonzero(np.diff(d["indptr"]) >= 2)[0])
    return r, int(d["indptr"][r])


def bump_co(d):
    r, s = long_row(d)
    d["co"][s] += 1
    d["cosine"][s] = cl.cosine(d["co"][s], pop[r], pop[d["neighbor"][s]])


def swap(d):
    _, s = long_row(d)
    for k in ("neighbor", "cosine", "co", "support_most", "support_low"):
        if k in d:
            d[k][[s, s + 1]] = d[k][[s + 1, s]]


def self_loop(d):
    r, s = long_row(d)
    d["neighbor"][s] = r


def other_params(d):
    d["params"] = np.array(json.dumps(dict(json.loads(str(d["params"])), top_n=9)))


def plant_same_artist(d):
    """Replace one other-artist neighbour by a same-artist track with the
    right count and cosine, placed where its cosine belongs."""
    rows = np.repeat(np.arange(NI), np.diff(d["indptr"]))
    for s in range(len(d["neighbor"])):
        x = int(rows[s])
        mates = [
            y
            for y in range(1, NI)
            if y != x
            and item_art[x] & item_art[y]
            and y not in d["neighbor"][d["indptr"][x] : d["indptr"][x + 1]]
            and int((P[:, x] & P[:, y]).sum()) >= 2
        ]
        if mates:
            y = mates[0]
            c = int((P[:, x] & P[:, y]).sum())
            e0, e1 = d["indptr"][x], d["indptr"][x + 1]
            seg = [
                (
                    float(cl.cosine(cc, pop[x], pop[n])),
                    int(n),
                    int(cc),
                    int(m1),
                    int(m2),
                )
                for n, cc, m1, m2 in zip(
                    d["neighbor"][e0:e1],
                    d["co"][e0:e1],
                    d["support_most"][e0:e1],
                    d["support_low"][e0:e1],
                )
            ]
            seg[-1] = (float(cl.cosine(c, pop[x], pop[y])), y, c, c, c)
            seg.sort(key=lambda t: (-t[0], t[1]))
            for k, t in enumerate(seg):
                (
                    d["cosine"][e0 + k],
                    d["neighbor"][e0 + k],
                    d["co"][e0 + k],
                    d["support_most"][e0 + k],
                    d["support_low"][e0 + k],
                ) = t
            return
    raise AssertionError("no place to plant a same-artist pair")


def drop_same_neighbour(d):
    """Remove the last neighbour of one full same-artist row. The row stays
    sorted, so only the full recomputation can see it."""
    r = int(np.flatnonzero(np.diff(d["indptr"]) == 3)[0])
    s = int(d["indptr"][r + 1]) - 1
    for k in ("neighbor", "cosine", "co"):
        d[k] = np.delete(d[k], s)
    d["indptr"][r + 1 :] -= 1


check(
    with_damage("other", bump_co, "independent recount")
    and with_damage("same", bump_co, "independent recount"),
    "verify() catches a co count that differs from the train data, in " "both lists",
)
check(
    with_damage("other", swap, "not sorted")
    and with_damage("same", swap, "not sorted"),
    "verify() catches a row out of order, in both lists",
)
check(with_damage("same", self_loop, "lists itself"), "verify() catches a self loop")
check(
    with_damage("other", other_params, "was built with"),
    "verify() catches a file built with other parameters",
)
check(
    with_damage("other", plant_same_artist, "share an artist"),
    "verify() catches an other-artist pair that shares an artist",
)
check(
    with_damage("same", drop_same_neighbour, "full recomputation"),
    "verify() catches a same-artist list with a neighbour missing",
)

# 7: report, with the DuckDB loaders replaced.
users = np.arange(1, NU, 3)
picks = np.stack([rng.integers(1, NI, size=len(users)) for _ in range(4)], axis=1)
stage._eval_picks = lambda: (users, picks)
stage._pop_centile = lambda n: np.clip((pop * 100) // (pop.max() + 1) + 1, 1, 100)
stage._names = lambda ids: {int(x): f"artist {x} - title {x}" for x in ids}
stage_p9b.SAMPLE_K, stage_p9b.CAND_PER_ITEM = 10**6, 10**6  # no sampling
stage.run({}, [])
stage.report({})
text = out["report"].read_text()
rows = [ln for ln in text.splitlines() if ln.startswith("| random")]
full = [ln for ln in rows if ln.split("|")[-2].strip() == "100.0%"]
check(
    "| test_last | all |" in text
    and "artist" in text
    and "## Validation" in text
    and "## Examples" in text
    and "same-artist neighbours:" in text
    and "other-artist neighbours:" in text
    and len(rows) >= 4
    and len(full) == len(rows),
    "report() writes coverage, validation (every list identical without "
    "sampling) and named examples of both kinds",
)
credit_line = next(ln for ln in text.splitlines() if ln.startswith("- artist credits:"))
check(
    "which do not count as an artist: 1, with 4 credit rows" in credit_line
    and "fisherman" not in credit_line
    and "## Special-purpose artists" in text
    and '| 99 | name "[unknown]" | 4 | 3 | A fisherman (2), '
    "Pipe\\|Name (1) |" in text,
    "report() counts the special artists in the credits line and lists "
    "each one with its matched rule part and printed names in a table",
)

print()
if failures:
    print(f"FAILED: {len(failures)} check(s)")
    raise SystemExit(1)
print("ALL P9B CHECKS PASSED")
