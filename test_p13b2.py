"""Checks for pipeline/stage_p11b.py, stage_p12b.py and stage_p13b2.py on
the synthetic world of p13_world.py. Every path points into a temporary
folder and the loaders are replaced, so nothing touches the real model
folder. The whole chain runs as on the real machine: P11, P12, P13a and
P13b (with part 1's real outcome, the monotone cell m1w0t0p0, forced),
then P11b, P12b and P13b part 2. P3's core leaves out some positive
tracks, as on the real data, so p_pop form B meets positives that its
proposal cannot draw.

1. P11b: every listener's draws equal the slow one-double reference of
   fusion_pairs.py, the first 16 draws equal P11's rows, the first wide
   draws of a reference listener equal its wide rows, the reference
   listeners follow the SHA-256 rule (recomputed here), n_allowed and
   pop_allowed follow from the exclusions. verify() passes and fails for
   planted faults.
2. P12b: every row of both files equals concept_data.user_rows() for its
   listener, the rows shared with P12 and between the files are equal
   bit for bit, a finished run resumes without work, a lost output is
   computed again, and verify() fails for planted faults.
3. P13b part 2, the helpers: warm starts, explanation points (and their
   shift invariance), gaps, the rules D1 to D4 on constructed inputs,
   part 1's final model in all three layouts of p13b_model.json, the log
   relabelling.
4. P13b part 2 on the chain: run(), verify() and report() pass. Recomputed
   here without the stage's code: the rows of every level, the loss of
   every fit from the raw rows (each arm's negatives, offsets and
   weights), the KKT conditions of a bounded form B fit and of a free fit
   by finite differences, the centering on each level's p_pop negatives,
   the explanation points, every selection rank by a dense count, and
   D1 to D4. Part 1's final fit returns at the smallest wide level.
5. One worker and three workers give the same outputs. A finished run
   fits nothing and scores nothing. A lost fit, lost background means and
   a lost selection block are computed again with the same outputs.
6. verify() fails for each planted fault in the outputs.

Usage, from the repository root:
    python3 test_p13b2.py
With P13B2_BASE_CACHE=<folder>, the outputs of P11 to P13b are kept in
that folder and reused while the code they come from stays the same, so a
repeated run checks P11b, P12b and P13b2 without fitting P11 to P13b
again. With FAIL_FAST=1, the test stops at its first failed check.
"""

import hashlib
import itertools
import json
import logging
import math
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

import p13_world as W
from pipeline import concept_data as cd
from pipeline import concepts as cn
from pipeline import evaluation as ev
from pipeline import fusion_pairs as fp
from pipeline import gam
from pipeline import gam_arms
from pipeline import stage_p11 as s11
from pipeline import stage_p11b as s11b
from pipeline import stage_p12 as s12
from pipeline import stage_p12b as s12b
from pipeline import stage_p13a as s13
from pipeline import stage_p13b as s13b
from pipeline import stage_p13b2 as s13b2
from pipeline.common import VerificationError

failures = []


def check(cond, msg):
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        failures.append(msg)
        if os.environ.get("FAIL_FAST"):  # FAIL_FAST=1: stop at the first failed check
            raise SystemExit(1)


def raises(fn, text):
    try:
        fn()
    except VerificationError as exc:
        if text in str(exc):
            return True
        print(f"      other message: {exc}")
        return False
    except Exception as exc:  # reported, then False
        print(f"      unexpected {type(exc).__name__}: {exc}")
        return False
    return False


print(W.summary())
HERE = Path(__file__).resolve().parent
TMP = Path(tempfile.mkdtemp(prefix="test_p13b2_"))
CHOSEN = (1, 0, 0, 0)  # part 1's real outcome, m1w0t0p0
KW, KR, K11 = 32, 128, 16
FULL, REF = (16, 24, 32), (16, 32, 128)
ZERO, MISS = gam.ZERO, gam.MISS
CUT1 = s13.RANK_CUT + 1

# P3's core without some of the positives' tracks, renormalised.
POS_ITEMS = np.unique(W.p_items)
OUT_CORE = POS_ITEMS[::12]
CORE = np.setdiff1d(W.core, OUT_CORE).astype(np.int64)
PP = W.p_pop_slot[CORE] / W.p_pop_slot[CORE].sum()
CUM = np.cumsum(PP)
PP_SLOT = np.zeros(W.NI)
PP_SLOT[CORE] = PP
SAMPLERS = {
    fp.KIND_UNIFORM: fp.uniform_sampler(1, W.NI),
    fp.KIND_POP: fp.pop_sampler(CORE, CUM),
}


# Stage harness


def configure(root, workers=1):
    for m in (s11, s11b, s12, s12b, s13, s13b, s13b2):
        m.MODEL_DIR = root / "model"
        m.SCRATCH_DIR = root / "scratch"
    (root / "model" / "reports").mkdir(parents=True, exist_ok=True)
    s11.WORKERS, s11.TASKS_PER_WORKER = workers, 2
    s11.VERIFY_REDRAW_USERS, s11.VERIFY_SPOT_PAIRS = 40, 300
    s12.WORKERS, s12.TASKS_PER_WORKER, s12.VERIFY_USERS = workers, 2, 10
    s13.LAMBDAS = (1.0, 0.1, 0.01, 1e-3)
    s13.N_BOOT = 30
    s13.FIT_BLOCKS = 5
    s13.FIT_WORKERS = workers
    s13.SELECTION_WORKERS = workers
    s13.TASKS_PER_WORKER = 2
    s13.VERIFY_USERS = 6
    s13.VERIFY_ROWS = 4000
    s13.VERIFY_MAINSTREAM = 10_000
    s13.NEUTRAL_MARGIN = 0.02
    s13.SELECTION_BLOCK = 7
    s13.SELECTION_CHUNK = 64
    s13.PROBE_USERS = 3
    s13b.EXT_LAMBDAS = (1e-4, 0.0)
    s13b.VERIFY_OFFSET_USERS = 60
    s11b.WIDE_PER_KIND, s11b.REF_PER_KIND, s11b.REF_USERS = KW, KR, 60
    s11b.VERIFY_WIDE_USERS, s11b.VERIFY_REF_USERS = 10, 3
    s13b2.FULL_LEVELS, s13b2.REF_LEVELS = FULL, REF
    s13b2.ROWS_PER_BLOCK, s13b2.MIN_BLOCKS = 4000, 5
    s13b2.VERIFY_RANK_USERS, s13b2.VERIFY_OFFSET_USERS = 6, 40
    return root


def roles():
    """P11's roles of the world: every listener with a positive, the
    selection sample of p13_world.py."""
    users, picks = W.p_users, W.picks4
    p_indptr, p_items, p_flags, counts = fp.positives(picks)
    has_pos = np.diff(p_indptr) > 0
    rows = np.repeat(has_pos, np.diff(p_indptr))
    p_items, p_flags = p_items[rows], p_flags[rows]
    p_indptr = np.concatenate(([0], np.cumsum(np.diff(p_indptr)[has_pos]))).astype(
        np.int64
    )
    users, picks = users[has_pos], picks[has_pos]
    split = np.where(np.isin(users, W.SEL), fp.SPLIT_SELECTION, fp.SPLIT_TRAIN).astype(
        np.int8
    )
    counts.update(
        eligible_users=int(len(W.p_users)),
        eval_users_left_out=0,
        selection_users=int(len(W.SEL)),
    )
    return users, split, picks, (p_indptr, p_items, p_flags), counts


TRAIN_SET = {
    (u, int(x))
    for u in range(1, len(W.u_indptr) - 1)
    for x in W.u_indices[W.u_indptr[u] : W.u_indptr[u + 1]]
}


def train_hits(users, items):
    return sum((int(u), int(x)) in TRAIN_SET for u, x in zip(users, items))


def p11_loaders(st):
    st._roles = roles
    st._id_slots = lambda: (W.NI, 1)
    st._csr = lambda: (W.u_indptr, W.u_indices)
    st._proposals = lambda: (CORE.copy(), PP.copy(), CUM.copy())
    st._eval_users = lambda: np.zeros(0, dtype=np.int64)
    st._selection_users = lambda: W.SEL.copy()
    st._train_hits = train_hits
    return st


def p12_loaders(st):
    st._id_slots = lambda: (W.NI, 1)
    st._sources = lambda n: W.SRC
    st._space = lambda src: W.space_pairs
    st._csr = lambda: (W.u_indptr, W.u_indices, W.u_data)
    st._upstream_outputs = lambda: {"p25_train": {"stats": "a" * 64}}
    return st


def p13_loaders(st):
    st._proposals = lambda: (CORE.copy(), PP.copy())
    st._id_slots = lambda: (W.NI, 1)
    st._csr = lambda: (W.u_indptr, W.u_indices, W.u_data)
    st._popularity = lambda n: W.popularity.copy()
    st._selection = lambda: (W.SEL.copy(), W.SEL_PICKS.copy())
    st._sources = lambda n: W.SRC
    st._space = lambda src: W.SPACE
    st._als_reference = lambda: {
        "popularity": 0.003,
        "als": 0.017,
        "als_label": "alpha 1, reg 100",
    }
    st._upstream_outputs = lambda: {"p11_fusion_pairs": {"pairs": "x"}}
    return st


class Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


def run_stage(st, report=True):
    """run(), verify() and report() of a stage, and the log lines."""
    cap = Capture()
    s13.log.addHandler(cap)
    s13.log.setLevel(logging.INFO)
    try:
        st.run({}, [])
        st.verify({})
        if report:
            st.report({})
    finally:
        s13.log.removeHandler(cap)
    return st, cap.lines


def build_base(root):
    """P11 to P13b in root, with part 1's chosen cell forced to CHOSEN."""
    configure(root)
    run_stage(p11_loaders(s11.StageP11()), report=False)
    run_stage(p12_loaders(s12.StageP12()), report=False)
    run_stage(p13_loaders(s13.StageP13a()))
    real = s13b.decide

    def forced(effects, base_score, sesoi=s13b.SESOI):
        return CHOSEN, real(effects, base_score, sesoi)[1]

    s13b.decide = forced
    try:
        run_stage(p13_loaders(s13b.StageP13b()))
    finally:
        s13b.decide = real


BASE_FILES = (
    "fusion_pairs.npz",
    "fusion_features.npz",
    "p13a_model.json",
    "p13a_arrays.npz",
    "p13b_model.json",
    "p13b_arrays.npz",
)


def base_key():
    """The code and the settings that the base outputs come from."""
    h = hashlib.sha256()
    for mod in (s11, fp, s12, cd, cn, s13, s13b, gam, gam_arms):
        h.update(Path(mod.__file__).read_bytes())
    for name in ("p13_world.py",):
        h.update((HERE / name).read_bytes())
    h.update(
        json.dumps(
            [
                CHOSEN,
                OUT_CORE.tolist(),
                s11._params(),
                s12._params(),
                s13._params(),
                s13b._params(),
                s13.N_BOOT,
                s13.FIT_BLOCKS,
                s13.NEUTRAL_MARGIN,
                s13.SELECTION_CHUNK,
                s13.VERIFY_ROWS,
            ],
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    )
    return h.hexdigest()


def base_into(root):
    """The base outputs in root/model, from the cache when it is current."""
    cache = os.environ.get("P13B2_BASE_CACHE")
    configure(root)
    if cache:
        cache = Path(cache)
        key = base_key()
        if (cache / "key.txt").exists() and (cache / "key.txt").read_text() == key:
            for name in BASE_FILES:
                shutil.copy(cache / name, root / "model" / name)
            print(f"      base outputs from the cache {cache}")
            return
    build_base(root)
    if cache:
        cache.mkdir(parents=True, exist_ok=True)
        for name in BASE_FILES:
            shutil.copy(root / "model" / name, cache / name)
        (cache / "key.txt").write_text(base_key())
    configure(root)


def load_npz(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


def strip_seconds(obj):
    if isinstance(obj, dict):
        return {k: strip_seconds(v) for k, v in obj.items() if k != "seconds"}
    if isinstance(obj, list):
        return [strip_seconds(v) for v in obj]
    return obj


# 1: the chain and P11b
root1 = TMP / "run"
base_into(root1)
MODEL = root1 / "model"
st11b, lines11b = run_stage(p11_loaders(s11b.StageP11b()))
P11 = load_npz(MODEL / "fusion_pairs.npz")
PW = load_npz(MODEL / "fusion_pairs_wide.npz")
PR = load_npz(MODEL / "fusion_pairs_ref.npz")
r_users, r_split, r_picks, (r_ptr, r_items, r_flags), _ = roles()
TR = r_split == fp.SPLIT_TRAIN
TR_USERS = r_users[TR]
PICKS_OF = {int(u): r_picks[j] for j, u in enumerate(r_users)}
check(
    np.array_equal(PW["user_id"], TR_USERS)
    and bool((PW["split"] == fp.SPLIT_TRAIN).all())
    and str(PW["pairs_sha256"]) == ev.npz_hash(MODEL / "fusion_pairs.npz")
    and str(PR["pairs_sha256"]) == str(PW["pairs_sha256"]),
    f"the wide file holds P11's {len(TR_USERS)} train listeners, and both "
    f"files name the fusion_pairs.npz they extend",
)


def excl_of(u):
    row = W.u_indices[W.u_indptr[u] : W.u_indptr[u + 1]]
    return set(int(x) for x in row) | set(int(x) for x in PICKS_OF[int(u)])


def check_file(P_, k, label):
    """Every listener of a P11b file: layout, positives, draws against the
    slow reference, raw counts, n_allowed and pop_allowed."""
    ok = {"layout": True, "draws": True, "p11": True, "counts": True}
    for j, u in enumerate(P_["user_id"]):
        u = int(u)
        a, b = int(P_["indptr"][j]), int(P_["indptr"][j + 1])
        i11 = int(np.searchsorted(P11["user_id"], u))
        a11, b11 = int(P11["indptr"][i11]), int(P11["indptr"][i11 + 1])
        n_p = int(np.count_nonzero(P11["kind"][a11:b11] == 0))
        kinds = P_["kind"][a:b]
        ok["layout"] &= (
            b - a == n_p + 2 * k
            and np.array_equal(kinds, np.repeat([0, 1, 2], [n_p, k, k]))
            and np.array_equal(
                P_["draw"][a:b],
                np.concatenate([np.zeros(n_p), np.arange(k), np.arange(k)]),
            )
            and np.array_equal(P_["item"][a : a + n_p], P11["item"][a11 : a11 + n_p])
            and np.array_equal(P_["pick"][a : a + n_p], P11["pick"][a11 : a11 + n_p])
            and not P_["pick"][a + n_p : b].any()
        )
        excl = excl_of(u)
        for col, kd in enumerate((fp.KIND_UNIFORM, fp.KIND_POP)):
            got = P_["item"][a + n_p + col * k : a + n_p + (col + 1) * k]
            want, used = fp.reference_draws(s11.SEED, u, kd, k, SAMPLERS[kd], excl)
            ok["draws"] &= (
                np.array_equal(got.astype(np.int64), want)
                and int(P_["consumed"][j, col]) == used
            )
            g11 = P11["item"][a11 + n_p + col * K11 : a11 + n_p + (col + 1) * K11]
            ok["p11"] &= np.array_equal(got[:K11], g11)
        ex = np.array(sorted(excl), dtype=np.int64)
        ok["counts"] &= (
            int(P_["n_allowed"][j]) == W.NI - 1 - len(ex)
            and abs(float(P_["pop_allowed"][j]) - (1.0 - float(PP_SLOT[ex].sum())))
            <= 1e-12
        )
    return ok


ok_w = check_file(PW, KW, "wide")
check(
    all(ok_w.values()),
    f"wide file: every listener holds its positives, then {KW} uniform "
    f"and {KW} p_pop draws that equal the slow one-double reference with "
    f"the same raw counts, the first {K11} of each kind equal P11's rows, "
    f"and n_allowed and pop_allowed follow from the exclusions ({ok_w})",
)
want_ref = np.sort(
    sorted(
        (int(u) for u in TR_USERS),
        key=lambda u: hashlib.sha256(
            f"{s11b.REF_SALT}:{u}".encode("utf-8")
        ).hexdigest(),
    )[:60]
)
check(
    np.array_equal(PR["user_id"], want_ref),
    "the reference listeners are the 60 train listeners with the "
    "smallest SHA-256 of 'salt:user_id', recomputed here",
)
ok_r = check_file(PR, KR, "ref")
pref_ok = True
for j, u in enumerate(PR["user_id"]):
    jw = int(np.searchsorted(PW["user_id"], u))
    a, aw = int(PR["indptr"][j]), int(PW["indptr"][jw])
    n_p = int(np.count_nonzero(PR["kind"][a : PR["indptr"][j + 1]] == 0))
    for col in range(2):
        pref_ok &= np.array_equal(
            PR["item"][a + n_p + col * KR : a + n_p + col * KR + KW],
            PW["item"][aw + n_p + col * KW : aw + n_p + (col + 1) * KW],
        )
check(
    all(ok_r.values()) and pref_ok,
    f"reference file: every listener's {KR} draws per kind equal the slow "
    f"reference, and the first {KW} equal its wide rows ({ok_r})",
)
rep11b = (MODEL / "reports" / "p11b_report.md").read_text()
check(
    st11b._verified["redrawn"] == 13
    and "| wide |" in rep11b
    and "| ref |" in rep11b
    and f"the first {K11} draws" in rep11b
    and ";" not in rep11b,
    "P11b's verify() passes and its report holds both files, without a " "semicolon",
)
n_rep = 0
for j in range(len(PW["user_id"])):
    s = (
        int(PW["indptr"][j])
        + int(np.count_nonzero(PW["kind"][PW["indptr"][j] : PW["indptr"][j + 1]] == 0))
        + KW
    )
    n_rep += KW - len(np.unique(PW["item"][s : s + KW]))
check(
    f"| wide | {len(TR_USERS):,} | {KW:,} | {len(PW['item']):,} |" in rep11b
    and rep11b.split("| wide |")[1].split("\n")[0].rstrip(" |").endswith(f"{n_rep:,}"),
    f"the report counts the rows and the repeated p_pop draws of the wide "
    f"file ({n_rep}), recomputed here",
)

# P11b faults.
st_v11 = p11_loaders(s11b.StageP11b())


def with_file(name, change, text):
    path = MODEL / name
    good = load_npz(path)
    bad = {k: v.copy() for k, v in good.items()}
    change(bad)
    np.savez(path, **bad)
    try:
        return raises(lambda: st_v11.verify({}), text)
    finally:
        np.savez(path, **good)


def first_neg(P_, j):
    a = int(P_["indptr"][j])
    return a + int(np.count_nonzero(P_["kind"][a : P_["indptr"][j + 1]] == 0))


NON_REF = next(
    j for j, u in enumerate(PW["user_id"]) if u not in set(PR["user_id"].tolist())
)


def wide_prefix(d):
    r = first_neg(d, NON_REF) + 3
    d["item"][r] = d["item"][r] % (W.NI - 2) + 1


def wide_train_item(d):
    u = int(d["user_id"][NON_REF])
    d["item"][first_neg(d, NON_REF) + 20] = int(W.u_indices[W.u_indptr[u]])


def ref_prefix(d):
    r = first_neg(d, 2) + 5
    d["item"][r] = d["item"][r] % (W.NI - 2) + 1


def ref_users(d):
    d["user_id"][0] = next(
        int(u) for u in TR_USERS if u not in set(d["user_id"].tolist())
    )


check(
    with_file(
        "fusion_pairs_wide.npz",
        wide_prefix,
        f"first {K11} draws of the wide file differ from P11's",
    ),
    "P11b verify() fails when a draw among the first 16 leaves P11's row",
)
check(
    with_file(
        "fusion_pairs_wide.npz", wide_train_item, "are train items of their listener"
    ),
    "P11b verify() fails when a later draw is a train item of its " "listener",
)
check(
    with_file(
        "fusion_pairs_ref.npz",
        ref_prefix,
        f"first {KW} draws of the reference file differ",
    ),
    "P11b verify() fails when a reference draw leaves the wide row",
)
check(
    with_file("fusion_pairs_ref.npz", ref_users, "do not follow the SHA-256 rule"),
    "P11b verify() fails when the reference listeners leave the rule",
)
check(
    with_file(
        "fusion_pairs_wide.npz",
        lambda d: d.update(pairs_sha256=np.array("0" * 64)),
        "extends another fusion_pairs.npz",
    ),
    "P11b verify() fails when a file names another fusion_pairs.npz",
)
check(
    with_file(
        "fusion_pairs_wide.npz",
        lambda d: d["consumed"].__setitem__((0, 1), KW - 1),
        "too few raw draws",
    ),
    "P11b verify() fails when a raw draw count falls below the number of " "draws",
)
check(
    with_file(
        "fusion_pairs_wide.npz",
        lambda d: d["n_allowed"].__setitem__(4, 7),
        "n_allowed differs from P11's",
    ),
    "P11b verify() fails when n_allowed leaves P11's value",
)
check(
    with_file(
        "fusion_pairs_wide.npz",
        lambda d: d.update(
            params=np.array(
                json.dumps(dict(json.loads(str(d["params"])), wide_per_kind=KW - 1))
            )
        ),
        "was built with",
    ),
    "P11b verify() fails for a file built with other parameters",
)


def wide_held_pick(d):
    u = int(d["user_id"][NON_REF])
    d["item"][first_neg(d, NON_REF) + 21] = int(PICKS_OF[u][0])


def wide_out_of_core(d):
    u = int(d["user_id"][NON_REF])
    d["item"][first_neg(d, NON_REF) + KW + 20] = next(
        int(i) for i in OUT_CORE if int(i) not in excl_of(u)
    )


REF_SET = set(PR["user_id"].tolist())
VS = np.sort(
    np.random.default_rng(s11b.VERIFY_SEED).choice(
        len(PW["user_id"]),
        size=min(s11b.VERIFY_WIDE_USERS, len(PW["user_id"])),
        replace=False,
    )
)
J_S = next(int(j) for j in VS if int(PW["user_id"][j]) not in REF_SET)


def wide_redraw(d):
    r = first_neg(d, J_S) + 25
    ex = excl_of(int(d["user_id"][J_S]))
    d["item"][r] = next(
        x for x in range(1, W.NI) if x not in ex and x != int(d["item"][r])
    )


check(
    with_file(
        "fusion_pairs_wide.npz", wide_held_pick, "held-out picks of their listener"
    ),
    "P11b verify() fails when a draw is a held-out pick of its listener",
)
check(
    with_file("fusion_pairs_wide.npz", wide_out_of_core, "is not a P3 proposal item"),
    "P11b verify() fails when a p_pop draw lies outside P3's core",
)
check(
    with_file(
        "fusion_pairs_wide.npz", wide_redraw, "differ from a fresh recomputation"
    ),
    "P11b verify() fails when an allowed draw of a checked listener "
    "leaves its random stream",
)


def wide_consumed(d):
    d["consumed"][J_S, 0] += 1


def drop_last_row(d):
    d["indptr"][-1] -= 1
    for key in ("item", "kind", "pick", "draw"):
        d[key] = d[key][:-1]


def wide_draw_number(d):
    d["draw"][first_neg(d, NON_REF) + 22] = 5


def wide_neg_pick(d):
    d["pick"][first_neg(d, NON_REF) + 23] = fp.PICK_VAL_LAST


def wide_pos_item(d):
    r = int(d["indptr"][NON_REF])
    d["item"][r] = d["item"][r] % (W.NI - 2) + 1


check(
    with_file(
        "fusion_pairs_wide.npz", wide_consumed, "differ from a fresh recomputation"
    ),
    "P11b verify() fails when the raw draw count of a checked listener "
    "leaves its random stream",
)
check(
    with_file(
        "fusion_pairs_wide.npz",
        drop_last_row,
        "does not hold its positives plus two sets",
    ),
    "P11b verify() fails when a listener holds a draw too few",
)
check(
    with_file("fusion_pairs_wide.npz", wide_draw_number, "wrong kind or draw number"),
    "P11b verify() fails when a draw carries another draw number",
)
check(
    with_file("fusion_pairs_wide.npz", wide_neg_pick, "differ from the val picks"),
    "P11b verify() fails when a negative carries a pick flag",
)
check(
    with_file("fusion_pairs_wide.npz", wide_pos_item, "differ from the val picks"),
    "P11b verify() fails when a positive leaves the val picks",
)
st_roles = p11_loaders(s11b.StageP11b())


def fewer_roles():
    u, sp, pk, (pi, it, fl), c = roles()
    n = int(pi[-2])
    return u[:-1], sp[:-1], pk[:-1], (pi[:-1], it[:n], fl[:n]), c


st_roles._roles = fewer_roles
check(
    raises(lambda: st_roles.verify({}), "P11's rules now give other listeners"),
    "P11b stops when P11's rules give other listeners than " "fusion_pairs.npz holds",
)
cons = PW["consumed"].astype(np.float64)
cells = [c.strip() for c in rep11b.split("| wide |")[1].split("\n")[0].split("|")]
check(
    cells[3] == f"{cons[:, 0].sum() / (KW * len(cons)):.4f}"
    and cells[4] == f"{cons[:, 1].sum() / (KW * len(cons)):.4f}",
    "the report's raw draws per accepted draw, per kind, follow from the "
    "stored counts",
)
check(
    not raises(lambda: st_v11.verify({}), "no such text")
    and st_v11._verified is not None,
    "P11b verify() passes again on the restored files",
)

# 2: P12b
st12b, lines12b = run_stage(p12_loaders(s12b.StageP12b()))
F12 = load_npz(MODEL / "fusion_features.npz")
FW = load_npz(MODEL / "fusion_features_wide.npz")
FR = load_npz(MODEL / "fusion_features_ref.npz")
rows_ok = True
for P_, F_ in ((PW, FW), (PR, FR)):
    for j, u in enumerate(P_["user_id"]):
        a, b = int(P_["indptr"][j]), int(P_["indptr"][j + 1])
        items_h, w_h = cd.history_of(W.u_indptr, W.u_indices, W.u_data, int(u))
        v, h, t = cd.user_rows(
            W.space_pairs, items_h, w_h, P_["item"][a:b].astype(np.int64)
        )
        rows_ok &= (
            np.array_equal(
                F_["values"][a:b].view(np.uint32), v.astype(np.float32).view(np.uint32)
            )
            and np.array_equal(F_["has"][a:b], h)
            and np.array_equal(F_["match_tier"][a:b], t.astype(np.int8))
        )
check(
    rows_ok
    and str(FW["pairs_sha256"]) == ev.npz_hash(MODEL / "fusion_pairs_wide.npz")
    and str(FR["pairs_sha256"]) == ev.npz_hash(MODEL / "fusion_pairs_ref.npz")
    and json.loads(str(FW["params"])) == s12._params(),
    f"P12b: all {len(FW['match_tier']) + len(FR['match_tier'])} rows of "
    f"both files equal concept_data.user_rows() for their listener, bit "
    f"for bit, and each file names its pairs",
)
shared_ok = True
n_shared = 0
for j, u in enumerate(PW["user_id"]):
    i11 = int(np.searchsorted(P11["user_id"], u))
    a11 = int(P11["indptr"][i11])
    n_p = int(np.count_nonzero(P11["kind"][a11 : P11["indptr"][i11 + 1]] == 0))
    a = int(PW["indptr"][j])
    parts_w = [
        np.arange(a, a + n_p),
        a + n_p + np.arange(K11),
        a + n_p + KW + np.arange(K11),
    ]
    rw = np.concatenate(parts_w)
    r11 = np.arange(a11, a11 + n_p + 2 * K11)
    for key in ("values", "has", "match_tier"):
        x, y = FW[key][rw], F12[key][r11]
        if key == "values":
            x, y = x.view(np.uint32), y.view(np.uint32)
        shared_ok &= np.array_equal(x, y)
    n_shared += len(rw)
check(
    shared_ok and st12b._verified_b["shared_p12"] == n_shared,
    f"P12b: the {n_shared} wide rows that P11 holds carry P12's stored "
    f"features bit for bit, as verify() counts",
)
rep12b = (MODEL / "reports" / "p12b_report.md").read_text()
check(
    "| wide |" in rep12b
    and "| ref |" in rep12b
    and ";" not in rep12b
    and "equal P12's stored features bit for bit" in rep12b,
    "P12b's report holds both files and the shared-row line",
)
sha_w = ev.npz_hash(MODEL / "fusion_features_wide.npz")
_, lines12r = run_stage(p12_loaders(s12b.StageP12b()), report=False)
check(
    sum("every block is done" in x for x in lines12r) == 2
    and ev.npz_hash(MODEL / "fusion_features_wide.npz") == sha_w,
    "P12b: a finished run resumes without work for both files",
)
(MODEL / "fusion_features_wide.npz").unlink()
_, lines12l = run_stage(p12_loaders(s12b.StageP12b()), report=False)
check(
    ev.npz_hash(MODEL / "fusion_features_wide.npz") == sha_w
    and sum("every block is done" in x for x in lines12l) == 1,
    "P12b: a lost output is computed again, with the same content",
)

# P12b faults. P12's own checks sample listeners with its seed. The faults
# sit in listeners outside that sample, so only the shared-row check can
# see them.
st_v12 = p12_loaders(s12b.StageP12b())


def p12_sample(n_users):
    rng = np.random.default_rng(s12.SEED)
    return set(
        rng.choice(n_users, size=min(s12.VERIFY_USERS, n_users), replace=False).tolist()
    )


def with_feat(name, change, text):
    path = MODEL / name
    good = load_npz(path)
    bad = {k: v.copy() for k, v in good.items()}
    change(bad)
    np.savez(path, **bad)
    try:
        return raises(lambda: st_v12.verify({}), text)
    finally:
        np.savez(path, **good)


TAG = cn.CONTINUOUS.index("tag_affinity")


def halve_shared(P_, d, sample, k_pref):
    for j in range(len(P_["user_id"])):
        if j in sample:
            continue
        a = first_neg(P_, j)
        for r in range(a, a + k_pref):
            if d["has"][r, TAG] and d["values"][r, TAG] > 0:
                d["values"][r, TAG] *= np.float32(0.5)
                return
    raise AssertionError("no row to change")


check(
    with_feat(
        "fusion_features_wide.npz",
        lambda d: halve_shared(PW, d, p12_sample(len(PW["user_id"])), K11),
        "differ from P12's stored features",
    ),
    "P12b verify() fails when a wide row that P11 holds leaves P12's " "features",
)
check(
    with_feat(
        "fusion_features_ref.npz",
        lambda d: halve_shared(PR, d, p12_sample(len(PR["user_id"])), KW),
        "reference features of the first draws differ",
    ),
    "P12b verify() fails when a reference row leaves its wide row",
)
check(
    with_feat(
        "fusion_features_wide.npz",
        lambda d: d.update(pairs_sha256=np.array("0" * 64)),
        "built from other fusion pairs",
    ),
    "P12b verify() fails for features of other pairs",
)
check(
    with_feat(
        "fusion_features.npz",
        lambda d: d.update(pairs_sha256=np.array("2" * 64)),
        "built from other pairs than the ones P11b extends",
    ),
    "P12b verify() fails when P12's features belong to other pairs",
)
row_w = rep12b.split("| wide |")[1].split("\n")[0]
check(
    ", ".join(
        f"{c} {100 * float(FW['has'][:, j].mean()):.1f}%"
        for j, c in enumerate(cn.CONTINUOUS)
    )
    in row_w
    and f"{len(FW['match_tier']):,}" in row_w,
    "the P12b report's data shares per concept follow from the wide file",
)
real_mp = s12b.marker_path
mk = TMP / "markers"
mk.mkdir()
for stage_ in s12b.StageP12b.upstream:
    (mk / f"{stage_}.json").write_text(
        json.dumps({"outputs": {"a": {"sha256": stage_ + "-sha"}, "b": "plain"}})
    )
s12b.marker_path = lambda name: mk / f"{name}.json"
try:
    up = s12b.StageP12b()._upstream_outputs()
finally:
    s12b.marker_path = real_mp
check(
    up
    == {
        s_: {"a": s_ + "-sha", "b": "plain"}
        for s_ in s12b.StageP12b.upstream
        if s_ not in ("p11_fusion_pairs", "p11b_more_pairs")
    },
    "P12b's resume key holds the output SHA-256 of every upstream stage "
    "but the two pairs stages, whose hash the key holds itself",
)

# 3: P13b part 2, the helpers
check(
    s13b2.n_negatives(("mix32", "A"), 24) == 48
    and s13b2.n_negatives(("p_pop", "B"), 24) == 24
    and s13b2.flip_monotone((1, 0, 0, 1)) == (0, 0, 0, 1)
    and s13b2.ref_arms(16) == s13b2.REF_MIX_ARMS + s13b2.REF_POP_ARMS
    and s13b2.ref_arms(32) == s13b2.REF_MIX_ARMS
    and s13b2.fit_blocks(1) == 5
    and s13b2.fit_blocks(20_001) == 6,
    "the helpers: negatives per arm and level, the twin cell, the arms of "
    "each reference level, the fit blocks by rows",
)
code_ = s13b2.StageP13b2()._code()
check(
    all(
        n in code_
        for n in (
            "stage_p13b2.py",
            "stage_p13b.py",
            "stage_p11b.py",
            "stage_p13a.py",
            "gam.py",
        )
    ),
    "the key of the saved fits holds the code of this stage, of P13b, "
    "P11b, P13a and the GAM library",
)
ARMS4 = set(s13b2.FULL_ARMS + s13b2.REF_POP_ARMS)
ids = {
    s13b2.fid2(w, k, c, a, 0.1)
    for w in ("ref", "wide")
    for k in (16, 24)
    for c in ((1, 0, 0, 0), (0, 0, 0, 0))
    for a in ARMS4
}
check(
    len(ARMS4) == 4 and len(ids) == 2 * 2 * 2 * 4,
    "fid2 is unique over files, levels, cells and arms",
)
c1, c0 = (1, 0, 0, 0), (0, 0, 0, 0)
A, B = ("mix32", "A"), ("mix32", "B")
PA, PB = ("p_pop", "A"), ("p_pop", "B")
t1, t2, t3, t4 = (np.array([float(i)]) for i in range(1, 5))
check(
    s13b2.warm_start({}, {}, c1, A, t1) is t1
    and s13b2.warm_start({}, {(c1, A): t2}, c1, B, t1) is t2
    and s13b2.warm_start({(c1, B): t3}, {(c1, A): t2}, c1, B, t1) is t3
    and s13b2.warm_start({(c1, PB): t4}, {(c1, B): t3}, c1, PB, t1) is t3
    and s13b2.warm_start({(c1, PB): t4}, {}, c1, PB, t1) is t4
    and s13b2.warm_start({(c1, A): t4}, {}, c0, A, t1) is t1,
    "warm_start: a mixture fit starts one level below, else from the "
    "level's form A fit, else from part 1, a p_pop fit from the level's "
    "mixture fit of its form",
)
term = {
    "name": "x",
    "knots": [0.0, 1.0, 3.0],
    "zero_state": True,
    "missing_state": True,
}
tab = np.array([1.0, 2.0, 5.0, -1.0, 0.5, 0.0, 0.0])
pts = s13b2.shape_points([term], tab, {"x": [0.5, 2.0, 4.0]})
pts_s = s13b2.shape_points(
    [term], tab + np.array([0.7, 0.7, 0.7, 0.7, 0.7, 0, 0]), {"x": [0.5, 2.0, 4.0]}
)
check(
    np.allclose(
        pts["x"], [1.5 - 3.5, 0.0, 5.0 - 3.5, -1.0 - 3.5, 0.5 - 3.5], rtol=0, atol=1e-15
    )
    and np.allclose(pts_s["x"], pts["x"], rtol=0, atol=1e-12)
    and s13b2.shape_points([term], tab, {"x": None}) == {},
    "shape_points: the values at the positions (flat beyond the last "
    "knot), the zero and the missing constant, each minus the middle "
    "value, unchanged by a shift of the term",
)
check(
    s13b2.points_gap({"x": [0, 1], "y": [2]}, {"x": [0.5, 1], "z": [9]}) == 0.5,
    "points_gap: the largest difference over the shared terms",
)


def fake_points(delta):
    return {"t": [0.0, delta]}


def fake_means(vals):
    """{(cell, arm, level): mean} from {(cell, arm label, level): mean},
    0 where not given."""
    out = {}
    for cell in (c1, c0):
        for arm in s13b2.FULL_ARMS:
            for k in FULL:
                out[(cell, arm, k)] = vals.get((cell, s13b.arm_label(arm), k), 0.0)
    return out


def ref_pts(bias_a, bias_b, spread=0.0):
    out = {}
    for k in REF:
        for arm in s13b2.ref_arms(k):
            out[(k, arm)] = fake_points(0.0)
    out[(REF[-1], PA)] = fake_points(spread)
    out[(REF[0], A)] = fake_points(bias_a)
    out[(REF[0], B)] = fake_points(bias_b)
    return out


m1 = fake_means(
    {
        (c1, "mix32-A", 16): 0.10,
        (c1, "mix32-A", 32): 0.2,
        (c1, "mix32-B", 16): 0.111,
        (c1, "mix32-B", 24): 0.112,
        (c1, "mix32-B", 32): 0.113,
        (c1, "p_pop-B", 16): 0.105,
        (c1, "p_pop-B", 24): 0.109,
        (c1, "p_pop-B", 32): 0.1115,
        (c0, "p_pop-B", 32): 0.1140,
    }
)
d_a = s13b2.decide2(ref_pts(0.3, 0.4, 0.2), m1, c1, c0)
check(
    d_a["form"] == "B"
    and d_a["form_by"].startswith("selection")
    and d_a["g_ref"] == 0.2
    and d_a["bias"] == {"A": 0.3, "B": 0.4}
    and (d_a["arm"], d_a["level"]) == (["p_pop", "B"], 32)
    and d_a["m"] == 32
    and d_a["cell"] == "m0w0t0p0"
    and not d_a["twin_kept_out"],
    "decide2: with both biases within tau the higher score at m = 32 "
    "picks the form, the fewest negatives within SESOI win and a tie in m "
    "goes to the higher score, and a twin more than SESOI above replaces "
    "the chosen cell",
)
d_b = s13b2.decide2(ref_pts(0.9, 0.6), m1, c1, c0)
d_c = s13b2.decide2(ref_pts(0.3, 0.9), m1, c1, c0)
d_t = s13b2.decide2(
    ref_pts(0.1, 0.2),
    fake_means(
        {
            (c1, "mix32-A", 16): 0.1,
            (c1, "mix32-B", 16): 0.1,
            (c0, "mix32-A", 16): 0.1019,
        }
    ),
    c1,
    c0,
)
check(
    d_b["form"] == "B"
    and d_b["form_by"] == "bias"
    and d_c["form"] == "A"
    and d_c["form_by"] == "bias"
    and d_a["scores_m32"] == {"A": 0.10, "B": 0.111}
    and d_t["form"] == "A"
    and (d_t["arm"], d_t["level"], d_t["cell"]) == (["mix32", "A"], 16, "m1w0t0p0")
    and d_t["twin_kept_out"],
    "decide2: unless both biases lie within tau the smaller bias picks the "
    "form, the scores at m = 32 come from the smallest wide level, a score "
    "tie goes to form A, and a twin within SESOI stays out",
)
rec_ = logging.LogRecord(
    "pipeline",
    logging.INFO,
    "",
    0,
    "p13a: swap. Start again with P13A_WORKERS=8 python3 "
    "-m pipeline.runner --only p13a_fit.",
    (),
    None,
)
rec2 = logging.LogRecord(
    "pipeline", logging.INFO, "", 0, "p13b: probe mean 0.5 s", (), None
)
rec3 = logging.LogRecord(
    "pipeline", logging.INFO, "", 0, "p13b2: a line of this stage", (), None
)
for r_ in (rec_, rec2, rec3):
    s13b2._Relabel().filter(r_)
check(
    rec_.msg == "p13b2: swap. Start again with P13A_WORKERS=8 python3 -m "
    "pipeline.runner --only p13b2_loss."
    and rec2.msg == "p13b2: probe mean 0.5 s"
    and rec3.msg == "p13b2: a line of this stage",
    "the log filter renames the pass lines of P13a and P13b, points the "
    "advice at this stage and leaves the stage's own lines",
)
# Part 1's final model in the three layouts of p13b_model.json.
fake_a = {
    "path": [
        {"lambda": 0.1 * (i + 1), "theta": [float(i)], "fit": {"objective": 10.0 + i}}
        for i in range(4)
    ]
}
fake_b = {
    "cells": [{"reuses_base_fit": True, "fingerprint": "fpb"}],
    "extension": {
        "fits": [
            {
                "lambda": 1e-4,
                "theta": [7.0],
                "fingerprint": "fpe",
                "fit": {"objective": 3.0},
            }
        ]
    },
    "final": {
        "chosen": 1,
        "cell": "m1w0t0p0",
        "reuses_base_path": False,
        "selection": [{"mean": 0.1}, {"mean": 0.2}],
        "path": [
            None,
            {
                "lambda": 0.5,
                "theta": [2.0],
                "fingerprint": "fpf",
                "fit": {"objective": 5.0},
            },
        ],
    },
}
st_h = s13b2.StageP13b2()
r1 = st_h._part1_final(fake_a, fake_b)
fake_b["final"].update(
    reuses_base_path=True,
    cell="m0w0t1p1",
    chosen=4,
    selection=[{"mean": 0.0}] * 4 + [{"mean": 0.4}],
)
r2 = st_h._part1_final(fake_a, fake_b)
fake_b["final"].update(chosen=2)
r3 = st_h._part1_final(fake_a, fake_b)
check(
    r1[0] == (1, 0, 0, 0)
    and r1[1] == 0.5
    and r1[3:] == ("fpf", 5.0, 0.2)
    and r2[0] == (0, 0, 1, 1)
    and r2[1] == 1e-4
    and r2[3:] == ("fpe", 3.0, 0.4)
    and r3[1] == 0.30000000000000004
    and r3[2].tolist() == [2.0]
    and r3[3:] == ("fpb", 12.0, 0.0),
    "part 1's final model: the final path, the extension of the base "
    "path, and P13a's path with the base cell's fingerprint",
)

# 4: P13b part 2 on the chain
st2, lines2 = run_stage(p13_loaders(s13b2.StageP13b2()))
DOC = json.loads((MODEL / "p13b2_model.json").read_text())
ARR = load_npz(MODEL / "p13b2_arrays.npz")
A_DOC = json.loads((MODEL / "p13a_model.json").read_text())
B_DOC = json.loads((MODEL / "p13b_model.json").read_text())
B_ARR = load_npz(MODEL / "p13b_arrays.npz")
rep2 = (MODEL / "reports" / "p13b2_report.md").read_text()
gaps2 = st2._verified
check(
    gaps2 is not None
    and gaps2["kkt"] <= s13.KKT_TOL
    and gaps2["centering"] <= s13b2.CENTER_TOL
    and gaps2["offsets"] == 0.0,
    f"run() and verify() pass on the chain (gaps: "
    f"{ {k: f'{v:.1e}' for k, v in gaps2.items()} })",
)
check(
    all(
        h in rep2
        for h in (
            "## 1. Data",
            "## 2. The reference",
            "## 3. The full data",
            "## 4. Decisions",
            "## 5. The final model",
            "- D1:",
            "- D2:",
        )
    )
    and ";" not in rep2,
    "report() writes every section, without a semicolon",
)
pass_lines = [x for x in lines2 if "selection pass" in x or "probe" in x]
check(
    pass_lines
    and not any(x.startswith(("p13a", "p13b:")) for x in lines2)
    and any(x.startswith("p13b2: probe mean") for x in lines2)
    and any(x.startswith("p13b2: fits (28): done") for x in lines2),
    f"while part 2 runs, every log line names P13b2 ({len(pass_lines)} "
    f"pass lines), and the fit progress closes",
)
check(
    len(DOC["ref_fits"]) == 10
    and len(DOC["wide_fits"]) == 18
    and ARR["ranks"].shape == (18, len(W.SEL), 2)
    and DOC["part1"]["cell"] == "m1w0t0p0"
    and DOC["part1"]["twin"] == "m0w0t0p0",
    "the model holds 10 reference fits, 18 full-data fits and their "
    "ranks, for the chosen cell m1w0t0p0 and its twin m0w0t0p0",
)
print(f"      decisions: {json.dumps(DOC['decisions'])}")

# The rows of every level, per listener, written here.
A_MC = A_DOC["mainstreaminess"]


def mainstream(items, weights):
    raw = float(np.sum(weights * W.popularity[items]) / np.sum(weights))
    return float(np.clip((raw - A_MC["mean"]) / A_MC["sd"], A_MC["lo"], A_MC["hi"]))


M_OF = {}
for u in set(PW["user_id"].tolist()) | set(W.SEL.tolist()):
    ih, wh = cd.history_of(W.u_indptr, W.u_indices, W.u_data, int(u))
    M_OF[int(u)] = mainstream(ih, wh.astype(np.float64))


def level_data(which, k):
    P_, F_, K_ = (PW, FW, KW) if which == "wide" else (PR, FR, KR)
    rows, lens, npos = [], [], []
    for j in range(len(P_["user_id"])):
        a, b = int(P_["indptr"][j]), int(P_["indptr"][j + 1])
        n_p = int(np.count_nonzero(P_["kind"][a:b] == 0))
        rows.append(
            np.concatenate(
                [
                    np.arange(a, a + n_p),
                    a + n_p + np.arange(k),
                    a + n_p + K_ + np.arange(k),
                ]
            )
        )
        lens.append(n_p + 2 * k)
        npos.append(n_p)
    rows = np.concatenate(rows)
    lens, npos = np.array(lens), np.array(npos)
    user = np.repeat(np.arange(len(lens)), lens)
    item = P_["item"][rows].astype(np.int64)
    return {
        "k": k,
        "rows": rows,
        "lens": lens,
        "npos": npos,
        "U": len(lens),
        "starts": np.concatenate(([0], np.cumsum(lens)[:-1])),
        "item": item,
        "kind": P_["kind"][rows],
        "q_u": 1.0 / P_["n_allowed"][user].astype(np.float64),
        "q_p": PP_SLOT[item] / P_["pop_allowed"][user],
        "tier": F_["match_tier"][rows].astype(np.int64),
        "m": np.array([M_OF[int(u)] for u in P_["user_id"]])[user],
        "cols": {
            c: (
                F_["values"][rows, i].astype(np.float64),
                F_["has"][rows, i].astype(bool),
            )
            for i, c in enumerate(cn.CONTINUOUS)
        },
    }


LEVELS = {
    (w, k): level_data(w, k)
    for w in ("ref", "wide")
    for k in (REF if w == "ref" else FULL)
}
same_rows = True
for (w, k), lv in LEVELS.items():
    st2._set_level(w, k)
    inp = st2._inputs()
    same_rows &= (
        np.array_equal(inp["item"], lv["item"])
        and np.array_equal(inp["kind"], lv["kind"])
        and np.array_equal(inp["lens"], lv["lens"])
        and np.array_equal(
            inp["f"]["values"].view(np.uint32),
            (FW if w == "wide" else FR)["values"][lv["rows"]].view(np.uint32),
        )
    )
st2._drop_inputs()
check(
    same_rows
    and DOC["data"]["wide:16"]["rows"] == len(LEVELS[("wide", 16)]["rows"])
    and DOC["data"]["ref:128"]["rows"] == len(LEVELS[("ref", 128)]["rows"]),
    "every level holds each listener's positives and its first k draws "
    "of each kind, rows and features, built here listener by listener",
)


def term_values(t, tab, v, h, tier, m):
    kn = np.asarray(t["knots"], dtype=np.float64)
    K = len(kn)
    x = np.clip(np.where(h, v, kn[0]), kn[0], kn[-1])
    val = np.interp(x, kn, tab[:K])
    if t["zero_state"]:
        val = np.where(h & (v == 0.0), tab[K + ZERO], val)
    miss = tab[K + MISS] if t["missing_state"] else 0.0
    val = np.where(h, val, miss)
    if t["tiers"] is not None:
        val = np.where(np.isin(tier, t["tiers"]), val, 0.0)
    if t["multiplier"] is not None:
        val = m * val
    return val


def scores_of(j_terms, tables, cols, tier, m):
    out = np.zeros(len(tier))
    off = 0
    for t in j_terms:
        size = len(t["knots"]) + 4
        v, h = cols[t["source"]]
        out += term_values(t, np.asarray(tables[off : off + size]), v, h, tier, m)
        off += size
    return out


def arm_loss(s, arm, lv):
    """The loss of an arm on a level, one listener at a time."""
    k, U = lv["k"], lv["U"]
    total = []
    for u in range(U):
        r0, npu = int(lv["starts"][u]), int(lv["npos"][u])
        negs = r0 + npu + (np.arange(2 * k) if arm[0] == "mix32" else k + np.arange(k))
        m = len(negs)
        q = lv["q_p"] if arm[0] == "p_pop" else 0.5 * lv["q_u"] + 0.5 * lv["q_p"]
        zn = s[negs] - np.log(m * q[negs])
        for p in range(r0, r0 + npu):
            zp = s[p]
            if arm[1] == "B":
                if q[p] <= 0.0:
                    continue
                zp = zp - math.log(m * q[p])
            total.append(
                (float(np.logaddexp.reduce(np.concatenate(([zp], zn)))) - zp)
                / (npu * U)
            )
    return math.fsum(total)


def lv_of(j):
    return LEVELS[(j["data"], j["level"])]


ALL_FITS = DOC["ref_fits"] + DOC["wide_fits"]
worst_loss = 0.0
for j in ALL_FITS:
    lv = lv_of(j)
    s = scores_of(j["terms"], j["tables_raw"], lv["cols"], lv["tier"], lv["m"])
    lo = arm_loss(s, tuple(j["arm"].split("-")), lv)
    worst_loss = max(worst_loss, abs(lo - j["fit"]["loss"]) / max(1.0, abs(lo)))
check(
    worst_loss < 1e-12,
    f"the stored loss of each of the {len(ALL_FITS)} fits equals the loss "
    f"recomputed here from the raw rows of its level, with each arm's "
    f"negatives, offsets and weights (largest relative gap "
    f"{worst_loss:.1e})",
)
n_q0 = int(
    np.count_nonzero(
        LEVELS[("wide", 16)]["q_p"][LEVELS[("wide", 16)]["kind"] == 0] == 0.0
    )
)
check(
    n_q0 > 0 and DOC["data"]["wide:16"]["pos_no_pop"] == n_q0,
    f"the positives outside P3's core get weight 0 under p_pop form B "
    f"({n_q0} at each wide level), and the model counts them",
)

# KKT conditions by finite differences of the loss written here.
BASE = st2._base_context()


def kkt_fd(j):
    cell = tuple(int(x) for x in j["cell"][1::2])
    lay, _ = st2._cell_layout(BASE, cell)
    theta = np.asarray(j["theta"], dtype=np.float64)
    lv = lv_of(j)
    arm = tuple(j["arm"].split("-"))

    def F(th):
        s_ = scores_of(j["terms"], lay.tables(th), lv["cols"], lv["tier"], lv["m"])
        return (
            arm_loss(s_, arm, lv)
            + j["lambda"] * float(th @ lay.Q @ th)
            + s13.RIDGE * float(th @ th)
        )

    eps = 1e-6
    worst, active = 0.0, 0
    for i, (x_, (lo, hi)) in enumerate(zip(theta, lay.bounds)):
        e = np.zeros(len(theta))
        e[i] = eps
        g = (F(theta + e) - F(theta - e)) / (2 * eps)
        if lo is not None and x_ <= lo:
            worst, active = max(worst, max(-g, 0.0)), active + 1
        elif hi is not None and x_ >= hi:
            worst, active = max(worst, max(g, 0.0)), active + 1
        else:
            worst = max(worst, abs(g))
    return worst, active


j_b = next(
    j
    for j in DOC["wide_fits"]
    if j["level"] == 24 and j["cell"] == "m1w0t0p0" and j["arm"] == "p_pop-B"
)
j_f = next(j for j in DOC["ref_fits"] if j["level"] == 128 and j["arm"] == "mix32-B")
k1, act1 = kkt_fd(j_b)
k2, _ = kkt_fd(j_f)
check(
    k1 < 1e-6 and k2 < 1e-6 and act1 > 0,
    f"finite differences of the loss written here meet the KKT "
    f"conditions at the bounded p_pop form B fit ({act1} parameters at a "
    f"bound, worst {k1:.1e}) and at a reference fit (worst {k2:.1e})",
)

# Part 1's final fit returns at the smallest wide level.
fin1 = B_DOC["final"]
j16 = next(
    j
    for j in DOC["wide_fits"]
    if j["level"] == 16 and j["cell"] == "m1w0t0p0" and j["arm"] == "mix32-A"
)
theta1 = np.asarray(fin1["path"][fin1["chosen"]]["theta"])
lv16 = LEVELS[("wide", 16)]
lay1, _ = st2._cell_layout(BASE, CHOSEN)
s1 = scores_of(j16["terms"], lay1.tables(theta1), lv16["cols"], lv16["tier"], lv16["m"])
loss1 = arm_loss(s1, ("mix32", "A"), lv16)
check(
    abs(loss1 - fin1["path"][fin1["chosen"]]["fit"]["loss"]) < 1e-12
    and np.array_equal(np.asarray(j16["theta"]), theta1)
    and j16["fit"]["newton_steps"] == 0
    and np.array_equal(
        ARR["ranks"][DOC["wide_fits"].index(j16)], B_ARR["ranks_final"][fin1["chosen"]]
    )
    and gaps2["part1"] < 1e-9,
    f"part 1's final fit returns at wide level 16 with mixture form A: "
    f"its loss on the level's rows, recomputed here, equals part 1's, the "
    f"refit starts at part 1's parameters and needs no Newton step, and "
    f"its selection ranks equal part 1's ranks",
)

# Centering on each level's p_pop negatives, and the explanation points.
worst_c, worst_p, worst_s = 0.0, 0.0, 0.0
SQ = A_DOC["shape_quantiles"]


def points_here(terms, tables):
    out = {}
    off = 0
    for t in terms:
        K = len(t["knots"])
        tab_ = np.asarray(tables[off : off + K + 4], dtype=np.float64)
        off += K + 4
        xs = SQ.get(t["name"])
        if not xs:
            continue
        kn = np.asarray(t["knots"], dtype=np.float64)
        vals = np.interp(np.clip(np.asarray(xs), kn[0], kn[-1]), kn, tab_[:K])
        mid = vals[len(xs) // 2]
        p_ = list(vals - mid)
        if t["zero_state"]:
            p_.append(tab_[K + ZERO] - mid)
        if t["missing_state"]:
            p_.append(tab_[K + MISS] - mid)
        out[t["name"]] = p_
    return out


for j in ALL_FITS:
    lv = lv_of(j)
    ref_rows = lv["kind"] == fp.KIND_POP
    off = 0
    for t in j["terms"]:
        size = len(t["knots"]) + 4
        if not t["centered"]:
            v_, h_ = lv["cols"][t["source"]]
            val = term_values(
                dict(t, multiplier=None),
                np.asarray(j["tables"][off : off + size]),
                v_[ref_rows],
                h_[ref_rows],
                lv["tier"][ref_rows],
                1.0,
            )
            worst_c = max(worst_c, abs(float(np.mean(val))))
        off += size
    ph = points_here(j["terms"], j["tables_raw"])
    pc = points_here(j["terms"], j["tables"])
    for name in ph:
        worst_p = max(
            worst_p,
            float(np.max(np.abs(np.asarray(ph[name]) - np.asarray(j["points"][name])))),
        )
        worst_s = max(
            worst_s, float(np.max(np.abs(np.asarray(ph[name]) - np.asarray(pc[name]))))
        )
check(
    worst_c < 1e-12 and worst_p < 1e-14 and worst_s < 1e-12,
    f"every stored table has mean 0 over its level's p_pop negatives "
    f"(largest {worst_c:.1e}), the stored explanation points equal the "
    f"points computed here ({worst_p:.1e}), and the raw and the centered "
    f"tables give the same points ({worst_s:.1e})",
)

# Every selection rank by a dense count.


def rank_bounds(s, pick, exclude, tol=1e-9):
    s = np.array(s, dtype=np.float64)
    s[0] = -np.inf
    s[exclude] = -np.inf
    sp_ = s[pick]
    s[pick] = -np.inf
    exact = 1 + int(np.count_nonzero(s > sp_)) + int(np.count_nonzero(s[:pick] == sp_))
    near = int(np.count_nonzero(np.abs(s - sp_) <= tol))
    if near == int(np.count_nonzero(s == sp_)):
        return exact, exact
    g = int(np.count_nonzero(s > sp_ + tol))
    return 1 + g, 1 + g + near


all_ok, n_exact, n_beyond = True, 0, 0
for jj, u in enumerate(W.SEL):
    items_h, w_h = cd.history_of(W.u_indptr, W.u_indices, W.u_data, int(u))
    full = cn.full_catalog(cn.build_user_state(items_h, w_h, W.SPACE), W.SPACE)
    for c, j in enumerate(DOC["wide_fits"]):
        s = scores_of(j["terms"], j["tables"], full, W.SPACE.tier, M_OF[int(u)])
        for p_ in range(2):
            lo, hi = rank_bounds(s, int(W.SEL_PICKS[jj, p_]), items_h)
            got = int(ARR["ranks"][c, jj, p_])
            all_ok &= min(lo, CUT1) <= got <= min(hi, CUT1)
            n_exact += int(lo == hi and lo <= s13.RANK_CUT)
            n_beyond += int(lo > s13.RANK_CUT)
check(
    all_ok and n_exact > 0 and n_beyond > 0,
    f"every stored rank of the 18 fits and {len(W.SEL)} selection "
    f"listeners equals the P5 count on a dense scorer written here, "
    f"censored at {CUT1} ({n_exact} exact ranks, {n_beyond} beyond)",
)

# D1 to D4 from the stored points and ranks, written here.


def ndcg(r):
    r = np.asarray(r, dtype=np.float64)
    return np.where(r <= 100, 1.0 / np.log2(r + 1.0), 0.0).mean(axis=-1)


SC = ndcg(ARR["ranks"])
MEAN = {
    (j["cell"], j["arm"], j["level"]): float(SC[i].mean())
    for i, j in enumerate(DOC["wide_fits"])
}
PTS = {
    (j["level"], j["arm"]): points_here(j["terms"], j["tables_raw"])
    for j in DOC["ref_fits"]
}


def gap(a, b):
    return max(float(np.max(np.abs(np.asarray(a[n]) - np.asarray(b[n])))) for n in a)


top_arms = ("mix32-A", "mix32-B", "p_pop-A", "p_pop-B")
g_ref = max(
    gap(PTS[(REF[-1], x)], PTS[(REF[-1], y)])
    for x, y in itertools.combinations(top_arms, 2)
)
R = PTS[(REF[-1], "mix32-A")]
bias = {f: gap(PTS[(REF[0], f"mix32-{f}")], R) for f in "AB"}
ch, tw = "m1w0t0p0", "m0w0t0p0"
if bias["A"] <= 0.5 and bias["B"] <= 0.5:
    form = "B" if MEAN[(ch, "mix32-B", 16)] > MEAN[(ch, "mix32-A", 16)] else "A"
else:
    form = "B" if bias["B"] < bias["A"] else "A"
cands = [
    (a, k)
    for a in (f"mix32-{form}", f"p_pop-{form}")
    for k in FULL
    if (ch, a, k) in MEAN
]
best = max(MEAN[(ch, a, k)] for a, k in cands)
ok_c = [c for c in cands if MEAN[(ch, *c)] >= best - 0.02 * best]
ok_c.sort(
    key=lambda c: ((2 if c[0].startswith("mix32") else 1) * c[1], -MEAN[(ch, *c)])
)
arm_w, lvl_w = ok_c[0]
cell_w = (
    tw
    if MEAN[(tw, arm_w, lvl_w)] - MEAN[(ch, arm_w, lvl_w)]
    > 0.02 * MEAN[(ch, arm_w, lvl_w)]
    else ch
)
d_ = DOC["decisions"]
fin_j = DOC["wide_fits"][DOC["final"]["index"]]
check(
    d_["form"] == form
    and "-".join(d_["arm"]) == arm_w
    and d_["level"] == lvl_w
    and d_["cell"] == cell_w
    and abs(d_["g_ref"] - g_ref) < 1e-14
    and d_["scores_m32"] == {f: MEAN[(ch, f"mix32-{f}", FULL[0])] for f in "AB"}
    and all(abs(d_["bias"][f] - bias[f]) < 1e-14 for f in "AB")
    and (fin_j["cell"], fin_j["arm"], fin_j["level"]) == (cell_w, arm_w, lvl_w)
    and all(
        abs(j["gap_to_reference"] - gap(PTS[(j["level"], j["arm"])], R)) < 1e-14
        for j in DOC["ref_fits"]
    ),
    f"D1 to D4 follow from the stored points and ranks by the rules "
    f"written here: form {form}, arm {arm_w}, level {lvl_w}, cell "
    f"{cell_w} (bias A {bias['A']:.3f}, B {bias['B']:.3f}, spread at the "
    f"largest level {g_ref:.3f})",
)
best_i = int(np.argmax(SC.mean(axis=1)))
sel_ok = DOC["best_index"] == best_i
for i, j in enumerate(DOC["wide_fits"]):
    dd = SC[best_i] - SC[i]
    sel_ok &= (
        abs(j["selection"]["mean"] - SC[i].mean()) < 1e-15
        and abs(j["selection"]["diff_to_best"] - dd.mean()) < 1e-15
        and abs(j["selection"]["se"] - dd.std(ddof=1) / math.sqrt(len(dd))) < 1e-15
    )
check(
    sel_ok,
    "each fit's score, difference to the best and standard error "
    "follow from the stored ranks",
)
mono_ok = True
for j in ALL_FITS:
    if not j["cell"].startswith("m1"):
        continue
    off = 0
    for t in j["terms"]:
        K = len(t["knots"])
        if t["monotone"]:
            hgt = np.asarray(j["tables_raw"][off : off + K]) * t["monotone"]
            mono_ok &= bool(np.all(np.diff(hgt) >= -1e-12))
            if t["zero_state"]:
                mono_ok &= (
                    j["tables_raw"][off + K + ZERO] * t["monotone"] <= hgt[0] + 1e-12
                )
        off += K + 4
check(
    mono_ok,
    "every fit of the monotone cell keeps its shapes "
    "non-decreasing and its zero states at or below the first "
    "knot, under both forms and both samplers",
)
sec3 = rep2.split("## 3. The full data")[1].split("## 4.")[0]
sec5 = rep2.split("## 5. The final model")[1]
check(
    sum(1 for x in sec3.splitlines() if x.startswith("| m")) == 18
    and sum(
        1
        for x in sec5.splitlines()
        if x.startswith("| ") and not x.startswith(("| term", "|---"))
    )
    == len(fin_j["terms"])
    and f"form A {d_['bias']['A']:.3f}" in rep2
    and f"form B {d_['bias']['B']:.3f}" in rep2
    and f"Final cell: {d_['cell']}." in rep2
    and [
        x.split("|")[1:3]
        for x in rep2.split("## 1. Data")[1].split("## 2.")[0].splitlines()
        if x.startswith(("| ref", "| wide"))
    ]
    == [
        [f" {w} ", f" {k} "]
        for w in ("ref", "wide")
        for k in (REF if w == "ref" else FULL)
    ]
    and f"lambda {DOC['part1']['lambda']:.3g}, selection score "
    f"{DOC['part1']['selection_mean']:.5f}. Twin cell: "
    f"{DOC['part1']['twin']}." in rep2
    and f"part 1's final model {DOC['part1']['selection_mean']:.5f}" in sec3
    and DOC["part1"]["selection_mean"]
    == B_DOC["final"]["selection"][B_DOC["final"]["chosen"]]["mean"],
    "the report holds a row per full-data fit and per term of the final "
    "model, the biases, the final cell and part 1's score",
)

# 5: workers and resume
root3 = TMP / "w3"
configure(root3, workers=3)
for f in MODEL.iterdir():
    if f.is_file() and not f.name.startswith("p13b2_"):
        shutil.copy(f, root3 / "model" / f.name)
run_stage(p13_loaders(s13b2.StageP13b2()))
d3 = json.loads((root3 / "model" / "p13b2_model.json").read_text())
a3 = load_npz(root3 / "model" / "p13b2_arrays.npz")
check(
    strip_seconds(d3) == strip_seconds(DOC)
    and all(np.array_equal(a3[k], ARR[k]) for k in ARR),
    "one worker and three workers give the same model and arrays",
)

configure(root1, workers=1)
fit_calls = []
real_fit, real_fb = gam.fit, gam_arms.fit_bounded


def count_fit(*a, **k):
    fit_calls.append("fit")
    return real_fit(*a, **k)


def count_fb(*a, **k):
    fit_calls.append("bounded")
    return real_fb(*a, **k)


def same_as_first():
    d = json.loads((MODEL / "p13b2_model.json").read_text())
    a = load_npz(MODEL / "p13b2_arrays.npz")
    return strip_seconds(d) == strip_seconds(DOC) and all(
        np.array_equal(a[k], ARR[k]) for k in ARR
    )


gam.fit, gam_arms.fit_bounded = count_fit, count_fb
try:
    _, lines_r = run_stage(p13_loaders(s13b2.StageP13b2()))
    check(
        not fit_calls
        and not any("catalog scorer ready" in x for x in lines_r)
        and any("selection pass resume" in x for x in lines_r)
        and same_as_first(),
        "a finished run fits nothing, scores no listener and writes the "
        "same outputs",
    )
    work = root1 / "scratch" / "p13b2"
    state = json.loads((work / "fits.json").read_text())
    lam = BASE["lambda"]
    lost = s13b2.fid2("wide", 24, (0, 0, 0, 0), ("mix32", "B"), lam)
    del state["fits"][lost]
    del state["means"]["wide:24"]
    del state["means"]["ref:32"]
    (work / "fits.json").write_text(json.dumps(state))
    parts = sorted((work / "pass_part2" / "selection").glob("part_*.npz"))
    parts[2].unlink()
    fit_calls.clear()
    _, lines_r2 = run_stage(p13_loaders(s13b2.StageP13b2()))
    state2 = json.loads((work / "fits.json").read_text())
    check(
        fit_calls == ["fit"]
        and sum("catalog scorer ready" in x for x in lines_r2) == 1
        and {"wide:24", "ref:32"} <= set(state2["means"])
        and same_as_first(),
        "a lost fit, lost background means (with and without a design) "
        "and a lost selection block are computed again, and the outputs "
        "stay the same",
    )
finally:
    gam.fit, gam_arms.fit_bounded = real_fit, real_fb

# 6: planted faults
st_v = p13_loaders(s13b2.StageP13b2())
mpath = MODEL / "p13b2_model.json"
apath = MODEL / "p13b2_arrays.npz"
good_doc = mpath.read_text()
good_arr = dict(np.load(apath))


def with_doc(change, text):
    d = json.loads(good_doc)
    change(d)
    mpath.write_text(json.dumps(d))
    try:
        return raises(lambda: st_v.verify({}), text)
    finally:
        mpath.write_text(good_doc)


def with_arr(change, text):
    a = {k: v.copy() for k, v in good_arr.items()}
    change(a)
    np.savez(apath, **a)
    try:
        return raises(lambda: st_v.verify({}), text)
    finally:
        np.savez(apath, **good_arr)


def wf(d, level, cell, arm):
    return next(
        j
        for j in d["wide_fits"]
        if j["level"] == level and j["cell"] == cell and j["arm"] == arm
    )


def bump(d, key, i, x):
    d[key][i] += x


check(
    with_doc(
        lambda d: bump(wf(d, 24, "m0w0t0p0", "mix32-A"), "theta", 3, 1e-3),
        "projected gradient",
    ),
    "verify() fails when a stored fit is moved off its optimum",
)
check(
    with_doc(
        lambda d: bump(d["ref_fits"][4], "tables_raw", 2, 1e-9),
        "stored raw tables differ",
    ),
    "verify() fails when stored raw tables differ from M theta",
)
check(
    with_doc(
        lambda d: bump(wf(d, 32, "m1w0t0p0", "p_pop-B"), "tables", 1, 1e-6),
        "centered tables differ",
    ),
    "verify() fails when stored centered tables leave the centering on "
    "the level's p_pop negatives",
)
check(
    with_doc(
        lambda d: d["wide_fits"][5]["points"]["popularity"].__setitem__(0, 9.0),
        "explanation points differ",
    ),
    "verify() fails when stored explanation points leave the tables",
)
check(
    with_doc(
        lambda d: d["ref_fits"][0]["fit"].__setitem__(
            "objective", d["ref_fits"][0]["fit"]["objective"] + 1e-6
        ),
        "objective differs",
    ),
    "verify() fails when a stored objective leaves its recomputation",
)
check(
    with_doc(
        lambda d: d["ref_fits"][3].__setitem__("gap_to_reference", 0.0),
        "gap to the reference",
    ),
    "verify() fails when a stored gap leaves the points",
)
check(
    with_doc(
        lambda d: d["decisions"].__setitem__(
            "form", "A" if d["decisions"]["form"] == "B" else "B"
        ),
        "decisions do not follow",
    ),
    "verify() fails when the stored decisions leave the rules",
)
check(
    with_doc(
        lambda d: d["final"].__setitem__("index", (d["final"]["index"] + 1) % 18),
        "final model is not",
    ),
    "verify() fails when the final model is another fit than the " "decisions name",
)
check(
    with_doc(
        lambda d: d["wide_fits"][7]["selection"].__setitem__("mean", 0.5),
        "score does not follow",
    ),
    "verify() fails when a stored score leaves the ranks",
)
check(
    with_doc(
        lambda d: d.__setitem__("best_index", (d["best_index"] + 1) % 18),
        "best configuration",
    ),
    "verify() fails when the stored best configuration leaves the ranks",
)
check(
    with_doc(lambda d: d["params"].__setitem__("tau", 0.25), "other parameters"),
    "verify() fails for outputs built with other parameters",
)
check(
    with_doc(lambda d: d["ref_fits"].pop(), "do not match the levels"),
    "verify() fails when a fit is missing",
)
check(
    with_arr(
        lambda a: a["ranks"].__setitem__((3, 0), np.where(a["ranks"][3, 0] == 1, 2, 1)),
        "does not follow from the ranks",
    ),
    "verify() fails when stored ranks change the scores",
)
# Swapping the two picks' ranks of a checked listener keeps every score,
# so only the dense reference can see it.
take_v = np.sort(
    np.random.default_rng(s13.SEED + 22).choice(
        len(W.SEL), size=min(s13b2.VERIFY_RANK_USERS, len(W.SEL)), replace=False
    )
)
fi = DOC["final"]["index"]
swap_j = next(
    int(j) for j in take_v if ARR["ranks"][fi, j, 0] != ARR["ranks"][fi, j, 1]
)


def swap_picks(a):
    a["ranks"][fi, swap_j] = a["ranks"][fi, swap_j][::-1].copy()


check(
    with_arr(swap_picks, "outside the reference range"),
    "verify() fails when the final model's ranks leave the dense "
    "reference while every score stays",
)
bpath = MODEL / "p13b_model.json"
good_b = bpath.read_text()


def with_part1(change, text):
    d = json.loads(good_b)
    change(d["final"]["path"][d["final"]["chosen"]])
    bpath.write_text(json.dumps(d))
    try:
        return raises(lambda: p13_loaders(s13b2.StageP13b2()).verify({}), text)
    finally:
        bpath.write_text(good_b)


def with_b(change, text):
    d = json.loads(good_b)
    change(d)
    bpath.write_text(json.dumps(d))
    try:
        return raises(lambda: p13_loaders(s13b2.StageP13b2()).verify({}), text)
    finally:
        bpath.write_text(good_b)


def twin_print(d):
    next(j for j in d["cells"] if j["cell"] == "m0w0t0p0")["fingerprint"] = "0" * 64


def chosen_print(d):
    d["final"]["path"][d["final"]["chosen"]]["fingerprint"] = "0" * 64


check(
    with_b(twin_print, "twin cell's layout differs")
    and with_b(chosen_print, "chosen cell's layout differs"),
    "verify() fails when part 1 fit the chosen or the twin cell on " "another layout",
)
check(
    with_part1(
        lambda j: j["theta"].__setitem__(2, j["theta"][2] + 1e-5),
        "differs from part 1's final fit",
    )
    and with_part1(
        lambda j: j["fit"].__setitem__("objective", j["fit"]["objective"] + 1e-6),
        "part 1's final model has objective",
    ),
    "verify() fails when part 1's final fit is not the fit at the "
    "smallest wide level, in its parameters or in its objective",
)
real_arm_block = s13b.arm_block


def shifted_arm_block(blk, arm):
    out = real_arm_block(blk, arm)
    if arm is not None and arm[0] == "p_pop" and "offset" in out:
        out = dict(out, offset=out["offset"] + 1e-9)
    return out


s13b.arm_block = shifted_arm_block
try:
    check(
        raises(
            lambda: p13_loaders(s13b2.StageP13b2()).verify({}), "an arm offset differs"
        ),
        "verify() fails when an arm's offsets leave the stored n_allowed "
        "and pop_allowed",
    )
finally:
    s13b.arm_block = real_arm_block
fpath = MODEL / "fusion_features_ref.npz"
good_f = load_npz(fpath)
np.savez(fpath, **dict(good_f, pairs_sha256=np.array("1" * 64)))
try:
    check(
        raises(
            lambda: p13_loaders(s13b2.StageP13b2()).verify({}),
            "was built from other pairs",
        ),
        "verify() fails when a features file belongs to other pairs",
    )
finally:
    np.savez(fpath, **good_f)
check(
    with_doc(
        lambda d: d["ref_fits"][1].__setitem__("m", 999),
        "the weight or the number of negatives",
    ),
    "verify() fails when a fit records another number of negatives",
)
check(
    with_arr(
        lambda a: a["sel_users"].__setitem__(0, a["sel_users"][0] + 1),
        "selection listeners",
    ),
    "verify() fails when the stored selection listeners differ",
)


def wrong_cols(blk, arm):
    out = real_arm_block(blk, arm)
    if arm is not None and arm[0] == "mix32":
        cols = s13b.neg_cols("mixture", blk["neg_rows"].shape[1] // 2)
        out = dict(out, neg_rows=blk["neg_rows"][:, cols])
    return out


def wrong_weight(blk, arm):
    out = real_arm_block(blk, arm)
    if arm is not None and arm[1] == "B":
        out = dict(out, pos_weight=blk["pos_weight"])
    return out


take_o = np.sort(
    np.random.default_rng(s13.SEED + 21).choice(
        len(PR["user_id"]),
        size=min(s13b2.VERIFY_OFFSET_USERS, len(PR["user_id"])),
        replace=False,
    )
)
n_out = sum(
    int(
        np.isin(
            PR["item"][PR["indptr"][j] : PR["indptr"][j + 1]][
                PR["kind"][PR["indptr"][j] : PR["indptr"][j + 1]] == 0
            ],
            OUT_CORE,
        ).sum()
    )
    for j in take_o
)
for patch, text, what in (
    (
        wrong_cols,
        "takes other negatives",
        "an arm takes other negatives than its draws",
    ),
    (
        wrong_weight,
        "wrong weight",
        "form B keeps the weight of a positive that its " "proposal cannot draw",
    ),
):
    s13b.arm_block = patch
    try:
        check(
            raises(lambda: p13_loaders(s13b2.StageP13b2()).verify({}), text)
            and (patch is wrong_cols or n_out > 0),
            f"verify() fails when {what}",
        )
    finally:
        s13b.arm_block = real_arm_block


def with_pairs(which, change, text):
    pp = MODEL / f"fusion_pairs_{which}.npz"
    pf = MODEL / f"fusion_features_{which}.npz"
    gp, gf = load_npz(pp), load_npz(pf)
    bad = {k: v.copy() for k, v in gp.items()}
    change(bad)
    np.savez(pp, **bad)
    np.savez(pf, **dict(gf, pairs_sha256=np.array(ev.npz_hash(pp))))
    try:
        return raises(lambda: p13_loaders(s13b2.StageP13b2()).verify({}), text)
    finally:
        np.savez(pp, **gp)
        np.savez(pf, **gf)


def swap_kinds(d):
    a = first_neg(d, 0)
    d["kind"][a], d["kind"][a + KR] = fp.KIND_POP, fp.KIND_UNIFORM


check(
    with_pairs(
        "ref",
        lambda d: d.update(
            params=np.array(
                json.dumps(dict(json.loads(str(d["params"])), ref_salt="other"))
            )
        ),
        "other P11b parameters",
    ),
    "verify() fails for pairs built with other P11b parameters",
)
check(
    with_pairs("ref", swap_kinds, "does not hold each listener's"),
    "verify() fails for pairs outside P11b's layout",
)
check(
    not raises(lambda: st_v.verify({}), "no such text") and st_v._verified is not None,
    "verify() passes again on the restored outputs",
)

# Summary
shutil.rmtree(TMP, ignore_errors=True)
print()
if failures:
    print(f"{len(failures)} check(s) FAILED:")
    for f in failures:
        print("  - " + f)
    raise SystemExit(1)
print("all checks passed")
