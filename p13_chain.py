"""The chain of P13 stages on the synthetic world of p13_world.py, as a
module for the tests of the scripts that read P13's outputs.

It runs, in a temporary model folder, what test_p13b2.py runs: P11, P12,
P13a and P13b (with part 1's real outcome, the monotone cell m1w0t0p0,
forced), then P11b, P12b and P13b part 2. Then it writes P9b's two lists
of the world into the folder and runs itemknn_select.run() on the
selection listeners. The settings equal test_p13b2.py's, so the base
outputs (P11 to P13b) share its cache.

Caches, both optional:
- P13B2_BASE_CACHE=<folder>: the outputs of P11 to P13b, with the key of
  test_p13b2.py.
- P13_CHAIN_CACHE=<folder>: the outputs of P11b to P13b part 2 and of
  ItemKNN, keyed by the base key and the code of those stages.
A cache is used only while its key equals the current one.
"""

import hashlib
import json
import logging
import os
import shutil
from pathlib import Path

import numpy as np

import p13_world as W
from pipeline import catalog as ct
from pipeline import concept_data as cd
from pipeline import concepts as cn
from pipeline import fusion_pairs as fp
from pipeline import gam
from pipeline import gam_arms
from pipeline import gam_audit as ga
from pipeline import itemknn as knn
from pipeline import stage_p11 as s11
from pipeline import stage_p11b as s11b
from pipeline import stage_p12 as s12
from pipeline import stage_p12b as s12b
from pipeline import stage_p13a as s13
from pipeline import stage_p13b as s13b
from pipeline import stage_p13b2 as s13b2

from pipeline import itemknn_select as iks

HERE = Path(__file__).resolve().parent
CHOSEN = (1, 0, 0, 0)  # part 1's real outcome, m1w0t0p0
KW, KR, K11 = 32, 128, 16
FULL, REF = (16, 24, 32), (16, 32, 128)

# P3's core without some of the positives' tracks, renormalised.
POS_ITEMS = np.unique(W.p_items)
OUT_CORE = POS_ITEMS[::12]
CORE = np.setdiff1d(W.core, OUT_CORE).astype(np.int64)
PP = W.p_pop_slot[CORE] / W.p_pop_slot[CORE].sum()
CUM = np.cumsum(PP)
PP_SLOT = np.zeros(W.NI)
PP_SLOT[CORE] = PP

U_ARRAYS = (W.u_indptr, W.u_indices, W.u_data)
LISTS = {
    k: (
        np.asarray(v[0], dtype=np.int64),
        np.asarray(v[1], np.int32),
        np.asarray(v[2], dtype=np.float64),
    )
    for k, v in W.SRC.colisten.items()
}
POP = np.asarray(W.listens, dtype=np.int64).copy()  # itemknn's "pop"
POP[0] = -1


def pop_centiles():
    """A popularity centile per slot (1 to 100, 0 for slot 0), from the
    world's listens: the stand-in for P4's item_covariates.parquet."""
    out = np.zeros(W.NI, dtype=np.int64)
    order = np.argsort(W.listens[1:], kind="stable")
    out[1 + order] = 1 + (np.arange(W.NI - 1) * 100) // (W.NI - 1)
    return out


# Stage harness


def configure(root, workers=1):
    """test_p13b2.configure(): small settings and paths into root."""
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
CHAIN_FILES = (
    "fusion_pairs_wide.npz",
    "fusion_pairs_ref.npz",
    "fusion_features_wide.npz",
    "fusion_features_ref.npz",
    "p13b2_model.json",
    "p13b2_arrays.npz",
    "itemknn_selection.json",
    "itemknn_selection_ranks.npz",
)


def base_key():
    """test_p13b2.base_key(): the code and the settings that the base
    outputs come from."""
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


def chain_key():
    """The base key, the code of P11b to P13b part 2 and of ItemKNN, and
    their settings."""
    h = hashlib.sha256(base_key().encode("utf-8"))
    for mod in (s11b, s12b, s13b2, ga, ct, knn, iks):
        h.update(Path(mod.__file__).read_bytes())
    h.update((HERE / "p13_chain.py").read_bytes())
    h.update(
        json.dumps(
            [KW, KR, FULL, REF, s11b._params(), s13b2._params()],
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    )
    return h.hexdigest()


def _from_cache(env, key, files, model):
    cache = os.environ.get(env)
    if not cache:
        return False
    cache = Path(cache)
    if (cache / "key.txt").exists() and (cache / "key.txt").read_text() == key:
        for name in files:
            shutil.copy(cache / name, model / name)
        print(f"      outputs from the cache {cache}")
        return True
    return False


def _to_cache(env, key, files, model):
    cache = os.environ.get(env)
    if not cache:
        return
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    for name in files:
        shutil.copy(model / name, cache / name)
    (cache / "key.txt").write_text(key)


def base_into(root):
    """The base outputs in root/model, from the cache when it is current."""
    configure(root)
    model = root / "model"
    if not _from_cache("P13B2_BASE_CACHE", base_key(), BASE_FILES, model):
        build_base(root)
        _to_cache("P13B2_BASE_CACHE", base_key(), BASE_FILES, model)
    configure(root)


def write_lists(model):
    """The world's two P9b lists in P9b's file layout."""
    for kind in knn.KINDS:
        ip, nb, cs = LISTS[kind]
        np.savez(
            model / knn.LIST_FILES[kind],
            indptr=ip.copy(),
            neighbor=nb.copy(),
            cosine=cs.copy(),
        )


def chain_into(root):
    """Every output that the scripts after P13b part 2 read, in
    root/model."""
    base_into(root)
    model = root / "model"
    write_lists(model)
    if not _from_cache("P13_CHAIN_CACHE", chain_key(), CHAIN_FILES, model):
        run_stage(p11_loaders(s11b.StageP11b()), report=False)
        run_stage(p12_loaders(s12b.StageP12b()), report=False)
        run_stage(p13_loaders(s13b2.StageP13b2()), report=False)
        iks.run(
            model,
            W.NI,
            U_ARRAYS,
            W.SEL.copy(),
            W.SEL_PICKS.copy(),
            POP,
            LISTS,
            workers=1,
            say=lambda msg: None,
            n_check=12,
            inputs={"marker:p6_als": None},
        )
        _to_cache("P13_CHAIN_CACHE", chain_key(), CHAIN_FILES, model)
    configure(root)
    return model
