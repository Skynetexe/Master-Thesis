"""Checks for pipeline/stage_p13a.py on a synthetic world. Every path
points into a temporary folder and the loaders are replaced, so nothing
touches the real model folder.

The world: a small catalog with every concept source, listeners with
histories, and held-out picks drawn from a planted additive preference
(artist match, tag match and popularity). The pairs come from P11's own
functions (fusion_pairs.user_rows) and the features from P12's own
function (concept_data.user_rows), so the stage reads data in the real
layout.

1. run() writes the model, the arrays and a report, verify() passes, and
   the report holds every section.
2. Every selection rank of every configuration equals a direct count
   under the P5 rule on the independent dense reference, censored at
   RANK_CUT + 1. Some listeners stop early, some score the whole catalog.
3. The fit recovers the planted directions: the artist match and the
   popularity shape rise, and the loss falls as the penalty weakens.
4. One worker and three workers give the same outputs.
5. Resume: a finished run does no work, a lost fit or selection block is
   computed again with the same result, and a parameter change starts
   again.
6. The neutral rule: a margin of 1 ties every missing state, a margin of
   0 ties none, and verify() passes in both cases.
7. verify() fails for each planted fault in the outputs.
8. The pool: an error in a worker ends it at once, and unordered() pairs
   each result with its task.

Usage, from the repository root:
    python3 test_p13a.py
"""

import copy
import json
import math
import multiprocessing as mp
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

from pipeline import catalog as ct
from pipeline import concept_data as cd
from pipeline import concepts as cn
from pipeline import fusion_pairs as fp
from pipeline import gam
from pipeline import gam_audit as ga
from pipeline import stage_p13a as s13
from pipeline.common import VerificationError

failures = []


def check(cond, msg):
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        failures.append(msg)


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


rng = np.random.default_rng(13013)
NI = 700
DA = len(cn.AUDIO_DIMS)

# The world


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
    "tags_direct": rows_csr(20, 0.5, 3),
    "tags_artist": rows_csr(20, 0.2, 5),
    "genres": rows_csr(8, 0.4, 2),
}
terms_src = {k: (*pack(v), 20 if k != "genres" else 8) for k, v in term_rows.items()}
credits, credit_ids = {}, {}
for name, n_ent, p_empty in (("artists", 70, 0.03), ("works", 40, 0.7)):
    ip, raw = pack(rows_csr(n_ent, p_empty, 2), np.int64)
    c = cd.compact(ip, raw)
    credits[name], credit_ids[name] = c[:3], c[3]
colisten = {}
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
    ip, nb = pack(rn)
    colisten[kind_] = (ip, nb, np.array([x for r in rc for x in r]))
audio = rng.normal(size=(NI, DA))
audio[0] = np.nan
audio[rng.choice(np.arange(1, NI), 120, replace=False)] = np.nan
listens = rng.integers(1, 3000, NI)
popularity = np.log1p(listens.astype(np.float64))
popularity[0] = 0.0
tier = rng.integers(0, 6, NI)
tier[np.isnan(audio).all(axis=1)] = cn.MATCH_TIERS.index("unmatched")
SRC = cd.Sources(
    NI, terms_src, credits, credit_ids, colisten, audio, popularity, tier, {}, {}
)
SPACE = cd.item_side(copy.deepcopy(SRC), transposed=True)

NU = 520
hist = [np.zeros(0, dtype=np.int32)]
for u in range(1, NU):
    h = rng.choice(np.arange(1, NI), size=int(rng.integers(4, 60)), replace=False)
    hist.append(np.sort(h).astype(np.int32))
u_indptr = np.concatenate(([0], np.cumsum([len(h) for h in hist])))
u_indices = np.concatenate(hist).astype(np.int32)
u_data = np.round(1000 * np.log1p(rng.integers(1, 300, size=len(u_indices)))).astype(
    np.uint16
)

# Held-out picks from a planted additive preference.
held = {}
for u in range(1, NU):
    items_h, w_h = cd.history_of(u_indptr, u_indices, u_data, u)
    st = cn.build_user_state(items_h, w_h, SPACE)
    full = cn.full_catalog(st, SPACE)
    art = np.nan_to_num(full["artist_affinity"][0])
    tag = np.nan_to_num(full["artist_tag_affinity"][0])
    score = 3.0 * (art > 0) + 2.0 * tag + 0.6 * popularity
    score[0] = -np.inf
    score[items_h] = -np.inf
    pr = np.exp(score - score.max())
    pr /= pr.sum()
    pk = rng.choice(NI, size=4, replace=False, p=pr)
    if u % 17 == 0:
        pk[1] = pk[2]  # val_last is the test_first pick (rule 3)
    if u % 23 == 0:
        pk[3] = pk[1]  # val_last and val_first coincide (rule 4)
    held[u] = pk
users_all = np.arange(1, NU)
SEL = np.sort(rng.choice(users_all, size=90, replace=False))
cols = [np.repeat(users_all, 4), np.concatenate([held[u] for u in users_all])]
flags = [np.tile(np.eye(4, dtype=bool)[j], len(users_all)) for j in range(4)]
p_users, picks4 = fp.pick_table(cols[0], cols[1], *flags)
p_indptr, p_items, p_flags, _ = fp.positives(picks4)
core = np.arange(1, NI)
p_pop = listens[core].astype(np.float64) ** 0.75
p_pop /= p_pop.sum()
cum = np.cumsum(p_pop)
p_pop_slot = np.zeros(NI)
p_pop_slot[core] = p_pop
samplers = {
    fp.KIND_UNIFORM: fp.uniform_sampler(1, NI),
    fp.KIND_POP: fp.pop_sampler(core, cum),
}
parts = {k: [] for k in ("item", "kind", "pick", "draw")}
keep_u, lens, n_allowed, pop_allowed = [], [], [], []
for j, u in enumerate(p_users):
    a, b = p_indptr[j], p_indptr[j + 1]
    if b == a:
        continue
    row = u_indices[u_indptr[u] : u_indptr[u + 1]]
    out = fp.user_rows(
        7,
        int(u),
        p_items[a:b],
        p_flags[a:b],
        row,
        picks4[j],
        16,
        samplers,
        NI - 1,
        p_pop_slot,
        float(p_pop.sum()),
    )
    for key, arr in zip(("item", "kind", "pick", "draw"), out[:4]):
        parts[key].append(arr)
    keep_u.append(int(u))
    lens.append(len(out[0]))
    n_allowed.append(out[4])
    pop_allowed.append(out[5])
keep_u = np.array(keep_u, dtype=np.int64)
P = {
    "user_id": keep_u,
    "split": np.where(np.isin(keep_u, SEL), fp.SPLIT_SELECTION, fp.SPLIT_TRAIN).astype(
        np.int8
    ),
    "indptr": np.concatenate(([0], np.cumsum(lens))).astype(np.int64),
    "item": np.concatenate(parts["item"]).astype(np.int32),
    "kind": np.concatenate(parts["kind"]),
    "n_allowed": np.array(n_allowed, dtype=np.int64),
    "pop_allowed": np.array(pop_allowed, dtype=np.float64),
}
P_PARAMS = {"version": "fusion-pairs-v1", "neg_per_kind": 16}
vals, hass, tiers = [], [], []
space_pairs = cd.item_side(copy.deepcopy(SRC), transposed=False)
for j, u in enumerate(keep_u):
    a, b = P["indptr"][j], P["indptr"][j + 1]
    items_h, w_h = cd.history_of(u_indptr, u_indices, u_data, int(u))
    v_, h_, t_ = cd.user_rows(
        space_pairs, items_h, w_h, P["item"][a:b].astype(np.int64)
    )
    vals.append(v_.astype(np.float32))
    hass.append(h_)
    tiers.append(t_.astype(np.int8))
F = {
    "values": np.concatenate(vals),
    "has": np.concatenate(hass),
    "match_tier": np.concatenate(tiers),
}
F_PARAMS = {"concepts": list(cn.CONTINUOUS)}
SEL_PICKS = np.array([[held[u][1], held[u][3]] for u in SEL], dtype=np.int64)
print(
    f"world: {NI} slots, {len(keep_u)} listeners with pairs "
    f"({int((P['split'] == 0).sum())} train), {len(P['item'])} rows, "
    f"{len(SEL)} selection listeners"
)

# Stage harness
TMP = Path(tempfile.mkdtemp(prefix="test_p13a_"))
REAL = {
    k: getattr(s13, k)
    for k in (
        "MODEL_DIR",
        "SCRATCH_DIR",
        "LAMBDAS",
        "N_BOOT",
        "FIT_BLOCKS",
        "FIT_WORKERS",
        "SELECTION_WORKERS",
        "TASKS_PER_WORKER",
        "VERIFY_USERS",
        "VERIFY_ROWS",
        "VERIFY_MAINSTREAM",
        "NEUTRAL_MARGIN",
        "SELECTION_BLOCK",
        "SELECTION_CHUNK",
        "PROBE_USERS",
    )
}


def configure(
    workers=1, lambdas=(1.0, 0.1, 0.01, 1e-3), boot=30, margin=0.02, sub="run"
):
    root = TMP / sub
    s13.MODEL_DIR = root / "model"
    s13.SCRATCH_DIR = root / "scratch"
    s13.LAMBDAS = tuple(lambdas)
    s13.N_BOOT = boot
    s13.FIT_BLOCKS = 5
    s13.FIT_WORKERS = workers
    s13.SELECTION_WORKERS = workers
    s13.TASKS_PER_WORKER = 2
    s13.VERIFY_USERS = 6
    s13.VERIFY_ROWS = 4000
    s13.VERIFY_MAINSTREAM = 10_000
    s13.NEUTRAL_MARGIN = margin
    s13.SELECTION_BLOCK = 7  # 13 blocks for the 90 listeners
    s13.SELECTION_CHUNK = 64  # 11 ranges, so listeners stop early
    s13.PROBE_USERS = 3
    return root


def make_stage():
    st = s13.StageP13a()
    st._pairs = lambda: (P, P_PARAMS, "pairs-sha")
    st._features = lambda: (F, F_PARAMS, "pairs-sha", "features-sha")
    st._proposals = lambda: (core.astype(np.int64), p_pop)
    st._id_slots = lambda: (NI, 1)
    st._csr = lambda: (u_indptr, u_indices, u_data)
    st._popularity = lambda n: popularity.copy()
    st._selection = lambda: (SEL.copy(), SEL_PICKS.copy())
    st._sources = lambda n: SRC
    st._space = lambda src: SPACE
    st._als_reference = lambda: {
        "popularity": 0.003,
        "als": 0.017,
        "als_label": "alpha 1, reg 100, 64 factors",
    }
    st._upstream_outputs = lambda: {"p11_fusion_pairs": {"pairs": "x"}}
    return st


def full_run(st):
    st.run({}, [])
    st.verify({})
    st.report({})
    return st


def model_doc(root):
    d = json.loads((root / "model" / "p13a_model.json").read_text())
    return d


def strip_seconds(obj):
    if isinstance(obj, dict):
        return {k: strip_seconds(v) for k, v in obj.items() if k != "seconds"}
    if isinstance(obj, list):
        return [strip_seconds(v) for v in obj]
    return obj


def arrays(root):
    z = np.load(root / "model" / "p13a_arrays.npz")
    return {k: z[k] for k in z.files}


# 1: first run
root1 = configure(workers=1)
st1 = full_run(make_stage())
doc = model_doc(root1)
arr = arrays(root1)
rep = (root1 / "model" / "reports" / "p13a_report.md").read_text()
check(
    st1._verified is not None and st1._verified["kkt"] <= s13.KKT_TOL,
    f"run() and verify() pass on the synthetic world (gaps: "
    f"{ {k: f'{v:.1e}' for k, v in st1._verified.items()} })",
)
check(
    all(
        h in rep
        for h in (
            "## 1. Knots",
            "## 2. Audit before the fit",
            "### 2a.",
            "### 2b.",
            "### 2c.",
            "## 3. Penalty path and selection",
            "## 4. The chosen model",
            "## 5. Explanation semantics",
        )
    )
    and "popularity scores 0.00300" in rep
    and ";" not in rep,
    "report() writes every section, the baseline line and no semicolon",
)
n_cfg = len(doc["path"])
for i, r in enumerate(doc["path"]):
    print(
        f"      fit {i}: lambda {r['lambda']:g}, {r['fit']['nit']} "
        f"iterations, {r['fit']['restarts']} restarts, projected gradient "
        f"{r['fit']['proj_grad_max']:.1e}, {r['fit']['message']}"
    )
print(
    f"      chosen {doc['choice']['chosen']}, cold fit "
    f"{doc['two_run']['cold_fit']['nit']} iterations, "
    f"{doc['two_run']['cold_fit']['message']}"
)
check(
    arr["ranks"].shape == (n_cfg, len(SEL), 2)
    and 0.0 < doc["two_run"]["max_table_diff"] <= s13.TWO_RUN_TOL,
    f"the arrays hold every configuration's ranks, and the two runs "
    f"agree within {doc['two_run']['max_table_diff']:.1e}",
)

# 2: every rank against brute
terms = [gam.Term.from_json(t) for t in doc["terms"]]
tabs = np.stack([np.asarray(r["tables"]) for r in doc["path"]])
mc = doc["mainstreaminess"]
all_ok, worst = True, 0.0
CUT1 = s13.RANK_CUT + 1
n_exact, n_beyond = 0, 0
pass_scorer = ct.CatalogScorer(terms, tabs, SPACE, chunk=s13.SELECTION_CHUNK)
pass_ok = True
for j, u in enumerate(SEL):
    items_h, w_h = cd.history_of(u_indptr, u_indices, u_data, int(u))
    st_ = cn.build_user_state(items_h, w_h, SPACE)
    m_u = float(
        gam.mainstreaminess(gam.mainstreaminess_raw(items_h, w_h, popularity), mc)
    )
    ref = ct.reference_scores(cn.full_catalog(st_, SPACE), terms, tabs, m_u, SPACE.tier)
    r_pass, sc_pass = pass_scorer.ranks_cut(
        st_, SPACE, m_u, SEL_PICKS[j], items_h, s13.RANK_CUT
    )
    pass_ok &= np.array_equal(r_pass, arr["ranks"][:, j, :]) and int(sc_pass) == int(
        arr["scored"][j]
    )
    for c in range(n_cfg):
        s_ = ref[c].copy()
        s_[0] = -np.inf
        s_[items_h] = -np.inf
        for p_ in range(2):
            t_ = int(SEL_PICKS[j, p_])
            lo, hi = ct.reference_rank(ref[c], t_, items_h, 1e-9)
            got = int(arr["ranks"][c, j, p_])
            all_ok &= min(lo, CUT1) <= got <= min(hi, CUT1)
            if lo == hi:
                brute = (
                    1
                    + int(np.count_nonzero(s_ > s_[t_]))
                    + int(np.count_nonzero(s_[:t_] == s_[t_]))
                )
                all_ok &= got == min(brute, CUT1)
                n_exact += int(brute <= s13.RANK_CUT)
                n_beyond += int(brute > s13.RANK_CUT)
check(
    pass_ok,
    "every selection listener's stored ranks and slot count "
    "equal a new run of ranks_cut() outside the stage",
)
check(
    all_ok and n_exact > 0 and n_beyond > 0,
    f"every stored rank of every configuration and selection listener "
    f"equals the P5 count on the dense reference, censored at "
    f"{CUT1} ({n_exact} exact ranks, {n_beyond} beyond the cut)",
)
sc_ = arr["scored"]
check(
    sc_.shape == (len(SEL),)
    and (sc_ < NI).any()
    and (sc_ == NI).any()
    and 0.0 < doc["selection_pass"]["full_pass_share"] < 1.0,
    f"some selection listeners stop early and some need the whole "
    f"catalog ({int((sc_ == NI).sum())} of {len(SEL)} whole passes)",
)
check(
    "Ranks are exact up to 100 and stored as 101 beyond" in rep,
    "the report states the rank cut",
)

# 2b: independent recomputations
# The stage's plumbing, recomputed here without the stage's code: the
# training rows and offsets, the knots, the loss of every configuration
# from the raw rows, one audit AUC by pair counting, and the choice.
tr_mask = P["split"] == fp.SPLIT_TRAIN
tr_users = P["user_id"][tr_mask]
tr_idx = np.flatnonzero(tr_mask)
rows_i, grp = [], []
for j in tr_idx:
    rows_i.append(np.arange(P["indptr"][j], P["indptr"][j + 1]))
rows_i = np.concatenate(rows_i)
kind_i = P["kind"][rows_i]
item_i = P["item"][rows_i].astype(np.int64)
user_of = np.repeat(np.arange(len(tr_idx)), np.diff(P["indptr"])[tr_idx])
q = (
    0.5 / P["n_allowed"][tr_idx][user_of]
    + 0.5 * p_pop_slot[item_i] / P["pop_allowed"][tr_idx][user_of]
)
off_want = np.where(kind_i == fp.KIND_POSITIVE, 0.0, -np.log(32 * q))
st_chk = make_stage()
inp_chk = st_chk._inputs()
check(
    np.array_equal(inp_chk["rows"], rows_i)
    and np.allclose(inp_chk["offset"], off_want, rtol=0, atol=1e-12),
    "the training rows and the form-A offsets -log(32 q_mix) equal a "
    "direct computation from the pairs and the proposals",
)


def knots_direct(v, pos_mask):
    parts = [
        np.array([0.0]),
        np.quantile(v[pos_mask], s13.KNOT_POS_LEVELS),
        np.quantile(v[~pos_mask], s13.KNOT_NEG_LEVELS),
    ]
    k = np.unique(np.concatenate(parts))
    gap = 1e-6 * (k[-1] - k[0])
    out = [k[0]]
    for x_ in k[1:]:
        if x_ - out[-1] > gap:
            out.append(x_)
    return out


knots_ok = True
pos_i = kind_i == fp.KIND_POSITIVE
for c in list(s13.BASE_CONCEPTS) + ["popularity"]:
    j = cn.CONTINUOUS.index(c)
    v_ = F["values"][rows_i, j].astype(np.float64)
    h_ = F["has"][rows_i, j]
    m_ = h_ & ((v_ != 0) if c in s13.ZERO_STATE else True)
    knots_ok &= knots_direct(v_[m_], pos_i[m_]) == doc["knots"][c]
check(
    knots_ok,
    "the stored knots equal quantiles of the positives' and "
    "the negatives' shape values, merged with 0",
)

m_raw_u = np.array(
    [
        math.fsum(
            float(wv) * float(popularity[it])
            for it, wv in zip(*cd.history_of(u_indptr, u_indices, u_data, int(u)))
        )
        / math.fsum(
            float(wv) for wv in cd.history_of(u_indptr, u_indices, u_data, int(u))[1]
        )
        for u in tr_users
    ]
)
z_u = (m_raw_u - m_raw_u.mean()) / m_raw_u.std()
m_u_i = np.clip(z_u, np.quantile(z_u, 0.005), np.quantile(z_u, 0.995))
n_pos_u = np.bincount(user_of[pos_i], minlength=len(tr_idx))


offs = np.cumsum([0] + [t.size for t in terms])[:-1]


def brute_loss(tables):
    s_ = np.zeros(len(rows_i))
    for t, off in zip(terms, offs):
        j = cn.CONTINUOUS.index(t.source)
        v_ = F["values"][rows_i, j].astype(np.float64)
        h_ = F["has"][rows_i, j]
        tab = tables[off : off + t.size]
        kn = t.knot_array
        val = np.interp(np.clip(np.where(h_, v_, kn[0]), kn[0], kn[-1]), kn, tab[: t.K])
        if t.zero_state:
            val = np.where(h_ & (v_ == 0), tab[t.K + gam.ZERO], val)
        val = np.where(h_, val, tab[t.K + gam.MISS] if t.missing_state else 0.0)
        if t.tiers is not None:
            val = np.where(np.isin(F["match_tier"][rows_i], t.tiers), val, 0.0)
        if t.multiplier is not None:
            val = m_u_i[user_of] * val
        s_ += val
    z_ = s_ + off_want
    tot = []
    starts_u = np.concatenate(([0], np.cumsum(np.diff(P["indptr"])[tr_idx])))
    for u in range(len(tr_idx)):
        zz = z_[starts_u[u] : starts_u[u + 1]]
        kk = kind_i[starts_u[u] : starts_u[u + 1]]
        neg = zz[kk != fp.KIND_POSITIVE]
        for zp in zz[kk == fp.KIND_POSITIVE]:
            grp_ = np.concatenate(([zp], neg))
            mx = grp_.max()
            tot.append((mx + math.log(math.fsum(np.exp(grp_ - mx))) - zp) / n_pos_u[u])
    return math.fsum(tot) / len(tr_idx)


worst_loss = max(
    abs(brute_loss(np.asarray(r["tables"])) - r["fit"]["loss"])
    / max(1.0, abs(r["fit"]["loss"]))
    for r in doc["path"]
)
check(
    worst_loss < 1e-10,
    f"every configuration's stored loss equals a brute-force recomputation "
    f"from the raw rows, the mainstreaminess and the listener weights "
    f"(largest relative gap {worst_loss:.1e})",
)

j_tag = cn.CONTINUOUS.index("genre_affinity")
j_pop = cn.CONTINUOUS.index("popularity")
pop_i = F["values"][rows_i, j_pop].astype(np.float64)
cuts = np.quantile(pop_i[pos_i], np.arange(1, 10) / 10)
st_i = np.searchsorted(cuts, pop_i, side="right")
flag_i = F["has"][rows_i, j_tag]
num = den = 0.0
for s_ in range(10):
    fpos = flag_i[pos_i & (st_i == s_)].astype(float)
    fneg = flag_i[(kind_i == fp.KIND_UNIFORM) & (st_i == s_)].astype(float)
    for a_ in (0.0, 1.0):
        na = float((fpos == a_).sum())
        num += na * (float((fneg < a_).sum()) + 0.5 * float((fneg == a_).sum()))
    den += len(fpos) * len(fneg)
auc_direct = num / den
auc_stored = doc["audit"]["missing"]["concepts"]["genre_affinity"]["by_kind"][
    "uniform"
]["auc"]
check(
    abs(auc_direct - auc_stored) < 1e-12,
    "the stored within-decile AUC of a data flag equals a pair count over "
    "the positive-negative pairs of each popularity decile",
)

sc_l = []
for c in range(n_cfg):
    r_ = arr["ranks"][c].astype(np.float64)
    sc_l.append(np.where(r_ <= 100, 1 / np.log2(r_ + 1), 0).mean(axis=1))
sc_l = np.array(sc_l)
best_c = 0
for c in range(n_cfg):
    if sc_l[c].mean() > sc_l[best_c].mean():
        best_c = c
chosen_c = next(
    c
    for c in range(n_cfg)
    if c == best_c
    or (sc_l[best_c] - sc_l[c]).mean()
    <= (sc_l[best_c] - sc_l[c]).std(ddof=1) / math.sqrt(len(SEL))
)
check(
    chosen_c == doc["choice"]["chosen"] and best_c == doc["choice"]["best"],
    "the choice equals the one-standard-error rule with the strongest "
    "penalty first, recomputed from the ranks",
)

# 3: planted directions
ch = doc["choice"]["chosen"]
T_ch = np.asarray(doc["path"][ch]["tables"])
offs = np.cumsum([0] + [t.size for t in terms])[:-1]


def term_table(name):
    k = [t.name for t in terms].index(name)
    return terms[k], T_ch[offs[k] : offs[k] + terms[k].size]


ta, tab_a = term_table("artist_affinity")
tp, tab_p = term_table("popularity")
qa = doc["shape_quantiles"]["artist_affinity"]
qp = doc["shape_quantiles"]["popularity"]
art_hi = np.interp(qa[2], ta.knot_array, tab_a[: ta.K])
pop_lo = np.interp(qp[0], tp.knot_array, tab_p[: tp.K])
pop_hi = np.interp(qp[-1], tp.knot_array, tab_p[: tp.K])
check(
    art_hi > tab_a[ta.K + gam.ZERO] + 0.5 and pop_hi > pop_lo + 0.5,
    f"the fit recovers the planted directions: an artist match beats "
    f'"no match" ({art_hi:+.2f} against {tab_a[ta.K]:+.2f}) and '
    f"popular tracks beat rare ones ({pop_hi:+.2f} against {pop_lo:+.2f})",
)
losses = [r["fit"]["loss"] for r in doc["path"]]
check(
    all(b <= a + 1e-12 for a, b in zip(losses, losses[1:])),
    "the training loss falls as the penalty weakens",
)
conc = doc["audit"]["concurvity"]
check(
    all(
        0.0 <= v["worst"] <= 1.0 and 0.0 <= v["estimate"] <= v["worst"] + 1e-12
        for v in conc["full"].values()
    )
    and all(0.0 <= v <= 1.0 for v in conc["observed"].values()),
    "the concurvity indices lie in [0, 1] and estimate <= worst",
)
imp = doc["after_fit"]["importance"]
check(
    abs(sum(v["cov_share"] for v in imp.values()) - 1.0) < 1e-9,
    "the covariance shares add to 1",
)

# 4: worker invariance
root3 = configure(workers=3, sub="run3")
full_run(make_stage())
doc3 = model_doc(root3)
arr3 = arrays(root3)
check(
    strip_seconds(doc3) == strip_seconds(doc)
    and all(np.array_equal(arr[k], arr3[k]) for k in arr),
    "one worker and three workers give the same model and arrays",
)

# 5: resume
fit_calls = {"n": 0}
real_fit = gam.fit


def counting_fit(*a, **k):
    fit_calls["n"] += 1
    return real_fit(*a, **k)


gam.fit = counting_fit
try:
    configure(workers=1)
    st_again = make_stage()
    st_again.run({}, [])
    check(
        fit_calls["n"] == 0 and strip_seconds(model_doc(root1)) == strip_seconds(doc),
        "a finished run resumes without a fit and writes the same model",
    )
    fits_path = root1 / "scratch" / "p13a" / "fits.json"
    state = json.loads(fits_path.read_text())
    state["fits"] = state["fits"][:-1]
    fits_path.write_text(json.dumps(state))
    parts = sorted((root1 / "scratch" / "p13a" / "selection").glob("part_*.npz"))
    parts[3].unlink()  # a block in the middle
    fit_calls["n"] = 0
    make_stage().run({}, [])
    check(
        fit_calls["n"] == 1
        and strip_seconds(model_doc(root1)) == strip_seconds(doc)
        and all(np.array_equal(arrays(root1)[k], arr[k]) for k in arr),
        "a lost fit and a lost selection block are computed again, with "
        "the same result",
    )
    fit_calls["n"] = 0
    configure(workers=1, boot=31)
    make_stage().run({}, [])
    check(
        fit_calls["n"] == len(doc["path"]) + 1,
        "a parameter change starts the fits again",
    )
finally:
    gam.fit = real_fit

# 6: neutral rule
for margin, want_all in ((1.0, True), (0.0, False)):
    rootm = configure(
        workers=1, margin=margin, lambdas=(0.1, 0.01), sub=f"margin{margin}"
    )
    stm = full_run(make_stage())
    dm = model_doc(rootm)
    got = dm["layout"]["neutral"]
    check(
        (got == list(s13.BASE_CONCEPTS)) if want_all else (got == []),
        f"margin {margin}: the neutral missing states are "
        f"{'all concepts' if want_all else 'none'}, and verify() passes",
    )

# 7: planted faults
root_f = configure(workers=1, sub="run")  # the first run's folder
mpath = root_f / "model" / "p13a_model.json"
apath = root_f / "model" / "p13a_arrays.npz"
configure(workers=1)
full_run(make_stage())  # restore after section 5
good_doc = mpath.read_text()
good_arr = arrays(root_f)


def with_doc(edit, text, label):
    d = json.loads(good_doc)
    edit(d)
    mpath.write_text(json.dumps(d))
    try:
        check(
            raises(lambda: make_stage().verify({}), text), f"verify() catches {label}"
        )
    finally:
        mpath.write_text(good_doc)


def with_arr(edit, text, label):
    a = {k: v.copy() for k, v in good_arr.items()}
    edit(a)
    np.savez(apath, **a)
    try:
        check(
            raises(lambda: make_stage().verify({}), text), f"verify() catches {label}"
        )
    finally:
        np.savez(apath, **good_arr)


def bump_raw(d):
    d["path"][0]["tables_raw"][5] += 1e-3


def bump_theta(d):
    d["path"][1]["theta"][3] += 1e-3


def bump_center(d):
    d["path"][0]["tables"][4] += 1e-3


def bump_knot(d):
    k = d["knots"]["tag_affinity"]
    k[3] = (k[2] + k[3]) / 2


def flip_neutral(d):
    d["layout"]["neutral"] = ["tag_affinity"]


def bump_auc(d):
    d["audit"]["missing"]["concepts"]["genre_affinity"]["by_kind"]["uniform"][
        "auc"
    ] += 1e-6


def flip_signal(d):
    e = d["audit"]["missing"]["concepts"]["work_affinity"]
    e["no_signal"] = not e["no_signal"]


def bad_two_run(d):
    d["two_run"]["max_table_diff"] = 1.0


def bump_consts(d):
    d["mainstreaminess"]["mean"] += 1e-6


def move_choice(d):
    d["choice"]["chosen"] = (d["choice"]["chosen"] + 1) % len(d["path"])


def bump_objective(d):
    d["path"][2]["fit"]["objective"] += 1e-6


with_doc(bump_raw, "raw tables differ", "a changed raw table")
with_doc(bump_theta, "configuration 1", "changed parameters")
with_doc(bump_center, "centered table", "a centered table off its mean")
with_doc(bump_knot, "stored knots differ", "a moved knot")
with_doc(flip_neutral, "neutral missing states differ", "a changed neutral list")
with_doc(bump_auc, "audit AUC differs", "a changed audit AUC")
with_doc(
    flip_signal,
    "neutral decision does not follow",
    "a decision against the neutral rule",
)
with_doc(bad_two_run, "two runs differ", "a failed two-run check")
with_doc(bump_consts, "mainstreaminess constants", "changed mainstreaminess constants")
with_doc(move_choice, "choice does not follow", "a moved choice")
with_doc(bump_objective, "objective differs", "a changed objective")


def bump_m(a):
    a["m_raw"][7] += 1e-6


def ranks_to_one(a):
    a["ranks"][0] = 1


def bump_rank(a):
    """A rank of the chosen configuration, for a listener that verify()
    samples, moved by one where the move keeps the choice: only the
    reference ranks can catch the change."""
    d = json.loads(good_doc)
    c = d["choice"]["chosen"]
    n = len(a["sel_users"])
    take = np.random.default_rng(s13.SEED + 2).choice(
        n, size=min(s13.VERIFY_USERS, n), replace=False
    )
    order = list(range(len(d["path"])))
    for j in take:
        for p_ in range(2):
            r = int(a["ranks"][c, j, p_])
            for new in (r + 1, r - 1):
                if not 1 <= new <= s13.RANK_CUT + 1:
                    continue
                trial = a["ranks"].copy()
                trial[c, j, p_] = new
                ch2, b2, _ = ga.one_se_choice(ga.per_listener_scores(trial), order)
                if ch2 == c and b2 == d["choice"]["best"]:
                    a["ranks"][c, j, p_] = new
                    return
    raise AssertionError("no rank move keeps the choice")


def bad_scored(a):
    a["scored"][3] = 0


def swap_scored(a):
    """Two listeners' slot counts swapped: each count stays in range."""
    d = json.loads(good_doc)
    n = len(a["sel_users"])
    take = np.random.default_rng(s13.SEED + 2).choice(
        n, size=min(s13.VERIFY_USERS, n), replace=False
    )
    sc = a["scored"]
    j = next(j for j in take if sc[j] != sc[take[0]])
    sc[take[0]], sc[j] = sc[j], sc[take[0]]


def rank_beyond(a):
    a["ranks"][1, 5, 1] = s13.RANK_CUT + 2


with_arr(bump_m, "mainstreaminess", "a changed raw mainstreaminess")
with_arr(ranks_to_one, "choice does not follow", "ranks that change the choice")
with_arr(
    bump_rank, "outside the reference range", "ranks that keep the choice but are wrong"
)
with_arr(bad_scored, "slot counts", "a slot count of 0")
with_arr(swap_scored, "another slot count", "two listeners' slot counts swapped")
with_arr(rank_beyond, "outside 1 to", "a rank beyond the cut plus one")

# 8: the pool's error path


def _boom(t):
    if t == 2:
        raise RuntimeError("planted failure")
    time.sleep(30.0 if t == 0 else 0.2)
    return t


t0 = time.time()
err = None
try:
    with s13._Pool(3) as pool_:
        for _ in pool_.unordered(_boom, list(range(8))):
            pass
except RuntimeError as exc:
    err = str(exc)
elapsed = time.time() - t0
# A worker that the pool ended can need a moment to exit. Wait up to 5 s
# for the workers before the check.
deadline = time.time() + 5.0
while mp.active_children() and time.time() < deadline:
    time.sleep(0.05)
check(
    err == "planted failure" and elapsed < 20 and not mp.active_children(),
    f"an error in a worker ends the pool at once, without waiting for a "
    f"running 30 s task ({elapsed:.1f} s, no worker left)",
)
order_seen = []
with s13._Pool(3) as pool_:
    for i_, r_ in pool_.unordered(_boom, [5, 1, 3]):
        order_seen.append((i_, r_))
check(
    sorted(order_seen) == [(0, 5), (1, 1), (2, 3)],
    "unordered() pairs every result with its task position",
)

for k, v in REAL.items():
    setattr(s13, k, v)
shutil.rmtree(TMP, ignore_errors=True)
print()
print(f"{len(failures)} failure(s)")
raise SystemExit(1 if failures else 0)
