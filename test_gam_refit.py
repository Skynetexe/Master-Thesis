"""Checks for gam_refit.py on the P13 chain of p13_chain.py (P11 to P13b
part 2 and ItemKNN on the synthetic world of p13_world.py) and stage 1 of
gam_knn_diag.py. Every path points into a temporary folder, and nothing
touches the real model folder.

1. The helpers: the configurations, the layouts against P13b part 2's
   stored layout, the warm start, the decision rules on constructed
   scores, the log filter and the Newton step lines.
2. A run on the chain. Recomputed here without the script's code: every
   fit's loss from the raw rows of its level and its gradient through a
   design built with np.interp (the KKT conditions), the centering on the
   level's p_pop negatives, the explanation points, every rank of every
   configuration by a dense count from the stored terms and tables, and
   every result and decision from the ranks.
3. A forced choice of a smaller configuration, to check the outputs and
   the report on that path.
4. Resume (a finished run, a lost refit, lost background means and a lost
   pass block), and three workers against one.
5. Refusals of inconsistent inputs, planted faults in the outputs, a
   planted scorer fault, saved fits that do not fit, and a rank changed on
   its way to the disk.

Usage, from the repository root:
    python3 test_gam_refit.py
With P13B2_BASE_CACHE=<folder> and P13_CHAIN_CACHE=<folder> the chain's
outputs are kept and reused while their code stays the same.
"""

import json
import logging
import math
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

import p13_chain as C
import p13_world as W
from pipeline import als as als_mod
from pipeline import catalog as ct
from pipeline import concept_data as cd
from pipeline import concepts as cn
from pipeline import fusion_pairs as fp
from pipeline import gam
from pipeline import gam_arms
from pipeline import stage_p13a as s13
from pipeline import stage_p13b2 as s13b2

from pipeline import gam_knn_diag as G
from pipeline import gam_refit as R

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


class Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


def close(a, b, tol=1e-12):
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= tol * max(1.0, abs(b))


print(W.summary())
TMP = Path(tempfile.mkdtemp(prefix="test_gam_refit_"))
CHAIN = C.chain_into(TMP / "chain")
G.run_slices(
    CHAIN, W.NI, C.U_ARRAYS, C.LISTS, C.pop_centiles(), workers=1, say=quiet, n_check=5
)
NI = W.NI
USERS = np.asarray(W.SEL, dtype=np.int64)
PICKS = np.asarray(W.SEL_PICKS, dtype=np.int64)
DOC2 = json.loads((CHAIN / "p13b2_model.json").read_text())
ARR2 = load_npz(CHAIN / "p13b2_arrays.npz")
A_DOC = json.loads((CHAIN / "p13a_model.json").read_text())
FIN = int(DOC2["final"]["index"])
FIT = DOC2["wide_fits"][FIN]
G_RANK = ARR2["ranks"][FIN]
NAMES = [t["name"] for t in FIT["terms"]]
LEVEL, LAM = int(FIT["level"]), float(FIT["lambda"])
ARM = tuple(FIT["arm"].split("-"))
assert ARM == ("p_pop", "B") and FIT["cell"] == "m0w0t0p0", FIT["cell"]
BASE = C.p13_loaders(s13b2.StageP13b2())._base_context()
SPECS = R.config_specs(NAMES)
SL = load_npz(CHAIN / "gam_knn_slices.npz")
KNN = load_npz(CHAIN / "itemknn_selection_ranks.npz")
OTHER, POP = "colisten_other_artists", "popularity"
print(
    f"      final model {FIT['cell']} {FIT['arm']} level {LEVEL}, lambda "
    f"{LAM}, {len(NAMES)} terms, {len(USERS)} selection listeners"
)


def fresh(name):
    """A copy of the chain's folder, with the paths of the stages pointed
    at it."""
    root = TMP / name
    if root.exists():
        shutil.rmtree(root)
    shutil.copytree(CHAIN.parent, root)
    C.configure(root)
    return root / "model"


def run_refit(model, workers=1, n_verify=90, st=None):
    C.configure(model.parent, workers)
    cap = Capture()
    R.log.addHandler(cap)
    R.log.setLevel(logging.INFO)
    try:
        st = st or C.p13_loaders(s13b2.StageP13b2())
        R.run(
            st,
            model,
            model.parent / "scratch" / R.WORK_NAME,
            say=R.log.info,
            n_verify=n_verify,
            inputs={"x": "y"},
        )
    finally:
        R.log.removeHandler(cap)
    return cap.lines


# 1: helpers
want_names = [
    "full",
    f"without {OTHER}",
    f"without {POP}",
    "without both",
    "artist terms only",
]
check(
    [s[0] for s in SPECS] == want_names
    and [s[2] for s in SPECS]
    == ["incumbent", "candidate", "candidate", "candidate", "reference"]
    and SPECS[0][1] == tuple(NAMES)
    and SPECS[1][1] == tuple(n for n in NAMES if n != OTHER)
    and SPECS[2][1] == tuple(n for n in NAMES if n != POP)
    and SPECS[3][1] == tuple(n for n in NAMES if n not in (OTHER, POP))
    and SPECS[4][1] == ("artist_affinity", "colisten_same_artist")
    and [len(s[1]) for s in SPECS] == [8, 7, 7, 6, 2],
    "config_specs(): five configurations in the declared order, roles and " "terms",
)
ok = all(
    raises(
        lambda t=t: R.config_specs([n for n in NAMES if n != t]),
        f"lacks the terms ['{t}']",
    )
    for t in (OTHER, POP, "artist_affinity", "colisten_same_artist")
)
check(
    ok and raises(lambda: R.config_specs(NAMES + NAMES[:1]), "names a term twice"),
    "config_specs() refuses a model without one of the four terms it "
    "needs, or with a term twice",
)

LAY_F, IDX_F = R.layout_of(BASE, NAMES)
THETA_F = np.asarray(FIT["theta"], dtype=np.float64)
check(
    gam_arms.layout_fingerprint(LAY_F) == FIT["fingerprint"]
    and [t.to_json() for t in LAY_F.terms] == FIT["terms"]
    and np.array_equal(IDX_F, np.arange(BASE["L_u"]))
    and np.array_equal(LAY_F.tables(THETA_F), np.asarray(FIT["tables_raw"])),
    "layout_of(): the full layout has P13b part 2's terms and fingerprint, "
    "covers the union and gives the stored raw tables",
)
ok, ok_start, worst_start = True, True, 0.0
pos_f = {lab: k for k, lab in enumerate(LAY_F.labels)}
T_F = LAY_F.tables(THETA_F)
for name, kept, _ in SPECS[1:]:
    lay, idx = R.layout_of(BASE, kept)
    want_idx = np.concatenate(
        [
            BASE["union_off"][n]
            + np.arange(
                len(next(t for t in FIT["terms"] if t["name"] == n)["knots"]) + 4
            )
            for n in kept
        ]
    )
    cols = [pos_f.get(lab, -1) for lab in lay.labels]
    ok &= (
        np.array_equal(idx, want_idx)
        and min(cols) >= 0
        and np.array_equal(lay.M, LAY_F.M[np.ix_(idx, cols)])
        and not np.any(
            LAY_F.M[np.ix_(idx, [k for k in range(LAY_F.P) if k not in cols])]
        )
        and np.allclose(
            lay.Q,
            LAY_F.Q[np.ix_(cols, cols)],
            rtol=0,
            atol=1e-12 * float(np.max(np.abs(LAY_F.Q))),
        )
        and [t.name for t in lay.terms] == list(kept)
    )
    start = R.warm_start(LAY_F, THETA_F, lay)
    ok_start &= np.array_equal(start, THETA_F[cols])
    worst_start = max(worst_start, float(np.max(np.abs(lay.tables(start) - T_F[idx]))))
check(
    ok,
    "layout_of(): every smaller layout is the full layout's block of "
    "its terms (the map M and the penalty Q), at the union positions "
    "of its terms",
)
check(
    ok_start and worst_start < 1e-12,
    f"warm_start(): the final model's parameters of the kept terms, which "
    f"give the final model's tables of those terms (largest gap "
    f"{worst_start:.1e})",
)
lay_z, _ = R.layout_of(dict(BASE, n_zero=dict(BASE["n_zero"], genre_affinity=0)), NAMES)
lay_m, _ = R.layout_of(dict(BASE, n_miss=dict(BASE["n_miss"], genre_affinity=0)), NAMES)
check(
    "genre_affinity:zero" in LAY_F.labels
    and "genre_affinity:miss" in LAY_F.labels
    and "genre_affinity:zero" not in lay_z.labels
    and "genre_affinity:miss" in lay_z.labels
    and "genre_affinity:zero" in lay_m.labels
    and "genre_affinity:miss" not in lay_m.labels,
    "layout_of() reads the counts of zero and missing rows of each term "
    "from the base context: without zero rows the zero state is tied, "
    "without missing rows the missing state",
)
odd = gam.Layout([gam.Term("x", "audio_fit", (0.0, 1.0, 2.0))])
check(
    raises(lambda: R.layout_of(BASE, NAMES[::-1]), "not in its order")
    and raises(lambda: R.layout_of(BASE, NAMES[:3] + ["nope"]), "not all in the union")
    and raises(
        lambda: R.warm_start(LAY_F, THETA_F, odd), "have no value in the final model"
    ),
    "layout_of() refuses terms out of order or outside the union, "
    "warm_start() a parameter the final model lacks",
)
ids = [R.fit_id(n, LEVEL, ARM, LAM) for n, _, _ in SPECS] + [
    R.fit_id(R.FULL_CHECK, LEVEL, ARM, LAM)
]
check(
    len(set(ids)) == 6
    and ids[1] == f"without {OTHER}|wide|{LEVEL}|" f"p_pop-B|{LAM!r}",
    "fit_id(): one id per configuration, with the level, arm and weight",
)

# The rules on constructed scores. Every configuration shares one
# standardized noise vector, so the standard error of a paired
# difference is |sigma_a - sigma_b| / sqrt(n).
N_C = 400
rng = np.random.default_rng(3)
Z0 = rng.normal(size=N_C)
Z0 = (Z0 - Z0.mean()) / Z0.std(ddof=1)
ZS = rng.normal(size=N_C)
ZS = (ZS - ZS.mean()) / ZS.std(ddof=1)
S0 = 0.1 + 0.05 * Z0


def scores_case(mus, sigmas):
    out = [S0]
    for mu, sg in zip(mus, sigmas):
        out.append(S0 + mu + sg * ZS)
    return np.stack(out)


def decide_here(S, sesoi=0.02):
    """R1 to R3 written here, with loops."""
    n = S.shape[1]
    mean = [float(np.mean(x)) for x in S]
    thr = sesoi * mean[0]
    names = [s[0] for s in SPECS]
    terms = [len(s[1]) for s in SPECS]

    def se(a, b):
        d = S[a] - S[b]
        return math.sqrt(sum((x - d.mean()) ** 2 for x in d) / (n - 1) / n)

    elig = []
    for c in (1, 2, 3):
        dm = float(np.mean(S[c] - S[0]))
        if dm > thr and dm - 1.959963984540054 * se(c, 0) > 0:
            elig.append(c)
    order = [0] + elig
    order.sort(key=lambda c: (-terms[c], -mean[c], c))
    best = order[0]
    for c in order:
        if mean[c] > mean[best]:
            best = c
    chosen = next(
        c for c in order if c == best or float(np.mean(S[best] - S[c])) <= se(best, c)
    )
    dr = float(np.mean(S[4] - S[chosen]))
    flag = dr > thr and dr - 1.959963984540054 * se(4, chosen) > 0
    return {
        "eligible": [names[c] for c in elig],
        "order": [names[c] for c in order],
        "best": names[best],
        "chosen": names[chosen],
        "flag": flag,
    }


CASES = {
    "none above SESOI": ((0.001, 0.0015, 0.0019, 0.0), (1e-4,) * 4, None),
    "above SESOI, interval reaches 0": (
        (0.003, 0.0, 0.0, 0.0),
        (0.04, 1e-4, 1e-4, 1e-4),
        None,
    ),
    "one eligible": ((0.0, 0.004, 0.0, 0.0), (1e-4,) * 4, "without " "popularity"),
    "7 terms within one SE of the best": (
        (0.004, 0.0, 0.0045, 0.0),
        (0.0, 0.0, 0.02, 0.0),
        f"without {OTHER}",
    ),
    "7 terms beyond one SE": (
        (0.004, 0.0, 0.0045, 0.0),
        (0.0, 0.0, 0.004, 0.0),
        "without both",
    ),
    "two 7-term cells within one SE": (
        (0.004, 0.0042, 0.006, 0.0),
        (0.0, 0.0, 0.04, 0.0),
        "without popularity",
    ),
    "a tie in terms and score": (
        (0.004, 0.004, 0.0, 0.0),
        (0.0,) * 4,
        f"without {OTHER}",
    ),
    "reference flagged": ((0.0, 0.0, 0.0, 0.01), (1e-4,) * 4, None),
    "reference above SESOI, interval reaches 0": (
        (0.0, 0.0, 0.0, 0.01),
        (1e-4, 1e-4, 1e-4, 0.2),
        None,
    ),
    "reference above the chosen, below SESOI": (
        (0.0, 0.0, 0.0, 0.001),
        (1e-4,) * 4,
        None,
    ),
    "reference above the full model, not the chosen": (
        (0.0, 0.004, 0.0, 0.005),
        (1e-4,) * 4,
        "without popularity",
    ),
}
ok = True
seen = set()
for label, (mus, sig, want) in CASES.items():
    S = scores_case(mus, sig)
    got = R.decide(S, SPECS)
    here = decide_here(S)
    one = (
        got["chosen"] == here["chosen"] == (want or "full")
        and got["eligible"] == here["eligible"]
        and got["order"] == here["order"]
        and got["best"] == here["best"]
        and got["reference"]["flag"] == here["flag"]
        and got["chosen_index"] == [s[0] for s in SPECS].index(got["chosen"])
        and got["chosen_terms"] == list(SPECS[got["chosen_index"]][1])
        and close(got["sesoi_abs"], 0.02 * float(S[0].mean()))
        and all(
            close(got["gate"][n]["difference"]["mean"], float(np.mean(S[c] - S[0])))
            for c, n in enumerate(want_names)
            if 1 <= c <= 3
        )
    )
    if not one:
        print(f"      {label}: got {got['chosen']}, {got['eligible']}, here " f"{here}")
    ok &= one
    seen.add((got["chosen"], got["reference"]["flag"]))
check(
    ok and len(seen) == 5,
    f"decide() on {len(CASES)} constructed cases equals the rules written "
    f"here: no eligible configuration, an interval that reaches 0, one "
    f"eligible, the one-standard-error rule toward more terms, two cells "
    f"with 7 terms, a tie, and the flag of the reference arm (on, and off "
    f"for an interval that reaches 0 or a gain below SESOI)",
)
S_I = np.full((5, N_C), 0.5)
S_I[1] += 0.125
S_I[2] += 0.125 + 2.0**-20
d_I = R.decide(S_I, SPECS, sesoi=0.25)
check(
    d_I["sesoi_abs"] == 0.125
    and d_I["eligible"] == [f"without {POP}"]
    and d_I["gate"][f"without {OTHER}"]["eligible"] is False
    and d_I["chosen"] == f"without {POP}",
    "R1 needs a difference strictly above SESOI: a gain of exactly SESOI "
    "is not eligible",
)
S_J = scores_case((0.004, 0.0, 0.0045, 0.0), (0.0, 0.0, 0.02, 0.0))
d_J = R.decide(S_J, SPECS)
row = d_J["one_se"]["without both"]
check(
    d_J["order"] == ["full", f"without {OTHER}", "without both"]
    and d_J["best"] == "without both"
    and close(d_J["one_se"][f"without {OTHER}"]["diff_to_best"], 0.0005)
    and close(d_J["one_se"][f"without {OTHER}"]["se"], 0.02 / math.sqrt(N_C), 1e-9)
    and row["diff_to_best"] == 0.0
    and row["terms"] == 6
    and d_J["reference"]["name"] == "artist terms only",
    "decide() stores the order, the best and each row of the "
    "one-standard-error rule",
)
rs = R.paired_scores(np.array([1.0]), np.array([0.5]))
check(
    rs == {"mean": 0.5, "se": None, "lo": None, "hi": None, "t": None, "n": 1},
    "paired_scores(): one listener gives no standard error, as None",
)
prov = R.provenance(CHAIN)
check(
    prov["p13b2_model.json"] == R.file_sha256(CHAIN / "p13b2_model.json")
    and prov["gam_knn_slices.npz"] == R.file_sha256(CHAIN / "gam_knn_slices.npz")
    and len(prov) == 13
    and all(
        prov[f"marker:{st_}"] is None or len(prov[f"marker:{st_}"]) == 64
        for st_ in (
            "p6_als",
            "p9b_colisten",
            "p11b_more_pairs",
            "p12b_more_features",
            "p13b2_loss",
        )
    ),
    "provenance(): the hashes of the input files, and of each stage marker "
    "or None where it is missing",
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
        "gam_refit: own line",
        "p13ab: not a stage",
        "loading p13a: not at the start",
    )
]
with R._relabel_logs():
    for r_ in [rec, rec2] + recs:
        R.log.filter(r_)
rec_after = logging.LogRecord("pipeline", logging.INFO, "", 0, "p13a: x", (), None)
R.log.filter(rec_after)
check(
    rec.getMessage() == "gam_refit: probe mean 1.00 s"
    and rec2.msg == "gam_refit: swap. Start again with P13A_WORKERS=8 "
    "python3 -m pipeline.runner --only gam_refit."
    and [r_.msg for r_ in recs]
    == [
        "gam_refit: probe listener 1 of 3",
        "gam_refit: a line",
        "gam_refit: own line",
        "p13ab: not a stage",
        "loading p13a: not at the start",
    ]
    and rec_after.msg == "p13a: x"
    and not R.log.filters,
    "the log filter renames the lines of P13a, P13b and P13b2, points the "
    "advice at this script, leaves other lines and goes away after the "
    "pass",
)
said, H0 = [], np.eye(2)


class FakeObjective:
    hess_fn = staticmethod(lambda T, D: H0)
    loss_grad = staticmethod(lambda T, D: (1.0, "g"))


fo = FakeObjective()
real_ps = R.PROGRESS_SECONDS
R.PROGRESS_SECONDS = 0.0
try:
    R.fit_lines(fo, "fit x", 0.0, said.append)
    same = all(fo.hess_fn(None, None) is H0 for _ in range(3)) and fo.loss_grad(
        None, None
    ) == (1.0, "g")
finally:
    R.PROGRESS_SECONDS = real_ps
check(
    same
    and len(said) == 4
    and said[2].startswith(
        "gam_refit: fit x: Newton step 3 done, 0 " "evaluations of the loss"
    )
    and said[3].startswith(
        "gam_refit: fit x: Newton step 3 done, 1 " "evaluations of the loss"
    )
    and R._text("NEWTON; 3 steps") == "NEWTON, 3 steps",
    "fit_lines(): a line per Newton step and per evaluation of the loss "
    "when due, with the Hessian and the loss unchanged, and _text() keeps "
    "semicolons out of the report",
)

# 2: a run, independent
M1 = fresh("main")
# The layouts of als_tune.py's and als_tune_d.py's results.
(M1 / "als_tuning.json").write_text(
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
(M1 / "als_tuning_d.json").write_text(
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
LINES1 = run_refit(M1)
DOC1 = json.loads((M1 / "gam_refit.json").read_text())
Z1 = load_npz(M1 / "gam_refit.npz")
RK = Z1["ranks"]
FITS = DOC1["fits"]
check(
    RK.shape == (5, len(USERS), 2)
    and np.array_equal(RK[0], G_RANK)
    and np.array_equal(Z1["user_id"], USERS)
    and Z1["tables"].shape == (5, BASE["L_u"])
    and Z1["scored"].shape == (len(USERS),)
    and DOC1["configs"] == [[n, list(k), r] for n, k, r in SPECS]
    and [j["name"] for j in FITS] == want_names
    and DOC1["inputs"] == {"x": "y"},
    "the outputs hold 5 configurations in the declared order, and the "
    "full model's ranks equal P13b part 2's stored ranks",
)
j0 = FITS[0]
ck0 = j0["check"]
check(
    j0["refit"] is False
    and j0["theta"] == FIT["theta"]
    and j0["tables"] == FIT["tables"]
    and j0["tables_raw"] == FIT["tables_raw"]
    and j0["center_consts"] == FIT["center_consts"]
    and ck0["proj_grad_max"] <= s13.KKT_TOL
    and ck0["relative_gap"] == 0.0
    and ck0["centering_gap"] <= 1e-12
    and ck0["objective_stored"] == FIT["fit"]["objective"]
    and all(j["refit"] is True for j in FITS[1:]),
    f"the full model is P13b part 2's fit, not refit: a stationary point "
    f"of the level's objective (largest projected gradient "
    f"{ck0['proj_grad_max']:.1e}, objective gap {ck0['relative_gap']}), "
    f"with its stored tables",
)

# The rows of the level, listener by listener, from the pair files.
PW = load_npz(CHAIN / "fusion_pairs_wide.npz")
FW = load_npz(CHAIN / "fusion_features_wide.npz")
assert bool(np.all(PW["split"] == fp.SPLIT_TRAIN))


def level_rows(k):
    rows, lens, npos = [], [], []
    for j in range(len(PW["user_id"])):
        a, b = int(PW["indptr"][j]), int(PW["indptr"][j + 1])
        n_p = int(np.count_nonzero(PW["kind"][a:b] == fp.KIND_POSITIVE))
        rows.append(
            np.concatenate(
                [
                    np.arange(a, a + n_p),
                    a + n_p + np.arange(k),
                    a + n_p + C.KW + np.arange(k),
                ]
            )
        )
        lens.append(n_p + 2 * k)
        npos.append(n_p)
    rows = np.concatenate(rows)
    lens, npos = np.array(lens), np.array(npos)
    user = np.repeat(np.arange(len(lens)), lens)
    item = PW["item"][rows].astype(np.int64)
    return {
        "k": k,
        "rows": rows,
        "lens": lens,
        "npos": npos,
        "U": len(lens),
        "user": user,
        "starts": np.concatenate(([0], np.cumsum(lens)[:-1])),
        "item": item,
        "kind": PW["kind"][rows],
        "q_p": C.PP_SLOT[item] / PW["pop_allowed"][user],
        "tier": FW["match_tier"][rows].astype(np.int64),
        "cols": {
            c: (
                FW["values"][rows, i].astype(np.float64),
                FW["has"][rows, i].astype(bool),
            )
            for i, c in enumerate(cn.CONTINUOUS)
        },
    }


LV = level_rows(LEVEL)


def term_values(t, tab, v, h, tier):
    assert t["multiplier"] is None
    kn = np.asarray(t["knots"], dtype=np.float64)
    K = len(kn)
    x = np.clip(np.where(h, v, kn[0]), kn[0], kn[-1])
    val = np.interp(x, kn, tab[:K])
    if t["zero_state"]:
        val = np.where(h & (v == 0.0), tab[K + gam.ZERO], val)
    miss = tab[K + gam.MISS] if t["missing_state"] else 0.0
    val = np.where(h, val, miss)
    if t["tiers"] is not None:
        val = np.where(np.isin(tier, t["tiers"]), val, 0.0)
    return val


def scores_of(terms, tables, cols, tier):
    out = np.zeros(len(tier))
    off = 0
    for t in terms:
        size = len(t["knots"]) + 4
        v, h = cols[t["source"]]
        out += term_values(t, np.asarray(tables[off : off + size]), v, h, tier)
        off += size
    return out


def loss_and_grad(s, lv):
    """p_pop form B on a level, written here: the k p_pop draws of the
    listener as negatives, -log(m q) on every candidate, weight 0 for a
    positive that p_pop cannot draw, 1 / (n_u U) for the others. Returns
    the loss and its gradient over the rows."""
    k, U = lv["k"], lv["U"]
    negs = (lv["starts"] + lv["npos"])[:, None] + k + np.arange(k)[None, :]
    assert bool(np.all(lv["kind"][negs] == fp.KIND_POP))
    assert bool(np.all(lv["q_p"][negs] > 0))
    pos_user = np.repeat(np.arange(U), lv["npos"])
    pos = np.concatenate([lv["starts"][u] + np.arange(lv["npos"][u]) for u in range(U)])
    q = np.where(lv["q_p"] > 0, lv["q_p"], 1.0)
    z = s - np.log(k * q)
    w = np.where(lv["q_p"][pos] > 0, 1.0, 0.0) / (lv["npos"][pos_user] * U)
    zp, zn = z[pos], z[negs][pos_user]
    mx = np.maximum(zp, zn.max(axis=1))
    lse = mx + np.log(np.exp(zp - mx) + np.exp(zn - mx[:, None]).sum(axis=1))
    loss = math.fsum(w * (lse - zp))
    g = np.zeros(len(s))
    np.add.at(g, pos, w * (np.exp(zp - lse) - 1.0))
    np.add.at(
        g, negs[pos_user].ravel(), (w[:, None] * np.exp(zn - lse[:, None])).ravel()
    )
    return loss, g, int(np.count_nonzero(w == 0.0))


def kkt_here(j):
    """(loss, largest gradient component over the parameters) of a fit
    record, with the design written here column by column."""
    lay, _ = R.layout_of(BASE, j["kept"])
    theta = np.asarray(j["theta"], dtype=np.float64)
    s = scores_of(j["terms"], lay.tables(theta), LV["cols"], LV["tier"])
    loss, g, n0 = loss_and_grad(s, LV)
    xg = np.empty(lay.L)
    for col in range(lay.L):
        e = np.zeros(lay.L)
        e[col] = 1.0
        xg[col] = scores_of(j["terms"], e, LV["cols"], LV["tier"]) @ g
    grad = lay.M.T @ xg + 2.0 * LAM * (lay.Q @ theta) + 2.0 * s13.RIDGE * theta
    return loss, float(np.max(np.abs(grad))), n0


worst_kkt, worst_loss, moved, n_zero_w = 0.0, 0.0, True, 0
for c, j in enumerate(FITS):
    loss, kkt, n0 = kkt_here(j)
    stored = j["check"]["loss"] if c == 0 else j["fit"]["loss"]
    worst_loss = max(worst_loss, abs(loss - stored) / max(1.0, abs(loss)))
    worst_kkt = max(worst_kkt, kkt)
    n_zero_w = n0
    if c:
        lay, _ = R.layout_of(BASE, j["kept"])
        start = R.warm_start(LAY_F, THETA_F, lay)
        moved &= (
            j["fit"]["newton_steps"] >= 1
            and float(np.max(np.abs(np.asarray(j["theta"]) - start))) > 1e-3
        )
check(
    worst_loss < 1e-12 and worst_kkt < 1e-8 and n_zero_w > 0,
    f"every fit's stored loss equals the loss written here on the raw rows "
    f"of level {LEVEL} (largest relative gap {worst_loss:.1e}), and its "
    f"gradient through a design written here meets the KKT conditions "
    f"(largest {worst_kkt:.1e}). {n_zero_w} positives outside P3's core "
    f"get weight 0",
)
check(moved, "every refit moves away from its start with at least one " "Newton step")
loss_full_here = kkt_here(FITS[0])[0]
check(
    abs(loss_full_here - FIT["fit"]["loss"]) < 1e-12,
    "the loss written here gives P13b part 2's stored loss of the final " "model",
)

worst_c, ok_tab = 0.0, True
pp_rows = LV["kind"] == fp.KIND_POP
for c, j in enumerate(FITS):
    lay, idx = R.layout_of(BASE, j["kept"])
    Tc = np.asarray(j["tables"])
    ok_tab &= (
        np.array_equal(lay.tables(np.asarray(j["theta"])), np.asarray(j["tables_raw"]))
        and np.array_equal(Z1["tables"][c][idx], Tc)
        and not np.any(np.delete(Z1["tables"][c], idx))
    )
    off = 0
    for t in j["terms"]:
        size = len(t["knots"]) + 4
        v, h = LV["cols"][t["source"]]
        val = term_values(
            t, Tc[off : off + size], v[pp_rows], h[pp_rows], LV["tier"][pp_rows]
        )
        worst_c = max(worst_c, abs(float(np.mean(val))))
        off += size
check(
    ok_tab and worst_c < 1e-12,
    f"every configuration's raw tables are M theta, its union tables hold "
    f"its own tables and 0 for the terms it drops, and each term has mean "
    f"0 over the level's p_pop negatives (largest {worst_c:.1e})",
)

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
            p_.append(tab_[K + gam.ZERO] - mid)
        if t["missing_state"]:
            p_.append(tab_[K + gam.MISS] - mid)
        out[t["name"]] = p_
    return out


worst_p = 0.0
for j in FITS:
    ph = points_here(j["terms"], j["tables_raw"])
    worst_p = max(
        worst_p,
        max(
            float(np.max(np.abs(np.asarray(ph[n]) - np.asarray(j["points"][n]))))
            for n in ph
        ),
    )
    worst_p = max(worst_p, float(set(ph) != set(j["points"])))
check(
    worst_p < 1e-14,
    f"the stored explanation points equal the points computed here " f"({worst_p:.1e})",
)


def rank_range(s, pick, excl, tol=1e-9):
    s = s.copy()
    s[0] = -np.inf
    s[excl] = -np.inf
    sp_ = s[pick]
    s[pick] = -np.inf
    near = np.abs(s - sp_) <= tol
    if near.sum() == (s == sp_).sum():
        r = 1 + int((s > sp_).sum()) + int((s[:pick] == sp_).sum())
        return r, r
    g = int((s > sp_ + tol).sum())
    return 1 + g, 1 + g + int(near.sum())


ok, n_exact, n_beyond, n_near = True, 0, 0, 0
for jj, u in enumerate(USERS):
    items_h, w_h = cd.history_of(W.u_indptr, W.u_indices, W.u_data, int(u))
    full = cn.full_catalog(cn.build_user_state(items_h, w_h, W.SPACE), W.SPACE)
    for c, j in enumerate(FITS):
        s = scores_of(j["terms"], j["tables"], full, W.SPACE.tier)
        for p in range(2):
            lo, hi = rank_range(s, int(PICKS[jj, p]), items_h)
            got = int(RK[c, jj, p])
            if not min(lo, 101) <= got <= min(hi, 101):
                ok = False
                print(
                    f"      {j['name']}, listener {u}, pick {p}: {got} "
                    f"not in [{lo}, {hi}]"
                )
            n_exact += int(lo == hi and lo <= 100)
            n_beyond += int(lo > 100)
            n_near += int(hi > lo)
check(
    ok
    and n_exact > 0
    and n_beyond > 0
    and DOC1["check"]["dense"] == {"listeners": len(USERS), "near_ties": n_near},
    f"every configuration's rank of every pick equals a dense count from "
    f"the stored terms and tables ({n_exact} exact, {n_beyond} beyond "
    f"100, censored at 101), and the run's dense check covered every "
    f"listener with the same count of near ties ({n_near})",
)


def ndcg(r):
    r = np.asarray(r, dtype=np.float64)
    return np.where(r <= 100, 1.0 / np.log2(r + 1.0), 0.0)


SC = ndcg(RK).mean(axis=2)
RES = DOC1["results"]
nU = SC.shape[1]


def sd_mean(d):
    return float(np.std(d, ddof=1) / math.sqrt(len(d)))


ok = True
for c, name in enumerate(want_names):
    cr = RES["configs"][name]
    d = SC[c] - SC[0]
    ok &= (
        close(cr["score"], float(SC[c].mean()))
        and close(cr["diff_to_full"]["mean"], float(d.mean()))
        and (c == 0 or close(cr["diff_to_full"]["se"], sd_mean(d)))
        and cr["terms"] == list(SPECS[c][1])
    )
    for p, pt in enumerate(("val_last", "val_first")):
        ok &= (
            close(cr[pt]["ndcg_at_100"], float(ndcg(RK[c, :, p]).mean()))
            and close(cr[pt]["recall_at_100"], float(np.mean(RK[c, :, p] <= 100)))
            and close(cr[pt]["recall_at_10"], float(np.mean(RK[c, :, p] <= 10)))
        )
f_o = 0.5 * (SC[0] + SC[2]) - 0.5 * (SC[1] + SC[3])
f_p = 0.5 * (SC[0] + SC[1]) - 0.5 * (SC[2] + SC[3])
f_x = 0.5 * (SC[0] - SC[1] - SC[2] + SC[3])
eff = RES["effects"]
ok &= set(eff) == {OTHER, POP, f"{OTHER} x {POP}"} and all(
    close(eff[k]["mean"], float(v.mean()))
    and close(eff[k]["se"], sd_mean(v))
    and eff[k]["n"] == nU
    for k, v in ((OTHER, f_o), (POP, f_p), (f"{OTHER} x {POP}", f_x))
)
here = decide_here(SC)
dec = RES["decision"]
ok &= (
    dec["chosen"] == here["chosen"]
    and dec["eligible"] == here["eligible"]
    and dec["order"] == here["order"]
    and dec["best"] == here["best"]
    and dec["reference"]["flag"] == here["flag"]
    and DOC1["chosen"]
    == {
        "name": dec["chosen"],
        "index": want_names.index(dec["chosen"]),
        "terms": list(SPECS[want_names.index(dec["chosen"])][1]),
    }
)
ch = want_names.index(dec["chosen"])
kb, ks = ndcg(KNN["both_id"]).mean(axis=1), ndcg(KNN["same_artist_id"]).mean(axis=1)
want_cmp = {
    "chosen minus ItemKNN, both lists": SC[ch] - kb,
    "chosen minus ItemKNN, same-artist list": SC[ch] - ks,
    "artist terms only minus ItemKNN, same-artist list": SC[4] - ks,
    "full minus ItemKNN, both lists": SC[0] - kb,
}
ok &= (
    set(RES["comparisons"]) == set(want_cmp)
    and all(
        close(RES["comparisons"][k]["mean"], float(v.mean()))
        and close(RES["comparisons"][k]["se"], sd_mean(v))
        for k, v in want_cmp.items()
    )
    and close(RES["itemknn"]["ItemKNN, both lists"], float(kb.mean()))
    and close(RES["itemknn"]["ItemKNN, same-artist list"], float(ks.mean()))
)
check(
    ok,
    f"the scores, differences, effects, decision ({dec['chosen']}) and "
    f"ItemKNN comparisons follow from the ranks, recomputed here",
)

band = np.digitize(SL["pop_centile"], [10.5, 50.5, 90.5])
masks = {f"popularity band: {nm}": band == c for c, nm in enumerate(G.POP_BANDS)}
masks.update(
    {
        f"co-listening evidence: {nm}": SL["evidence"] == c
        for c, nm in enumerate(G.EVIDENCE)
    }
)


def cluster_here(x, m):
    per = [[x[j, p] for p in range(2) if m[j, p]] for j in range(len(x))]
    per = [v for v in per if v]
    n_p = sum(len(v) for v in per)
    if not n_p:
        return None, None
    mean = sum(sum(v) for v in per) / n_p
    if len(per) < 2:
        return mean, None
    num = sum((sum(v) - mean * len(v)) ** 2 for v in per)
    return mean, math.sqrt(len(per) / (len(per) - 1) * num) / n_p


ok = RES["slice_picks"] == {k: int(m.sum()) for k, m in masks.items()}
D = ndcg(RK)
for c, name in enumerate(want_names):
    for k, m in masks.items():
        ok &= close(RES["slices"][name][k], cluster_here(D[c], m)[0])
for k, m in masks.items():
    mean, se = cluster_here(D[ch] - D[0], m)
    ok &= close(RES["chosen_minus_full"][k]["mean"], mean) and close(
        RES["chosen_minus_full"][k]["se"], se
    )
check(
    ok and len(masks) == 8,
    "the slice picks and every slice mean follow from loops over the "
    "listeners, with stage 1's bands and classes",
)

REP1 = (M1 / "reports" / "gam_refit.md").read_text()
heads = (
    "## 1. Fits",
    "## 2. Selection scores",
    "## 3. Decision",
    "## 4. Other models on the same picks",
    "## 5. Slices",
    "## 6. The chosen model (full)",
)
check(
    ";" not in REP1
    and all(h in REP1 for h in heads)
    and f"| full | 8 | {RES['configs']['full']['score']:.5f} |" in REP1
    and "- R1, the gate" in REP1
    and "- R2, the one-standard-error rule" in REP1
    and "- R3, the reference arm" in REP1
    and "minus full |" not in REP1
    and REP1.count("- full minus ItemKNN, both lists") == 1
    and f"| ItemKNN, same-artist list | {float(ks.mean()):.5f} |" in REP1
    and "| ALS, phase D winner (256 factors, nu 0) | 0.02674 |" in REP1
    and DOC1["references"]
    == [
        ["popularity", 0.003],
        ["ALS, 64 factors (phases A to C)", 0.017],
        ["ALS, phase D winner (256 factors, nu 0)", 0.02674],
    ]
    and ("No flag: the artist terms alone" in REP1) == (not dec["reference"]["flag"]),
    "the report has every section and the stored scores, without a "
    "semicolon, and no row of the full model against itself",
)
rows6 = REP1.split("## 6. The chosen model (full)")[1]
ok = True
for t in FIT["terms"]:
    if not SQ.get(t["name"]):
        continue
    off = sum(len(x["knots"]) + 4 for x in FIT["terms"][: FIT["terms"].index(t)])
    tab = np.asarray(FIT["tables"][off : off + len(t["knots"]) + 4])
    kn = np.asarray(t["knots"])
    vals = np.interp(np.clip(SQ[t["name"]], kn[0], kn[-1]), kn, tab[: len(kn)])
    ok &= f"| {t['name']} | " + " | ".join(f"{v:+.3f}" for v in vals) in rows6
moves_ok = True
for name in NAMES:
    for j in FITS[1:]:
        if name in j["points"]:
            mv = float(
                np.max(
                    np.abs(
                        np.asarray(j["points"][name])
                        - np.asarray(FITS[0]["points"][name])
                    )
                )
            )
            moves_ok &= f"{mv:.3f}" in rows6.split(f"| {name} | ")[-1]
check(
    ok and moves_ok,
    "the report's shape table reads the chosen model's tables, and the "
    "moves of the explanation points follow from the stored points",
)
check(
    not any(x.startswith(("p13a", "p13b")) for x in LINES1)
    and any(x.startswith("gam_refit: probe mean") for x in LINES1)
    and any(x.startswith("gam_refit: selection pass, 5 configurations") for x in LINES1)
    and any("refit 4 of 4, artist terms only (2 terms)" in x for x in LINES1)
    and any("design ready in" in x for x in LINES1)
    and any(x.startswith("gam_refit: chosen: ") for x in LINES1),
    "every line of the run names this script, with the probe, the pass, "
    "the design and each refit",
)

# 3: a forced smaller choice
real_decide = R.decide


def boosted(scores, specs, sesoi=R.SESOI):
    s = np.array(scores, dtype=np.float64, copy=True)
    s[1] = s[1] + 0.05
    return real_decide(s, specs, sesoi)


M3 = fresh("forced")
shutil.rmtree(M3.parent / "scratch" / R.WORK_NAME, ignore_errors=True)
shutil.copytree(
    M1.parent / "scratch" / R.WORK_NAME, M3.parent / "scratch" / R.WORK_NAME
)
sp3 = M3.parent / "scratch" / R.WORK_NAME / "fits.json"
sd3 = json.loads(sp3.read_text())
sd3["fits"][R.fit_id("without both", LEVEL, ARM, LAM)]["message"] = "NEWTON; 3 steps"
sp3.write_text(json.dumps(sd3))
R.decide = boosted
try:
    lines3 = run_refit(M3)
    DOC3 = json.loads((M3 / "gam_refit.json").read_text())
    rerun_ok = not raises(
        lambda: R.check_outputs(
            M3,
            BASE,
            FIT,
            SPECS,
            {"pop_centile": SL["pop_centile"], "evidence": SL["evidence"]},
            {"both": KNN["both_id"], "same": KNN["same_artist_id"]},
            G_RANK,
        ),
        "no such text",
    )
finally:
    R.decide = real_decide
REP3 = (M3 / "reports" / "gam_refit.md").read_text()
name1 = f"without {OTHER}"
cmf = DOC3["results"]["chosen_minus_full"]
check(
    DOC3["chosen"]["name"] == name1
    and DOC3["chosen"]["index"] == 1
    and rerun_ok
    and f"## 6. The chosen model ({name1})" in REP3
    and f"| {name1} minus full |" in REP3
    and f"- {name1} minus ItemKNN, both lists:" in REP3
    and all(
        close(cmf[k]["mean"], cluster_here(D[1] - D[0], m)[0]) for k, m in masks.items()
    )
    and close(
        DOC3["results"]["comparisons"]["chosen minus ItemKNN, both lists"]["mean"],
        float((SC[1] - kb).mean()),
    )
    and np.array_equal(load_npz(M3 / "gam_refit.npz")["ranks"], RK)
    and not any("Newton step" in x or "loading level" in x for x in lines3),
    "a forced choice of the model without colisten_other_artists: the "
    "outputs, the slices against the full model and the report follow it, "
    "from the saved fits and blocks",
)
rows63 = REP3.split(f"## 6. The chosen model ({name1})")[1]
check(
    "| without both | 6 |" in REP3 and "NEWTON, 3 steps" in REP3,
    "a solver message with a semicolon reaches the report without it",
)
check(
    "| colisten_other_artists |" not in rows63.split("How far")[0]
    and "| popularity | " in rows63.split("How far")[0],
    "the report's shape table holds the chosen model's terms only",
)

# 4: resume and workers
lines_again = run_refit(M1)
Z_again = load_npz(M1 / "gam_refit.npz")
DOC_again = json.loads((M1 / "gam_refit.json").read_text())
n_blocks = len(
    list(
        (M1.parent / "scratch" / R.WORK_NAME / "pass" / "selection").glob("part_*.npz")
    )
)
check(
    any("stay on disk" in x for x in lines_again)
    and any(
        f"selection pass resume, {n_blocks} of {n_blocks} blocks done" in x
        for x in lines_again
    )
    and not any("probe listener" in x or "Newton step" in x for x in lines_again)
    and np.array_equal(Z_again["ranks"], RK)
    and DOC_again["fits"] == FITS
    and DOC_again["results"] == RES,
    "a finished run fits and scores nothing when it runs again, with the "
    "same outputs",
)
store_path = M1.parent / "scratch" / R.WORK_NAME / "fits.json"
st_doc = json.loads(store_path.read_text())
fid2 = R.fit_id(f"without {POP}", LEVEL, ARM, LAM)
check(
    set(st_doc["fits"])
    == {R.fit_id(n, LEVEL, ARM, LAM) for n in want_names[1:] + [R.FULL_CHECK]}
    and set(st_doc["means"]) == {f"wide:{LEVEL}"},
    "the store holds the four refits, the full model's check and the "
    "level's background means",
)
del st_doc["fits"][fid2]
store_path.write_text(json.dumps(st_doc))
lines_lost = run_refit(M1)


def untimed(fits):
    out = json.loads(json.dumps(fits))
    for j in out:
        if j["refit"]:
            del j["fit"]["seconds"]
    return out


check(
    any("refit 1 of 1, without popularity (7 terms)" in x for x in lines_lost)
    and untimed(json.loads((M1 / "gam_refit.json").read_text())["fits"])
    == untimed(FITS)
    and np.array_equal(load_npz(M1 / "gam_refit.npz")["ranks"], RK),
    "a lost refit is fit again, bit for bit, and nothing else is",
)
st_doc = json.loads(store_path.read_text())
del st_doc["means"]
store_path.write_text(json.dumps(st_doc))
lines_means = run_refit(M1)
check(
    any("loading level" in x for x in lines_means)
    and not any("building the design" in x for x in lines_means)
    and untimed(json.loads((M1 / "gam_refit.json").read_text())["fits"])
    == untimed(FITS),
    "lost background means come back from the level's rows without a "
    "design, with the same tables",
)
st_doc = json.loads(store_path.read_text())
del st_doc["fits"][R.fit_id(R.FULL_CHECK, LEVEL, ARM, LAM)]
store_path.write_text(json.dumps(st_doc))
lines_check = run_refit(M1)
check(
    any("building the design" in x for x in lines_check)
    and any("the final model is a stationary point" in x for x in lines_check)
    and not any("refit 1 of" in x for x in lines_check)
    and untimed(json.loads((M1 / "gam_refit.json").read_text())["fits"])
    == untimed(FITS),
    "a lost check of the full model is done again on a new design, with " "no refit",
)
part = sorted(
    (M1.parent / "scratch" / R.WORK_NAME / "pass" / "selection").glob("part_*.npz")
)
part[3].unlink()
lines_part = run_refit(M1)
check(
    any(
        f"selection pass resume, {n_blocks - 1} of {n_blocks} blocks " f"done" in x
        for x in lines_part
    )
    and np.array_equal(load_npz(M1 / "gam_refit.npz")["ranks"], RK),
    "a lost pass block is scored again, with the same ranks",
)
M5 = fresh("keys")
shutil.rmtree(M5.parent / "scratch" / R.WORK_NAME, ignore_errors=True)
shutil.copytree(
    M1.parent / "scratch" / R.WORK_NAME, M5.parent / "scratch" / R.WORK_NAME
)
st_up = C.p13_loaders(s13b2.StageP13b2())
st_up._upstream_outputs = lambda: {"p11_fusion_pairs": {"pairs": "other"}}
lines_up = run_refit(M5, st=st_up)
check(
    sum("refit " in x and " of 4, " in x and "objective" in x for x in lines_up) == 4
    and any("the final model is a stationary point" in x for x in lines_up)
    and any("selection pass resume" in x for x in lines_up)
    and untimed(json.loads((M5 / "gam_refit.json").read_text())["fits"])
    == untimed(FITS),
    "a change in P13b part 2's key (here its upstream outputs) starts the "
    "check and the refits again, with the same fits, and the pass keeps "
    "its blocks for the same tables",
)
d_note = json.loads((M5 / "p13b2_model.json").read_text())
d_note["note"] = "an edit that changes no number"
(M5 / "p13b2_model.json").write_text(json.dumps(d_note))
lines_note = run_refit(M5)
check(
    sum("refit " in x and " of 4, " in x and "objective" in x for x in lines_note) == 4
    and not any("selection pass resume" in x for x in lines_note)
    and any("gam_refit: probe mean" in x for x in lines_note)
    and np.array_equal(load_npz(M5 / "gam_refit.npz")["ranks"], RK),
    "a changed final model file starts the refits and the pass again, "
    "with the same ranks",
)
M4 = fresh("three")
real_ps2 = als_mod.PROGRESS_SECONDS
als_mod.PROGRESS_SECONDS = 0.0
R.PROGRESS_SECONDS = 0.0
try:
    lines4 = run_refit(M4, workers=3, n_verify=4)
finally:
    als_mod.PROGRESS_SECONDS = real_ps2
    R.PROGRESS_SECONDS = real_ps2
check(
    any(x.startswith("gam_refit: refits: 25% (1 of 4 fits)") for x in lines4)
    and any(x.startswith("gam_refit: refits: 75% (3 of 4 fits)") for x in lines4)
    and any(
        x.startswith(
            f"gam_refit: refit 1 of 4, without {OTHER} (7 "
            f"terms): Newton step 1 done"
        )
        for x in lines4
    )
    and any(
        x.startswith(
            "gam_refit: refit 4 of 4, artist terms only (2 "
            "terms): Newton step 1 done"
        )
        for x in lines4
    ),
    "the refits print their share done and each Newton step when the "
    "progress interval allows",
)
DOC4 = json.loads((M4 / "gam_refit.json").read_text())
check(
    np.array_equal(load_npz(M4 / "gam_refit.npz")["ranks"], RK)
    and [j["theta"] for j in DOC4["fits"]] == [j["theta"] for j in FITS]
    and DOC4["results"] == RES,
    "three workers give the same fits, bit for bit, and the same ranks "
    "and results as one",
)

# 5: refusals and faults


def with_input(name, change, text, n_verify=4):
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
    return raises(lambda: run_refit(m, n_verify=n_verify), text)


def fin_fit(change):
    def f(d):
        change(d["wide_fits"][FIN])

    return f


def set_term(name, key, value):
    def f(j):
        for t in j["terms"]:
            if t["name"] == name:
                t[key] = value

    return f


def bump_theta(j):
    j["theta"][5] += 1e-3


def bump_obj(j):
    j["fit"]["objective"] *= 1 + 1e-8


def bump_raw(j):
    j["tables_raw"][3] += 1e-12


def bump_centered(j):
    j["tables"][3] += 1e-9


def drop_other(j):
    j["terms"] = [t for t in j["terms"] if t["name"] != OTHER]


def other_lambda(j):
    j["lambda"] = LAM * 2


def other_print(j):
    j["fingerprint"] = "0" * 64


def other_knot(j):
    t = next(t for t in j["terms"] if t["name"] == "audio_fit")
    t["knots"] = t["knots"][:-1] + [t["knots"][-1] + 1.0]


def edit_params(d):
    d["params"]["tau"] = 0.4


def bump_rank(key, idx):
    def f(d):
        d[key] = d[key].copy()
        d[key][idx] += 1

    return f


def shuffle(key):
    def f(d):
        d[key] = d[key][::-1].copy()

    return f


def edit_knn_score(d):
    d["variants"]["both/id"]["score"] += 1e-9


j_hit = next(j for j in range(len(USERS)) if G_RANK[j, 0] <= 99)
cases = [
    ("p13b2_model.json", edit_params, "other P13b part 2 parameters"),
    (
        "p13b2_model.json",
        fin_fit(set_term("audio_fit", "monotone", 1)),
        "has a monotone term",
    ),
    (
        "p13b2_model.json",
        fin_fit(set_term("audio_fit", "name", "audio_x")),
        "terms outside the union: ['audio_x']",
    ),
    (
        "p13b2_model.json",
        fin_fit(set_term("audio_fit", "tiers", [3])),
        "does not apply to every candidate",
    ),
    (
        "p13b2_model.json",
        fin_fit(drop_other),
        "lacks the terms ['colisten_other_artists']",
    ),
    ("p13b2_model.json", fin_fit(other_lambda), "penalty weight differs"),
    ("p13b2_model.json", fin_fit(other_print), "rebuilt here, differs"),
    ("p13b2_model.json", fin_fit(other_knot), "rebuilt here, differs"),
    ("p13b2_model.json", fin_fit(bump_theta), "not a stationary point"),
    (
        "p13b2_model.json",
        fin_fit(lambda j: j["theta"].pop()),
        "the final model's parameters do not match its layout",
    ),
    (
        "p13b2_model.json",
        fin_fit(bump_obj),
        "objective on the rebuilt " "design differs",
    ),
    ("p13b2_model.json", fin_fit(bump_raw), "stored raw tables differ from " "M theta"),
    ("p13b2_model.json", fin_fit(bump_centered), "stored centered tables " "differ"),
    (
        "gam_knn_slices.npz",
        bump_rank("picks", (0, 0)),
        "belongs to other listeners, picks or another final model",
    ),
    (
        "gam_knn_slices.npz",
        bump_rank("gam_rank", (j_hit, 0)),
        "belongs to other listeners, picks or another final model",
    ),
    (
        "itemknn_selection_ranks.npz",
        shuffle("user_id"),
        "ItemKNN's stored listeners differ",
    ),
    (
        "itemknn_selection_ranks.npz",
        lambda d: d.update(both_id=d["both_id"][:-1].copy()),
        "ItemKNN's stored both/id ranks do not match the picks",
    ),
    (
        "gam_knn_slices.npz",
        shuffle("user_id"),
        "belongs to other listeners, picks or another final model",
    ),
    ("itemknn_selection.json", edit_knn_score, "does not follow from its stored ranks"),
    (
        "p13b2_arrays.npz",
        shuffle("sel_users"),
        "p13b2_arrays.npz holds other listeners",
    ),
]
ok = True
for name, fn, text in cases:
    one = with_input(name, fn, text)
    if not one:
        print(f"      not refused: {name}, {text}")
    ok &= one
m_ns = fresh("no_slices")
(m_ns / "gam_knn_slices.npz").unlink()
ok &= raises(
    lambda: run_refit(m_ns),
    "gam_knn_slices.npz is missing: run "
    "python3 -m pipeline.runner --only "
    "gam_knn_slices",
)
m_br = fresh("both_ranks")
for name, key in (("p13b2_arrays.npz", "ranks"), ("gam_knn_slices.npz", "gam_rank")):
    d_ = load_npz(m_br / name)
    d_[key] = d_[key].copy()
    if key == "ranks":
        d_[key][FIN, j_hit, 0] += 1
    else:
        d_[key][j_hit, 0] += 1
    np.savez(m_br / name, **d_)
ok &= raises(
    lambda: run_refit(m_br, n_verify=4),
    "the full model's ranks differ from P13b part 2's stored ranks "
    "of the final model in 1 picks",
)
check(
    ok,
    f"the run refuses {len(cases) + 2} kinds of inconsistent input: "
    f"parameters, a monotone, foreign, partial or missing term, the "
    f"weight, the layout, parameters off the optimum, a changed "
    f"objective or table, foreign stage 1 or ItemKNN outputs, and "
    f"stored ranks that the full model does not give",
)

# The full model keeps P13b part 2's stored tables and constants. A planted
# shift of one term by 1e-13, inside the centering tolerance and consistent
# with its constant, must reach the outputs as stored.
m_sh = fresh("shift")
d_sh = json.loads((m_sh / "p13b2_model.json").read_text())
f_sh = d_sh["wide_fits"][FIN]
T_sh = np.asarray(f_sh["tables_raw"], dtype=np.float64)
Tc_sh = np.asarray(f_sh["tables"], dtype=np.float64)
c_new = float(f_sh["center_consts"]["genre_affinity"]) - 1e-13
off_g = sum(len(t["knots"]) + 4 for t in f_sh["terms"][:2])
K_g = len(f_sh["terms"][2]["knots"])
seg = T_sh[off_g : off_g + K_g + 2].copy()
seg -= c_new
Tc_sh[off_g : off_g + K_g + 2] = seg
f_sh["center_consts"]["genre_affinity"] = c_new
f_sh["tables"] = Tc_sh.tolist()
(m_sh / "p13b2_model.json").write_text(json.dumps(d_sh))
run_refit(m_sh, n_verify=4)
j_sh = json.loads((m_sh / "gam_refit.json").read_text())["fits"][0]
check(
    j_sh["tables"] == f_sh["tables"]
    and j_sh["center_consts"] == f_sh["center_consts"]
    and 0.0 < j_sh["check"]["centering_gap"] <= 1e-12,
    f"the full model keeps P13b part 2's stored tables and constants: a "
    f"consistent shift of {j_sh['check']['centering_gap']:.1e} inside the "
    f"centering tolerance reaches the outputs as stored",
)

real_fit = R.gam.fit


def short_fit(obj, theta0, settings):
    """gam.fit() stopped after one Newton step: a refit that does not
    converge."""
    return real_fit(obj, theta0, dict(settings, newton_steps=1, newton_accept=1.0))


R.gam.fit = short_fit
try:
    m_cs = fresh("solver")
    ok_cs = raises(lambda: run_refit(m_cs, n_verify=4), "above KKT_TOL")
finally:
    R.gam.fit = real_fit
st_cs = json.loads((m_cs.parent / "scratch" / R.WORK_NAME / "fits.json").read_text())
check(
    ok_cs and set(st_cs["fits"]) == {R.fit_id(R.FULL_CHECK, LEVEL, ARM, LAM)},
    "a refit that stops above KKT_TOL stops the run and is not saved",
)

# Saved fits that do not fit.
for change, text in (
    (lambda r_: r_.update(fingerprint="0" * 64), "belongs to another layout"),
    (lambda r_: r_.update(proj_grad_max=2 * s13.KKT_TOL), "above KKT_TOL"),
):
    m = fresh("store_fault")
    shutil.rmtree(m.parent / "scratch" / R.WORK_NAME, ignore_errors=True)
    shutil.copytree(
        M1.parent / "scratch" / R.WORK_NAME, m.parent / "scratch" / R.WORK_NAME
    )
    sp_ = m.parent / "scratch" / R.WORK_NAME / "fits.json"
    sd = json.loads(sp_.read_text())
    change(sd["fits"][fid2])
    sp_.write_text(json.dumps(sd))
    check(
        raises(lambda: run_refit(m, n_verify=4), text),
        f"a saved refit that does not fit stops the run: {text}",
    )

# The output check against planted faults.
S1_IN = {"pop_centile": SL["pop_centile"], "evidence": SL["evidence"]}
KNN_IN = {"both": KNN["both_id"], "same": KNN["same_artist_id"]}


def out_fault(change, text, npz=False):
    m = fresh("out_fault")
    for name in R.OUT_FILES:
        shutil.copy(M1 / name, m / name)
    if npz:
        d_ = load_npz(m / R.OUT_FILES[1])
        change(d_)
        np.savez(m / R.OUT_FILES[1], **d_)
    else:
        d_ = json.loads((m / R.OUT_FILES[0]).read_text())
        change(d_)
        (m / R.OUT_FILES[0]).write_text(json.dumps(d_))
    return raises(
        lambda: R.check_outputs(m, BASE, FIT, SPECS, S1_IN, KNN_IN, G_RANK), text
    )


def set_rank(c, val):
    def f(d):
        d["ranks"] = d["ranks"].copy()
        d["ranks"][c, 3, 1] = val(d["ranks"][c, 3, 1])

    return f


def bump_table(d):
    d["tables"] = d["tables"].copy()
    d["tables"][2, 7] += 1e-12


def flip(r_):
    return 1 if r_ > 1 else 101


def jfit(c, change):
    def f(d):
        change(d["fits"][c])

    return f


def bump_list(key, i, by):
    def f(j):
        j[key][i] += by

    return f


def bump_const(j):
    k = next(iter(j["center_consts"]))
    j["center_consts"][k] += 1e-12


def bump_point(j):
    k = next(iter(j["points"]))
    j["points"][k][0] += 1e-12


def other_choice(d):
    d["chosen"]["name"] = "without both"


def edit_result(d):
    d["results"]["configs"]["full"]["score"] += 1e-12


def edit_configs(d):
    d["configs"] = d["configs"][::-1]


def other_terms(j):
    j["kept"] = j["kept"][:-1]


def full_like_refit(d):
    d["fits"][0]["theta"] = (
        d["fits"][0]["theta"][:1]
        + [d["fits"][0]["theta"][0]]
        + d["fits"][0]["theta"][2:]
    )


faults = [
    (set_rank(2, flip), "results do not follow from the stored ranks", True),
    (set_rank(0, flip), "the stored full model's ranks differ", True),
    (shuffle("user_id"), "the stored listeners differ", True),
    (bump_table, "the stored union tables differ", True),
    (
        jfit(2, bump_list("theta", 4, 1e-9)),
        "raw tables do not follow from " "the stored parameters",
        False,
    ),
    (
        jfit(3, bump_list("tables", 6, 1e-12)),
        "are not the raw tables minus " "the stored constant",
        False,
    ),
    (jfit(1, bump_const), "are not the raw tables minus the stored " "constant", False),
    (jfit(4, bump_point), "explanation points do not follow", False),
    (
        jfit(1, lambda j: j["fit"].update(proj_grad_max=1e-7)),
        "the stored refit is not stationary",
        False,
    ),
    (jfit(2, other_terms), "terms or layout differ", False),
    (other_choice, "the stored choice does not follow", False),
    (edit_result, "results do not follow from the stored ranks", False),
    (edit_configs, "configurations differ from the declared ones", False),
    (full_like_refit, "raw tables do not follow", False),
    (
        lambda d: d.update(fits=d["fits"][:-1]),
        "the stored tables do not match the configurations",
        False,
    ),
    (jfit(4, lambda j: j.update(role="candidate")), "terms or layout differ", False),
    (jfit(1, lambda j: j.update(name="other")), "terms or layout differ", False),
    (
        jfit(2, lambda j: j.update(fingerprint="0" * 64)),
        "terms or layout differ",
        False,
    ),
    (
        jfit(
            3,
            lambda j: j["terms"][0]["knots"].__setitem__(
                1, j["terms"][0]["knots"][1] * 0.5
            ),
        ),
        "terms or layout differ",
        False,
    ),
    (
        lambda d: d.update(ranks=d["ranks"][:, :-1].copy()),
        "the stored ranks do not match the configurations",
        True,
    ),
]
ok = True
for fn, text, npz in faults:
    one = out_fault(fn, text, npz)
    if not one:
        print(f"      fault not caught: {text}")
    ok &= one
check(ok, f"the output check fails for {len(faults)} planted faults in the " f"outputs")


def consistent_full(m):
    """The full model's record changed in its parameters, with tables,
    points and union tables that follow from them."""
    d_ = json.loads((m / R.OUT_FILES[0]).read_text())
    z_ = load_npz(m / R.OUT_FILES[1])
    j = d_["fits"][0]
    lay, idx = R.layout_of(BASE, j["kept"])
    th = np.asarray(j["theta"], dtype=np.float64)
    th[4] += 1e-6
    T = lay.tables(th)
    Tc = T.copy()
    for t, off in zip(lay.terms, lay.offsets):
        Tc[off : off + t.K + 2] -= j["center_consts"][t.name]
    j.update(
        theta=th.tolist(),
        tables_raw=T.tolist(),
        tables=Tc.tolist(),
        points=s13b2.shape_points(j["terms"], T.tolist(), BASE["shape_q"]),
    )
    z_["tables"] = z_["tables"].copy()
    z_["tables"][0][idx] = Tc
    (m / R.OUT_FILES[0]).write_text(json.dumps(d_))
    np.savez(m / R.OUT_FILES[1], **z_)


m_cf = fresh("out_full")
for name in R.OUT_FILES:
    shutil.copy(M1 / name, m_cf / name)
consistent_full(m_cf)
check(
    raises(
        lambda: R.check_outputs(m_cf, BASE, FIT, SPECS, S1_IN, KNN_IN, G_RANK),
        "the stored full model differs from P13b part 2's final fit",
    ),
    "the output check fails for a full model whose stored parameters, "
    "tables and points agree with each other but not with P13b part 2's",
)
m_nj = fresh("out_nojson")
shutil.copy(M1 / R.OUT_FILES[1], m_nj / R.OUT_FILES[1])
check(
    raises(
        lambda: R.check_outputs(m_nj, BASE, FIT, SPECS, S1_IN, KNN_IN, G_RANK),
        "gam_refit.json is missing",
    ),
    "the output check fails without its json file",
)
check(
    not raises(
        lambda: R.check_outputs(M1, BASE, FIT, SPECS, S1_IN, KNN_IN, G_RANK),
        "no such text",
    ),
    "the output check passes on the untouched outputs",
)

# A planted scorer fault in one configuration.
real_cut = ct.CatalogScorer.ranks_cut
C_REF = 4
J_REF = next(j for j in range(len(USERS)) if RK[C_REF, j, 0] <= 99)
J_MISS = next(j for j in range(len(USERS)) if RK[C_REF, j, 0] == 101)
J_REF2 = next(j for j in range(len(USERS)) if 2 <= RK[C_REF, j, 0] <= 100)
PLANT = {}


def planted_cut(self, state, space, m_u, picks, exclude, cut, chunk=None):
    rk, sc_ = real_cut(self, state, space, m_u, picks, exclude, cut, chunk)
    if self.C == 5 and np.array_equal(picks, PICKS[PLANT["j"]]):
        rk = rk.copy()
        rk[C_REF, 0] = PLANT["rank"](int(rk[C_REF, 0]))
    return rk, sc_


ct.CatalogScorer.ranks_cut = planted_cut
ok = True
try:
    for j_, fn_ in (
        (J_REF, lambda r_: r_ + 1),
        (J_REF2, lambda r_: r_ - 1),
        (J_REF, lambda r_: 101),
        (J_MISS, lambda r_: 1),
    ):
        PLANT.update(j=j_, rank=fn_)
        ok &= raises(
            lambda: run_refit(fresh("scorer"), n_verify=90),
            "lies outside the dense reference's range",
        )
finally:
    ct.CatalogScorer.ranks_cut = real_cut
check(
    ok,
    "the dense check fails when the scorer gives the reference arm a "
    "rank one too high or one too low, 101 for a hit or 1 for a miss",
)

# A rank changed on its way to the disk.
real_savez = R._atomic_savez


def bad_savez(path, **arrays):
    a = np.array(arrays["ranks"], copy=True)
    a[2, 3, 1] = flip(a[2, 3, 1])
    arrays["ranks"] = a
    real_savez(path, **arrays)


R._atomic_savez = bad_savez
try:
    ok = raises(
        lambda: run_refit(fresh("write"), n_verify=4),
        "results do not follow from the stored ranks",
    )
finally:
    R._atomic_savez = real_savez
check(
    ok,
    "the run checks what it wrote: a rank changed on the way to the " "disk stops it",
)

print()
if failures:
    print(f"{len(failures)} check(s) failed:")
    for f_ in failures:
        print(f"  - {f_}")
    raise SystemExit(1)
print("all checks passed")
