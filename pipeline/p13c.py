"""P13c: the stability and the exports of the chosen GAM.

The model is the configuration that gam_refit.py chose (gam_refit.json,
"chosen"). On the real data it is "without both": tag_affinity,
artist_tag_affinity, genre_affinity, artist_affinity, colisten_same_artist
and audio_fit, at the final model's level of the wide file (m = 64), loss
arm (p_pop form B) and penalty weight (lambda 3.16e-8). P13c never changes
the model, and it never reads a test pick.

Protocol, fixed before any result.

A. Sampler bias. The model's terms are fit on the reference file of P11b and
P12b (the reference listeners) with P13b part 2's machinery:
- at the largest reference level (4,096 draws per kind on the real data)
  under the four arms mixture form A, mixture form B, p_pop form A and p_pop
  form B. The reference R is mixture form A at this level (m = 8,192), as in
  P13b part 2.
- at the model's level under the same four arms. The arm of the model gives
  the model's configuration on the reference listeners.
Explanation points are stage_p13b2.shape_points(): a term's values at the
positives' percentiles (P13a's shape quantiles), then its zero and missing
constants, each minus the value at the median. A shift of the term does not
change them.
- A1, R's convergence: the largest gap of explanation points between two of
  the four fits at the largest level is at most TAU (0.5 log-odds points),
  P13b part 2's rule D1. Otherwise the report marks table A2 as descriptive.
- A2, the sampler-bias table: per term and point, the model's configuration
  on the reference listeners minus R. A point with an absolute bias above
  TAU is flagged. P14's detail view must show such a point with its bias.

B. The near-optimal set. The family is the model's terms at the model's
level of the wide file and its penalty weight, under the four arms. One
selection pass of P13 (StageP13a._selection_pass) scores the four, the model
included: full-catalog NDCG@100 of the val_last and val_first picks of the
selection listeners, ranks exact up to 100.
- B1: the near-optimal set holds the family members whose score lies within
  SESOI (2 percent) of the family's best score. The report gives, across the
  set, the range of each term's centered values and importances, each
  member's gap to R (its arm at the model's level on the reference
  listeners), and the signed top-3 agreement of the reasons (D). A member
  that scores more than SESOI above the model, with the 95 percent paired
  interval above 0, is flagged. P13c does not change the model.

C. The bootstrap.
- listener: N_BOOT["listener"] = 100 replicates. Replicate b draws the
  training listeners of the level with replacement (keyed stream SEED, 0,
  b). Each positive's weight is multiplied by its listener's count.
- twoway: N_BOOT["twoway"] = 50 replicates. Replicate b draws the listeners
  and, independently, the distinct tracks of the positives with replacement
  (keyed stream SEED, 1, b). A positive's weight is multiplied by its
  listener's count times its track's count: Owen's (2007) pigeonhole
  bootstrap, where an observation is a listener and a picked track. The
  draws of a choice set stay with it.
In both, the multipliers are scaled so that the positives' total weight
equals the full fit's, so the penalty keeps its weight. Each replicate
starts from the model's parameters (a warm start) and must reach KKT_TOL.
Its tables are centered with the frozen rule: the model's background, the
level's p_pop negatives with the full fit's weights.
- C1: per term and point (centered values at the positives' percentiles,
  zero and missing): the value, the 2.5 and 97.5 percentiles and the SD over
  the listener replicates, the same over the twoway replicates, and the share
  of listener replicates with the model's sign. Each band comes with the
  term's importance (D), because a flat term is stable and says nothing.

D. The stable-reason rule, estimated on the val picks of the selection
listeners in P11 (rows that no fit uses). A pick's background is its
listener's p_pop draws in P11, a frozen popularity-weighted sample of the
listener's candidates (background 2 of the P13 design). A term's
contribution is its value at the pick minus its mean over those draws.
artist_affinity and colisten_same_artist form one artist reason (the sum of
the two). Every other term is a reason of its own. A reason is a candidate
when one of its terms is on its shape at the pick, so "no data" and "data,
but no match" never make a reason. Candidates rank by contribution, largest
first, with ties by the reason order.
- D1 (K_TOP = 3): a reason is shown when it is a candidate, its contribution
  under the model is positive, it is positive in at least 95 percent of the
  listener replicates, and it ranks in the top 3 in at least 80 percent of
  them. The twoway replicates give a sensitivity run of the same rule.
- A term's importance: its mean absolute contribution over the listeners'
  p_pop draws, against the same background.
P14 applies the rule to the eval listeners' top-10 lists with background 1
(the listener's own candidates), from the exported replicate parameters.

Checks:
- gam_refit's chosen record: its layout, rebuilt here, has the stored terms
  and fingerprint. Its raw tables equal M theta, its centered tables equal a
  centering on the level's p_pop negatives, and its parameters are a
  stationary point of the level's objective with the stored objective.
- the weighted objective with unit multipliers equals the stage's objective
  bit for bit.
- every fit reaches KKT_TOL before it is saved.
- the model's ranks in the pass equal gam_refit's stored ranks.
- VERIFY_USERS listeners get each family member's ranks again from the
  dense reference of catalog.py.
- the script reads its outputs back: the tables follow from the stored
  parameters, and the bands, the set and the rule from the stored tables,
  ranks and reason values.

Outputs: MODEL_DIR/p13c_model.json (the protocol, the model's tables for
P14, the results), MODEL_DIR/p13c_arrays.npz (the parameters of every fit
and replicate, the ranks, the reason sample) and
MODEL_DIR/reports/p13c_report.md. Every line goes to the console and to
MODEL_DIR/logs/p13c.log. Long steps print a line at most every 30 seconds:
each fit its Newton steps, the replicates their share done with the time
left, and the pass its share done with the time left (P13's lines).

Resume: each fit is saved in SCRATCH_DIR/p13c/fits.json and the pass in
SCRATCH_DIR/p13c/pass, keyed by their inputs. After Ctrl+C, the same command
continues with the fits and blocks still missing.

Expected run on the real data, about 5 to 6 hours: loading the wide level
and its design a few minutes (peak memory near P13b part 2's 18 GB), the 3
family refits about 15 minutes, the 150 replicates about 3 to 5 hours (3 to
5 Newton steps of about 30 seconds each), the reference fits about 20
minutes, the pass about 30 minutes and the reason sample a few minutes.

Model switch (added for P13d, with P13d's protocol).
--model gam_refit, the default, runs the protocol above on gam_refit's
chosen model, unchanged. --model p13d runs it on the model record in
p13d.json: the model that P13d's rules S1 and S2 chose. A bounded model
keeps its bounds in every fit (gam_arms.fit_bounded), every bounded fit
must keep its order, and the model's ranks must equal P13d's stored
ranks. The outputs then carry the record's tag:
MODEL_DIR/p13c_<tag>_model.json, p13c_<tag>_arrays.npz,
reports/p13c_<tag>_report.md and logs/p13c_<tag>.log, with the saved work
in SCRATCH_DIR/p13c_<tag>. The outputs of the first run stay.

Runner stages: p13c_stability runs the first
run, on gam_refit's model. p13c_final runs the model switch when P13d
changed the model, and it always writes MODEL_DIR/p13c_final.json, the
pointer to the exports that P14 reads: the tagged outputs, or
p13c_stability's outputs when P13d kept the model.

Usage, from the repository root:
    python3 -m pipeline.runner --only p13c_stability
    python3 -m pipeline.runner --only p13c_final
The module's own entry runs the same pass without a marker (the runner
then asks for --adopt):
    python3 -m pipeline.p13c [--model p13d]
The pass uses P13's worker setting. To use fewer workers:
    P13A_WORKERS=8 python3 -m pipeline.runner --only p13c_final
"""

import argparse
import contextlib
import dataclasses
import logging
import math
import os
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from pipeline import catalog as ct
from pipeline import concept_data as cd
from pipeline import concepts as cn
from pipeline import evaluation as ev
from pipeline import fusion_pairs as fp
from pipeline import gam
from pipeline import gam_arms
from pipeline import gam_audit as ga
from pipeline import stage_p13a as s13
from pipeline import stage_p13b as s13b
from pipeline import stage_p13b2 as s13b2
from pipeline.als import Progress, _clock
from pipeline.common import (
    atomic_json_write,
    atomic_text_write,
    file_sha256,
    load_json,
    log,
)
from pipeline.config import MODEL_DIR, SCRATCH_DIR

from pipeline import gam_knn_diag as gkd
from pipeline import gam_refit as gr
from pipeline.script_stage import ScriptStage, log_file

LOG_PATH = MODEL_DIR / "logs" / "p13c.log"
VERSION = "p13c-v1"
WORK_NAME = "p13c"
DECLARED = "before any result"
SESOI = 0.02
TAU = s13b2.TAU
ALL_ARMS = (("p_pop", "B"), ("p_pop", "A"), ("mix32", "B"), ("mix32", "A"))
REF_ARM = ("mix32", "A")
SCHEMES = ("listener", "twoway")
N_BOOT = {"listener": 100, "twoway": 50}
SEED = 2026100103
K_TOP = 3
SIGN_SHARE = 0.95
TOP_SHARE = 0.80
BAND = (0.025, 0.975)
ARTIST = ("artist_affinity", "colisten_same_artist")
POINTS = ("p5", "p25", "p50", "p75", "p95", "zero", "missing")
ZERO_TOL = 1e-9  # a value this close to 0 (a neutral tie) has no sign
VERIFY_USERS = 6
OUT_FILES = ("p13c_model.json", "p13c_arrays.npz")
REPORT_NAME = "p13c_report.md"
MODELS = ("gam_refit", "p13d")  # the model switch (P13d)
P13D_VERSION = "p13d-v1"
P13D_FILES = ("p13d.json", "p13d_model_set.npz")

PROTOCOL = {
    "version": VERSION,
    "declared": DECLARED,
    "model": "gam_refit.json's chosen configuration at the final model's "
    "level, arm and penalty weight",
    "reference": "R = mixture form A at the largest reference level",
    "family_arms": [s13b.arm_label(a) for a in ALL_ARMS],
    "n_boot": N_BOOT,
    "seed": SEED,
    "sesoi": SESOI,
    "tau": TAU,
    "rules": {
        "A1": "R is converged when the four fits at the largest reference "
        "level agree within TAU (largest gap of explanation points)",
        "A2": "sampler bias per term and point: the model's configuration on "
        "the reference listeners minus R, flagged above TAU",
        "B1": "near-optimal set: family members within SESOI of the family's "
        "best score. A member more than SESOI above the model, with "
        "the interval above 0, is flagged. The model does not change",
        "C1": "bands: 2.5 and 97.5 percentiles and SD over the replicates, "
        "sign share over the listener replicates, with importances",
        "D1": f"a reason is shown when it is a candidate, positive under the "
        f"model, positive in at least {SIGN_SHARE:.0%} of the listener "
        f"replicates and in the top {K_TOP} in at least "
        f"{TOP_SHARE:.0%}. The twoway replicates give a sensitivity "
        f"run",
    },
}


# Helpers


def _num(x):
    return None if x is None or not math.isfinite(float(x)) else float(x)


def fid(which, level, arm, lam):
    return f"{which}|{int(level)}|{s13b.arm_label(arm)}|{float(lam)!r}"


def m_of(arm, level):
    return s13b2.n_negatives(arm, level)


@contextlib.contextmanager
def relabel_logs(command="python3 -m pipeline.runner --only p13c_stability"):
    """P13's pass lines carry this module's name while it runs them, and
    their advice names command."""

    class Relabel(logging.Filter):
        def filter(self, record):
            if isinstance(record.msg, str):
                record.msg = gr._STAGE_PREFIX.sub(
                    WORK_NAME, record.msg, count=1
                ).replace("python3 -m pipeline.runner --only p13a_fit", command)
            return True

    flt = Relabel()
    log.addFilter(flt)
    try:
        yield
    finally:
        log.removeFilter(flt)


def provenance(model_dir, model="gam_refit"):
    out = gr.provenance(model_dir)
    names = [
        "gam_refit.json",
        "gam_refit.npz",
        "fusion_pairs.npz",
        "fusion_features.npz",
    ]
    if model == "p13d":
        names += list(OUT_FILES) + list(P13D_FILES)
    for name in names:
        p = Path(model_dir) / name
        out[name] = file_sha256(p) if p.exists() else None
    return out


# Inputs


def load_model(d, base):
    """gam_refit's chosen configuration and the final model's level, arm
    and weight, checked. Returns (model dict, its stored selection ranks)."""
    doc = load_json(d / "gam_refit.json")
    if doc is None:
        raise ValueError(
            "gam_refit.json is missing: run python3 -m "
            "pipeline.runner --only gam_refit first"
        )
    if doc.get("version") != gr.VERSION:
        raise ValueError(
            f"gam_refit.json has version {doc.get('version')}, " f"not {gr.VERSION}"
        )
    doc2, _, _, fit = gkd.load_final(d)
    if doc2["params"] != s13b2._params():
        raise ValueError(
            "p13b2_model.json was built with other P13b part 2 " "parameters"
        )
    for key in ("arm", "level", "lambda"):
        if doc["final"][key] != fit[key]:
            raise ValueError(
                f"gam_refit.json's final model has another {key} "
                f"than p13b2_model.json's"
            )
    ch = doc["chosen"]
    i = int(ch["index"])
    rec = doc["fits"][i]
    if rec["name"] != ch["name"] or list(rec["kept"]) != list(ch["terms"]):
        raise ValueError("gam_refit.json's chosen record differs from its " "choice")
    arr = gkd.load_npz(d / "gam_refit.npz")
    if not np.array_equal(np.asarray(arr["user_id"]), base["sel_users"]):
        raise ValueError(
            "gam_refit.npz holds other listeners than P13a's " "selection sample"
        )
    lay, idx = gr.layout_of(base, rec["kept"])
    if [t.to_json() for t in lay.terms] != rec["terms"] or gam_arms.layout_fingerprint(
        lay
    ) != rec["fingerprint"]:
        raise ValueError(
            "the chosen configuration's layout, rebuilt here, "
            "differs from the stored one"
        )
    theta = np.asarray(rec["theta"], dtype=np.float64)
    if not np.array_equal(lay.tables(theta), np.asarray(rec["tables_raw"])):
        raise ValueError("the chosen configuration's raw tables differ from " "M theta")
    arm = tuple(fit["arm"].split("-"))
    stored = (rec["fit"] if rec["refit"] else rec["p13b2_fit"])["objective"]
    M = {
        "name": rec["name"],
        "kept": list(rec["kept"]),
        "lay": lay,
        "idx": idx,
        "theta": theta,
        "tables": np.asarray(rec["tables"]),
        "center_consts": rec["center_consts"],
        "fingerprint": rec["fingerprint"],
        "objective": float(stored),
        "level": int(fit["level"]),
        "arm": arm,
        "lambda": float(fit["lambda"]),
        "family": [arm] + [a for a in ALL_ARMS if a != arm],
    }
    if arm not in ALL_ARMS:
        raise ValueError(
            f"the model's arm {fit['arm']} is not one of the " f"family's arms"
        )
    return M, np.asarray(arr["ranks"][i])


def layout_of(base, kept, bounded=()):
    """gam_refit.layout_of() with monotone = +1 on the terms in bounded and
    their zero states ordered against the first knot
    (gam_arms.order_zero_states), as in the monotone arm of P13b
    (stage_p13b._cell_layout). The missing states and the other terms
    stay free. A bounded term must be an affinity term without multiplier,
    tiers or centering, as there."""
    lay, idx = gr.layout_of(base, kept)
    bounded = tuple(bounded)
    if not bounded:
        return lay, idx
    names = [t.name for t in lay.terms]
    unknown = [n for n in bounded if n not in names]
    if unknown:
        raise ValueError(
            f"the bounded terms {unknown} are not in the " f"configuration"
        )
    terms = []
    for t in lay.terms:
        if t.name in bounded:
            if (
                t.source not in s13b.AFFINITY
                or t.multiplier is not None
                or t.centered
                or t.tiers is not None
            ):
                raise ValueError(
                    f"term {t.name} cannot carry P13b's " f"monotone bounds"
                )
            t = dataclasses.replace(t, monotone=1)
        terms.append(t)
    blay = gam.Layout(
        terms,
        {n: base["n_zero"][n] for n in names},
        {n: base["n_miss"][n] for n in names},
        [n for n in base["neutral"] if n in names],
        {n: v for n, v in base["data_means"].items() if n in names},
        {n: v for n, v in base["dev_means"].items() if n in names},
    )
    gam_arms.order_zero_states(blay)
    return blay, idx


def load_p13d_model(d, base):
    """The model record of p13d.json (P13d's rules S1 and S2), checked as
    load_model() checks gam_refit's record. Returns (model dict, its ranks
    in P13d's pass, its tag)."""
    doc = load_json(d / P13D_FILES[0])
    if doc is None:
        raise ValueError(
            "p13d.json is missing: run python3 -m "
            "pipeline.runner --only p13d_bounds first"
        )
    if doc.get("version") != P13D_VERSION:
        raise ValueError(
            f"p13d.json has version {doc.get('version')}, not " f"{P13D_VERSION}"
        )
    rec = doc["model"]
    lay, idx = layout_of(base, rec["kept"], rec["bounded"])
    if [t.to_json() for t in lay.terms] != rec["terms"] or gam_arms.layout_fingerprint(
        lay
    ) != rec["fingerprint"]:
        raise ValueError(
            "p13d.json's model layout, rebuilt here, differs " "from the stored one"
        )
    theta = np.asarray(rec["theta"], dtype=np.float64)
    if not np.array_equal(lay.tables(theta), np.asarray(rec["tables_raw"])):
        raise ValueError("p13d.json's model has raw tables other than M " "theta")
    z = gkd.load_npz(d / P13D_FILES[1])
    names = [str(x) for x in z["names"].tolist()]
    if rec["fit_name"] not in names:
        raise ValueError(f"p13d_model_set.npz holds no fit " f"{rec['fit_name']}")
    i = names.index(rec["fit_name"])
    if not np.array_equal(np.asarray(z["theta"][i]), theta):
        raise ValueError(
            "p13d.json's model parameters differ from " "p13d_model_set.npz's"
        )
    if not np.array_equal(np.asarray(z["user_id"]), base["sel_users"]):
        raise ValueError(
            "p13d_model_set.npz holds other listeners than " "P13a's selection sample"
        )
    arm = tuple(rec["arm"].split("-"))
    if arm not in ALL_ARMS:
        raise ValueError(
            f"the model's arm {rec['arm']} is not one of the " f"family's arms"
        )
    M = {
        "name": rec["name"],
        "kept": list(rec["kept"]),
        "lay": lay,
        "idx": idx,
        "theta": theta,
        "tables": np.asarray(rec["tables"]),
        "center_consts": rec["center_consts"],
        "fingerprint": rec["fingerprint"],
        "objective": float(rec["objective"]),
        "level": int(rec["level"]),
        "arm": arm,
        "lambda": float(rec["lambda"]),
        "family": [arm] + [a for a in ALL_ARMS if a != arm],
        "source": {
            "file": P13D_FILES[0],
            "fit": rec["fit_name"],
            "bounded": list(rec["bounded"]),
            "sha256": file_sha256(d / P13D_FILES[0]),
        },
    }
    return M, np.asarray(z["ranks"][i]), rec["tag"]


def out_names(tag=None):
    """((json, npz), report): the first run's names, or the model switch's
    names with the tag."""
    if tag is None:
        return OUT_FILES, REPORT_NAME
    return (
        (f"p13c_{tag}_model.json", f"p13c_{tag}_arrays.npz"),
        f"p13c_{tag}_report.md",
    )


def theta_of(rec, M):
    if rec.get("fingerprint") != M["fingerprint"]:
        raise ValueError("a saved fit belongs to another layout")
    if rec["proj_grad_max"] > s13.KKT_TOL:
        raise ValueError(
            f"a saved fit stopped at a largest projected gradient "
            f"of {rec['proj_grad_max']:.2e}, above KKT_TOL"
        )
    return np.asarray(rec["theta"], dtype=np.float64)


# Weighted objective (bootstrap)


def _weighted(task):
    b, arm, mult = task
    blk = s13b.arm_block(s13b._CTX["blocks"][b], arm)
    return dict(blk, pos_weight=blk["pos_weight"] * mult)


def _task_loss_w(task):
    b, T, D, arm, mult = task
    return gam.block_loss_grad(T, D, _weighted((b, arm, mult)), s13b._CTX["L"])


def _task_hess_w(task):
    b, T, D, arm, mult = task
    with threadpool_limits(1):
        return gam.block_hessian(T, D, _weighted((b, arm, mult)), s13b._CTX["L"])


def weighted_objective(pool, ctx, lay, idx, arm, lam, mults):
    """stage_p13b's objective with the positive weights of block b times
    mults[b]. The tasks run on the blocks that the forked workers hold."""
    L_u, terms_u = ctx["L_u"], ctx["union"]
    off_u = ctx["union_layout"].offsets
    n_blocks = len(s13b._CTX["blocks"])

    def lift(T):
        Tu = np.zeros(L_u, dtype=np.float64)
        Tu[idx] = T
        return Tu, gam.diff_table(Tu, terms_u, off_u)

    def loss_grad(T, D):
        Tu, Du = lift(T)
        loss, g = 0.0, np.zeros(L_u, dtype=np.float64)
        for l_b, g_b in pool.map(
            _task_loss_w, [(b, Tu, Du, arm, mults[b]) for b in range(n_blocks)]
        ):
            loss += l_b
            g += g_b
        return loss, g[idx]

    def hess(T, D):
        Tu, Du = lift(T)
        H = np.zeros((L_u, L_u), dtype=np.float64)
        for h in pool.map(
            _task_hess_w, [(b, Tu, Du, arm, mults[b]) for b in range(n_blocks)]
        ):
            H += h
        return H[np.ix_(idx, idx)]

    return gam.Objective(lay, loss_grad, lam, s13.RIDGE, hess)


def block_positives(inp):
    """Per fit block: the level's listener index and the item of each
    positive, in the block's order. Every positive appears once."""
    lens = np.asarray(inp["lens"], dtype=np.int64)
    starts = np.concatenate(([0], np.cumsum(lens)[:-1]))
    item, kind = np.asarray(inp["item"]), np.asarray(inp["kind"])
    out, n = [], 0
    for blk in s13b._CTX["blocks"]:
        lo, _ = blk["users"]
        rows = int(starts[lo]) + np.asarray(blk["pos_rows"], dtype=np.int64)
        if not np.all(kind[rows] == fp.KIND_POSITIVE):
            raise ValueError(
                "a block's positive rows are not positives of the " "level"
            )
        out.append((int(lo) + np.asarray(blk["pos_user"], dtype=np.int64), item[rows]))
        n += len(rows)
    if n != int(np.count_nonzero(kind == fp.KIND_POSITIVE)):
        raise ValueError("the fit blocks do not hold every positive once")
    return out


def replicate_mults(scheme, b, pos, w_arm, n_listeners, tracks):
    """The positive weight multipliers of one replicate, per block."""
    rng = np.random.default_rng([SEED, SCHEMES.index(scheme), int(b)])
    c_u = np.bincount(
        rng.integers(0, n_listeners, size=n_listeners), minlength=n_listeners
    ).astype(np.float64)
    c_t = None
    if scheme == "twoway":
        c_t = np.bincount(
            rng.integers(0, len(tracks), size=len(tracks)), minlength=len(tracks)
        ).astype(np.float64)
    mults = []
    for u, it in pos:
        m = c_u[u]
        if c_t is not None:
            m = m * c_t[np.searchsorted(tracks, it)]
        mults.append(m)
    tot0 = math.fsum(float(w.sum()) for w in w_arm)
    tot1 = math.fsum(float((w * m).sum()) for w, m in zip(w_arm, mults))
    if not tot1 > 0:
        raise ValueError(f"replicate {b} ({scheme}) has no weight")
    return [m * (tot0 / tot1) for m in mults]


# Fits


def solve(obj, start, label, say):
    """One fit to KKT_TOL. A layout with bounds (a bounded model of the
    model switch, or P13d) fits with gam_arms.fit_bounded(), as
    stage_p13b._fit() does, and must keep its monotone order."""
    t0 = time.time()
    gr.fit_lines(obj, label, t0, say)
    lay = obj.layout
    bounded = any(lo is not None or hi is not None for lo, hi in lay.bounds)
    if bounded:
        res = gam_arms.fit_bounded(
            obj, np.asarray(start, dtype=np.float64), s13b.SOLVER
        )
    else:
        res = gam.fit(obj, np.asarray(start, dtype=np.float64), s13b.SOLVER)
    res["seconds"] = time.time() - t0
    if res["proj_grad_max"] > s13.KKT_TOL:
        raise ValueError(
            f"{label}: the fit stopped at a largest projected "
            f"gradient of {res['proj_grad_max']:.2e}, above "
            f"KKT_TOL ({res['message']}). It is not saved"
        )
    if bounded:
        viol = gam_arms.monotone_violation(lay.tables(res["theta"]), lay)
        if viol > s13b.MONOTONE_TOL:
            raise ValueError(
                f"{label}: the fit breaks its monotone order by "
                f"{viol:.2e}. It is not saved"
            )
        res["monotone_violation"] = float(viol)
    return res


def done_line(label, res):
    return (
        f"{WORK_NAME}: {label}: objective {res['objective']:.8f}, largest "
        f"projected gradient {res['proj_grad_max']:.1e}, "
        f"{res['message']}, {_clock(res['seconds'])}"
    )


def check_model(st, pool, ctx, M):
    """The stored parameters on the rebuilt design: stationary, with the
    stored objective, and the weighted objective with unit multipliers
    gives the same loss and gradient bit for bit."""
    lay, idx, theta = M["lay"], M["idx"], M["theta"]
    obj = st._objective(pool, ctx, lay, idx, M["arm"], M["lambda"])
    loss, smooth, ridge, grad = obj.parts(theta)
    F = loss + obj.lam_smooth * smooth + obj.lam_ridge * ridge
    rel = abs(F - M["objective"]) / max(1.0, abs(F))
    pg = float(np.max(np.abs(gam.projected_gradient(theta, grad, lay.bounds))))
    if pg > s13.KKT_TOL:
        raise ValueError(
            f"the model's stored parameters are not a stationary "
            f"point of the level's objective: largest projected "
            f"gradient {pg:.2e}"
        )
    if rel > s13.OBJ_RTOL:
        raise ValueError(
            f"the model's objective on the rebuilt design differs "
            f"from the stored one by {rel:.2e}"
        )
    ones = [np.ones(len(b["pos_weight"])) for b in s13b._CTX["blocks"]]
    l1, _, _, g1 = weighted_objective(
        pool, ctx, lay, idx, M["arm"], M["lambda"], ones
    ).parts(theta)
    if l1 != loss or not np.array_equal(g1, grad):
        raise ValueError(
            "the weighted objective with unit multipliers differs "
            "from the stage's objective"
        )
    return {
        "theta": theta,
        "objective": F,
        "objective_stored": M["objective"],
        "relative_gap": rel,
        "proj_grad_max": pg,
        "unit_weights": True,
    }


def wide_part(st, base, M, store, n_boot, say):
    """The model's check, the family refits and the replicates at the
    model's level of the wide file, each saved in the store. Returns (check
    record, {arm: theta}, {scheme: (B, P) thetas}, the level's means)."""
    level, lam = M["level"], M["lambda"]
    check_id = "check|" + fid("wide", level, M["arm"], lam)
    fam_ids = {a: fid("wide", level, a, lam) for a in M["family"][1:]}
    boot_ids = {(s, b): f"boot|{s}|{b}" for s in SCHEMES for b in range(n_boot[s])}
    ids = [check_id] + list(fam_ids.values()) + list(boot_ids.values())
    level_id = f"wide:{level}"
    if all(i in store for i in ids) and store.means(level_id) is not None:
        say(
            f"{WORK_NAME}: every fit at level {level} of the wide file is "
            f"saved, so its rows stay on disk"
        )
        means = store.means(level_id)
    else:
        t0 = time.time()
        say(
            f"{WORK_NAME}: loading level {level} of the wide file (no progress "
            f"lines in this step)"
        )
        ctx = st._level_context(base, "wide", level, store)
        say(
            f"{WORK_NAME}: level {level}: {ctx['rows']:,} rows, "
            f"{ctx['listeners']:,} listeners, loaded in "
            f"{_clock(time.time() - t0)}. Building the design (no progress "
            f"lines in this step)"
        )
        t1 = time.time()
        with st._design_for(ctx, store, ids) as pool:
            if pool is not None:
                say(f"{WORK_NAME}: design ready in {_clock(time.time() - t1)}")
            if check_id not in store:
                rec = check_model(st, pool, ctx, M)
                store.put(check_id, rec)
                say(
                    f"{WORK_NAME}: the model is a stationary point of the "
                    f"level's objective (largest projected gradient "
                    f"{rec['proj_grad_max']:.1e}, objective gap "
                    f"{rec['relative_gap']:.1e}), and the weighted objective "
                    f"with unit weights equals the stage's"
                )
            todo = [a for a in M["family"][1:] if fam_ids[a] not in store]
            for k, a in enumerate(todo, start=1):
                obj = st._objective(pool, ctx, M["lay"], M["idx"], a, lam)
                label = (
                    f"family refit {k} of {len(todo)}, arm "
                    f"{s13b.arm_label(a)} (m = {m_of(a, level)})"
                )
                res = solve(obj, M["theta"], label, say)
                res["fingerprint"] = M["fingerprint"]
                store.put(fam_ids[a], res)
                say(done_line(label, res))
            todo_b = [k for k in boot_ids if boot_ids[k] not in store]
            if todo_b:
                inp = st._inputs()
                pos = block_positives(inp)
                w_arm = [
                    np.asarray(
                        s13b.arm_block(blk, M["arm"])["pos_weight"], dtype=np.float64
                    )
                    for blk in s13b._CTX["blocks"]
                ]
                tracks = np.unique(np.concatenate([it for _, it in pos]))
                n_u = int(ctx["listeners"])
                if max(int(u.max()) for u, _ in pos if len(u)) >= n_u:
                    raise ValueError(
                        "a positive's listener index lies outside "
                        "the level's listeners"
                    )
                say(
                    f"{WORK_NAME}: {len(todo_b)} bootstrap replicates to fit "
                    f"({n_u:,} listeners, {len(tracks):,} distinct tracks "
                    f"among the positives)"
                )
                prog = Progress(
                    f"{WORK_NAME}: bootstrap replicates", len(todo_b), say, unit=" fits"
                )
                for s, b in todo_b:
                    mults = replicate_mults(s, b, pos, w_arm, n_u, tracks)
                    obj = weighted_objective(
                        pool, ctx, M["lay"], M["idx"], M["arm"], lam, mults
                    )
                    label = f"replicate {b + 1} of {n_boot[s]} ({s})"
                    res = solve(obj, M["theta"], label, say)
                    res["fingerprint"] = M["fingerprint"]
                    store.put(boot_ids[(s, b)], res)
                    say(done_line(label, res))
                    prog.advance(1)
                prog.finish()
        st._ensure_means(ctx, store)
        means = ctx["means_ref"]
        del ctx
    st._drop_inputs()
    fam = {M["arm"]: M["theta"]}
    fam.update({a: theta_of(store.get(fam_ids[a]), M) for a in M["family"][1:]})
    boot = {
        s: np.array(
            [theta_of(store.get(boot_ids[(s, b)]), M) for b in range(n_boot[s])]
        ).reshape(n_boot[s], len(M["theta"]))
        for s in SCHEMES
    }
    check = {k: v for k, v in store.get(check_id).items() if k != "theta"}
    return check, fam, boot, means


def ref_part(st, base, M, store, say):
    """The four arms at the largest reference level and at the model's
    level, on the reference listeners. Returns {(level, arm): theta}."""
    lam = M["lambda"]
    out = {}
    for k in sorted({max(s13b2.REF_LEVELS), M["level"]}, reverse=True):
        ids = {a: fid("ref", k, a, lam) for a in ALL_ARMS}
        todo = [a for a in ALL_ARMS if ids[a] not in store]
        if todo:
            t0 = time.time()
            say(
                f"{WORK_NAME}: loading level {k} of the reference file (no "
                f"progress lines in this step)"
            )
            ctx = st._level_context(base, "ref", k, store)
            say(
                f"{WORK_NAME}: reference level {k}: {ctx['rows']:,} rows, "
                f"{ctx['listeners']:,} listeners, loaded in "
                f"{_clock(time.time() - t0)}"
            )
            with st._design_for(ctx, store, list(ids.values())) as pool:
                for j, a in enumerate(todo, start=1):
                    obj = st._objective(pool, ctx, M["lay"], M["idx"], a, lam)
                    label = (
                        f"reference level {k}, fit {j} of {len(todo)}, "
                        f"arm {s13b.arm_label(a)} (m = {m_of(a, k)})"
                    )
                    res = solve(obj, M["theta"], label, say)
                    res["fingerprint"] = M["fingerprint"]
                    store.put(ids[a], res)
                    say(done_line(label, res))
            del ctx
        out.update({(k, a): theta_of(store.get(ids[a]), M) for a in ALL_ARMS})
    st._drop_inputs()
    return out


# Shapes, bands


def centered(M, theta, means):
    Tc, _ = gam.center(
        M["lay"].tables(theta),
        M["lay"],
        {t.name: means[t.name] for t in M["lay"].terms},
    )
    return Tc


def value_array(M, thetas, means, shape_q):
    """(n, J, 7) centered values at POINTS (NaN where a term has none)."""
    lay = M["lay"]
    out = np.full((len(thetas), len(lay.terms), len(POINTS)), np.nan)
    for i, th in enumerate(thetas):
        Tc = centered(M, th, means)
        for j, (t, off) in enumerate(zip(lay.terms, lay.offsets)):
            kn = t.knot_array
            xs = shape_q.get(t.name) or []
            for p, x in enumerate(xs[:5]):
                out[i, j, p] = np.interp(
                    min(max(float(x), kn[0]), kn[-1]), kn, Tc[off : off + t.K]
                )
            if t.zero_state:
                out[i, j, 5] = Tc[off + t.K + gam.ZERO]
            if t.missing_state:
                out[i, j, 6] = Tc[off + t.K + gam.MISS]
    return out


def band_rows(names, full, reps):
    """C1 per term: [{point, value, <scheme>: {lo, hi, sd}, sign_share}]."""
    out = {}
    for j, name in enumerate(names):
        rows = []
        for p, label in enumerate(POINTS):
            v = full[j, p]
            if not np.isfinite(v):
                continue
            row = {"point": label, "value": float(v)}
            for s, r in reps.items():
                x = r[:, j, p]
                lo, hi = np.quantile(x, BAND)
                row[s] = {
                    "lo": float(lo),
                    "hi": float(hi),
                    "sd": _num(x.std(ddof=1)) if len(x) > 1 else None,
                }
            row["sign_share"] = (
                None
                if abs(v) <= ZERO_TOL
                else float(np.mean(np.sign(reps["listener"][:, j, p]) == np.sign(v)))
            )
            rows.append(row)
        out[name] = rows
    return out


def reference_summary(M, ref, shape_q):
    """A1 and A2 from the reference fits."""
    tj = [t.to_json() for t in M["lay"].terms]
    top = max(s13b2.REF_LEVELS)
    pts = {
        (k, a): s13b2.shape_points(tj, M["lay"].tables(th).tolist(), shape_q)
        for (k, a), th in ref.items()
    }
    tops = [pts[(top, a)] for a in ALL_ARMS]
    g_ref = max(
        s13b2.points_gap(x, y) for i, x in enumerate(tops) for y in tops[i + 1 :]
    )
    R, fin = pts[(top, REF_ARM)], pts[(M["level"], M["arm"])]
    bias, flagged = {}, []
    for t in M["lay"].terms:
        if t.name not in R:
            continue
        labels = (
            list(POINTS[:5])
            + (["zero"] if t.zero_state else [])
            + (["missing"] if t.missing_state else [])
        )
        b = [float(x - y) for x, y in zip(fin[t.name], R[t.name])]
        bias[t.name] = dict(zip(labels, b))
        flagged += [[t.name, lab] for lab, v in zip(labels, b) if abs(v) > TAU]
    largest = max((abs(v) for d in bias.values() for v in d.values()), default=0.0)
    gaps = {
        s13b.arm_label(a): s13b2.points_gap(pts[(M["level"], a)], R) for a in ALL_ARMS
    }
    return {
        "top_level": top,
        "level": M["level"],
        "g_ref": g_ref,
        "converged": bool(g_ref <= TAU),
        "bias": bias,
        "largest_bias": largest,
        "flagged": flagged,
        "member_gaps": gaps,
        "points": {f"{k}|{s13b.arm_label(a)}": p for (k, a), p in pts.items()},
    }


def near_optimal(scores, family):
    """B1 from the per-listener scores of the family (the model first)."""
    means = scores.mean(axis=1)
    best = float(means.max())
    thr = SESOI * best
    rows = []
    for c, a in enumerate(family):
        rows.append(
            {
                "arm": s13b.arm_label(a),
                "score": float(means[c]),
                "diff_to_model": (
                    gr.paired_scores(scores[c], scores[0]) if c else None
                ),
                "member": bool(means[c] >= best - thr),
            }
        )
    flag = [
        r["arm"]
        for r in rows[1:]
        if r["diff_to_model"]["mean"] > SESOI * means[0]
        and r["diff_to_model"]["lo"] > 0
    ]
    return {
        "best": best,
        "threshold": thr,
        "rows": rows,
        "members": [r["arm"] for r in rows if r["member"]],
        "flagged": flag,
    }


# Reasons


def reason_sample(d, base):
    """P11's val picks of the selection listeners and their p_pop draws,
    encoded for the union terms."""
    pp, pf = d / "fusion_pairs.npz", d / "fusion_features.npz"
    zp, zf = np.load(pp), np.load(pf)
    if str(zf["pairs_sha256"]) != ev.npz_hash(pp):
        raise ValueError("fusion_features.npz was built from other pairs")
    split = np.asarray(zp["split"])
    indptr = np.asarray(zp["indptr"], dtype=np.int64)
    kind, item = np.asarray(zp["kind"]), np.asarray(zp["item"])
    sel = np.flatnonzero(split == fp.SPLIT_SELECTION)
    users = np.asarray(zp["user_id"])[sel]
    if not np.all(np.isin(users, base["sel_users"])):
        raise ValueError(
            "P11 holds selection listeners outside P13a's " "selection sample"
        )
    lens = indptr[sel + 1] - indptr[sel]
    off = np.repeat(np.cumsum(lens) - lens, lens)
    rows = np.repeat(indptr[sel], lens) + (np.arange(int(lens.sum())) - off)
    lst = np.repeat(np.arange(len(sel)), lens)
    k = kind[rows]
    keep = (k == fp.KIND_POSITIVE) | (k == fp.KIND_POP)
    rows, lst, k = rows[keep], lst[keep], k[keep]
    f = {
        "values": zf["values"][rows],
        "has": zf["has"][rows],
        "match_tier": zf["match_tier"][rows],
    }
    slot, w = s13.encode_rows(base["union"], f, np.arange(len(rows)))
    is_pos = k == fp.KIND_POSITIVE
    n_pop = np.bincount(lst[~is_pos], minlength=len(sel))
    if np.any(n_pop == 0):
        raise ValueError("a selection listener has no p_pop draw in P11")
    return {
        "users": users,
        "lst": lst,
        "is_pos": is_pos,
        "item": item[rows],
        "slot": slot,
        "w": w,
        "n_pop": n_pop,
    }


def contributions(base, Tu, S):
    """(J_union, picks) contributions against each listener's p_pop draws,
    and each term's importance (mean absolute contribution over the
    draws)."""
    terms, offs = base["union"], base["union_layout"].offsets
    D = gam.diff_table(Tu, terms, offs)
    pos, lst = S["is_pos"], S["lst"]
    phi = np.empty((len(terms), int(pos.sum())))
    imp = np.empty(len(terms))
    for j, (t, o) in enumerate(zip(terms, offs)):
        v = gam.pwl_value(Tu, D, int(o) + S["slot"][j].astype(np.int64), S["w"][j])
        bg = (
            np.bincount(lst[~pos], weights=v[~pos], minlength=len(S["n_pop"]))
            / S["n_pop"]
        )
        phi[j] = v[pos] - bg[lst[pos]]
        imp[j] = float(np.mean(np.abs(v[~pos] - bg[lst[~pos]])))
    return phi, imp


def reason_groups(kept):
    art = tuple(t for t in ARTIST if t in kept)
    return ([("artist", art)] if art else []) + [
        (t, (t,)) for t in kept if t not in ARTIST
    ]


def ranks_of(val, cand):
    """Rank of each candidate reason, 1 = largest contribution, ties by the
    reason order. Non-candidates get G + 1."""
    G = val.shape[0]
    r = np.ones(val.shape, dtype=np.int64)
    for g in range(G):
        for h in range(G):
            if h != g:
                r[g] += cand[h] & ((val[h] > val[g]) | ((val[h] == val[g]) & (h < g)))
    r[~cand] = G + 1
    return r


class Reasons:
    """The reason values of any parameter vector of the model on the
    sample."""

    def __init__(self, base, M, S):
        self.base, self.M, self.S = base, M, S
        names = [t.name for t in base["union"]]
        self.jx = {n: j for j, n in enumerate(names)}
        self.groups = reason_groups(M["kept"])
        pos = S["is_pos"]
        on = np.stack([S["slot"][self.jx[t.name]][pos] < t.K for t in base["union"]])
        self.cand = np.stack(
            [on[[self.jx[n] for n in g]].any(axis=0) for _, g in self.groups]
        )

    def values(self, Tc):
        """(G, picks) reason values and the model terms' importances."""
        Tu = np.zeros(self.base["L_u"], dtype=np.float64)
        Tu[self.M["idx"]] = Tc
        phi, imp = contributions(self.base, Tu, self.S)
        val = np.stack(
            [phi[[self.jx[n] for n in g]].sum(axis=0) for _, g in self.groups]
        )
        return val, {t.name: float(imp[self.jx[t.name]]) for t in self.M["lay"].terms}

    def stable(self, val_full, vals, B):
        """D1 over B replicate reason values (an iterator)."""
        pos_n = np.zeros(self.cand.shape, dtype=np.int64)
        top_n = np.zeros(self.cand.shape, dtype=np.int64)
        n = 0
        for vb in vals:
            pos_n += self.cand & (vb > 0)
            top_n += self.cand & (ranks_of(vb, self.cand) <= K_TOP)
            n += 1
        if n != B:
            raise ValueError(f"{n} replicate values, expected {B}")
        shown = (
            self.cand
            & (val_full > 0)
            & (pos_n >= math.ceil(SIGN_SHARE * B - 1e-9))
            & (top_n >= math.ceil(TOP_SHARE * B - 1e-9))
        )
        return shown, pos_n, top_n

    def signed_top(self, val):
        top = self.cand & (ranks_of(val, self.cand) <= K_TOP)
        return top, val > 0


def reason_summary(R, val_full, shown, shown2):
    c = R.cand
    top3 = c & (ranks_of(val_full, c) <= K_TOP) & (val_full > 0)
    rows = []
    for g, (name, terms) in enumerate(R.groups):
        cg, sg = c[g], shown[g]
        rows.append(
            {
                "reason": name,
                "terms": list(terms),
                "candidate": float(cg.mean()),
                "top3_model": float(top3[g].mean()),
                "shown": float(sg.mean()),
                "shown_twoway": float(shown2[g].mean()),
                "mean_abs_candidate": (
                    _num(np.abs(val_full[g][cg]).mean()) if cg.any() else None
                ),
                "mean_shown": _num(val_full[g][sg].mean()) if sg.any() else None,
            }
        )
    n = shown.sum(axis=0)
    return {
        "picks": int(c.shape[1]),
        "rows": rows,
        "n_shown": {str(k): float(np.mean(n == k)) for k in range(K_TOP + 1)},
        "n_top3_model": {
            str(k): float(np.mean(top3.sum(axis=0) == k)) for k in range(K_TOP + 1)
        },
    }


def agreement(a, b):
    (ta, sa), (tb, sb) = a, b
    return ((ta == tb) & (~ta | (sa == sb))).all(axis=0)


def run(
    st,
    model_dir,
    work,
    say=print,
    n_verify=VERIFY_USERS,
    inputs=None,
    n_boot=None,
    model="gam_refit",
):
    """P13c (module docstring). st: a StageP13b2 whose loaders give the
    levels, the item side, the CSR and the slots. model: "gam_refit" (the
    default) or "p13d" (the model switch). Returns the output doc."""
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}, not {model!r}")
    prev = gr.WORK_NAME
    gr.WORK_NAME = WORK_NAME  # labels of gam_refit's fit lines
    try:
        return _run(st, model_dir, work, say, n_verify, inputs, n_boot, model)
    finally:
        gr.WORK_NAME = prev


def _run(st, model_dir, work, say, n_verify, inputs, n_boot, model):
    n_boot = dict(n_boot or N_BOOT)
    d, work, t0 = Path(model_dir), Path(work), time.time()
    base = st._base_context()
    users, picks = base["sel_users"], base["sel_picks"]
    if model == "p13d":
        M, stored_ranks, tag = load_p13d_model(d, base)
    else:
        (M, stored_ranks), tag = load_model(d, base), None
    files, report_name = out_names(tag)
    work.mkdir(parents=True, exist_ok=True)
    key = {
        "version": VERSION,
        "gam_refit": file_sha256(d / "gam_refit.json"),
        "p13b2_key": st._key(),
        "n_boot": n_boot,
        "seed": SEED,
        "arms": [s13b.arm_label(a) for a in ALL_ARMS],
        "ref_levels": [max(s13b2.REF_LEVELS), M["level"]],
    }
    if tag is not None:
        key["p13d"] = M["source"]["sha256"]
    store = s13b2._Store(work / "fits.json", s13._sha(key))
    say(
        f"{WORK_NAME}: model '{M['name']}' ({len(M['kept'])} terms), arm "
        f"{s13b.arm_label(M['arm'])}, level {M['level']}, lambda "
        f"{M['lambda']:.3g}. Fits: 3 family refits, "
        f"{sum(n_boot.values())} replicates, 8 reference fits. Saved work: "
        f"{work}"
    )
    if tag is not None:
        say(
            f"{WORK_NAME}: the model switch: {M['source']['fit']} from "
            f"p13d.json, bounded terms: "
            f"{', '.join(M['source']['bounded']) or 'none'}. Outputs: "
            f"{files[0]}, {files[1]} and reports/{report_name}"
        )
    t1 = time.time()
    check, fam, boot, means = wide_part(st, base, M, store, n_boot, say)
    Tc_model = centered(M, M["theta"], means)
    gap = float(np.max(np.abs(Tc_model - M["tables"])))
    if gap > s13b2.CENTER_TOL:
        raise ValueError(
            f"the model's stored centered tables differ from a "
            f"centering on the level's p_pop negatives by "
            f"{gap:.2e}"
        )
    check["centering_gap"] = gap
    ref = ref_part(st, base, M, store, say)
    fit_secs = time.time() - t1

    say(
        f"{WORK_NAME}: the fits are ready. Selection pass over the "
        f"{len(M['family'])} family members"
    )
    fam_tables = np.zeros((len(M["family"]), base["L_u"]))
    for c, a in enumerate(M["family"]):
        fam_tables[c, M["idx"]] = M["tables"] if c == 0 else centered(M, fam[a], means)
    pkey = {
        "version": VERSION,
        "stage": "pass",
        "p13a_params": s13._params(),
        "code": {
            Path(m.__file__).name: file_sha256(Path(m.__file__))
            for m in (ct, cn, cd, gam, s13)
        },
    }
    t2 = time.time()
    stage_name = "p13c_stability" if tag is None else "p13c_final"
    with relabel_logs(f"python3 -m pipeline.runner --only {stage_name}"):
        ranks, scored = st._selection_pass(
            base["union"], fam_tables, users, picks, base["m_sel"], work / "pass", pkey
        )
    pass_secs = time.time() - t2
    if not np.array_equal(ranks[0], stored_ranks):
        raise ValueError(
            f"the model's ranks differ from "
            f"{'gam_refit' if tag is None else 'P13d'}'s stored "
            f"ranks in {int(np.count_nonzero(ranks[0] != stored_ranks))} "
            f"picks"
        )
    dense = gr.verify_dense(st, base, fam_tables, ranks, say, n_verify)
    st._cache.pop("space", None)
    st._cache.pop("sources", None)

    say(f"{WORK_NAME}: reason sample from P11's selection rows")
    S = reason_sample(d, base)
    R = Reasons(base, M, S)
    out, arrays = summarize(base, M, fam, boot, ref, means, ranks, R)
    out.update(
        version=VERSION,
        protocol=(
            PROTOCOL
            if tag is None
            else dict(
                PROTOCOL,
                model="p13d.json's model record (P13d's rules S1 and "
                "S2) at its level, arm and penalty weight",
            )
        ),
        model=model_export(M),
        check={
            "model": check,
            ("ranks_equal_gam_refit" if tag is None else "ranks_equal_p13d"): True,
            "dense": dense,
        },
        pass_info={
            "members": len(M["family"]),
            "listeners": int(len(users)),
            "mean_scored_share": float(np.mean(scored) / int(st._id_slots()[0])),
        },
        inputs=dict(inputs or {}),
        exports={
            files[0]: "model.table_dict: the centered tables "
            "of each term (knots, heights, zero, missing) for P14",
            files[1]: "theta_boot_listener and "
            "theta_boot_twoway rebuild the replicate tables with "
            + (
                "gam_refit.layout_of(base, model.kept)"
                if tag is None
                else "p13c.layout_of(base, model.kept, the bounded terms "
                "of p13d.json's model)"
            )
            + ", for P14's stable-reason rule",
        },
        seconds={"fits_this_run": fit_secs, "pass_this_run": pass_secs},
    )
    if tag is not None:
        out["model_source"] = M["source"]
    arrays.update(
        user_id=users,
        ranks=ranks.astype(np.int32),
        scored=scored.astype(np.int64),
        theta_model=M["theta"],
        pick_user=S["users"][S["lst"][S["is_pos"]]],
        pick_item=S["item"][S["is_pos"]],
    )
    gr._atomic_savez(d / files[1], **arrays)
    atomic_json_write(out, d / files[0])
    check_outputs(d, base, M, R, means, files)
    out["seconds"]["total_this_run"] = time.time() - t0
    atomic_json_write(out, d / files[0])
    write_report(out, d / "reports" / report_name)
    say(
        f"{WORK_NAME}: largest sampler bias {out['reference']['largest_bias']:.3f}"
        f" log-odds points, near-optimal set "
        f"{out['near_optimal']['members']}, done in {_clock(time.time() - t0)}"
    )
    return out


def model_export(M):
    lay = M["lay"]
    return {
        "name": M["name"],
        "kept": M["kept"],
        "level": M["level"],
        "arm": s13b.arm_label(M["arm"]),
        "m": m_of(M["arm"], M["level"]),
        "lambda": M["lambda"],
        "terms": [t.to_json() for t in lay.terms],
        "fingerprint": M["fingerprint"],
        "theta": M["theta"].tolist(),
        "tables_raw": lay.tables(M["theta"]).tolist(),
        "tables": M["tables"].tolist(),
        "center_consts": M["center_consts"],
        "background": "the level's p_pop negatives, full-fit weights",
        "table_dict": gam.table_dict(M["tables"], lay),
    }


def summarize(base, M, fam, boot, ref, means, ranks, R):
    """Every result from the parameters, the ranks and the reason sample.
    A pure function, so check_outputs() recomputes it."""
    names = [t.name for t in M["lay"].terms]
    sq = base["shape_q"]
    full = value_array(M, [M["theta"]], means, sq)[0]
    reps = {s: value_array(M, list(boot[s]), means, sq) for s in SCHEMES}
    val_full, imp = R.values(M["tables"])

    def vals(s):
        return (R.values(centered(M, th, means))[0] for th in boot[s])

    shown, pos_n, top_n = R.stable(val_full, vals("listener"), len(boot["listener"]))
    shown2, _, _ = R.stable(val_full, vals("twoway"), len(boot["twoway"]))
    scores = ga.per_listener_scores(ranks)
    no = near_optimal(scores, M["family"])
    refs = reference_summary(M, ref, sq)
    for r in no["rows"]:
        r["ref_gap"] = refs["member_gaps"][r["arm"]]
    members = [a for a in M["family"] if s13b.arm_label(a) in no["members"]]
    mv = value_array(M, [fam[a] for a in members], means, sq)
    mimp = [R.values(centered(M, fam[a], means))[1] for a in members]
    mv = np.where(np.isfinite(mv), mv, 0.0)  # absent states: same in all
    rng_v = mv.max(axis=0) - mv.min(axis=0)
    no["ranges"] = {
        n: {
            "values": float(rng_v[j].max()),
            "importance": float(max(m[n] for m in mimp) - min(m[n] for m in mimp)),
        }
        for j, n in enumerate(names)
    }
    ref_top = R.signed_top(val_full)
    no["agreement"] = {
        s13b.arm_label(a): float(
            agreement(
                ref_top, R.signed_top(R.values(centered(M, fam[a], means))[0])
            ).mean()
        )
        for a in members
        if a != M["arm"]
    }
    out = {
        "reference": refs,
        "near_optimal": no,
        "bootstrap": {
            "n": {s: int(len(boot[s])) for s in SCHEMES},
            "bands": band_rows(names, full, reps),
            "importance": imp,
        },
        "reasons": {
            "rule": {
                "k": K_TOP,
                "sign_share": SIGN_SHARE,
                "top_share": TOP_SHARE,
                "replicates": "listener",
            },
            **reason_summary(R, val_full, shown, shown2),
        },
    }
    arrays = {
        "theta_boot_listener": boot["listener"],
        "theta_boot_twoway": boot["twoway"],
        "theta_family": np.array([fam[a] for a in M["family"]]),
        "family_arms": np.array([s13b.arm_label(a) for a in M["family"]]),
        "theta_ref": np.array(
            [ref[k] for k in sorted(ref, key=lambda x: (x[0], ALL_ARMS.index(x[1])))]
        ),
        "ref_keys": np.array(
            [
                f"{k}|{s13b.arm_label(a)}"
                for k, a in sorted(ref, key=lambda x: (x[0], ALL_ARMS.index(x[1])))
            ]
        ),
        "reason_value": val_full,
        "candidate": R.cand,
        "shown": shown,
        "shown_twoway": shown2,
        "positive_count": pos_n,
        "top_count": top_n,
    }
    return out, arrays


def check_outputs(d, base, M, R, means, files=OUT_FILES):
    """Read the outputs back: the tables follow from the stored parameters,
    and every result from the stored parameters and ranks."""
    doc = load_json(d / files[0])
    z = gkd.load_npz(d / files[1])
    theta = np.asarray(doc["model"]["theta"])
    if not (
        np.array_equal(theta, M["theta"])
        and np.array_equal(
            np.asarray(doc["model"]["tables_raw"]), M["lay"].tables(theta)
        )
        and np.array_equal(np.asarray(doc["model"]["tables"]), M["tables"])
    ):
        raise ValueError("the stored model differs from its parameters")
    fam = {
        tuple(a.split("-")): th
        for a, th in zip(z["family_arms"].tolist(), z["theta_family"])
    }
    ref = {
        (int(k.split("|")[0]), tuple(k.split("|")[1].split("-"))): th
        for k, th in zip(z["ref_keys"].tolist(), z["theta_ref"])
    }
    boot = {"listener": z["theta_boot_listener"], "twoway": z["theta_boot_twoway"]}
    again, arr = summarize(base, M, fam, boot, ref, means, z["ranks"], R)
    for k in ("reference", "near_optimal", "bootstrap", "reasons"):
        if again[k] != doc[k]:
            raise ValueError(
                f"the stored results '{k}' do not follow from the "
                f"stored parameters and ranks"
            )
    for k in ("reason_value", "shown", "shown_twoway", "candidate"):
        if not np.array_equal(arr[k], z[k]):
            raise ValueError(
                f"the stored array {k} does not follow from the " f"stored parameters"
            )


# Report


def _f(x, fmt="+.3f"):
    return "" if x is None else format(x, fmt)


def write_report(out, path):
    m, rf, no = out["model"], out["reference"], out["near_optimal"]
    bs, rs = out["bootstrap"], out["reasons"]
    L = [
        f"# P13c: stability and exports of the chosen GAM",
        "",
        f"- model: '{m['name']}' ({len(m['kept'])} terms), arm {m['arm']}, "
        f"level {m['level']} (m = {m['m']}), lambda {m['lambda']:.3g}. "
        f"Protocol: the docstring of p13c.py, fixed {DECLARED}.",
        f"- checks: the model is a stationary point of its level's objective "
        f"(largest projected gradient {out['check']['model']['proj_grad_max']:.1e}, "
        f"objective gap {out['check']['model']['relative_gap']:.1e}, centering "
        f"gap {out['check']['model']['centering_gap']:.1e}). Every fit reached "
        f"KKT_TOL. The model's ranks equal "
        f"{'P13d' if 'ranks_equal_p13d' in out['check'] else 'gam_refit'}'s. "
        f"{out['check']['dense']['listeners']} listeners matched the dense "
        f"reference ({out['check']['dense']['near_ties']} near ties). The "
        f"results follow from the stored outputs.",
        f"- time in this run: fits {_clock(out['seconds']['fits_this_run'])}, pass "
        f"{_clock(out['seconds']['pass_this_run'])}, total "
        f"{_clock(out['seconds']['total_this_run'])}.",
        "",
        "## A. Sampler bias",
        "",
        f"A1: the four fits at reference level {rf['top_level']} agree within "
        f"{rf['g_ref']:.3f} log-odds points (TAU {TAU}): "
        f"{'R is converged' if rf['converged'] else 'R is not converged, so table A2 is descriptive'}.",
        "",
        f"A2: explanation points of the model's configuration on the "
        f"reference listeners minus R, in log-odds points. Largest: "
        f"{rf['largest_bias']:.3f}.",
        "",
        "| term | p5 | p25 | p75 | p95 | zero | missing |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    if out.get("model_source"):
        s = out["model_source"]
        L.insert(
            3,
            f"- model source: {s['file']}, fit {s['fit']}, bounded "
            f"terms: {', '.join(s['bounded']) or 'none'}. P13d's rules "
            f"S1 and S2 chose it.",
        )
    for t, b in rf["bias"].items():
        L.append(
            f"| {t} | "
            + " | ".join(
                _f(b.get(p)) for p in ("p5", "p25", "p75", "p95", "zero", "missing")
            )
            + " |"
        )
    L += [
        "",
        "Flagged above TAU: "
        + (", ".join(f"{t} {p}" for t, p in rf["flagged"]) or "none")
        + ".",
        "",
        "## B. Near-optimal set",
        "",
        f"B1: members within SESOI ({no['threshold']:.5f}) of the family's "
        f"best score ({no['best']:.5f}).",
        "",
        "| arm | score | difference to the model (95% interval) | gap to R | "
        "in the set |",
        "|---|---:|---|---:|---|",
    ]
    for r in no["rows"]:
        L.append(
            f"| {r['arm']} | {r['score']:.5f} | "
            f"{gr._interval(r['diff_to_model']) if r['diff_to_model'] else ''}"
            f" | {r['ref_gap']:.3f} | {'yes' if r['member'] else 'no'} |"
        )
    L += [
        "",
        "Flagged (more than SESOI above the model): "
        + (", ".join(no["flagged"]) or "none")
        + ".",
        "",
        "Across the set, the largest range of each term's centered values "
        "and of its importance, in log-odds points:",
        "",
        "| term | values | importance |",
        "|---|---:|---:|",
    ]
    for t, r in no["ranges"].items():
        L.append(f"| {t} | {r['values']:.3f} | {r['importance']:.3f} |")
    if no["agreement"]:
        L += [
            "",
            "Signed top-3 reasons equal to the model's, share of picks: "
            + ", ".join(f"{a} {v:.1%}" for a, v in no["agreement"].items())
            + ".",
        ]
    L += [
        "",
        "## C. Bootstrap bands",
        "",
        f"{bs['n']['listener']} listener and {bs['n']['twoway']} twoway "
        f"replicates. Centered values in log-odds points, background: the "
        f"level's p_pop negatives. Band: 2.5 to 97.5 percent. Sign: the "
        f"share of listener replicates with the model's sign (blank for a "
        f"value within {ZERO_TOL:g} of 0).",
        "",
        "| term | importance | point | value | listener band | twoway band " "| sign |",
        "|---|---:|---|---:|---|---|---:|",
    ]
    for t, rows in bs["bands"].items():
        for i, r in enumerate(rows):
            L.append(
                f"| {t if i == 0 else ''} | "
                f"{_f(bs['importance'][t], '.3f') if i == 0 else ''} | "
                f"{r['point']} | {r['value']:+.3f} | "
                f"{r['listener']['lo']:+.3f} to {r['listener']['hi']:+.3f}"
                f" | {r['twoway']['lo']:+.3f} to {r['twoway']['hi']:+.3f}"
                f" | {_f(r['sign_share'], '.2f')} |"
            )
    L += [
        "",
        "## D. Stable reasons",
        "",
        f"{rs['picks']:,} val picks of the selection listeners. Rule D1: "
        f"positive in {SIGN_SHARE:.0%} and top {K_TOP} in {TOP_SHARE:.0%} of "
        f"the listener replicates.",
        "",
        "| reason | candidate | model's top 3 | shown | shown, twoway | "
        "mean abs. value | mean value when shown |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rs["rows"]:
        L.append(
            f"| {r['reason']} | {r['candidate']:.1%} | "
            f"{r['top3_model']:.1%} | {r['shown']:.1%} | "
            f"{r['shown_twoway']:.1%} | {_f(r['mean_abs_candidate'], '.3f')}"
            f" | {_f(r['mean_shown'], '+.3f')} |"
        )
    L += [
        "",
        "Reasons per pick, share of picks: "
        + ", ".join(
            f"{k}: {v:.1%} shown ({rs['n_top3_model'][k]:.1%} in the " f"model's top 3)"
            for k, v in rs["n_shown"].items()
        )
        + ".",
        "",
        "## Exports",
        "",
    ]
    L += [f"- {k}: {v}" for k, v in out["exports"].items()]
    atomic_text_write("\n".join(L) + "\n", path)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="P13c: stability and exports of the chosen GAM "
        "(protocol in the module docstring)"
    )
    parser.add_argument(
        "--model",
        choices=MODELS,
        default="gam_refit",
        help="gam_refit (default): gam_refit's chosen model. "
        "p13d: the model record in p13d.json, with "
        "tagged outputs",
    )
    args = parser.parse_args(argv)
    tag = None
    if args.model == "p13d":
        doc = load_json(MODEL_DIR / P13D_FILES[0])
        if doc is None:
            raise SystemExit(
                f"{WORK_NAME}: p13d.json is missing: run "
                f"python3 -m pipeline.runner --only "
                f"p13d_bounds first"
            )
        tag = doc["model"]["tag"]
    log_path = LOG_PATH if tag is None else LOG_PATH.with_name(f"p13c_{tag}.log")
    work = SCRATCH_DIR / (WORK_NAME if tag is None else f"{WORK_NAME}_{tag}")
    gr.setup_logging(log_path)
    say = log.info
    try:
        say(f"{WORK_NAME}: started, process id {os.getpid()}, log file " f"{log_path}")
        run(
            s13b2.StageP13b2(),
            MODEL_DIR,
            work,
            say=say,
            inputs=provenance(MODEL_DIR, args.model),
            model=args.model,
        )
    except KeyboardInterrupt:
        log.info(
            f"{WORK_NAME}: stopped by Ctrl+C. Run it again: it skips the "
            f"saved fits and scores the blocks still missing."
        )
        raise SystemExit(130)
    except Exception:
        log.exception(f"{WORK_NAME}: stopped by an error")
        raise SystemExit(1)


# Runner stages
FINAL_FILE = "p13c_final.json"
FINAL_VERSION = "p13c-final-v1"
POINTER_MODEL = ("fit_name", "tag", "changed", "bounded", "fingerprint")


def stored_means(st, base, M, work):
    """The background means of the model's level of the wide file, for a
    read-back: from the saved work in work when it holds them, else from
    the level's rows. Other means than the run's make the read-back
    fail."""
    state = load_json(Path(work) / "fits.json") or {}
    saved = (state.get("means") or {}).get(f"wide:{M['level']}")
    if saved is not None:
        return {k: np.asarray(v, dtype=np.float64) for k, v in saved.items()}
    ctx = st._level_context(base, "wide", M["level"], None)
    st._ensure_means(ctx, None)
    means = ctx["means_ref"]
    del ctx
    st._drop_inputs()
    return means


def pointer_files(rec):
    """(json, npz, report) of the P13c outputs of p13d.json's model record:
    the tagged outputs when P13d changed the model, else the first run's."""
    files, report = out_names(rec["tag"] if rec["changed"] else None)
    return files[0], files[1], f"reports/{report}"


def write_pointer(d, rec, inputs):
    """p13c_final.json: the P13c exports of P13d's model. P14 reads this
    file."""
    d = Path(d)
    res, arr, rep = pointer_files(rec)
    doc = {
        "version": FINAL_VERSION,
        "model": {k: rec[k] for k in POINTER_MODEL},
        "source": (
            "P13c on p13d.json's model (the model switch)"
            if rec["changed"]
            else "p13c_stability's outputs: " "P13d kept gam_refit's model"
        ),
        "files": {"results": res, "arrays": arr, "report": rep},
        "sha256": {"results": file_sha256(d / res), "arrays": file_sha256(d / arr)},
        "inputs": dict(inputs or {}),
    }
    atomic_json_write(doc, d / FINAL_FILE)
    return doc


def check_pointer(d, rec):
    """Raise ValueError unless p13c_final.json names p13d.json's model and
    points to the P13c outputs of that model, unchanged since."""
    d = Path(d)
    doc = load_json(d / FINAL_FILE)
    if doc is None:
        raise ValueError(f"{FINAL_FILE} is missing")
    if doc.get("version") != FINAL_VERSION:
        raise ValueError(
            f"{FINAL_FILE} has version {doc.get('version')}, " f"not {FINAL_VERSION}"
        )
    if doc.get("model") != {k: rec[k] for k in POINTER_MODEL}:
        raise ValueError(f"{FINAL_FILE} names another model than p13d.json")
    res, arr, rep = pointer_files(rec)
    if doc.get("files") != {"results": res, "arrays": arr, "report": rep}:
        raise ValueError(
            f"{FINAL_FILE} points to other files than the P13c "
            f"outputs of p13d.json's model"
        )
    for key, name in (("results", res), ("arrays", arr)):
        p = d / name
        if not p.exists() or file_sha256(p) != doc["sha256"].get(key):
            raise ValueError(
                f"{name} is missing or changed after " f"{FINAL_FILE} was written"
            )
    m = load_json(d / res)["model"]
    if m["fingerprint"] != rec["fingerprint"] or not np.array_equal(
        np.asarray(m["theta"], dtype=np.float64),
        np.asarray(rec["theta"], dtype=np.float64),
    ):
        raise ValueError(f"{res} holds another model than p13d.json's")
    return doc


class StageP13c(ScriptStage):
    """P13c as two runner stages:
    p13c_stability on gam_refit's model, and p13c_final on P13d's model.
    p13c_final runs the model switch only when P13d changed the model, and
    it always writes the pointer p13c_final.json."""

    def __init__(self, model="gam_refit", model_dir=None, scratch_dir=None):
        super().__init__(model_dir, scratch_dir)
        if model not in MODELS:
            raise ValueError(f"model must be one of {MODELS}, not {model!r}")
        self.model = model
        final = model == "p13d"
        self.name = "p13c_final" if final else "p13c_stability"
        self.upstream = ["p13d_bounds"] if final else ["gam_refit"]
        self.n_boot = None  # N_BOOT
        self.n_verify = VERIFY_USERS

    def _stage(self):
        return s13b2.StageP13b2()

    def _record(self):
        """p13d.json's model record."""
        doc = load_json(self.model_dir / P13D_FILES[0])
        if doc is None:
            raise ValueError(
                "p13d.json is missing: run python3 -m "
                "pipeline.runner --only p13d_bounds first"
            )
        return doc["model"]

    def outputs(self):
        d = self.model_dir
        if self.model == "gam_refit":
            return {
                "results": d / OUT_FILES[0],
                "arrays": d / OUT_FILES[1],
                "report": d / "reports" / REPORT_NAME,
            }
        out = {"pointer": d / FINAL_FILE}
        doc = load_json(d / P13D_FILES[0])
        rec = (doc or {}).get("model") or {}
        if rec.get("changed"):
            res, arr, rep = pointer_files(rec)
            out.update(results=d / res, arrays=d / arr, report=d / rep)
        return out

    def provenance(self):
        return provenance(self.model_dir, self.model)

    def recorded_inputs(self):
        key = "results" if self.model == "gam_refit" else "pointer"
        doc = load_json(self.outputs()[key])
        return None if doc is None else doc.get("inputs")

    def run(self, params, files):
        d, st, inputs = self.model_dir, self._stage(), self.provenance()
        if self.model == "gam_refit":
            with log_file(d / "logs" / LOG_PATH.name):
                run(
                    st,
                    d,
                    self.scratch_dir / WORK_NAME,
                    say=log.info,
                    n_verify=self.n_verify,
                    inputs=inputs,
                    n_boot=self.n_boot,
                )
            return self.output_info()
        rec = self._record()
        if rec["changed"]:
            tag = rec["tag"]
            with log_file(d / "logs" / f"p13c_{tag}.log"):
                run(
                    st,
                    d,
                    self.scratch_dir / f"{WORK_NAME}_{tag}",
                    say=log.info,
                    n_verify=self.n_verify,
                    inputs=inputs,
                    n_boot=self.n_boot,
                    model="p13d",
                )
        else:
            log.info(
                f"{WORK_NAME}: P13d kept gam_refit's model, so "
                f"p13c_final points to p13c_stability's outputs and "
                f"fits nothing"
            )
        doc = write_pointer(d, rec, inputs)
        log.info(
            f"{WORK_NAME}: {FINAL_FILE} points to "
            f"{doc['files']['results']} and {doc['files']['arrays']}"
        )
        return self.output_info()

    def check(self):
        d, st = self.model_dir, self._stage()
        if self.model == "p13d":
            rec = self._record()
            check_pointer(d, rec)
            if not rec["changed"]:
                return
        base = st._base_context()
        if self.model == "gam_refit":
            M, _ = load_model(d, base)
            files, work = OUT_FILES, self.scratch_dir / WORK_NAME
        else:
            M, _, tag = load_p13d_model(d, base)
            files = out_names(tag)[0]
            work = self.scratch_dir / f"{WORK_NAME}_{tag}"
        means = stored_means(st, base, M, work)
        R = Reasons(base, M, reason_sample(d, base))
        check_outputs(d, base, M, R, means, files)


if __name__ == "__main__":
    main()
