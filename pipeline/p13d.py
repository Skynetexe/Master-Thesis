"""P13d: the bounded content shapes, the model, the near-optimal set and
the reason rule S4. The protocol was declared before any result, and the
code was checked against it.

The model is gam_refit's chosen configuration ("without both", six terms
on the real data) at P13c's level of the wide file (m = 64), loss arm
(p_pop form B) and penalty weight lambda (3.16e-8). P13c's four family
arms at lambda come from p13c_arrays.npz. P13d never reads a test pick.

Fits. Ten new fits at the model's level, each to KKT_TOL and each saved:
- free: p_pop-B and mix32-B at lambda / 10 and lambda x 10. Each starts
  from P13c's fit of its arm at lambda.
- bounded: p_pop-B and mix32-B at lambda / 10, lambda and lambda x 10.
  tag_affinity, artist_tag_affinity, genre_affinity and audio_fit get
  P13b's monotone bounds: non-decreasing knot heights, and a zero state
  at or below the first knot (gam_arms.order_zero_states). The missing
  states and the two artist terms stay free. The fit at lambda starts
  from the free fit of its arm (gam_arms.theta_from_tables), the other
  two from the bounded fit of their arm at lambda. gam_arms.fit_bounded()
  fits them, as stage_p13b._fit() does.
The objective is strictly convex (ridge), so a start changes the time
only.

Scores. One selection pass of P13 ranks 12 configurations: the model,
P13c's mix32-B at lambda and the ten new fits. Score: full-catalog
NDCG@100 of the val_last and val_first picks of the selection listeners,
ranks exact up to 100. P13c's two form-A arms keep their stored ranks. A
difference is paired over the listeners, with gam_audit's 95 percent
normal interval. SESOI = 0.02 x the model's score, one fixed number.

Rules, declared before any result:
- S1, the bounds: the bounded p_pop-B at lambda replaces the free model
  when the lower end of the interval of (bounded - free) lies above
  -SESOI.
- S2, the model: the family that S1 keeps is the six free form-B fits
  (P13c's two at lambda and the four new ones) or the six bounded fits.
  Its start is the model, or the bounded p_pop-B at lambda. A fit of the
  family is eligible when it scores more than SESOI above the start, with
  the interval above 0 (R1 of gam_refit). With an eligible fit, R2 of
  gam_refit chooses: gam_audit.one_se_choice() over the start and the
  eligible fits, in P13's order: the strongest penalty first, and at
  equal lambda the model's arm first. With none, the start is the model.
- S3, the near-optimal set: every fit of P13c (its four arms at lambda)
  and of P13d (its ten) whose score lies within SESOI of the best.
- S4, the reasons that P14 shows: a reason that passes D1 of P13c is
  shown only when it is positive under every S3 member and in the top 3
  of at least 80 percent of them, against D1's background (the pick's
  p_pop draws in P11). No minimum. Each shown reason carries its odds
  factor exp(value). If S1 or S2 changes the model, these shares still
  use D1 of the free model, and the report says so. P14 applies D1 from
  the rerun of P13c, then S4.

Descriptive results:
- the shapes, flat shares and order of the bounded fits,
- concurvity as in P13a (gam_audit: worst and estimate of each term
  against the other terms, and the observed concurvity of the free model
  and of the bounded p_pop-B at lambda), on the within-listener Gram
  matrix of the rows of the model's loss: the positives and the p_pop
  draws,
- the signed top-3 agreement of the free model's reasons with each P13c
  fit at the largest reference level and with each S3 member.

Checks:
- the model is a stationary point of the level's objective with its
  stored objective (p13c.check_model), and P13c's mix32-B fit is a
  stationary point too.
- every new fit reaches KKT_TOL before it is saved, and every bounded fit
  keeps its order (monotone_violation at most MONOTONE_TOL).
- the pass gives the model and mix32-B the ranks that P13c stored, and
  the model the ranks of gam_refit.
- VERIFY_USERS listeners get every configuration's ranks again from the
  dense reference of catalog.py.
- D1, rebuilt from P13c's listener replicates, equals P13c's stored D1.
- the script reads its outputs back: the tables follow from the stored
  parameters, and every rule, share and agreement from the stored
  tables, ranks and reason sample.

Outputs: MODEL_DIR/p13d.json (the protocol, the fits, the scores, the
rules, the model record for the model switch of p13c.py, the S4 shares
and the descriptive results), MODEL_DIR/p13d_model_set.npz (every fit's
parameters, centered tables and ranks, the S3 members and the S4 mask,
for P14) and MODEL_DIR/reports/p13d_report.md. Every line goes to the
console and to MODEL_DIR/logs/p13d.log. Long steps print a line at most
every 30 seconds: each fit its Newton steps, the fits their share done
with the time left, and the pass its share done with the time left.

Resume: each fit is saved in SCRATCH_DIR/p13d/fits.json and the pass in
SCRATCH_DIR/p13d/pass, keyed by their inputs. After Ctrl+C, the same
command continues with the fits and blocks still missing.

Pilot: --pilot fits the bounded p_pop-B at lambda only (the S1 fit, the
slowest kind), saves it, prints its time and the projected total, and
stops. The full command continues from it.

Expected run on the real data, about 1 h 45 min: loading the level and
its design about a minute (peak memory near P13c's 18 GB), the six
bounded fits about 5 to 7 minutes each, the four free fits about 3
minutes each, the Gram matrix about a minute, the pass over 12
configurations about 50 minutes and the reasons a few minutes.

Usage, from the repository root, through the runner:
    python3 -m pipeline.runner --only p13d_bounds
The pilot uses the module's own entry, which writes no marker:
    python3 -m pipeline.p13d --pilot
The pass uses P13's worker setting. To use fewer workers:
    P13A_WORKERS=8 python3 -m pipeline.runner --only p13d_bounds
"""

import argparse
import contextlib
import json
import logging
import math
import os
import time
from pathlib import Path

import numpy as np

from pipeline import catalog as ct
from pipeline import concept_data as cd
from pipeline import concepts as cn
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
from pipeline import p13c as pc
from pipeline.script_stage import ScriptStage, log_file

LOG_PATH = MODEL_DIR / "logs" / "p13d.log"
VERSION = "p13d-v1"
WORK_NAME = "p13d"
DECLARED = "before any result"
SESOI = 0.02
CONTENT = ("tag_affinity", "artist_tag_affinity", "genre_affinity", "audio_fit")
FORM_B = (("p_pop", "B"), ("mix32", "B"))
STEP_FACTOR = {"lambda*10": 10.0, "lambda": 1.0, "lambda/10": 0.1}
S4_TOP_SHARE = 0.80
VERIFY_USERS = 6
OUT_FILES = ("p13d.json", "p13d_model_set.npz")
REPORT_NAME = "p13d_report.md"
# The selection pass on the real data takes about a + b x C minutes for C
# configurations (the logs of P13c, gam_refit and P13b part 2), and a free
# fit about half the time of a bounded one (the logs of P13c and P13b
# part 2). The pilot's projection uses both.
PASS_MINUTES = (26.0, 2.0)
FREE_SHARE = 0.5
SECTIONS = ("scores", "sesoi", "s1", "s2", "s3", "s4", "shapes", "agreement")


# Names


def fit_name(family, arm, step):
    return f"{family}|{s13b.arm_label(arm)}|{step}"


def parse(name):
    """(family, arm, step) of a fit name."""
    family, arm, step = name.split("|")
    return family, tuple(arm.split("-")), step


def tag_of(name):
    """A file-name form of a fit name, for the outputs of p13c.py."""
    return name.replace("|", "_").replace("/", "d").replace("*", "x")


def lam_at(lam, step):
    return {"lambda/10": lam / 10.0, "lambda": lam, "lambda*10": lam * 10.0}[step]


def label(name):
    """A fit name for the report (a Markdown table cell holds no "|")."""
    family, arm, step = parse(name)
    step = {"lambda/10": "lambda / 10", "lambda": "lambda", "lambda*10": "lambda x 10"}[
        step
    ]
    return f"{family} {s13b.arm_label(arm)} at {step}" + (
        " (the model)" if name == MODEL else ""
    )


MODEL = fit_name("free", ("p_pop", "B"), "lambda")
MIX = fit_name("free", ("mix32", "B"), "lambda")
PILOT = fit_name("bounded", ("p_pop", "B"), "lambda")
P13C_FITS = [fit_name("free", a, "lambda") for a in pc.ALL_ARMS]
NEW_FITS = (
    [fit_name("bounded", a, "lambda") for a in FORM_B]
    + [fit_name("bounded", a, s) for a in FORM_B for s in ("lambda*10", "lambda/10")]
    + [fit_name("free", a, s) for a in FORM_B for s in ("lambda*10", "lambda/10")]
)
PASS_FITS = [MODEL, MIX] + NEW_FITS
ALL_FITS = P13C_FITS + NEW_FITS
BOUNDED_FITS = [n for n in NEW_FITS if parse(n)[0] == "bounded"]

PROTOCOL = {
    "version": VERSION,
    "declared": DECLARED,
    "source": "the docstring of p13d.py, and its code check",
    "model": "gam_refit's chosen configuration at P13c's level, arm and "
    "penalty weight",
    "bounded_terms": list(CONTENT),
    "sesoi": SESOI,
    "new_fits": NEW_FITS,
    "rules": {
        "S1": "the bounded p_pop-B at lambda replaces the free model when "
        "the lower end of the interval of (bounded - free) lies "
        "above -SESOI",
        "S2": "in the family that S1 keeps, a fit more than SESOI above the "
        "start with the interval above 0 is eligible (R1 of "
        "gam_refit). With an eligible fit, gam_audit.one_se_choice() "
        "over the start and the eligible fits, the strongest penalty "
        "first and at equal lambda the model's arm first (R2 of "
        "gam_refit). Otherwise the start",
        "S3": "every fit of P13c and P13d within SESOI of the best",
        "S4": f"a reason that passes D1 of P13c is shown only when it is "
        f"positive under every S3 member and in the top {pc.K_TOP} "
        f"of at least {S4_TOP_SHARE:.0%} of them",
    },
}


# Helpers


def _num(x):
    return None if x is None or not math.isfinite(float(x)) else float(x)


def _norm(x):
    """x as the json file holds it, for a comparison after a read."""
    return json.loads(json.dumps(x, sort_keys=True, default=str))


def _record(store, rid):
    """A saved record as stored (lists, not arrays)."""
    return dict(store.state["fits"][rid])


@contextlib.contextmanager
def relabel_logs():
    """P13's pass lines carry this script's name while it runs them."""

    class Relabel(logging.Filter):
        def filter(self, record):
            if isinstance(record.msg, str):
                record.msg = gr._STAGE_PREFIX.sub(
                    WORK_NAME, record.msg, count=1
                ).replace(
                    "python3 -m pipeline.runner --only p13a_fit",
                    "python3 -m pipeline.runner --only p13d_bounds",
                )
            return True

    flt = Relabel()
    log.addFilter(flt)
    try:
        yield
    finally:
        log.removeFilter(flt)


def provenance(model_dir):
    out = pc.provenance(model_dir)
    for name in pc.OUT_FILES:
        p = Path(model_dir) / name
        out[name] = file_sha256(p) if p.exists() else None
    return out


def centered(lay, theta, means):
    """(centered tables, centering constants) on the level's background."""
    Tc, consts = gam.center(
        lay.tables(theta), lay, {t.name: means[t.name] for t in lay.terms}
    )
    return Tc, {k: float(v) for k, v in consts.items()}


def lift(base, M, T):
    Tu = np.zeros(base["L_u"], dtype=np.float64)
    Tu[M["idx"]] = T
    return Tu


def done_line(label, res):
    return (
        f"{WORK_NAME}: {label}: objective {res['objective']:.8f}, largest "
        f"projected gradient {res['proj_grad_max']:.1e}, "
        f"{gr._text(res['message'])}, {_clock(res['seconds'])}"
    )


# Inputs


def load_p13c(d, base, M, stored_ranks):
    """P13c's outputs for gam_refit's model, checked: its four family fits
    at lambda, their ranks, D1, the replicates and the reference fits."""
    doc = load_json(d / pc.OUT_FILES[0])
    if doc is None:
        raise ValueError(
            "p13c_model.json is missing: run python3 -m "
            "pipeline.runner --only p13c_stability first"
        )
    if doc.get("version") != pc.VERSION:
        raise ValueError(
            f"p13c_model.json has version {doc.get('version')}, " f"not {pc.VERSION}"
        )
    z = gkd.load_npz(d / pc.OUT_FILES[1])
    m = doc["model"]
    if (
        m["fingerprint"] != M["fingerprint"]
        or list(m["kept"]) != M["kept"]
        or int(m["level"]) != M["level"]
        or m["arm"] != s13b.arm_label(M["arm"])
        or float(m["lambda"]) != M["lambda"]
        or not np.array_equal(np.asarray(m["theta"]), M["theta"])
        or not np.array_equal(np.asarray(z["theta_model"]), M["theta"])
    ):
        raise ValueError(
            "P13c's outputs belong to another model than " "gam_refit's chosen one"
        )
    if not np.array_equal(np.asarray(z["user_id"]), base["sel_users"]):
        raise ValueError(
            "p13c_arrays.npz holds other listeners than P13a's " "selection sample"
        )
    arms = [str(a) for a in z["family_arms"].tolist()]
    want = [s13b.arm_label(a) for a in M["family"]]
    if arms != want:
        raise ValueError(f"P13c's family arms {arms} differ from {want}")
    names = [fit_name("free", tuple(a.split("-")), "lambda") for a in arms]
    fam = {
        n: np.asarray(th, dtype=np.float64) for n, th in zip(names, z["theta_family"])
    }
    ranks = {n: np.asarray(r) for n, r in zip(names, z["ranks"])}
    if not np.array_equal(ranks[MODEL], stored_ranks):
        raise ValueError("P13c's stored ranks of the model differ from " "gam_refit's")
    ref = {
        str(k): np.asarray(th, dtype=np.float64)
        for k, th in zip(z["ref_keys"].tolist(), z["theta_ref"])
    }
    return {"doc": doc, "z": z, "fam": fam, "ranks": ranks, "ref": ref}


# Fits


def check_fit(st, pool, ctx, M, arm, theta):
    """A stored fit of the model's layout is a stationary point of its
    arm's objective on the rebuilt design."""
    obj = st._objective(pool, ctx, M["lay"], M["idx"], arm, M["lambda"])
    loss, smooth, ridge, grad = obj.parts(theta)
    pg = float(np.max(np.abs(gam.projected_gradient(theta, grad, M["lay"].bounds))))
    if pg > s13.KKT_TOL:
        raise ValueError(
            f"P13c's {s13b.arm_label(arm)} fit is not a "
            f"stationary point of the level's objective: largest "
            f"projected gradient {pg:.2e}"
        )
    return {
        "theta": theta,
        "proj_grad_max": pg,
        "objective": loss + obj.lam_smooth * smooth + obj.lam_ridge * ridge,
    }


def warm_start(name, lays, M, P, store):
    """(start, its source) of a new fit (module docstring)."""
    family, arm, step = parse(name)
    own = fit_name("free", arm, "lambda")
    if family == "free":
        return P["fam"][own], own
    if step == "lambda":
        T = M["lay"].tables(P["fam"][own])
        return gam_arms.theta_from_tables(lays["bounded"], T), own
    src = fit_name("bounded", arm, "lambda")
    return store.get(src)["theta"], src


def loss_rows(blk, arm):
    """The rows of one block that an arm's loss uses: the positives and
    the arm's negatives, in row order."""
    cols = s13b.neg_cols(arm[0], blk["neg_rows"].shape[1] // 2)
    return np.unique(
        np.concatenate(
            [
                np.asarray(blk["pos_rows"], dtype=np.int64),
                np.asarray(blk["neg_rows"][:, cols], dtype=np.int64).ravel(),
            ]
        )
    )


def _task_gram(task):
    """P13a's within-listener Gram matrix (stage_p13a.gram_block) of one
    block, on the rows of the arm's loss only."""
    b, arm = task
    blk = s13b._CTX["blocks"][b]
    rows = loss_rows(blk, arm)
    ru = np.asarray(blk["row_user"])[rows]
    sub = {
        "slot": blk["slot"][:, rows],
        "w": blk["w"][:, rows],
        "off": blk["off"],
        "mult": {k: np.asarray(v)[rows] for k, v in (blk.get("mult") or {}).items()},
        "row_user": ru,
        "user_len": np.bincount(ru, minlength=len(blk["user_len"])),
    }
    if np.any(sub["user_len"] == 0):
        raise ValueError(
            f"block {b}: a listener has no row in the loss of "
            f"arm {s13b.arm_label(arm)}"
        )
    return s13.gram_block(sub, s13b._CTX["L"]), int(len(rows))


def gram(pool, arm, L, say):
    n_blocks = len(s13b._CTX["blocks"])
    say(
        f"{WORK_NAME}: concurvity: the Gram matrix of the rows of the "
        f"{s13b.arm_label(arm)} loss, {n_blocks} blocks (one line at the end)"
    )
    t0 = time.time()
    G = np.zeros((L, L), dtype=np.float64)
    n = 0
    for g, k in pool.map(_task_gram, [(b, arm) for b in range(n_blocks)]):
        G += g
        n += k
    say(
        f"{WORK_NAME}: concurvity: Gram matrix of {n:,} rows ready in "
        f"{_clock(time.time() - t0)}"
    )
    return 0.5 * (G + G.T), n


def concurvity_record(G, base, M, tables_u):
    """P13a's measures for the model's terms: each term's table entries
    (knots, zero, missing) against those of the other terms."""
    groups = {
        t.name: [int(base["union_off"][t.name]) + k for k in range(t.K + 2)]
        for t in M["lay"].terms
    }
    full, names, pair = ga.concurvity(G, groups)
    observed = {}
    for label, Tu in tables_u.items():
        o = ga.observed_concurvity(G, groups, {n: Tu[groups[n]] for n in groups})
        observed[label] = {n: float(v) for n, v in o.items()}
    return {
        "terms": {
            n: {
                "worst": float(r["worst"]),
                "estimate": float(r["estimate"]),
                "rank": int(r["rank"]),
            }
            for n, r in full.items()
        },
        "names": list(names),
        "pair": [[float(x) for x in row] for row in pair],
        "observed": observed,
    }


def fit_part(st, base, M, P, lays, fps, store, say, pilot):
    """The model checks, the new fits (only the pilot's with pilot) and
    the Gram matrix, at the model's level, each saved in the store.
    Returns the level's background means."""
    level, lam = M["level"], M["lambda"]
    names = [PILOT] if pilot else list(NEW_FITS)
    ids = ["check|model", "check|mix32-B"] + names + ([] if pilot else ["concurvity"])
    level_id = f"wide:{level}"
    if all(i in store for i in ids) and store.means(level_id) is not None:
        say(
            f"{WORK_NAME}: every fit at level {level} is saved, so its rows "
            f"stay on disk"
        )
        return store.means(level_id)
    t0 = time.time()
    say(
        f"{WORK_NAME}: loading level {level} of the wide file (no progress "
        f"lines in this step)"
    )
    ctx = st._level_context(base, "wide", level, store)
    say(
        f"{WORK_NAME}: level {level}: {ctx['rows']:,} rows, "
        f"{ctx['listeners']:,} listeners, loaded in "
        f"{_clock(time.time() - t0)}. Building the design (no progress lines "
        f"in this step)"
    )
    t1 = time.time()
    with st._design_for(ctx, store, ids) as pool:
        if pool is not None:
            say(f"{WORK_NAME}: design ready in {_clock(time.time() - t1)}")
            if "check|model" not in store:
                rec = pc.check_model(st, pool, ctx, M)
                store.put("check|model", rec)
                say(
                    f"{WORK_NAME}: the model is a stationary point of the "
                    f"level's objective (largest projected gradient "
                    f"{rec['proj_grad_max']:.1e}, objective gap "
                    f"{rec['relative_gap']:.1e})"
                )
            if "check|mix32-B" not in store:
                rec = check_fit(st, pool, ctx, M, ("mix32", "B"), P["fam"][MIX])
                store.put("check|mix32-B", rec)
                say(
                    f"{WORK_NAME}: P13c's mix32-B fit is a stationary point "
                    f"(largest projected gradient {rec['proj_grad_max']:.1e})"
                )
            todo = [n for n in names if n not in store]
            if todo:
                prog = Progress(f"{WORK_NAME}: new fits", len(todo), say, unit=" fits")
                for k, name in enumerate(todo, start=1):
                    family, arm, step = parse(name)
                    lay, lam_k = lays[family], lam_at(lam, step)
                    start, src = warm_start(name, lays, M, P, store)
                    obj = st._objective(pool, ctx, lay, M["idx"], arm, lam_k)
                    label = (
                        f"fit {k} of {len(todo)}, {name} (m = "
                        f"{pc.m_of(arm, level)}, lambda {lam_k:.3g})"
                    )
                    res = pc.solve(obj, start, label, say)
                    res.update(
                        {"fingerprint": fps[family], "start": src, "lambda": lam_k}
                    )
                    if family == "bounded":
                        res["flat_share"] = gam_arms.active_share(res["theta"], lay)
                    store.put(name, res)
                    say(done_line(label, res))
                    prog.advance(1)
                prog.finish()
            if not pilot and "concurvity" not in store:
                G, n_rows = gram(pool, M["arm"], ctx["L_u"], say)
                tabs = {
                    "free model": lift(base, M, M["lay"].tables(M["theta"])),
                    "bounded p_pop-B at lambda": lift(
                        base, M, lays["bounded"].tables(store.get(PILOT)["theta"])
                    ),
                }
                rec = concurvity_record(G, base, M, tabs)
                rec.update(rows=n_rows, theta=[])
                store.put("concurvity", rec)
                del G
    st._ensure_means(ctx, store)
    means = ctx["means_ref"]
    del ctx
    return means


def pilot_report(rec, P, say):
    """The pilot's time and the projected time of the full command."""
    t_b = float(rec["seconds"])
    a, b = PASS_MINUTES
    p13c_pass = float((P["doc"].get("seconds") or {}).get("pass_this_run") or 0.0)
    n_pass, n_p13c = len(PASS_FITS), len(pc.ALL_ARMS)
    pass_s = (
        p13c_pass * (a + b * n_pass) / (a + b * n_p13c)
        if p13c_pass > 60.0
        else 60.0 * (a + b * n_pass)
    )
    n_b = len(BOUNDED_FITS) - 1
    n_f = len(NEW_FITS) - len(BOUNDED_FITS)
    fits_s = n_b * t_b + n_f * FREE_SHARE * t_b
    rest_s = 300.0
    total = fits_s + pass_s + rest_s
    say(
        f"{WORK_NAME}: pilot done. The bounded p_pop-B at lambda took "
        f"{_clock(t_b)} ({gr._text(rec['message'])}). Projected for the full "
        f"command: the other {n_b + n_f} fits about {_clock(fits_s)} ({n_b} "
        f"bounded, and {n_f} free at about half that time), the pass over "
        f"{n_pass} configurations about {_clock(pass_s)} (from P13c's pass), "
        f"loading, checks and reasons about {_clock(rest_s)}: about "
        f"{_clock(total)} in all. Next: python3 -m pipeline.runner --only "
        f"p13d_bounds"
    )
    return {"pilot": PILOT, "seconds": t_b, "projected_total": total}


# Rules


def simplicity(name, model_arm):
    """P13's order: the strongest penalty first, then the model's arm."""
    _, arm, step = parse(name)
    return (-STEP_FACTOR[step], arm != tuple(model_arm), name)


def decide(scores, names, model_arm):
    """S1 to S3 from the per-listener scores (C, n) of the fits in names
    (module docstring)."""
    S = np.asarray(scores, dtype=np.float64)
    ix = {n: i for i, n in enumerate(names)}
    means = S.mean(axis=1)
    sesoi_abs = SESOI * float(means[ix[MODEL]])
    d1 = gr.paired_scores(S[ix[PILOT]], S[ix[MODEL]])
    keep = bool(d1["lo"] is not None and d1["lo"] > -sesoi_abs)
    family = "bounded" if keep else "free"
    start = PILOT if keep else MODEL
    gate, eligible = {}, []
    for n in names:
        fam, arm, _ = parse(n)
        if fam != family or arm not in FORM_B or n == start:
            continue
        dd = gr.paired_scores(S[ix[n]], S[ix[start]])
        ok = bool(
            dd["mean"] is not None
            and dd["lo"] is not None
            and dd["mean"] > sesoi_abs
            and dd["lo"] > 0.0
        )
        gate[n] = {"difference": dd, "eligible": ok}
        if ok:
            eligible.append(n)
    order = sorted([start] + eligible, key=lambda n: simplicity(n, model_arm))
    best, one_se = None, {}
    if eligible:
        c, b, rows = ga.one_se_choice(S, [ix[n] for n in order])
        chosen, best = names[c], names[b]
        one_se = {
            n: {
                "mean": float(rows[ix[n]]["mean"]),
                "diff_to_best": float(rows[ix[n]]["diff"]),
                "se": _num(rows[ix[n]]["se"]),
            }
            for n in order
        }
    else:
        chosen = start
    i_best = int(np.argmax(means))
    thr = float(means[i_best]) - sesoi_abs
    members = [n for i, n in enumerate(names) if means[i] >= thr]
    return {
        "sesoi": {
            "share": SESOI,
            "abs": sesoi_abs,
            "base_score": float(means[ix[MODEL]]),
        },
        "s1": {"difference": d1, "threshold": -sesoi_abs, "bounds_kept": keep},
        "s2": {
            "family": family,
            "start": start,
            "gate": gate,
            "eligible": eligible,
            "order": order,
            "best": best,
            "one_se": one_se,
            "chosen": chosen,
        },
        "s3": {
            "best": names[i_best],
            "best_score": float(means[i_best]),
            "threshold": thr,
            "members": members,
        },
    }


def s4_rule(R, d1_shown, member_vals):
    """S4: D1's reasons that are positive under every member and in the
    top K_TOP of at least S4_TOP_SHARE of the members."""
    cand = R.cand
    pos_all = cand.copy()
    top_n = np.zeros(cand.shape, dtype=np.int64)
    for v in member_vals:
        pos_all &= v > 0
        top_n += cand & (pc.ranks_of(v, cand) <= pc.K_TOP)
    need = math.ceil(S4_TOP_SHARE * len(member_vals) - 1e-9)
    return d1_shown & pos_all & (top_n >= need), top_n, need


def s4_summary(R, d1_shown, shown4, val_model):
    rows = []
    for g, (name, terms) in enumerate(R.groups):
        s = shown4[g]
        rows.append(
            {
                "reason": name,
                "terms": list(terms),
                "candidate": float(R.cand[g].mean()),
                "shown_d1": float(d1_shown[g].mean()),
                "shown_s4": float(s.mean()),
                "removed_by_s4": float((d1_shown[g] & ~s).mean()),
                "mean_shown": _num(val_model[g][s].mean()) if s.any() else None,
                "odds_shown": (
                    _num(np.exp(val_model[g][s]).mean()) if s.any() else None
                ),
            }
        )
    k = shown4.sum(axis=0)
    return {
        "picks": int(shown4.shape[1]),
        "rows": rows,
        "n_shown": {str(j): float(np.mean(k == j)) for j in range(len(R.groups) + 1)},
    }


def shape_summary(M, lays, thetas, means, shape_q):
    """Centered values at P13c's points of the model and of each bounded
    fit, with the bounded fits' flat shares and order."""
    names_t = [t.name for t in M["lay"].terms]
    fits = {}
    for n in [MODEL] + BOUNDED_FITS:
        Mx = dict(M, lay=lays[parse(n)[0]])
        v = pc.value_array(Mx, [thetas[n]], means, shape_q)[0]
        fits[n] = {t: [_num(x) for x in v[j]] for j, t in enumerate(names_t)}
    lay_b = lays["bounded"]
    return {
        "points": list(pc.POINTS),
        "fits": fits,
        "flat_share": {
            n: {k: float(x) for k, x in gam_arms.active_share(thetas[n], lay_b).items()}
            for n in BOUNDED_FITS
        },
        "monotone_violation": {
            n: float(gam_arms.monotone_violation(lay_b.tables(thetas[n]), lay_b))
            for n in BOUNDED_FITS
        },
    }


def agreement_summary(M, R, val_model, member_vals, ref, means):
    """Share of picks whose signed top-3 reasons equal the free model's."""
    top_m = R.signed_top(val_model)
    top_level = max(s13b2.REF_LEVELS)
    refs = {}
    for k, th in sorted(ref.items()):
        if int(k.split("|")[0]) != top_level:
            continue
        v = R.values(pc.centered(M, th, means))[0]
        refs[k] = float(pc.agreement(top_m, R.signed_top(v)).mean())
    mem = {
        n: float(pc.agreement(top_m, R.signed_top(v)).mean())
        for n, v in member_vals.items()
        if n != MODEL
    }
    return {"top_level": int(top_level), "reference": refs, "members": mem}


def summarize(
    M, lays, thetas, tables, ranks_all, R, d1_shown, val_model, ref, means, shape_q
):
    """Every rule and result from the parameters, the ranks and the reason
    sample. A pure function, so check_outputs() recomputes it."""
    names = list(ALL_FITS)
    S = np.stack([ga.per_listener_scores(ranks_all[n]) for n in names])
    rules = decide(S, names, M["arm"])
    members = rules["s3"]["members"]
    member_vals = {n: R.values(tables[n])[0] for n in members}
    shown4, top_n, need = s4_rule(R, d1_shown, [member_vals[n] for n in members])
    i0 = names.index(MODEL)
    out = {
        "scores": {
            n: {
                "score": float(S[i].mean()),
                "diff_to_model": (None if i == i0 else gr.paired_scores(S[i], S[i0])),
                "in_set": n in members,
            }
            for i, n in enumerate(names)
        }
    }
    out.update(rules)
    out["s4"] = dict(
        s4_summary(R, d1_shown, shown4, val_model),
        members=len(members),
        need=int(need),
        model_changed=bool(rules["s2"]["chosen"] != MODEL),
        d1_source="D1 of P13c, for the free model (p_pop-B at " "lambda)",
    )
    out["shapes"] = shape_summary(M, lays, thetas, means, shape_q)
    out["agreement"] = agreement_summary(M, R, val_model, member_vals, ref, means)
    return out, {"s4_shown": shown4, "s4_top_count": top_n}


# Records


def fit_info(store, M):
    """Per fit: its family, arm, penalty weight and stop record."""
    info = {}
    chk, mix = _record(store, "check|model"), _record(store, "check|mix32-B")
    for n in ALL_FITS:
        family, arm, step = parse(n)
        row = {
            "family": family,
            "arm": s13b.arm_label(arm),
            "step": step,
            "lambda": float(lam_at(M["lambda"], step)),
            "source": "p13d" if n in NEW_FITS else "p13c",
        }
        if n in (MODEL, MIX):
            r = chk if n == MODEL else mix
            row.update(
                objective=float(r["objective"]),
                proj_grad_max=float(r["proj_grad_max"]),
                fingerprint=M["fingerprint"],
            )
        elif n in NEW_FITS:
            r = _record(store, n)
            row.update(
                {
                    k: r.get(k)
                    for k in (
                        "objective",
                        "proj_grad_max",
                        "message",
                        "seconds",
                        "newton_steps",
                        "method",
                        "fingerprint",
                        "start",
                        "monotone_violation",
                        "flat_share",
                    )
                }
            )
        else:
            row.update(objective=None, proj_grad_max=None, fingerprint=M["fingerprint"])
        info[n] = row
    return info


def model_record(chosen, M, lays, fps, thetas, tables, consts, info):
    """The chosen model in the fields of gam_refit's fit record, for the
    model switch of p13c.py."""
    family, arm, step = parse(chosen)
    lay, theta = lays[family], thetas[chosen]
    return {
        "name": M["name"],
        "fit_name": chosen,
        "tag": tag_of(chosen),
        "changed": chosen != MODEL,
        "kept": list(M["kept"]),
        "bounded": list(CONTENT) if family == "bounded" else [],
        "level": int(M["level"]),
        "arm": s13b.arm_label(arm),
        "m": int(pc.m_of(arm, M["level"])),
        "lambda": float(lam_at(M["lambda"], step)),
        "terms": [t.to_json() for t in lay.terms],
        "fingerprint": fps[family],
        "theta": theta.tolist(),
        "tables_raw": lay.tables(theta).tolist(),
        "tables": tables[chosen].tolist(),
        "center_consts": consts[chosen],
        "objective": float(info[chosen]["objective"]),
        "background": "the level's p_pop negatives, full-fit weights",
        "table_dict": gam.table_dict(tables[chosen], lay),
    }


def check_outputs(d, base, M, lays, fps, R, d1_shown, val_model, ref, means, info):
    """Read the outputs back: the tables follow from the stored
    parameters, and every rule and result from the stored parameters,
    ranks and reason sample."""
    doc = load_json(d / OUT_FILES[0])
    z = gkd.load_npz(d / OUT_FILES[1])
    names = [str(x) for x in z["names"].tolist()]
    if names != ALL_FITS:
        raise ValueError("p13d_model_set.npz holds other fits")
    thetas, tables, consts = {}, {}, {}
    for i, n in enumerate(names):
        lay = lays[parse(n)[0]]
        thetas[n] = np.asarray(z["theta"][i], dtype=np.float64)
        tables[n], consts[n] = centered(lay, thetas[n], means)
        if not np.array_equal(tables[n], z["tables"][i]):
            raise ValueError(f"the stored tables of {n} differ from its " f"parameters")
        if doc["fits"][n]["fingerprint"] != fps[parse(n)[0]]:
            raise ValueError(f"the stored layout of {n} differs")
    ranks_all = {n: np.asarray(z["ranks"][i]) for i, n in enumerate(names)}
    again, arr = summarize(
        M,
        lays,
        thetas,
        tables,
        ranks_all,
        R,
        d1_shown,
        val_model,
        ref,
        means,
        base["shape_q"],
    )
    for k in SECTIONS:
        if _norm(again[k]) != _norm(doc[k]):
            raise ValueError(
                f"the stored results '{k}' do not follow from "
                f"the stored parameters and ranks"
            )
    for k, v in arr.items():
        if not np.array_equal(v, z[k]):
            raise ValueError(
                f"the stored array {k} does not follow from the " f"stored parameters"
            )
    rec = model_record(
        again["s2"]["chosen"], M, lays, fps, thetas, tables, consts, info
    )
    if _norm(rec) != _norm(doc["model"]):
        raise ValueError(
            "the stored model record does not follow from the " "stored parameters"
        )
    members = again["s3"]["members"]
    if [str(x) for x in z["members"].tolist()] != members or not np.array_equal(
        z["member_tables"], np.stack([tables[n] for n in members])
    ):
        raise ValueError("the stored model set differs from S3")


def run(
    st, model_dir, work, say=print, n_verify=VERIFY_USERS, inputs=None, pilot=False
):
    """P13d (module docstring). st: a StageP13b2 whose loaders give the
    level, the item side, the CSR and the slots. Returns the output doc,
    or the pilot's record with pilot."""
    prev = gr.WORK_NAME
    gr.WORK_NAME = WORK_NAME  # labels of gam_refit's fit lines
    try:
        return _run(st, Path(model_dir), Path(work), say, n_verify, inputs, pilot)
    finally:
        gr.WORK_NAME = prev


def _run(st, d, work, say, n_verify, inputs, pilot):
    t0 = time.time()
    base = st._base_context()
    users, picks = base["sel_users"], base["sel_picks"]
    M, stored_ranks = pc.load_model(d, base)
    if tuple(M["arm"]) != ("p_pop", "B"):
        raise ValueError(
            f"P13d's protocol is declared for a p_pop form-B "
            f"model, not {s13b.arm_label(M['arm'])}"
        )
    P = load_p13c(d, base, M, stored_ranks)
    lay_b, idx_b = pc.layout_of(base, M["kept"], CONTENT)
    if not np.array_equal(idx_b, M["idx"]):
        raise ValueError("the bounded layout maps to other table entries")
    lays = {"free": M["lay"], "bounded": lay_b}
    fps = {"free": M["fingerprint"], "bounded": gam_arms.layout_fingerprint(lay_b)}
    work.mkdir(parents=True, exist_ok=True)
    key = {
        "version": VERSION,
        "gam_refit": file_sha256(d / "gam_refit.json"),
        "p13c": [file_sha256(d / f) for f in pc.OUT_FILES],
        "p13b2_key": st._key(),
        "bounded": list(CONTENT),
        "fits": NEW_FITS,
    }
    store = s13b2._Store(work / "fits.json", s13._sha(key))
    say(
        f"{WORK_NAME}: model '{M['name']}' ({len(M['kept'])} terms), arm "
        f"{s13b.arm_label(M['arm'])}, level {M['level']}, lambda "
        f"{M['lambda']:.3g}. "
        + (
            "Pilot: the bounded p_pop-B at lambda only. "
            if pilot
            else f"{len(NEW_FITS)} new fits ({len(NEW_FITS) - len(BOUNDED_FITS)} "
            f"free, {len(BOUNDED_FITS)} bounded on {', '.join(CONTENT)}), then "
            f"one selection pass over {len(PASS_FITS)} configurations. "
        )
        + f"Saved work: {work}"
    )
    t1 = time.time()
    means = fit_part(st, base, M, P, lays, fps, store, say, pilot)
    st._drop_inputs()
    if pilot:
        return pilot_report(store.get(PILOT), P, say)
    fit_secs = time.time() - t1

    thetas = {n: P["fam"][n] for n in P13C_FITS}
    for n in NEW_FITS:
        rec = store.get(n)
        if rec["fingerprint"] != fps[parse(n)[0]] or rec["proj_grad_max"] > s13.KKT_TOL:
            raise ValueError(
                f"the saved fit {n} belongs to another layout "
                f"or stopped above KKT_TOL"
            )
        thetas[n] = np.asarray(rec["theta"], dtype=np.float64)
    tables, consts = {}, {}
    for n in ALL_FITS:
        tables[n], consts[n] = centered(lays[parse(n)[0]], thetas[n], means)
    gap = float(np.max(np.abs(tables[MODEL] - M["tables"])))
    if gap > s13b2.CENTER_TOL:
        raise ValueError(
            f"the model's stored centered tables differ from a "
            f"centering on the level's p_pop negatives by "
            f"{gap:.2e}"
        )
    tables_u = np.zeros((len(PASS_FITS), base["L_u"]))
    for c, n in enumerate(PASS_FITS):
        tables_u[c, M["idx"]] = tables[n]
    say(
        f"{WORK_NAME}: the fits are ready. Selection pass over "
        f"{len(PASS_FITS)} configurations"
    )
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
    with relabel_logs():
        ranks, scored = st._selection_pass(
            base["union"], tables_u, users, picks, base["m_sel"], work / "pass", pkey
        )
    pass_secs = time.time() - t2
    for c, n in ((0, MODEL), (1, MIX)):
        if not np.array_equal(ranks[c], P["ranks"][n]):
            raise ValueError(
                f"the ranks of {n} differ from P13c's stored "
                f"ranks in {int(np.count_nonzero(ranks[c] != P['ranks'][n]))} "
                f"picks"
            )
    if not np.array_equal(ranks[0], stored_ranks):
        raise ValueError("the model's ranks differ from gam_refit's stored " "ranks")
    dense = gr.verify_dense(st, base, tables_u, ranks, say, n_verify)
    st._cache.pop("space", None)
    st._cache.pop("sources", None)
    ranks_all = {n: np.asarray(ranks[c]) for c, n in enumerate(PASS_FITS)}
    for n in P13C_FITS:
        ranks_all.setdefault(n, P["ranks"][n])

    say(
        f"{WORK_NAME}: reason sample from P11's selection rows, and D1 "
        f"again from P13c's listener replicates"
    )
    S = pc.reason_sample(d, base)
    R = pc.Reasons(base, M, S)
    z = P["z"]
    if not (
        np.array_equal(S["users"][S["lst"][S["is_pos"]]], z["pick_user"])
        and np.array_equal(S["item"][S["is_pos"]], z["pick_item"])
    ):
        raise ValueError("the reason sample differs from P13c's")
    val_model = R.values(M["tables"])[0]
    if not np.array_equal(val_model, z["reason_value"]):
        raise ValueError("the model's reason values differ from P13c's")
    boot = np.asarray(z["theta_boot_listener"], dtype=np.float64)
    d1_shown, _, _ = R.stable(
        val_model, (R.values(pc.centered(M, th, means))[0] for th in boot), len(boot)
    )
    if not np.array_equal(d1_shown, z["shown"]):
        raise ValueError(
            "D1, rebuilt from P13c's replicates, differs from " "P13c's stored D1"
        )
    out, arrays = summarize(
        M,
        lays,
        thetas,
        tables,
        ranks_all,
        R,
        d1_shown,
        val_model,
        P["ref"],
        means,
        base["shape_q"],
    )
    info = fit_info(store, M)
    chosen = out["s2"]["chosen"]
    members = out["s3"]["members"]
    chk = {k: v for k, v in _record(store, "check|model").items() if k != "theta"}
    mix = {k: v for k, v in _record(store, "check|mix32-B").items() if k != "theta"}
    conc = {k: v for k, v in _record(store, "concurvity").items() if k != "theta"}
    out.update(
        version=VERSION,
        protocol=PROTOCOL,
        model=model_record(chosen, M, lays, fps, thetas, tables, consts, info),
        fits=info,
        concurvity=conc,
        check={
            "model": chk,
            "mix32_B": mix,
            "centering_gap": gap,
            "ranks_equal_p13c": True,
            "ranks_equal_gam_refit": True,
            "dense": dense,
            "d1_equals_p13c": True,
            "largest_kkt_new": max(float(info[n]["proj_grad_max"]) for n in NEW_FITS),
            "largest_monotone_break": max(out["shapes"]["monotone_violation"].values()),
        },
        pass_info={
            "configurations": len(PASS_FITS),
            "listeners": int(len(users)),
            "mean_scored_share": float(np.mean(scored) / int(st._id_slots()[0])),
        },
        inputs=dict(inputs or {}),
        exports={
            "p13d.json": "model: the chosen model in the fields of "
            "gam_refit's fit record, for the stage p13c_final",
            "p13d_model_set.npz": "members and member_tables: the S3 set "
            "(centered tables in the model's table order, idx maps them "
            "to the union) for P14's rule S4. names, theta, tables and "
            "ranks: every fit",
        },
        seconds={"fits_this_run": fit_secs, "pass_this_run": pass_secs},
    )
    arrays.update(
        names=np.array(ALL_FITS),
        bounded=np.array([parse(n)[0] == "bounded" for n in ALL_FITS]),
        theta=np.stack([thetas[n] for n in ALL_FITS]),
        tables=np.stack([tables[n] for n in ALL_FITS]),
        ranks=np.stack([ranks_all[n] for n in ALL_FITS]).astype(np.int32),
        user_id=users,
        idx=np.asarray(M["idx"], dtype=np.int64),
        members=np.array(members),
        member_tables=np.stack([tables[n] for n in members]),
        member_bounded=np.array([parse(n)[0] == "bounded" for n in members]),
        d1_shown=d1_shown,
        candidate=R.cand,
        reason_value=val_model,
        pick_user=S["users"][S["lst"][S["is_pos"]]],
        pick_item=S["item"][S["is_pos"]],
        scored=scored.astype(np.int64),
    )
    gr._atomic_savez(d / OUT_FILES[1], **arrays)
    atomic_json_write(out, d / OUT_FILES[0])
    check_outputs(d, base, M, lays, fps, R, d1_shown, val_model, P["ref"], means, info)
    out["seconds"]["total_this_run"] = time.time() - t0
    atomic_json_write(out, d / OUT_FILES[0])
    write_report(out, d / "reports" / REPORT_NAME)
    s1 = out["s1"]
    say(
        f"{WORK_NAME}: S1: bounded minus free "
        f"{gr._interval(s1['difference'])}, the bounds "
        f"{'stay' if s1['bounds_kept'] else 'do not stay'}. S2: model "
        f"{chosen}. S3: {len(members)} members. Done in "
        f"{_clock(time.time() - t0)}"
    )
    if chosen != MODEL:
        say(
            f"{WORK_NAME}: the model changed, so P13c runs again on it: "
            f"python3 -m pipeline.runner --only p13c_final"
        )
    return out


# Report


def _f(x, fmt="+.3f"):
    return "" if x is None else format(x, fmt)


def write_report(out, path):
    m, sc = out["model"], out["scores"]
    s1, s2, s3, s4 = out["s1"], out["s2"], out["s3"], out["s4"]
    sh, cc, ag, ck = (out["shapes"], out["concurvity"], out["agreement"], out["check"])
    ses = out["sesoi"]["abs"]
    L = [
        "# P13d: bounds, model, near-optimal set and reason rule S4",
        "",
        f"- model in: '{m['name']}' ({len(m['kept'])} terms), arm p_pop-B, "
        f"free shapes, level {m['level']}, lambda "
        f"{out['fits'][MODEL]['lambda']:.3g}. Protocol: "
        f"the docstring of p13d.py, declared {DECLARED}, with its code "
        f"check.",
        f"- checks: the model and P13c's mix32-B fit are stationary points "
        f"of the level's objective (largest projected gradients "
        f"{ck['model']['proj_grad_max']:.1e} and "
        f"{ck['mix32_B']['proj_grad_max']:.1e}). Every new fit reached "
        f"KKT_TOL (largest {ck['largest_kkt_new']:.1e}), and every bounded "
        f"fit keeps its order (largest break "
        f"{ck['largest_monotone_break']:.1e}). The model's and mix32-B's "
        f"ranks equal P13c's, and the model's equal gam_refit's. "
        f"{ck['dense']['listeners']} listeners matched the dense reference "
        f"({ck['dense']['near_ties']} near ties). D1, rebuilt from P13c's "
        f"replicates, equals P13c's. The results follow from the stored "
        f"outputs.",
        f"- time in this run: fits {_clock(out['seconds']['fits_this_run'])}"
        f", pass {_clock(out['seconds']['pass_this_run'])}, total "
        f"{_clock(out['seconds']['total_this_run'])}.",
        "",
        "## Rules",
        "",
        f"SESOI: {ses:.5f} (2 percent of the model's score "
        f"{out['sesoi']['base_score']:.5f}).",
        "",
        f"- S1 (bounds): the bounded p_pop-B at lambda minus the free model: "
        f"{gr._interval(s1['difference'])}. The lower end must lie above "
        f"{s1['threshold']:+.5f}. Result: the bounds "
        f"{'stay' if s1['bounds_kept'] else 'do not stay'}.",
    ]
    elig = ", ".join(label(n) for n in s2["eligible"]) or "none"
    L.append(
        f"- S2 (model): family {s2['family']}, start "
        f"{label(s2['start'])}. Eligible (more than SESOI above the "
        f"start, interval above 0): {elig}."
    )
    if s2["eligible"]:
        L.append(
            f"  One-standard-error rule over "
            f"{', '.join(label(n) for n in s2['order'])}: best "
            f"{label(s2['best'])}, chosen {label(s2['chosen'])}."
        )
    L.append(
        f"- Model: {label(s2['chosen'])}, "
        f"{'changed' if m['changed'] else 'unchanged'}."
    )
    if m["changed"]:
        L.append(
            "  The model changed, so P13c runs again on it: "
            "python3 -m pipeline.runner --only p13c_final. Its outputs "
            f"carry the tag {m['tag']}."
        )
    L += [
        f"- S3 (near-optimal set): best {label(s3['best'])} "
        f"({s3['best_score']:.5f}). Members, at or above "
        f"{s3['threshold']:.5f}: "
        f"{', '.join(label(n) for n in s3['members'])}.",
        "",
        "## Scores",
        "",
        "| fit | score | difference to the model (95% interval) | " "in S3 |",
        "|---|---:|---|---|",
    ]
    for n in ALL_FITS:
        r = sc[n]
        L.append(
            f"| {label(n)} | {r['score']:.5f} | "
            f"{gr._interval(r['diff_to_model']) if r['diff_to_model'] else ''}"
            f" | {'yes' if r['in_set'] else 'no'} |"
        )
    L += [
        "",
        "## S4: the reasons that P14 shows",
        "",
        f"{s4['picks']:,} val picks of the selection listeners. D1 of "
        f"P13c, then S4 over the {s4['members']} members of S3: positive "
        f"under every member and in the top {pc.K_TOP} of at least "
        f"{s4['need']} of them. Values and odds factors are the free "
        f"model's."
        + (
            " The model changed, so these shares use D1 of the free model."
            if s4["model_changed"]
            else ""
        ),
        "",
        "| reason | candidate | shown by D1 | shown by S4 | removed by S4 "
        "| mean value when shown | mean odds factor when shown |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in s4["rows"]:
        L.append(
            f"| {r['reason']} | {r['candidate']:.1%} | "
            f"{r['shown_d1']:.1%} | {r['shown_s4']:.1%} | "
            f"{r['removed_by_s4']:.1%} | {_f(r['mean_shown'])} | "
            f"{_f(r['odds_shown'], '.2f')} |"
        )
    L += [
        "",
        "Reasons per pick under S4, share of picks: "
        + ", ".join(f"{k}: {v:.1%}" for k, v in s4["n_shown"].items() if v > 0)
        + ".",
        "",
        "## Bounded shapes",
        "",
        "Centered values in log-odds points, at the positives' "
        "percentiles and at the zero and missing states. Free: the model. "
        "Bounded: p_pop-B at lambda.",
        "",
        "| term | point | free | bounded |",
        "|---|---|---:|---:|",
    ]
    for t in CONTENT + tuple(x for x in m["kept"] if x not in CONTENT):
        if t not in sh["fits"][MODEL]:
            continue
        for p, point in enumerate(sh["points"]):
            a, b = sh["fits"][MODEL][t][p], sh["fits"][PILOT][t][p]
            if a is None and b is None:
                continue
            L.append(f"| {t if p == 0 else ''} | {point} | {_f(a)} | " f"{_f(b)} |")
    L += [
        "",
        "Flat share: the share of a bounded term's increments held at "
        "the bound (a flat segment of the shape).",
        "",
        "| fit | " + " | ".join(CONTENT) + " | largest break |",
        "|---|" + "---:|" * (len(CONTENT) + 1),
    ]
    for n in BOUNDED_FITS:
        fs = sh["flat_share"][n]
        L.append(
            f"| {label(n)} | "
            + " | ".join(_f(fs.get(t), ".0%") for t in CONTENT)
            + f" | {sh['monotone_violation'][n]:.1e} |"
        )
    obs = cc["observed"]
    labels = [x for x in ("free model", "bounded p_pop-B at lambda") if x in obs]
    L += [
        "",
        "## Concurvity",
        "",
        f"As P13a, on the within-listener Gram matrix of the rows of the "
        f"model's loss ({cc['rows']:,} rows: the positives and the p_pop "
        f"draws). worst and estimate: each term against the other terms. "
        f"observed: the share of the fitted term that the other terms "
        f"reproduce.",
        "",
        "| term | worst | estimate | "
        + " | ".join(f"observed, {x}" for x in labels)
        + " |",
        "|---|---:|---:|" + "---:|" * len(labels),
    ]
    for t in m["kept"]:
        r = cc["terms"][t]
        L.append(
            f"| {t} | {r['worst']:.3f} | {r['estimate']:.3f} | "
            + " | ".join(f"{obs[x][t]:.3f}" for x in labels)
            + " |"
        )
    L += [
        "",
        "## Agreement of the signed top-3 reasons with the free model",
        "",
        "Share of picks with the same signed top-3 reasons.",
        "",
        f"- each P13c fit at reference level {ag['top_level']}: "
        + (
            ", ".join(f"{k.split('|')[1]} {v:.1%}" for k, v in ag["reference"].items())
            or "none"
        )
        + ".",
        "- each other member of S3: "
        + (", ".join(f"{label(k)} {v:.1%}" for k, v in ag["members"].items()) or "none")
        + ".",
        "",
        "## Exports",
        "",
    ]
    L += [f"- {k}: {v}" for k, v in out["exports"].items()]
    atomic_text_write("\n".join(L) + "\n", path)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="P13d: bounds, model, near-optimal set and reason rule "
        "S4 (protocol in the module docstring)"
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="fit the bounded p_pop-B at lambda only, print "
        "its time and the projected total, and stop",
    )
    args = parser.parse_args(argv)
    gr.setup_logging(LOG_PATH)
    say = log.info
    try:
        say(f"{WORK_NAME}: started, process id {os.getpid()}, log file " f"{LOG_PATH}")
        run(
            s13b2.StageP13b2(),
            MODEL_DIR,
            SCRATCH_DIR / WORK_NAME,
            say=say,
            inputs=provenance(MODEL_DIR),
            pilot=args.pilot,
        )
    except KeyboardInterrupt:
        log.info(
            f"{WORK_NAME}: stopped by Ctrl+C. Run the same command "
            f"again: it skips the saved fits and scores the blocks "
            f"still missing."
        )
        raise SystemExit(130)
    except Exception:
        log.exception(f"{WORK_NAME}: stopped by an error")
        raise SystemExit(1)


# Runner stage


class StageP13d(ScriptStage):
    """P13d as a runner stage. The pilot stays with the module's own
    entry: python3 -m pipeline.p13d --pilot."""

    name = "p13d_bounds"
    upstream = ["p13c_stability"]

    def outputs(self):
        d = self.model_dir
        return {
            "results": d / OUT_FILES[0],
            "model_set": d / OUT_FILES[1],
            "report": d / "reports" / REPORT_NAME,
        }

    def provenance(self):
        return provenance(self.model_dir)

    def recorded_inputs(self):
        doc = load_json(self.outputs()["results"])
        return None if doc is None else doc.get("inputs")

    def _stage(self):
        return s13b2.StageP13b2()

    def run(self, params, files):
        with log_file(self.model_dir / "logs" / LOG_PATH.name):
            run(
                self._stage(),
                self.model_dir,
                self.scratch_dir / WORK_NAME,
                say=log.info,
                inputs=self.provenance(),
            )
        return self.output_info()

    def check(self):
        """check_outputs() with what run() gives it: P13c's D1 and
        reference fits, the model's reason values and the stored fit
        records."""
        d, st = self.model_dir, self._stage()
        base = st._base_context()
        M, stored_ranks = pc.load_model(d, base)
        P = load_p13c(d, base, M, stored_ranks)
        lay_b, _ = pc.layout_of(base, M["kept"], CONTENT)
        lays = {"free": M["lay"], "bounded": lay_b}
        fps = {"free": M["fingerprint"], "bounded": gam_arms.layout_fingerprint(lay_b)}
        doc = load_json(d / OUT_FILES[0])
        means = pc.stored_means(st, base, M, self.scratch_dir / WORK_NAME)
        R = pc.Reasons(base, M, pc.reason_sample(d, base))
        val_model = R.values(M["tables"])[0]
        check_outputs(
            d,
            base,
            M,
            lays,
            fps,
            R,
            np.asarray(P["z"]["shown"]),
            val_model,
            P["ref"],
            means,
            doc["fits"],
        )


if __name__ == "__main__":
    main()
