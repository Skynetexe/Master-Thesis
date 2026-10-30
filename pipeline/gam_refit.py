"""The final GAM refit without its two weakest terms, in a 2 x 2 design.

The diagnostic in gam_knn_diag.py scored variants of the
final model of P13b part 2 from its stored tables, without a refit. Two
variants scored more than SESOI above the full model: the model without
colisten_other_artists (+0.00336) and the model without popularity
(+0.00137). The two artist terms alone (artist_affinity and
colisten_same_artist) scored 0.04653, the best selection score so far. A
variant without a refit cannot show what the model learns without a
term, because the other terms keep the shapes that they learned next to
it. This script refits. It never reads a test pick.

Protocol, fixed before any result.

Model and data. Every fit uses the final model's level of the wide file
of P11b and P12b (the training listeners, their positives and the first
k draws of each kind), the final model's loss arm, its penalty weight
and P13a's knots and layout constants. On the real data the arm is
p_pop form B (the k p_pop draws, -log(m q) on every candidate, weight 0
for a positive outside P3's core), the level is 64 (m = 64) and the
weight is lambda 3.16e-8. Only the terms change.

Configurations, in this order:
1. full: the final model as P13b part 2 stored it. It is not refit. The
   script checks that its stored parameters are a stationary point of
   the level's objective: the largest projected gradient is at most
   KKT_TOL, and the objective equals the stored one.
2. without colisten_other_artists.
3. without popularity.
4. without both.
5. artist terms only: artist_affinity and colisten_same_artist, the
   reference arm.
A configuration's fit leaves out the terms that it drops. Each refit
starts from the final model's parameters of its own terms. The layouts
name their parameters, and the parameters of one term do not depend on
the other terms, so the start reproduces the final model's tables of
those terms. The solver is P13b part 2's (stage_p13b.SOLVER: damped
Newton steps with the exact Hessian). The objective is strictly convex,
so the start changes the path of the solver, not the optimum. The tables
are centered on the level's p_pop negatives, as in P13b part 2.

Selection: one pass of P13 (StageP13a._selection_pass) over the five
configurations. Full-catalog NDCG@100 of the val_last and val_first
picks of the 20,000 selection listeners, ranks exact up to 100. A
configuration's score is the mean over the listeners of the mean over
their two picks.

Rules. SESOI is 2 percent of the full model's score, as in P13b and the
diagnostic. An interval is the 95 percent normal interval of the paired
per-listener differences (gam_audit.paired_summary).
- R1, the gate: a configuration without one or both terms is eligible
  when its score lies more than SESOI above the full model's score and
  the interval of the difference lies above 0.
- R2, the choice: the one-standard-error rule of P13
  (gam_audit.one_se_choice) over the full model and the eligible
  configurations. The order runs from the most terms to the fewest, and
  a tie in terms goes to the higher score. The best is the highest
  score. The chosen configuration is the first in the order whose score
  lies within one standard error of the paired difference below the
  best. With no eligible configuration, the full model stays.
  Why the order runs from the most terms: each term carries an
  explanation line, and colisten_other_artists is the only concept that
  reaches past the listener's own artists. Among the configurations
  that the data cannot separate from the best, the rule keeps the most
  terms. So it makes the smallest change to the explanations.
- R3, the reference arm, reported only: the artist terms alone are not a
  candidate. The report flags them when they score more than SESOI above
  the chosen configuration, with the interval above 0. That accuracy is
  then weighed against the explanation lines of the other terms.
These rules choose a model on the selection listeners. They are not
confirmatory tests. P14 reports the chosen model on the eval listeners.

Also reported: the main effect of each of the two terms and their
interaction (stage_p13b.main_effects), every configuration by
popularity band and by co-listening evidence (the per-pick classes of
gam_knn_diag.py's stage 1), the chosen configuration against ItemKNN on
the same picks, the shapes of the chosen configuration, and how far the
explanation points of each term move from the full model.

Checks:
- the final model: free shapes, terms in P13b part 2's union, and every
  term applies to every candidate (gam_knn_diag.check_every_candidate).
  Its layout, rebuilt here, has the stored terms and fingerprint.
- the full model on the rebuilt design: stationarity and objective, as
  above. Its stored raw tables equal M theta, and its stored centered
  tables equal a centering on the level's p_pop negatives.
- every refit reaches KKT_TOL before it is saved.
- the full model's ranks equal P13b part 2's stored ranks for every
  listener and pick.
- VERIFY_USERS listeners get every configuration's ranks again from the
  dense reference of catalog.py.
- the script reads its outputs back. The tables follow from the stored
  parameters, the explanation points from the tables, and every result
  and decision from the stored ranks.

Outputs: MODEL_DIR/gam_refit.json (the protocol, the fits, the results
and the decisions), MODEL_DIR/gam_refit.npz (the ranks, the slots scored
per listener and the union tables) and MODEL_DIR/reports/gam_refit.md.
Every line goes to the console and to MODEL_DIR/logs/gam_refit.log. Long
steps print a line at most every 30 seconds: each refit its Newton steps
and evaluations of the loss, the refits their share done with the time
left, the pass its share done with the time left (P13's lines), and the
dense check its listeners.

Resume: each refit, the full model's check and the level's background
means are saved in SCRATCH_DIR/gam_refit/fits.json. Their key holds P13b
part 2's key (its parameters, upstream outputs and code), the final
model file and the configurations. The pass saves its blocks in
SCRATCH_DIR/gam_refit/pass, keyed by the tables, the listeners and the
scoring code. After Ctrl+C, the same command continues: it skips the
saved fits and scores the blocks still missing. When every fit is
saved, the level's rows are not loaded again.

Expected run on the real data: loading the level and building the design
take a few minutes, with peak memory near P13b part 2's 18 GB. A Newton
step at level 64 takes about one minute and a refit 4 to 10 steps, so
the 4 refits take about 20 to 40 minutes. The pass takes about 1 hour
(the diagnostic's pass over 20 configurations took 1h04m), and the dense
check about 2 minutes.

Usage, from the repository root, through the runner:
    python3 -m pipeline.runner --only gam_refit
The module's own entry runs it without a marker (the runner then asks for
--adopt):
    python3 -m pipeline.gam_refit
The pass uses P13's worker setting. To use fewer workers:
    P13A_WORKERS=8 python3 -m pipeline.runner --only gam_refit
"""

import argparse
import contextlib
import json
import logging
import math
import os
import re
import time
from pathlib import Path

import numpy as np

from pipeline import catalog as ct
from pipeline import concept_data as cd
from pipeline import concepts as cn
from pipeline import gam
from pipeline import gam_arms
from pipeline import gam_audit as ga
from pipeline import itemknn as knn
from pipeline import stage_p13a as s13
from pipeline import stage_p13b as s13b
from pipeline import stage_p13b2 as s13b2
from pipeline.als import PROGRESS_SECONDS, Progress, _clock
from pipeline.common import (
    atomic_json_write,
    atomic_text_write,
    file_sha256,
    load_json,
    log,
    marker_path,
)
from pipeline.config import MODEL_DIR, SCRATCH_DIR

from pipeline import gam_knn_diag as gkd
from pipeline import itemknn_select as iks
from pipeline.script_stage import ScriptStage, log_file

LOG_PATH = MODEL_DIR / "logs" / "gam_refit.log"
LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"  # as in runner.py
VERSION = "gam-refit-v1"
WORK_NAME = "gam_refit"
DECLARED = "before any result"
SESOI = 0.02
OTHER, POP = "colisten_other_artists", "popularity"
ARTIST = ("artist_affinity", "colisten_same_artist")
FULL, REFERENCE = "full", "artist terms only"
CONFIG_NAMES = (FULL, f"without {OTHER}", f"without {POP}", "without both", REFERENCE)
ROLES = ("incumbent", "candidate", "candidate", "candidate", "reference")
# (colisten_other_artists in, popularity in) of the first four, the 2 x 2.
FACTOR_CELLS = ((1, 1), (0, 1), (1, 0), (0, 0))
FULL_CHECK = "full model check"
FIT_KEYS = (
    "objective",
    "loss",
    "smooth",
    "ridge",
    "proj_grad_max",
    "proj_grad_before_newton",
    "nit",
    "nfev",
    "newton_steps",
    "method",
    "message",
    "seconds",
)
VERIFY_USERS = 6
VERIFY_SEED = 2026100101
MEAN_TOL = 1e-12
OUT_FILES = ("gam_refit.json", "gam_refit.npz")
REPORT_NAME = "gam_refit.md"

PROTOCOL = {
    "version": VERSION,
    "declared": DECLARED,
    "model": "the final model of P13b part 2: its level, loss arm and "
    "penalty weight, P13a's knots and layout constants",
    "configurations": list(CONFIG_NAMES),
    "start": "each refit starts from the final model's parameters of its " "own terms",
    "selection": "full-catalog NDCG@100 of the val_last and val_first "
    "picks of the selection listeners",
    "sesoi": SESOI,
    "rules": {
        "R1": "a configuration without one or both terms is eligible when "
        "it scores more than SESOI above the full model, with the "
        "95 percent paired interval above 0",
        "R2": "the one-standard-error rule over the full model and the "
        "eligible configurations, ordered from the most terms to "
        "the fewest (a tie in terms by the higher score)",
        "R3": "the artist terms alone are reported, not chosen, and "
        "flagged when they score more than SESOI above the chosen "
        "configuration with the interval above 0",
    },
}


# Helpers


def _num(x):
    """A float for the json, or None when it is not finite."""
    x = float(x)
    return x if math.isfinite(x) else None


def paired_scores(a, b):
    """gam_audit.paired_summary of per-listener scores a - b, with NaN as
    None."""
    out = ga.paired_summary(a, b)
    return {k: (int(v) if k == "n" else _num(v)) for k, v in out.items()}


def config_specs(names):
    """[(name, kept terms, role)] of the five configurations, in the
    declared order. names: the final model's terms, in its order."""
    names = list(names)
    if len(set(names)) != len(names):
        raise ValueError("the final model names a term twice")
    missing = [t for t in (OTHER, POP) + ARTIST if t not in names]
    if missing:
        raise ValueError(
            f"the final model lacks the terms {missing} that "
            f"the configurations need"
        )
    kept = [
        tuple(names),
        tuple(t for t in names if t != OTHER),
        tuple(t for t in names if t != POP),
        tuple(t for t in names if t not in (OTHER, POP)),
        tuple(t for t in names if t in ARTIST),
    ]
    return list(zip(CONFIG_NAMES, kept, ROLES))


def layout_of(base, kept):
    """The layout of a configuration from P13a's stored constants, and the
    positions of its table entries in the union tables. kept: term names
    in the union's order."""
    kept = list(kept)
    terms = [t for t in base["union"] if t.name in kept]
    if [t.name for t in terms] != kept:
        raise ValueError(
            f"the terms {kept} are not all in the union, or " f"not in its order"
        )
    lay = gam.Layout(
        terms,
        {n: base["n_zero"][n] for n in kept},
        {n: base["n_miss"][n] for n in kept},
        [n for n in base["neutral"] if n in kept],
        {n: v for n, v in base["data_means"].items() if n in kept},
        {n: v for n, v in base["dev_means"].items() if n in kept},
    )
    idx = np.concatenate([base["union_off"][t.name] + np.arange(t.size) for t in terms])
    return lay, idx


def warm_start(full_layout, theta, layout):
    """The final model's parameters of the layout's terms, by name."""
    pos = {lab: k for k, lab in enumerate(full_layout.labels)}
    missing = [lab for lab in layout.labels if lab not in pos]
    if missing:
        raise ValueError(
            f"the parameters {missing[:3]} have no value in " f"the final model"
        )
    return np.asarray(theta, dtype=np.float64)[[pos[lab] for lab in layout.labels]]


def fit_id(name, level, arm, lam):
    return f"{name}|wide|{int(level)}|{s13b.arm_label(arm)}|" f"{float(lam)!r}"


def check_final_model(fit, base):
    """Raise ValueError unless the refit can use the final model (module
    docstring, checks). Returns the configurations."""
    terms = gkd.final_terms(fit)
    union = {t.name for t in base["union"]}
    extra = [t.name for t in terms if t.name not in union]
    if extra:
        raise ValueError(f"the final model has terms outside the union: " f"{extra}")
    if any(t.monotone for t in terms):
        raise ValueError(
            "the final model has a monotone term. The refit "
            "supports free shapes only"
        )
    gkd.check_every_candidate(terms)
    if float(fit["lambda"]) != float(base["lambda"]):
        raise ValueError(
            "the final model's penalty weight differs from P13b " "part 2's"
        )
    specs = config_specs([t.name for t in terms])
    lay, _ = layout_of(base, specs[0][1])
    if [t.to_json() for t in lay.terms] != fit["terms"] or gam_arms.layout_fingerprint(
        lay
    ) != fit["fingerprint"]:
        raise ValueError(
            "the final model's layout, rebuilt here, differs " "from the stored one"
        )
    return specs


def fit_lines(obj, label, t0, say):
    """Wraps the objective's Hessian and its loss and gradient with a line
    at most every PROGRESS_SECONDS. One Hessian is one Newton step, and an
    evaluation of the loss is one step of a line search or of the
    L-BFGS-B fallback of gam.fit(), so the line tells how far a long fit
    has come on either path. The wrappers only count, so they cannot change
    a result."""
    state = {"steps": 0, "evals": 0, "last": time.time()}
    hess_fn, loss_grad = obj.hess_fn, obj.loss_grad

    def tick():
        now = time.time()
        if now - state["last"] >= PROGRESS_SECONDS:
            state["last"] = now
            say(
                f"{WORK_NAME}: {label}: Newton step {state['steps']} done, "
                f"{state['evals']} evaluations of the loss, "
                f"{_clock(now - t0)} in this fit"
            )

    def hess(T, D):
        H = hess_fn(T, D)
        state["steps"] += 1
        tick()
        return H

    def loss(T, D):
        out = loss_grad(T, D)
        state["evals"] += 1
        tick()
        return out

    obj.hess_fn, obj.loss_grad = hess, loss
    return obj


# The stage names at the start of the lines that P13a's pass and P13b's
# probe write (StageP13b2 inherits both).
_STAGE_PREFIX = re.compile(r"^p13(?:a|b2|b)\b")


@contextlib.contextmanager
def _relabel_logs():
    """Renames the log lines of P13's selection pass while this script runs
    it, and points the advice of its swap warning at this script."""

    class Relabel(logging.Filter):
        def filter(self, record):
            msg = record.msg
            if isinstance(msg, str):
                msg = _STAGE_PREFIX.sub(WORK_NAME, msg, count=1)
                record.msg = msg.replace(
                    "python3 -m pipeline.runner --only p13a_fit",
                    "python3 -m pipeline.runner --only gam_refit",
                )
            return True

    flt = Relabel()
    log.addFilter(flt)
    try:
        yield
    finally:
        log.removeFilter(flt)


def setup_logging(path):
    """INFO lines to the console and to path (as gam_knn_diag.py). Each
    handler is added once."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter(LOG_FORMAT)
    fmt.converter = time.gmtime
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not any(type(h) is logging.StreamHandler for h in root.handlers):
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        root.addHandler(console)
    if not any(
        isinstance(h, logging.FileHandler) and Path(h.baseFilename) == path
        for h in root.handlers
    ):
        to_file = logging.FileHandler(path, encoding="utf-8")
        to_file.setFormatter(fmt)
        root.addHandler(to_file)
    if not (log.isEnabledFor(logging.INFO) and log.hasHandlers()):
        raise SystemExit(
            f"{WORK_NAME}: the pipeline logger drops INFO "
            f"lines, so this run would print nothing. Stopping."
        )


def _atomic_savez(path, **arrays):
    """As stage_p13a._atomic_savez: a stopped write leaves no npz."""
    s13._atomic_savez(path, **arrays)


def provenance(model_dir):
    """SHA-256 of the small input files, and of the markers of the stages
    behind the large ones."""
    d = Path(model_dir)
    out = {}
    for name in (
        "p13a_model.json",
        "p13a_arrays.npz",
        "p13b_model.json",
        "p13b2_model.json",
        "p13b2_arrays.npz",
        "itemknn_selection.json",
        "itemknn_selection_ranks.npz",
        "gam_knn_slices.npz",
    ):
        p = d / name
        out[name] = file_sha256(p) if p.exists() else None
    for stage in (
        "p6_als",
        "p9b_colisten",
        "p11b_more_pairs",
        "p12b_more_features",
        "p13b2_loss",
    ):
        try:
            out[f"marker:{stage}"] = file_sha256(marker_path(stage))
        except FileNotFoundError:
            out[f"marker:{stage}"] = None
    return out


# Inputs


def load_slices(model_dir, users, picks, stored_final):
    """The popularity centile and the co-listening class of every pick,
    from gam_knn_diag.py's stage 1, checked against P13's listeners and
    the final model's ranks."""
    p = Path(model_dir) / "gam_knn_slices.npz"
    if not p.exists():
        raise ValueError(
            "gam_knn_slices.npz is missing: run python3 "
            "-m pipeline.runner --only gam_knn_slices first"
        )
    z = gkd.load_npz(p)
    if not (
        np.array_equal(z["user_id"], users)
        and np.array_equal(z["picks"], picks)
        and np.array_equal(z["gam_rank"], stored_final)
    ):
        raise ValueError(
            "gam_knn_slices.npz belongs to other listeners, "
            "picks or another final model: run python3 "
            "-m pipeline.runner --only gam_knn_slices --force"
        )
    return {
        "pop_centile": np.asarray(z["pop_centile"], dtype=np.int64),
        "evidence": np.asarray(z["evidence"], dtype=np.int64),
    }


def load_knn(model_dir, users, picks):
    """ItemKNN's stored ranks of the same picks (both lists and the
    same-artist list, P5's tie rule), checked against its stored scores."""
    res, ranks, _, k_users = iks.load_outputs(model_dir)
    if not np.array_equal(k_users, users):
        raise ValueError(
            "ItemKNN's stored listeners differ from P13's " "selection listeners"
        )
    out = {}
    for key, v in (("both", ("both", "id")), ("same", ("same_artist", "id"))):
        name = knn.variant_name(*v)
        r = ranks[v]
        if r.shape != picks.shape:
            raise ValueError(
                f"ItemKNN's stored {name} ranks do not match " f"the picks"
            )
        stored = float(res["variants"][name]["score"])
        if abs(iks.score_picks(r)[0] - stored) > MEAN_TOL:
            raise ValueError(
                f"ItemKNN's stored {name} score does not follow "
                f"from its stored ranks"
            )
        out[key] = r
    return out


def fit_key(st, specs, model_sha):
    """What the saved fits depend on: P13b part 2's key (its parameters,
    the upstream outputs and the code of every module it runs), the final
    model file and the configurations."""
    return {
        "version": VERSION,
        "stage": "fits",
        "configs": [[n, list(k)] for n, k, _ in specs],
        "p13b2_model": model_sha,
        "p13b2_key": st._key(),
    }


def pass_key(specs, model_sha):
    """What the saved blocks depend on beyond the tables, the listeners and
    the blocks (StageP13a._selection_pass adds those)."""
    return {
        "version": VERSION,
        "stage": "pass",
        "configs": [[n, list(k)] for n, k, _ in specs],
        "p13b2_model": model_sha,
        "p13a_params": s13._params(),
        "code": {
            Path(m.__file__).name: file_sha256(Path(m.__file__))
            for m in (ct, cn, cd, gam, s13)
        },
    }


# Fits


def check_full(st, pool, ctx, lay, idx, arm, lam, theta, fit):
    """The final model's stored parameters on the rebuilt design: a
    stationary point with the stored objective, or ValueError."""
    obj = st._objective(pool, ctx, lay, idx, arm, lam)
    loss, smooth, ridge, grad = obj.parts(theta)
    F = loss + obj.lam_smooth * smooth + obj.lam_ridge * ridge
    stored = float(fit["fit"]["objective"])
    rel = abs(F - stored) / max(1.0, abs(F))
    pg = float(np.max(np.abs(gam.projected_gradient(theta, grad, lay.bounds))))
    if pg > s13.KKT_TOL:
        raise ValueError(
            f"the final model's stored parameters are not a "
            f"stationary point of the level's objective: the "
            f"largest projected gradient is {pg:.2e}"
        )
    if rel > s13.OBJ_RTOL:
        raise ValueError(
            f"the final model's objective on the rebuilt design "
            f"differs from the stored one by {rel:.2e}"
        )
    return {
        "theta": theta,
        "objective": F,
        "objective_stored": stored,
        "relative_gap": rel,
        "loss": loss,
        "smooth": smooth,
        "ridge": ridge,
        "proj_grad_max": pg,
    }


def fit_configs(st, base, fit, specs, store, say):
    """The full model's check and the refits, saved in the store, then the
    background means of the level. Returns (check record, {name: refit
    record}, means)."""
    level = int(fit["level"])
    arm = tuple(fit["arm"].split("-"))
    lam = float(fit["lambda"])
    lay_full, idx_full = layout_of(base, specs[0][1])
    theta_full = np.asarray(fit["theta"], dtype=np.float64)
    if theta_full.shape != (lay_full.P,):
        raise ValueError("the final model's parameters do not match its " "layout")
    check_id = fit_id(FULL_CHECK, level, arm, lam)
    ids = {name: fit_id(name, level, arm, lam) for name, _, _ in specs[1:]}
    todo = [(name, kept) for name, kept, _ in specs[1:] if ids[name] not in store]
    level_id = f"wide:{level}"
    means = store.means(level_id)
    if check_id in store and not todo and means is not None:
        say(
            f"{WORK_NAME}: the full model's check, every refit and the "
            f"background means are saved, so the rows of level {level} stay "
            f"on disk"
        )
    else:
        t0 = time.time()
        say(
            f"{WORK_NAME}: loading level {level} of the wide file (no "
            f"progress lines in this step)"
        )
        ctx = st._level_context(base, "wide", level, store)
        say(
            f"{WORK_NAME}: level {level}: {ctx['rows']:,} rows, "
            f"{ctx['listeners']:,} listeners, loaded in "
            f"{_clock(time.time() - t0)}"
        )
        fids = [check_id] + [ids[name] for name, _, _ in specs[1:]]
        if not all(f in store for f in fids):
            say(
                f"{WORK_NAME}: building the design of {len(base['union'])} "
                f"terms (no progress lines in this step)"
            )
        t1 = time.time()
        with st._design_for(ctx, store, fids) as pool:
            if pool is not None:
                say(f"{WORK_NAME}: design ready in {_clock(time.time() - t1)}")
            if check_id not in store:
                rec = check_full(
                    st, pool, ctx, lay_full, idx_full, arm, lam, theta_full, fit
                )
                store.put(check_id, rec)
                say(
                    f"{WORK_NAME}: the final model is a stationary point of "
                    f"the level's objective: largest projected gradient "
                    f"{rec['proj_grad_max']:.1e}, objective gap "
                    f"{rec['relative_gap']:.1e}"
                )
            prog = Progress(f"{WORK_NAME}: refits", len(todo), say, unit=" fits")
            for k, (name, kept) in enumerate(todo, start=1):
                lay, idx = layout_of(base, kept)
                start = warm_start(lay_full, theta_full, lay)
                obj = st._objective(pool, ctx, lay, idx, arm, lam)
                label = f"refit {k} of {len(todo)}, {name} ({len(kept)} " f"terms)"
                t2 = time.time()
                fit_lines(obj, label, t2, say)
                res = gam.fit(obj, start, s13b.SOLVER)
                res["seconds"] = time.time() - t2
                if res["proj_grad_max"] > s13.KKT_TOL:
                    raise ValueError(
                        f"the refit {name} stopped at a largest projected "
                        f"gradient of {res['proj_grad_max']:.2e}, above "
                        f"KKT_TOL ({res['message']}). It is not saved"
                    )
                res["fingerprint"] = gam_arms.layout_fingerprint(lay)
                store.put(ids[name], res)
                say(
                    f"{WORK_NAME}: {label}: objective {res['objective']:.8f},"
                    f" largest projected gradient {res['proj_grad_max']:.1e}, "
                    f"{res['message']}, {_clock(res['seconds'])}"
                )
                prog.advance(1)
            if todo:
                prog.finish()
        st._ensure_means(ctx, store)
        means = ctx["means_ref"]
        del ctx
    st._drop_inputs()
    check = store.get(check_id)
    refits = {}
    for name, kept, _ in specs[1:]:
        r = store.get(ids[name])
        lay, _ = layout_of(base, kept)
        if r["fingerprint"] != gam_arms.layout_fingerprint(lay):
            raise ValueError(f"the saved refit {name} belongs to another " f"layout")
        if r["proj_grad_max"] > s13.KKT_TOL:
            raise ValueError(
                f"the refit {name} stopped at a largest "
                f"projected gradient of "
                f"{r['proj_grad_max']:.2e}, above KKT_TOL"
            )
        refits[name] = r
    return check, refits, means


def fit_records(base, fit, specs, check, refits, means):
    """The json record of every configuration, and its union tables. The
    full model's centered tables are the stored ones, after the check that
    a centering here gives them again."""
    records, tables = [], np.zeros((len(specs), base["L_u"]))
    for c, (name, kept, role) in enumerate(specs):
        lay, idx = layout_of(base, kept)
        if c == 0:
            theta = np.asarray(fit["theta"], dtype=np.float64)
            extra = {
                "refit": False,
                "p13b2_fit": fit["fit"],
                "check": {k: v for k, v in check.items() if k != "theta"},
            }
        else:
            r = refits[name]
            theta = np.asarray(r["theta"], dtype=np.float64)
            extra = {"refit": True, "fit": {k: r[k] for k in FIT_KEYS}}
        T = lay.tables(theta)
        Tc, consts = gam.center(T, lay, {t.name: means[t.name] for t in lay.terms})
        if c == 0:
            if not np.array_equal(T, np.asarray(fit["tables_raw"])):
                raise ValueError(
                    "the final model's stored raw tables differ " "from M theta"
                )
            gap = float(np.max(np.abs(Tc - np.asarray(fit["tables"]))))
            if gap > s13b2.CENTER_TOL:
                raise ValueError(
                    f"the final model's stored centered tables "
                    f"differ from a centering on the level's "
                    f"p_pop negatives by {gap:.2e}"
                )
            # The model as P13b part 2 stored it: its tables and constants.
            Tc = np.asarray(fit["tables"], dtype=np.float64)
            consts = dict(fit["center_consts"])
            extra["check"]["centering_gap"] = gap
        terms_json = [t.to_json() for t in lay.terms]
        records.append(
            {
                "name": name,
                "role": role,
                "kept": list(kept),
                "terms": terms_json,
                "fingerprint": gam_arms.layout_fingerprint(lay),
                "theta": theta.tolist(),
                "tables_raw": T.tolist(),
                "tables": Tc.tolist(),
                "center_consts": consts,
                "points": s13b2.shape_points(terms_json, T.tolist(), base["shape_q"]),
                **extra,
            }
        )
        tables[c, idx] = Tc
    return records, tables


# The choice


def decide(scores, specs, sesoi=SESOI):
    """R1 to R3 of the module docstring from per-listener scores (C, n) in
    the order of specs."""
    s = np.asarray(scores, dtype=np.float64)
    names = [n for n, _, _ in specs]
    roles = [r for _, _, r in specs]
    n_terms = [len(k) for _, k, _ in specs]
    means = s.mean(axis=1)
    thr = sesoi * float(means[0])
    gate, eligible = {}, []
    for c, role in enumerate(roles):
        if role != "candidate":
            continue
        d = paired_scores(s[c], s[0])
        ok = bool(
            d["mean"] is not None
            and d["lo"] is not None
            and d["mean"] > thr
            and d["lo"] > 0.0
        )
        gate[names[c]] = {"difference": d, "eligible": ok}
        if ok:
            eligible.append(c)
    order = sorted([0] + eligible, key=lambda c: (-n_terms[c], -float(means[c]), c))
    chosen, best, rows = ga.one_se_choice(s, order)
    one_se = {
        names[c]: {
            "terms": n_terms[c],
            "mean": rows[c]["mean"],
            "diff_to_best": rows[c]["diff"],
            "se": _num(rows[c]["se"]),
        }
        for c in order
    }
    ref = roles.index("reference")
    d_ref = paired_scores(s[ref], s[chosen])
    flag = bool(
        d_ref["mean"] is not None
        and d_ref["lo"] is not None
        and d_ref["mean"] > thr
        and d_ref["lo"] > 0.0
    )
    return {
        "sesoi_abs": thr,
        "gate": gate,
        "eligible": [names[c] for c in eligible],
        "order": [names[c] for c in order],
        "best": names[best],
        "one_se": one_se,
        "chosen": names[chosen],
        "chosen_index": int(chosen),
        "chosen_terms": list(specs[chosen][1]),
        "reference": {"name": names[ref], "difference_to_chosen": d_ref, "flag": flag},
    }


def summarize(specs, ranks, s1, knn_ranks):
    """The results from the ranks (C, n, 2), stage 1's per-pick classes
    and ItemKNN's ranks. A pure function, so the check recomputes it."""
    names = [n for n, _, _ in specs]
    scores = ga.per_listener_scores(ranks)
    configs = {}
    for c, (name, kept, role) in enumerate(specs):
        configs[name] = {
            "terms": list(kept),
            "role": role,
            "score": float(scores[c].mean()),
            **gkd.type_blocks(ranks[c]),
            "diff_to_full": paired_scores(scores[c], scores[0]),
        }
    effects = {
        k: {x: (int(v) if x == "n" else _num(v)) for x, v in e.items()}
        for k, e in s13b.main_effects(
            scores[: len(FACTOR_CELLS)], FACTOR_CELLS, (OTHER, POP)
        ).items()
    }
    dec = decide(scores, specs)
    ch = dec["chosen_index"]
    ref = names.index(REFERENCE)
    k_both = ga.per_listener_scores(knn_ranks["both"])
    k_same = ga.per_listener_scores(knn_ranks["same"])
    itemknn = {
        "ItemKNN, both lists": float(k_both.mean()),
        "ItemKNN, same-artist list": float(k_same.mean()),
    }
    comparisons = {
        "chosen minus ItemKNN, both lists": paired_scores(scores[ch], k_both),
        "chosen minus ItemKNN, same-artist list": paired_scores(scores[ch], k_same),
        "artist terms only minus ItemKNN, same-artist list": paired_scores(
            scores[ref], k_same
        ),
        "full minus ItemKNN, both lists": paired_scores(scores[0], k_both),
    }
    band = gkd.pop_bands(s1["pop_centile"])
    ev_code = s1["evidence"]
    masks = [
        ("popularity band", lab, band == c) for c, lab in enumerate(gkd.POP_BANDS)
    ] + [
        ("co-listening evidence", lab, ev_code == c)
        for c, lab in enumerate(gkd.EVIDENCE)
    ]
    dcg = gkd.pick_dcg(ranks)
    slice_picks = {f"{dim}: {lab}": int(np.count_nonzero(m)) for dim, lab, m in masks}
    slices = {
        name: {
            f"{dim}: {lab}": gkd.cluster_mean(dcg[c], m)["mean"]
            for dim, lab, m in masks
        }
        for c, name in enumerate(names)
    }
    chosen_minus_full = {}
    for dim, lab, m in masks:
        x = gkd.cluster_mean(dcg[ch] - dcg[0], m)
        chosen_minus_full[f"{dim}: {lab}"] = {
            k: x[k] for k in ("mean", "se", "lo", "hi")
        }
    return {
        "listeners": int(scores.shape[1]),
        "sesoi_abs": dec["sesoi_abs"],
        "configs": configs,
        "effects": effects,
        "decision": dec,
        "itemknn": itemknn,
        "comparisons": comparisons,
        "slice_picks": slice_picks,
        "slices": slices,
        "chosen_minus_full": chosen_minus_full,
    }


# Checks


def verify_dense(st, base, tables, ranks, say, n_verify=VERIFY_USERS):
    """Raise ValueError unless every configuration's stored ranks of
    n_verify listeners lie in the range of catalog.reference_scores() and
    reference_rank(), censored at RANK_CUT + 1 (as gam_knn_diag.py)."""
    n_slots, _ = st._id_slots()
    space = st._space(st._sources(n_slots))
    u_indptr, u_indices, u_data = st._csr()
    users, picks, m_sel = base["sel_users"], base["sel_picks"], base["m_sel"]
    rng = np.random.default_rng(VERIFY_SEED)
    take = np.sort(
        rng.choice(len(users), size=min(n_verify, len(users)), replace=False)
    )
    cut1 = s13.RANK_CUT + 1
    near = 0
    prog = Progress(
        f"{WORK_NAME}: ranks against the dense reference",
        len(take),
        say,
        unit=" listeners",
    )
    for j in take:
        u = int(users[j])
        items, weights = cd.history_of(u_indptr, u_indices, u_data, u)
        state = cn.build_user_state(items, weights, space)
        full = cn.full_catalog(state, space)
        ref = ct.reference_scores(
            full, base["union"], tables, float(m_sel[j]), space.tier
        )
        del full
        for c in range(len(tables)):
            for p in range(picks.shape[1]):
                lo, hi = ct.reference_rank(ref[c], picks[j, p], items, s13.SCORE_TOL)
                near += int(hi > lo)
                got = int(ranks[c, j, p])
                if not min(lo, cut1) <= got <= min(hi, cut1):
                    raise ValueError(
                        f"listener {u}, configuration {c}, pick {p}: the "
                        f"rank {got} lies outside the dense reference's "
                        f"range {lo} to {hi}, censored at {cut1}"
                    )
        prog.advance(1)
    prog.finish()
    return {"listeners": int(len(take)), "near_ties": int(near)}


def check_outputs(model_dir, base, fit, specs, s1, knn_ranks, stored_final):
    """Raise ValueError unless the stored outputs agree with each other:
    the tables with the parameters, the points with the tables, the full
    model with P13b part 2's, and the results with the ranks."""
    d = Path(model_dir)
    doc = load_json(d / OUT_FILES[0])
    if doc is None:
        raise ValueError(f"{OUT_FILES[0]} is missing")
    z = gkd.load_npz(d / OUT_FILES[1])
    ranks = z["ranks"]
    if ranks.shape != (len(specs),) + tuple(stored_final.shape):
        raise ValueError("the stored ranks do not match the configurations")
    if not np.array_equal(z["user_id"], base["sel_users"]):
        raise ValueError("the stored listeners differ from P13's")
    if not np.array_equal(ranks[0], stored_final):
        raise ValueError(
            "the stored full model's ranks differ from P13b "
            "part 2's ranks of the final model"
        )
    if doc["configs"] != [[n, list(k), r] for n, k, r in specs]:
        raise ValueError("the stored configurations differ from the " "declared ones")
    T_u = z["tables"]
    if T_u.shape != (len(specs), base["L_u"]) or len(doc["fits"]) != len(specs):
        raise ValueError("the stored tables do not match the configurations")
    for c, ((name, kept, role), j) in enumerate(zip(specs, doc["fits"])):
        lay, idx = layout_of(base, kept)
        if (
            j["name"] != name
            or j["kept"] != list(kept)
            or j["role"] != role
            or j["terms"] != [t.to_json() for t in lay.terms]
            or j["fingerprint"] != gam_arms.layout_fingerprint(lay)
        ):
            raise ValueError(
                f"{name}: the stored terms or layout differ from "
                f"the configuration's"
            )
        theta = np.asarray(j["theta"], dtype=np.float64)
        T = lay.tables(theta)
        Tc = np.asarray(j["tables"], dtype=np.float64)
        if not np.array_equal(T, np.asarray(j["tables_raw"])):
            raise ValueError(
                f"{name}: the stored raw tables do not follow "
                f"from the stored parameters"
            )
        for t, off in zip(lay.terms, lay.offsets):
            cst = 0.0 if t.centered else float(j["center_consts"][t.name])
            want = T[off : off + t.size].copy()
            want[: t.K + 2] -= cst
            if not np.array_equal(Tc[off : off + t.size], want):
                raise ValueError(
                    f"{name}: the stored centered tables of "
                    f"{t.name} are not the raw tables minus "
                    f"the stored constant"
                )
        row = np.zeros(base["L_u"])
        row[idx] = Tc
        if not np.array_equal(row, T_u[c]):
            raise ValueError(
                f"{name}: the stored union tables differ from " f"the stored tables"
            )
        pts = s13b2.shape_points(j["terms"], j["tables_raw"], base["shape_q"])
        if json.dumps(pts, sort_keys=True) != json.dumps(j["points"], sort_keys=True):
            raise ValueError(
                f"{name}: the stored explanation points do not "
                f"follow from the tables"
            )
        if c == 0:
            if j["theta"] != fit["theta"] or j["tables"] != fit["tables"]:
                raise ValueError(
                    "the stored full model differs from P13b " "part 2's final fit"
                )
        elif j["fit"]["proj_grad_max"] > s13.KKT_TOL:
            raise ValueError(f"{name}: the stored refit is not stationary")
    again = summarize(specs, ranks, s1, knn_ranks)
    if json.dumps(again, sort_keys=True) != json.dumps(doc["results"], sort_keys=True):
        raise ValueError("the stored results do not follow from the stored " "ranks")
    ch = again["decision"]
    if doc["chosen"] != {
        "name": ch["chosen"],
        "index": ch["chosen_index"],
        "terms": ch["chosen_terms"],
    }:
        raise ValueError("the stored choice does not follow from the stored " "ranks")


def run(st, model_dir, work, say=print, n_verify=VERIFY_USERS, inputs=None):
    """The refit (module docstring). st: a StageP13b2, whose loaders give
    the level's rows, the item side, the CSR and the slots. Returns the
    output doc and writes the three outputs."""
    d = Path(model_dir)
    work = Path(work)
    t0 = time.time()
    doc, arr, i_final, fit = gkd.load_final(d)
    if doc["params"] != s13b2._params():
        raise ValueError(
            "p13b2_model.json was built with other P13b part 2 " "parameters"
        )
    base = st._base_context()
    users, picks = base["sel_users"], base["sel_picks"]
    stored_final = np.asarray(arr["ranks"][i_final])
    if not np.array_equal(arr["sel_users"], users):
        raise ValueError(
            "p13b2_arrays.npz holds other listeners than " "p13a_arrays.npz"
        )
    specs = check_final_model(fit, base)
    s1 = load_slices(d, users, picks, stored_final)
    knn_ranks = load_knn(d, users, picks)
    refs = [
        [name, float(score)]
        for name, score, _, _ in iks.references(d)
        if not name.startswith("GAM")
    ]
    model_sha = file_sha256(d / "p13b2_model.json")
    work.mkdir(parents=True, exist_ok=True)
    store = s13b2._Store(work / "fits.json", s13._sha(fit_key(st, specs, model_sha)))
    say(
        f"{WORK_NAME}: final model {fit['cell']}, arm {fit['arm']}, level "
        f"{fit['level']} (m = {fit['m']}), lambda {fit['lambda']:.3g}. "
        f"{len(specs) - 1} refits ({len(specs) - 2} candidates and the "
        f"reference arm), then one selection pass over {len(specs)} "
        f"configurations for {len(users):,} listeners. Saved work: {work}"
    )
    t_fit = time.time()
    check, refits, means = fit_configs(st, base, fit, specs, store, say)
    records, tables = fit_records(base, fit, specs, check, refits, means)
    fit_secs = time.time() - t_fit
    say(
        f"{WORK_NAME}: the fits are ready. Selection pass over "
        f"{len(specs)} configurations"
    )
    t_pass = time.time()
    with _relabel_logs():
        ranks, scored = st._selection_pass(
            base["union"],
            tables,
            users,
            picks,
            base["m_sel"],
            work / "pass",
            pass_key(specs, model_sha),
        )
    pass_secs = time.time() - t_pass
    if not np.array_equal(ranks[0], stored_final):
        n_bad = int(np.count_nonzero(ranks[0] != stored_final))
        raise ValueError(
            f"the full model's ranks differ from P13b part 2's "
            f"stored ranks of the final model in {n_bad} picks"
        )
    dense = verify_dense(st, base, tables, ranks, say, n_verify)
    n_slots = int(st._id_slots()[0])
    st._cache.pop("space", None)
    st._cache.pop("sources", None)
    results = summarize(specs, ranks, s1, knn_ranks)
    dec = results["decision"]
    out = {
        "version": VERSION,
        "protocol": PROTOCOL,
        "final": {
            "index": i_final,
            "cell": fit["cell"],
            "arm": fit["arm"],
            "level": fit["level"],
            "m": fit.get("m"),
            "lambda": fit["lambda"],
        },
        "configs": [[n, list(k), r] for n, k, r in specs],
        "fits": records,
        "results": results,
        "chosen": {
            "name": dec["chosen"],
            "index": dec["chosen_index"],
            "terms": dec["chosen_terms"],
        },
        "pass": {
            "configurations": len(specs),
            "listeners": int(len(users)),
            "full_pass_share": float(np.mean(scored == n_slots)),
            "mean_scored_share": float(np.mean(scored) / n_slots),
            "seconds": float(pass_secs),
        },
        "check": {"full_equals_stored": True, "dense": dense},
        "references": refs,
        "shape_quantiles": base["shape_q"],
        "inputs": dict(inputs or {}),
        "seconds": {
            "fits_this_run": float(fit_secs),
            "pass_this_run": float(pass_secs),
        },
    }
    _atomic_savez(
        d / OUT_FILES[1],
        user_id=users,
        ranks=ranks.astype(np.int32),
        scored=scored.astype(np.int64),
        tables=tables,
    )
    atomic_json_write(out, d / OUT_FILES[0])
    check_outputs(d, base, fit, specs, s1, knn_ranks, stored_final)
    out["seconds"]["total_this_run"] = time.time() - t0
    atomic_json_write(out, d / OUT_FILES[0])
    write_report(out, d / "reports" / REPORT_NAME)
    for name, c in results["configs"].items():
        say(f"{WORK_NAME}: {name}: score {c['score']:.5f}")
    say(f"{WORK_NAME}: chosen: {dec['chosen']}. Done in " f"{_clock(time.time() - t0)}")
    return out


# Report


def _f(x, fmt=".5f"):
    return "n/a" if x is None else format(x, fmt)


def _text(x):
    """Text from outside this script (a solver message) for the report,
    which holds no semicolon."""
    return str(x).replace(";", ",")


def _interval(dd, fmt="+.5f"):
    if dd.get("mean") is None:
        return "n/a"
    if dd.get("lo") is None:
        return format(dd["mean"], fmt)
    return (
        f"{format(dd['mean'], fmt)} ({format(dd['lo'], fmt)} to "
        f"{format(dd['hi'], fmt)})"
    )


def shape_row(j, name, shape_q):
    """(values at the positives' percentiles, zero, missing) of one term
    of a fit record, from its centered tables, as P13b's report."""
    terms = [gam.Term.from_json(t) for t in j["terms"]]
    T = np.asarray(j["tables"], dtype=np.float64)
    off = 0
    for t in terms:
        if t.name == name:
            tab = T[off : off + t.size]
            kn = t.knot_array
            xs = shape_q.get(name) or []
            vals = [
                float(np.interp(np.clip(x, kn[0], kn[-1]), kn, tab[: t.K])) for x in xs
            ]
            zero = f"{tab[t.K + gam.ZERO]:+.3f}" if t.zero_state else ""
            miss = f"{tab[t.K + gam.MISS]:+.3f}" if t.missing_state else ""
            return vals, zero, miss
        off += t.size
    raise KeyError(f"{j['name']} has no term {name}")


def point_moves(fits):
    """{term: {configuration: the largest absolute change of the term's
    explanation points against the full model}}."""
    full = fits[0]["points"]
    out = {}
    for name in full:
        out[name] = {}
        for j in fits[1:]:
            if name in j["points"]:
                out[name][j["name"]] = float(
                    np.max(
                        np.abs(np.asarray(j["points"][name]) - np.asarray(full[name]))
                    )
                )
    return out


def write_report(out, path):
    r = out["results"]
    dec = r["decision"]
    fin, ps, ck = out["final"], out["pass"], out["check"]
    fits = out["fits"]
    full_ck = fits[0]["check"]
    lines = ["# The final GAM refit without its two weakest terms", ""]
    add = lines.append
    add(
        f"- final model of P13b part 2: {fin['cell']}, arm {fin['arm']}, "
        f"level {fin['level']} (m = {fin['m']}), lambda "
        f"{fin['lambda']:.3g}. {r['listeners']:,} selection listeners, two "
        f"val picks each."
    )
    add(
        f"- protocol: the docstring of gam_refit.py, fixed "
        f"{out['protocol']['declared']}. SESOI {r['sesoi_abs']:.5f}, 2 "
        f"percent of the full model's score."
    )
    add(
        f"- selection pass: {ps['configurations']} configurations, "
        f"{100 * ps['full_pass_share']:.1f} percent of the listeners over "
        f"the whole catalog, {100 * ps['mean_scored_share']:.1f} percent of "
        f"all slots scored, {_clock(ps['seconds'])} in this run. Ranks are "
        f"exact up to 100 and stored as 101 beyond."
    )
    add(
        f"- checks: the full model's stored parameters are a stationary "
        f"point of the level's objective (largest projected gradient "
        f"{full_ck['proj_grad_max']:.1e}, objective gap "
        f"{full_ck['relative_gap']:.1e}, centering gap "
        f"{full_ck['centering_gap']:.1e}). Every refit reached the KKT "
        f"tolerance {s13.KKT_TOL:.0e}. The full model's ranks equal P13b "
        f"part 2's stored ranks for every pick. {ck['dense']['listeners']} "
        f"listeners got every configuration's ranks again from the dense "
        f"reference ({ck['dense']['near_ties']} checked ranks with a near "
        f"tie). The results follow from the stored ranks."
    )
    add("- intervals: 95 percent, paired over listeners.")
    add("")
    add("## 1. Fits")
    add("")
    add(
        "The full model is P13b part 2's fit, checked here and not refit. "
        "Each refit starts from the final model's parameters of its own "
        "terms."
    )
    add("")
    add(
        "| configuration | terms | objective | loss | fit | largest "
        "projected gradient | time |"
    )
    add("|---|---:|---:|---:|---|---:|---:|")
    for j in fits:
        if j["refit"]:
            f = j["fit"]
            add(
                f"| {j['name']} | {len(j['kept'])} | {f['objective']:.8f} | "
                f"{f['loss']:.8f} | {_text(f['message'])} | "
                f"{f['proj_grad_max']:.1e} | {_clock(f['seconds'])} |"
            )
        else:
            c = j["check"]
            add(
                f"| {j['name']} | {len(j['kept'])} | {c['objective']:.8f} | "
                f"{c['loss']:.8f} | P13b part 2: "
                f"{_text(j['p13b2_fit']['message'])} | "
                f"{c['proj_grad_max']:.1e} | "
                f"|"
            )
    add("")
    add("## 2. Selection scores")
    add("")
    add(
        "| configuration | terms | score | difference to the full model "
        "(95% interval) | val_last NDCG@100 | val_first NDCG@100 | val_last "
        "R@100 | val_first R@100 |"
    )
    add("|---|---:|---:|---|---:|---:|---:|---:|")
    for name, c in r["configs"].items():
        diff = "" if name == FULL else _interval(c["diff_to_full"])
        add(
            f"| {name} | {len(c['terms'])} | {c['score']:.5f} | {diff} | "
            f"{c['val_last']['ndcg_at_100']:.5f} | "
            f"{c['val_first']['ndcg_at_100']:.5f} | "
            f"{c['val_last']['recall_at_100']:.4f} | "
            f"{c['val_first']['recall_at_100']:.4f} |"
        )
    add("")
    add(
        "Effects over the 2 x 2: for a term, the mean over listeners of the "
        "average score of the two configurations with it minus that of the "
        "two without it. A positive effect means that the term helps. The "
        "interaction is the matching contrast of both terms."
    )
    add("")
    add("| effect | estimate | standard error | relative to the full model |")
    add("|---|---:|---:|---:|")
    full_score = r["configs"][FULL]["score"]
    for name, e in r["effects"].items():
        rel = f"{100 * e['mean'] / full_score:+.1f}%" if full_score else "n/a"
        add(f"| {name} | {e['mean']:+.5f} | {_f(e['se'])} | {rel} |")
    add("")
    add("## 3. Decision")
    add("")
    add(
        f"- R1, the gate: a configuration is eligible when it scores more "
        f"than SESOI ({dec['sesoi_abs']:.5f}) above the full model, with the "
        f"interval above 0."
    )
    for name, g in dec["gate"].items():
        add(
            f"  - {name}: {_interval(g['difference'])}. "
            + ("Eligible." if g["eligible"] else "Not eligible.")
        )
    add(
        f"- R2, the one-standard-error rule with the most terms first, over: "
        f"{', '.join(dec['order'])}. Best: {dec['best']}. Chosen: "
        f"{dec['chosen']}, the first configuration in this order whose score "
        f"lies within one standard error of the best. P13's own rule puts "
        f"the strongest penalty first. Here the order puts the smallest "
        f"change to the explanations first."
    )
    if len(dec["order"]) > 1:
        add("")
        add(
            "| configuration | terms | score | difference to the best | "
            "standard error |"
        )
        add("|---|---:|---:|---:|---:|")
        for name, row in dec["one_se"].items():
            add(
                f"| {name} | {row['terms']} | {row['mean']:.5f} | "
                f"{row['diff_to_best']:.5f} | {_f(row['se'])} |"
            )
        add("")
    rf = dec["reference"]
    add(
        f"- R3, the reference arm: {rf['name']} minus {dec['chosen']}: "
        f"{_interval(rf['difference_to_chosen'])}. "
        + (
            "Flag: the artist terms alone score more than SESOI above the "
            "chosen configuration. This gain is weighed against the "
            "explanation lines of the other terms."
            if rf["flag"]
            else "No flag: the artist terms alone do not score more than SESOI "
            "above the chosen configuration with the interval above 0."
        )
    )
    add(
        f"- the chosen model: {dec['chosen']}, with the terms "
        f"{', '.join(dec['chosen_terms'])}."
    )
    add("")
    add("## 4. Other models on the same picks")
    add("")
    add("| model | score |")
    add("|---|---:|")
    for name, c in r["configs"].items():
        add(f"| GAM, {name} | {c['score']:.5f} |")
    for name, score in r["itemknn"].items():
        add(f"| {name} | {score:.5f} |")
    for name, score in out["references"]:
        add(f"| {name} | {score:.5f} |")
    add("")
    shown = set()
    for name, dd in r["comparisons"].items():
        label = name.replace("chosen", dec["chosen"], 1)
        if label not in shown:  # the chosen model can be the full one
            shown.add(label)
            add(f"- {label}: {_interval(dd)}")
    add("")
    add("## 5. Slices")
    add("")
    add(
        "Mean NDCG@100 per pick in each slice of gam_knn_diag.py's stage 1. "
        "The first row gives the picks in the slice. When the chosen "
        "configuration is not the full model, the last row gives the chosen "
        "configuration minus the full model, with a 95 percent interval for "
        "listener clusters."
    )
    keys = list(r["slice_picks"])
    for dim in ("popularity band", "co-listening evidence"):
        cols = [k for k in keys if k.split(": ")[0] == dim]
        add("")
        add("| configuration | " + " | ".join(k.split(": ", 1)[1] for k in cols) + " |")
        add("|---|" + "---:|" * len(cols))
        add("| picks | " + " | ".join(f"{r['slice_picks'][k]:,}" for k in cols) + " |")
        for name, row in r["slices"].items():
            add(f"| {name} | " + " | ".join(_f(row[k]) for k in cols) + " |")
        if dec["chosen"] != FULL:
            add(
                f"| {dec['chosen']} minus full | "
                + " | ".join(_interval(r["chosen_minus_full"][k]) for k in cols)
                + " |"
            )
    add("")
    j = fits[dec["chosen_index"]]
    sq = out["shape_quantiles"]
    add(f"## 6. The chosen model ({dec['chosen']})")
    add("")
    add(
        "Values in log-odds points, centered on the p_pop negatives of the "
        "level, at percentiles of the positives' values. zero: data, but no "
        "match. missing: no data."
    )
    add("")
    add("| term | p5 | p25 | p50 | p75 | p95 | zero | missing |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|")
    for t in j["terms"]:
        vals, zero, miss = shape_row(j, t["name"], sq)
        add(
            f"| {t['name']} | "
            + " | ".join(f"{v:+.3f}" for v in vals)
            + f" | {zero} | {miss} |"
        )
    add("")
    add(
        "How far each term's explanation points move from the full model: "
        "the largest absolute change, in log-odds points, of the term's "
        "values at the positives' percentiles and of its zero and missing "
        "constants, each against the term's median value. An empty cell: "
        "the configuration drops the term."
    )
    add("")
    moves = point_moves(fits)
    others = [x["name"] for x in fits[1:]]
    add("| term | " + " | ".join(others) + " |")
    add("|---|" + "---:|" * len(others))
    for name, row in moves.items():
        add(
            f"| {name} | "
            + " | ".join(f"{row[o]:.3f}" if o in row else "" for o in others)
            + " |"
        )
    add("")
    text = "\n".join(lines) + "\n"
    if ";" in text:
        raise ValueError("the report holds a semicolon")
    atomic_text_write(text, path)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="The final GAM refit without colisten_other_artists, "
        "popularity or both (protocol in the module docstring)"
    )
    parser.parse_args(argv)
    setup_logging(LOG_PATH)
    say = log.info
    try:
        say(f"{WORK_NAME}: started, process id {os.getpid()}, log file " f"{LOG_PATH}")
        run(
            s13b2.StageP13b2(),
            MODEL_DIR,
            SCRATCH_DIR / WORK_NAME,
            say=say,
            inputs=provenance(MODEL_DIR),
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


# Runner stage


class StageGamRefit(ScriptStage):
    """The refit as a runner stage."""

    name = "gam_refit"
    upstream = ["p13b2_loss", "itemknn_select", "gam_knn_slices"]

    def outputs(self):
        d = self.model_dir
        return {
            "results": d / OUT_FILES[0],
            "arrays": d / OUT_FILES[1],
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
        d = self.model_dir
        _, arr, i_final, fit = gkd.load_final(d)
        base = self._stage()._base_context()
        users, picks = base["sel_users"], base["sel_picks"]
        stored_final = np.asarray(arr["ranks"][i_final])
        specs = check_final_model(fit, base)
        s1 = load_slices(d, users, picks, stored_final)
        knn_ranks = load_knn(d, users, picks)
        check_outputs(d, base, fit, specs, s1, knn_ranks, stored_final)


if __name__ == "__main__":
    main()
