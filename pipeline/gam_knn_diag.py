"""The final GAM against ItemKNN: a diagnostic in two stages, before P13c.

On the 20,000 selection listeners, ItemKNN (itemknn_select.py) ties the
final GAM of P13b part 2 (0.04111 against 0.04117), and its same-artist
list alone beats the GAM by 9.0 percent (0.04488). ItemKNN ranks by the
numerators of the GAM's two co-listening concepts, so the GAM reads all of
ItemKNN's evidence. This script looks for the reason. It fits nothing and
never reads a test pick.

Protocol, fixed before any result.

Listeners and picks: the selection listeners of P13 and their val_last
and val_first picks (p13a_arrays.npz). They are the listeners and picks of
itemknn_select.py and of P13b part 2. Metric: NDCG@100 per pick, with the
P5 candidate and tie rules. A model's score is the mean over listeners of
the mean over their two picks, the score of P13 and of itemknn_select.py.

Stage 1, "slices" (about 10 to 25 minutes, all cores, as ItemKNN):
- It reads the stored ranks of the final GAM (p13b2_arrays.npz, exact up
  to 100) and of ItemKNN (itemknn_selection_ranks.npz, exact).
- For every listener it computes the two co-listening values of every
  candidate from P9b's lists with pipeline/itemknn.py. The numerators
  equal the GAM's bit for bit. The values divide them by the listener's
  total weight, as the GAM's concepts do.
- Per pick: its co-listening evidence (same-artist, other-artist only,
  lists without overlap, no lists), whether its same-artist value lies on
  the flat top of the GAM's shape (at or above the top knot, where the
  shape is constant), its popularity band (the bands of P5) and the
  listener's history length (tertiles over the listeners).
- Two rankings of its own: ItemKNN's same-artist list and both lists, with
  each value clamped at the GAM's top knot of its concept and ties by item
  id (the P5 rule). They show what the flat top alone costs co-listening.
- Per slice: both models' mean NDCG@100, the paired difference with a 95
  percent interval for listener clusters, the picks that only one model
  puts in its top 100, and the mean of the better model per pick. The
  last is an upper bound for any switch between the two lists.
- Checks: the listeners of the three files are equal, the stored scores
  follow from the stored ranks, the recomputed ItemKNN ranks and counts
  equal the stored ones for every listener (so ItemKNN ranked the same
  picks), and CHECK_USERS listeners get their values again from a second
  route and their clamped ranks from a full sort.

Stage 2, "variants" (about 1 to 1.5 hours on all cores, then about 5
minutes of checks):
- One selection pass of P13 (StageP13a._selection_pass) over variants of
  the final model. The variants come from its stored tables, without a
  refit: the full model, the model without each term, each term alone,
  the two co-listening terms alone, the two artist terms alone
  (artist_affinity and colisten_same_artist) and the model without both
  co-listening terms. A variant sets the tables of the terms it drops to
  0. Every term of the final model applies to every candidate, so
  centering moves each term by one constant for all candidates, and a
  dropped term changes the ranks as the removal of the term itself does.
- Checks: the full variant's ranks equal P13b part 2's stored ranks of the
  final model for every listener and pick, and VERIFY_USERS listeners get
  every variant's ranks again from the dense reference of catalog.py.
- The pass saves its blocks under SCRATCH_DIR/gam_knn_diag. After Ctrl+C,
  the same command continues with the blocks still missing. The key of
  the saved blocks holds the variant tables, the listeners and the code
  that scores them (not this file), so a change to a report line keeps
  the blocks.

Reading rules (SESOI 2 percent, as in P13b part 2 and ALS phase D):
- S1, the flat top in co-listening: ItemKNN's same-artist list minus its
  clamped form, and the same for both lists. The flat top costs
  co-listening accuracy when the difference exceeds SESOI times the
  unclamped score and its interval lies above 0.
- S2, where the GAM loses: the slices whose interval for GAM minus
  ItemKNN (both lists) lies below 0.
- V1, the same-artist term: the term alone minus the clamped same-artist
  list. Within SESOI of the clamped list's score, the term ranks as well
  as its clamped values, and the gap to ItemKNN comes from the flat top
  (S1). Below that, with the interval below 0, the shape or the zero and
  missing constants lose more.
- V2, the other terms: the two co-listening terms alone minus the full
  model. When this exceeds SESOI times the full model's score and its
  interval lies above 0, the other terms push co-listening's good
  candidates down.
- V3, single terms: a term hurts when the model without it scores more
  than the full model and the interval lies above 0. The variants are not
  refits, so such a term is a candidate to drop, and a refit decides.
The results are read against four options: extend the top of the
shapes and refit (S1, V1), drop the terms that hurt (V3), combine the GAM
with ItemKNN (S2, V2 and the better-of-two bound), or keep the model and
state that it matches co-listening on accuracy and adds the other
reasons.

Outputs:
- MODEL_DIR/gam_knn_slices.json, gam_knn_slices.npz (per pick) and
  reports/gam_knn_slices.md
- MODEL_DIR/gam_knn_variants.json, gam_knn_variants.npz (the ranks) and
  reports/gam_knn_variants.md
Every line goes to the console and to MODEL_DIR/logs/gam_knn_diag.log.
Long steps print a progress line at most every 30 seconds, with the time
left.

Usage, from the repository root. Stage 1 runs through the runner,
stage 2 stays a diagnostic:
    python3 -m pipeline.runner --only gam_knn_slices
    python3 -m pipeline.gam_knn_diag variants
The module's own entry also runs stage 1, without a marker (the runner
then asks for --adopt):
    python3 -m pipeline.gam_knn_diag slices
The variants stage uses P13's worker setting. To use fewer workers:
    P13A_WORKERS=8 python3 -m pipeline.gam_knn_diag variants
"""

import argparse
import contextlib
import json
import logging
import math
import multiprocessing as mp
import os
import queue
import re
import signal
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from pipeline import catalog as ct
from pipeline import concept_data as cd
from pipeline import concepts as cn
from pipeline import evaluation as ev
from pipeline import gam
from pipeline import gam_audit as ga
from pipeline import itemknn as knn
from pipeline import stage_p13a as s13
from pipeline import stage_p13b2 as s13b2
from pipeline.als import Progress, _clock
from pipeline.common import (
    atomic_json_write,
    atomic_text_write,
    connect,
    file_sha256,
    load_json,
    log,
    marker_path,
)
from pipeline.config import MODEL_DIR, SCRATCH_DIR

from pipeline import itemknn_select as iks
from pipeline.script_stage import ScriptStage, log_file

LOG_PATH = MODEL_DIR / "logs" / "gam_knn_diag.log"
LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"  # as in runner.py
VERSION = "gam-knn-diag-v1"
SESOI = 0.02
Z95 = 1.959963984540054  # as gam_audit.paired_summary
PICK_TYPES = ("val_last", "val_first")
EVIDENCE = ("same-artist", "other-artist only", "lists, no overlap", "no lists")
TOP_LABELS = ("below the top knot", "on the flat top")
POP_BANDS = ("01-10", "11-50", "51-90", "91-100")
HISTORY_LABELS = ("short", "middle", "long")
COLISTEN = {
    "same_artist": "colisten_same_artist",
    "other_artists": "colisten_other_artists",
}
SHAPE_TERMS = ("colisten_same_artist", "colisten_other_artists", "artist_affinity")
SHAPE_TAIL = 4  # knots shown at the top of a shape
GROUPS = {
    "co-listening": ("colisten_same_artist", "colisten_other_artists"),
    "artist": ("artist_affinity", "colisten_same_artist"),
}
# Stage 1's rankings, in the order of the npz keys and the report.
MODELS = (
    ("gam_rank", "GAM, final model"),
    ("knn_both", "ItemKNN, both lists"),
    ("knn_same", "ItemKNN, same-artist list"),
    ("clamp_both", "ItemKNN, both lists, clamped at the top knots"),
    ("clamp_same", "ItemKNN, same-artist list, clamped at the top " "knot"),
)
USERS_PER_TASK = 32
HEARTBEAT_SECONDS = 60.0
CHECK_USERS = 20
CHECK_SEED = 2026092902
VERIFY_USERS = 6
VERIFY_SEED = 2026092903
SCORE_RTOL = 1e-12
MEAN_TOL = 1e-12
SLICES_FILES = ("gam_knn_slices.json", "gam_knn_slices.npz")
VARIANTS_FILES = ("gam_knn_variants.json", "gam_knn_variants.npz")
WORK_NAME = "gam_knn_diag"


# Helpers


def pick_dcg(ranks):
    """DCG@100 of each pick: 1 / log2(rank + 1) up to rank 100, else 0.
    A rank censored at 101 gives the same value as the exact rank."""
    r = np.asarray(ranks, dtype=np.float64)
    return np.where(r <= 100, 1.0 / np.log2(r + 1.0), 0.0)


def _num(x):
    """A float for the json, or None when it is not finite."""
    x = float(x)
    return x if math.isfinite(x) else None


def cluster_mean(x, mask):
    """The mean of the per-pick values x over the picks in mask, with its
    standard error for listener clusters: the linearized ratio estimator
    with the small-sample factor G / (G - 1), G the listeners with a pick
    in mask. x and mask: (listeners, picks). With every pick in mask this
    is the paired standard error of gam_audit.paired_summary()."""
    x = np.asarray(x, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    n_u = mask.sum(axis=1).astype(np.float64)
    tot = np.where(mask, x, 0.0).sum(axis=1)
    n = float(n_u.sum())
    g = int(np.count_nonzero(n_u))
    if n == 0:
        return {
            "mean": None,
            "se": None,
            "lo": None,
            "hi": None,
            "picks": 0,
            "listeners": 0,
        }
    m = float(tot.sum() / n)
    se = math.nan
    if g > 1:
        r = (tot - m * n_u)[n_u > 0]
        se = math.sqrt(g / (g - 1) * float(np.sum(r * r))) / n
    return {
        "mean": _num(m),
        "se": _num(se),
        "lo": _num(m - Z95 * se),
        "hi": _num(m + Z95 * se),
        "picks": int(n),
        "listeners": g,
    }


def paired(a_ranks, b_ranks):
    """gam_audit.paired_summary of the listeners' scores under two
    rankings of the same picks, with NaN as None."""
    out = ga.paired_summary(
        ga.per_listener_scores(a_ranks), ga.per_listener_scores(b_ranks)
    )
    return {k: (_num(v) if k != "n" else int(v)) for k, v in out.items()}


def score_of(ranks):
    return float(ga.per_listener_scores(ranks).mean())


def type_blocks(ranks):
    """{pick type: NDCG@100, R@10 and R@100}. Censored ranks suffice."""
    out = {}
    for p, name in enumerate(PICK_TYPES):
        r = np.asarray(ranks)[:, p]
        out[name] = {
            "ndcg_at_100": float(pick_dcg(r).mean()),
            "recall_at_10": float(np.mean(r <= 10)),
            "recall_at_100": float(np.mean(r <= 100)),
        }
    return out


def pop_bands(centile):
    """P5's popularity bands of pop_centile (stage_p5.variant_block)."""
    c = np.asarray(centile)
    return np.where(c <= 10, 0, np.where(c <= 50, 1, np.where(c <= 90, 2, 3)))


def evidence_codes(value_same, value_other):
    """Index into EVIDENCE per pick. A value is NaN where the pick has no
    list row of that kind."""
    vs = np.asarray(value_same, dtype=np.float64)
    vo = np.asarray(value_other, dtype=np.float64)
    has_s, has_o = ~np.isnan(vs), ~np.isnan(vo)
    code = np.full(vs.shape, 3, dtype=np.int8)
    code[has_s | has_o] = 2
    code[has_o & (np.nan_to_num(vo) > 0.0)] = 1
    code[has_s & (np.nan_to_num(vs) > 0.0)] = 0
    return code


def on_top(values, top):
    """True where a value (NaN: no data) lies at or above the top knot.
    The GAM clamps such values to the top knot, so they share one shape
    value."""
    v = np.asarray(values, dtype=np.float64)
    out = np.zeros(v.shape, dtype=bool)
    ok = ~np.isnan(v)
    out[ok] = v[ok] >= top
    return out


def has_rows(indptr):
    """True for the slots whose list row holds a neighbour. Slot 0 never
    has data, as in concepts.ItemSide."""
    has = np.diff(np.asarray(indptr, dtype=np.int64)) > 0
    has[0] = False
    return has


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
                    "python3 -m pipeline.gam_knn_diag variants",
                )
            return True

    flt = Relabel()
    log.addFilter(flt)
    try:
        yield
    finally:
        log.removeFilter(flt)


def _worker_init():
    """Workers ignore Ctrl+C. The main process handles it and ends them."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _stop_pool(pool):
    """End a pool at once: cancel the tasks not yet started and terminate
    the workers (as als_freq._stop_pool)."""
    try:
        pool.terminate_workers()  # Python 3.14 and later
    except Exception:  # older Python, or already gone
        for p in list((getattr(pool, "_processes", None) or {}).values()):
            try:
                p.terminate()
            except Exception:  # already ended
                pass
    pool.shutdown(wait=False, cancel_futures=True)


def run_tasks(fn, tasks, workers, prog, label, say):
    """fn over every (lo, hi) task. Returns the results in task order, so
    the order in which tasks finish cannot change a result. A task counts
    toward the progress as soon as it finishes. While no task finishes, a
    line appears every HEARTBEAT_SECONDS."""
    parts = [None] * len(tasks)
    if workers <= 1 or len(tasks) <= 1:
        for k, task in enumerate(tasks):
            parts[k] = fn(task)
            prog.advance(task[1] - task[0])
        return parts
    t0 = time.time()
    finished = queue.SimpleQueue()
    pool = ProcessPoolExecutor(
        max_workers=workers, mp_context=mp.get_context("fork"), initializer=_worker_init
    )
    try:
        futures = []
        for k, task in enumerate(tasks):
            f = pool.submit(fn, task)
            f.add_done_callback(lambda _f, k=k: finished.put(k))
            futures.append(f)
        left = len(tasks)
        while left:
            try:
                k = finished.get(timeout=HEARTBEAT_SECONDS)
            except queue.Empty:
                say(
                    f"{label}: working, {len(tasks) - left:,} of "
                    f"{len(tasks):,} tasks done, "
                    f"{_clock(time.time() - t0)} elapsed"
                )
                continue
            parts[k] = futures[k].result()  # raises a worker's error
            prog.advance(tasks[k][1] - tasks[k][0])
            left -= 1
    except BaseException:
        _stop_pool(pool)
        raise
    pool.shutdown(wait=True)
    return parts


def setup_logging(path):
    """INFO lines to the console and to path (as itemknn_select.py). Each
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
            "gam_knn_diag: the pipeline logger drops INFO "
            "lines, so this run would print nothing. Stopping."
        )


def _atomic_savez(path, **arrays):
    """As stage_p13a._atomic_savez: a stopped write leaves no npz."""
    s13._atomic_savez(path, **arrays)


def load_npz(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


def provenance(model_dir):
    """SHA-256 of the small input files, and of the markers of the stages
    behind the large ones (P6's CSR, P9b's lists)."""
    d = Path(model_dir)
    out = {}
    for name in (
        "p13a_arrays.npz",
        "p13b2_model.json",
        "p13b2_arrays.npz",
        "itemknn_selection.json",
        "itemknn_selection_ranks.npz",
    ):
        p = d / name
        out[name] = file_sha256(p) if p.exists() else None
    for stage in ("p6_als", "p9b_colisten"):
        try:
            out[f"marker:{stage}"] = file_sha256(marker_path(stage))
        except FileNotFoundError:
            out[f"marker:{stage}"] = None
    return out


# The final model


def load_final(model_dir):
    """(P13b part 2's model doc, its arrays, the final index, the final
    fit), checked against each other."""
    d = Path(model_dir)
    doc = load_json(d / "p13b2_model.json")
    if doc is None:
        raise ValueError("p13b2_model.json is missing: run P13b part 2 " "first")
    arr = load_npz(d / "p13b2_arrays.npz")
    fits = doc["wide_fits"]
    i = int(doc["final"]["index"])
    if not 0 <= i < len(fits):
        raise ValueError(
            f"the final index {i} lies outside the "
            f"{len(fits)} fits of p13b2_model.json"
        )
    ranks = arr["ranks"]
    if (
        ranks.ndim != 3
        or ranks.shape[0] != len(fits)
        or ranks.shape[2] != 2
        or ranks.shape[1] != len(arr["sel_users"])
    ):
        raise ValueError(
            f"p13b2_arrays.npz holds ranks of shape "
            f"{ranks.shape}, the model has {len(fits)} fits and "
            f"{len(arr['sel_users'])} listeners"
        )
    fit = fits[i]
    for key in ("cell", "arm", "level"):
        if fit[key] != doc["final"][key]:
            raise ValueError(
                f"the final fit's {key} differs from the " f"final model's"
            )
    return doc, arr, i, fit


def final_terms(fit):
    return [gam.Term.from_json(t) for t in fit["terms"]]


def top_knots(terms):
    """{list kind: the top knot of the GAM's term for it}."""
    out = {}
    for kind, name in COLISTEN.items():
        t = next((t for t in terms if t.name == name), None)
        if t is None:
            raise ValueError(f"the final model has no term {name}")
        if t.source != name or not (t.zero_state and t.missing_state):
            raise ValueError(
                f"the term {name} does not read its concept "
                f"with a zero and a missing state"
            )
        out[kind] = float(t.knots[-1])
    return out


def shape_summary(fit):
    """For SHAPE_TERMS: the top SHAPE_TAIL knots and their heights, the
    largest height and its knot, the zero and the missing constant, from
    the final fit's centered tables in its own layout."""
    terms = final_terms(fit)
    T = np.asarray(fit["tables"], dtype=np.float64)
    out, off = {}, 0
    for t in terms:
        if t.name in SHAPE_TERMS:
            K = t.K
            h = T[off : off + K]
            k_max = int(np.argmax(h))
            out[t.name] = {
                "n_knots": K,
                "knots": [float(x) for x in t.knots[-SHAPE_TAIL:]],
                "heights": [float(x) for x in h[-SHAPE_TAIL:]],
                "top_knot": float(t.knots[-1]),
                "max_height": float(h[k_max]),
                "max_knot": float(t.knots[k_max]),
                "top_minus_max": float(h[-1] - h[k_max]),
                "zero": float(T[off + K + gam.ZERO]) if t.zero_state else None,
                "missing": float(T[off + K + gam.MISS]) if t.missing_state else None,
            }
        off += t.size
    if off != len(T):
        raise ValueError("the final fit's tables do not match its terms")
    return out


# Stage 1: slices
_CTX = {}


def listener_facts(t_mats, u_arrays, user, picks, top):
    """Stage 1's numbers for one listener.

    Returns (ranks (4, picks), values (2, picks), counts (4,), total):
    - ranks: ItemKNN's same-artist list, both lists (as itemknn.py ranks
      them), and the clamped same-artist and both lists, all with ties by
      item id.
    - values: the same-artist and other-artist values of the picks,
      numerator / total, the GAM's concept values. A pick without a list
      row gets 0 here (the caller marks it as no data).
    - counts over the candidates: values on the flat top (same-artist,
      other-artist), and positive numerators (same-artist, both lists).
    - total: the listener's total weight W."""
    items, weights = knn.history(*u_arrays, user)
    if len(items) == 0:
        raise ValueError(f"listener {user} has no train items")
    picks = np.asarray(picks, dtype=np.int64)
    numer = knn.user_scores(t_mats, items, weights)
    total = float(weights.sum())  # the order of concepts.build_user_state
    cand = np.ones(len(numer["both"]), dtype=bool)
    cand[0] = False
    cand[items] = False
    ranks = np.empty((4, len(picks)), dtype=np.int64)
    counts = np.empty(4, dtype=np.int64)
    ranks[0] = knn.pick_ranks(
        knn.candidate_scores(numer["same_artist"], items), picks, "id"
    )
    ranks[1] = knn.pick_ranks(knn.candidate_scores(numer["both"], items), picks, "id")
    counts[2] = np.count_nonzero(cand & (numer["same_artist"] > 0.0))
    counts[3] = np.count_nonzero(cand & (numer["both"] > 0.0))
    v_same = numer["same_artist"] / total
    v_other = numer["other_artists"] / total
    del numer
    values = np.stack([v_same[picks], v_other[picks]])
    counts[0] = np.count_nonzero(cand & (v_same >= top["same_artist"]))
    counts[1] = np.count_nonzero(cand & (v_other >= top["other_artists"]))
    np.minimum(v_same, top["same_artist"], out=v_same)
    np.minimum(v_other, top["other_artists"], out=v_other)
    ranks[2] = knn.pick_ranks(knn.candidate_scores(v_same, items), picks, "id")
    both = v_same + v_other
    del v_other
    ranks[3] = knn.pick_ranks(knn.candidate_scores(both, items), picks, "id")
    return ranks, values, counts, total


def _slice_task(task):
    lo, hi = task
    c = _CTX
    k_p = c["picks"].shape[1]
    ranks = np.empty((4, hi - lo, k_p), dtype=np.int64)
    values = np.empty((2, hi - lo, k_p), dtype=np.float64)
    counts = np.empty((4, hi - lo), dtype=np.int64)
    totals = np.empty(hi - lo, dtype=np.float64)
    for j, k in enumerate(range(lo, hi)):
        r, v, n, w = listener_facts(
            c["t_mats"], c["u_arrays"], int(c["users"][k]), c["picks"][k], c["top"]
        )
        ranks[:, j], values[:, j], counts[:, j], totals[j] = r, v, n, w
    return ranks, values, counts, totals


def facts_all(t_mats, u_arrays, users, picks, top, workers, say):
    """listener_facts() for every listener: (ranks (4, n, 2), values
    (2, n, 2), counts (4, n), totals (n,)). The result does not depend on
    the worker count."""
    n = len(users)
    if n == 0:
        raise ValueError("there are no listeners to score")
    tasks = [(lo, min(lo + USERS_PER_TASK, n)) for lo in range(0, n, USERS_PER_TASK)]
    label = f"{WORK_NAME}: co-listening values of every candidate"
    prog = Progress(label, n, say, unit=" listeners")
    _CTX.update(t_mats=t_mats, u_arrays=u_arrays, users=users, picks=picks, top=top)
    try:
        parts = run_tasks(_slice_task, tasks, workers, prog, label, say)
    finally:
        _CTX.clear()
    prog.finish()
    ranks = np.concatenate([p[0] for p in parts], axis=1)
    values = np.concatenate([p[1] for p in parts], axis=1)
    counts = np.concatenate([p[2] for p in parts], axis=1)
    totals = np.concatenate([p[3] for p in parts])
    return ranks, values, counts, totals


def summarize_slices(z, top):
    """The results of stage 1 from its per-pick arrays z. A pure function
    of z and top, so the check recomputes it from the stored npz."""
    g, k = z["gam_rank"], z["knn_both"]
    dg, dk = pick_dcg(g), pick_dcg(k)
    hg, hk = g <= 100, k <= 100
    models = {}
    for key, name in MODELS:
        models[name] = {"key": key, "score": score_of(z[key]), **type_blocks(z[key])}
    diffs = {
        "GAM minus ItemKNN, both lists": paired(g, k),
        "GAM minus ItemKNN, same-artist list": paired(g, z["knn_same"]),
        "ItemKNN same-artist list minus its clamped form": paired(
            z["knn_same"], z["clamp_same"]
        ),
        "ItemKNN both lists minus their clamped form": paired(k, z["clamp_both"]),
    }
    better = np.maximum(dg, dk)

    def row(mask):
        out = cluster_mean(dg - dk, mask)
        n_p = out["picks"]
        return {
            "picks": n_p,
            "listeners": out["listeners"],
            "gam_ndcg": cluster_mean(dg, mask)["mean"],
            "knn_ndcg": cluster_mean(dk, mask)["mean"],
            "diff": {x: out[x] for x in ("mean", "se", "lo", "hi")},
            "gam_r100": _num(np.count_nonzero(mask & hg) / n_p) if n_p else None,
            "knn_r100": _num(np.count_nonzero(mask & hk) / n_p) if n_p else None,
            "both": int(np.count_nonzero(mask & hg & hk)),
            "gam_only": int(np.count_nonzero(mask & hg & ~hk)),
            "knn_only": int(np.count_nonzero(mask & ~hg & hk)),
            "neither": int(np.count_nonzero(mask & ~hg & ~hk)),
            "better": cluster_mean(better, mask)["mean"],
        }

    n_users = g.shape[0]
    every = np.ones(g.shape, dtype=bool)
    hist = np.asarray(z["history"], dtype=np.float64)
    cuts = [float(x) for x in np.quantile(hist, [1.0 / 3.0, 2.0 / 3.0])]
    h_code = np.where(hist <= cuts[0], 0, np.where(hist <= cuts[1], 1, 2))
    h_code = np.repeat(h_code[:, None], g.shape[1], axis=1)
    ev_code = z["evidence"]
    band = pop_bands(z["pop_centile"])
    same_ev = ev_code == 0
    slices = {
        "all picks": {"all": row(every)},
        "pick type": {
            name: row(every & (np.arange(g.shape[1])[None, :] == p))
            for p, name in enumerate(PICK_TYPES)
        },
        "co-listening evidence": {
            name: row(ev_code == c) for c, name in enumerate(EVIDENCE)
        },
        "same-artist value": {
            TOP_LABELS[0]: row(same_ev & ~z["flat_same"]),
            TOP_LABELS[1]: row(same_ev & z["flat_same"]),
        },
        "popularity band": {name: row(band == c) for c, name in enumerate(POP_BANDS)},
        "history length": {
            name: row(h_code == c) for c, name in enumerate(HISTORY_LABELS)
        },
    }
    flat = z["flat_same"]
    n_flat = np.repeat(z["n_flat_same"][:, None], g.shape[1], axis=1)
    q = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)
    flat_stats = {
        "picks_on_flat_top": int(np.count_nonzero(flat)),
        "picks_with_same_artist_evidence": int(np.count_nonzero(same_ev)),
        "n_flat_of_their_listener": (
            {f"q{int(100 * x)}": float(np.quantile(n_flat[flat], x)) for x in q}
            if flat.any()
            else None
        ),
        "share_with_100_or_more_on_top": (
            float(np.mean(n_flat[flat] >= 100)) if flat.any() else None
        ),
        "listeners_median_flat_same": float(np.median(z["n_flat_same"])),
        "listeners_median_flat_other": float(np.median(z["n_flat_other"])),
        "listeners_median_positive_same": float(np.median(z["n_pos_same"])),
        "listeners_median_positive_both": float(np.median(z["n_pos_both"])),
        "listeners_with_any_on_top": int(np.count_nonzero(z["n_flat_same"] > 0)),
    }
    s1 = {}
    for label, a, b in (
        ("same-artist list", "knn_same", "clamp_same"),
        ("both lists", "knn_both", "clamp_both"),
    ):
        d = paired(z[a], z[b])
        base = score_of(z[a])
        s1[label] = {
            "difference": d,
            "unclamped_score": base,
            "holds": bool(
                d["mean"] is not None
                and d["lo"] is not None
                and d["mean"] > SESOI * base
                and d["lo"] > 0.0
            ),
        }
    s2 = []
    for dim, rows in slices.items():
        for name, r in rows.items():
            if r["picks"] and r["diff"]["hi"] is not None and r["diff"]["hi"] < 0.0:
                s2.append(f"{dim}: {name}")
    return {
        "listeners": int(n_users),
        "top_knots": dict(top),
        "models": models,
        "differences": diffs,
        "better_of_two": float(np.mean(better.mean(axis=1))),
        "history_cuts": cuts,
        "slices": slices,
        "flat_top": flat_stats,
        "readings": {"S1": s1, "S2": s2},
    }


def check_slices(
    model_dir, t_mats, lists, u_arrays, n_slots, pop_centile, say, n_check=CHECK_USERS
):
    """Raise ValueError unless the stored outputs of stage 1 agree with
    their inputs and follow from a second computation. Returns the largest
    relative value gap."""
    d = Path(model_dir)
    doc = load_json(d / SLICES_FILES[0])
    if doc is None:
        raise ValueError(f"{SLICES_FILES[0]} is missing")
    z = load_npz(d / SLICES_FILES[1])
    _, arr, i_final, fit = load_final(d)
    a_arr = load_npz(d / "p13a_arrays.npz")
    _, k_ranks, k_pos, _ = iks.load_outputs(d)
    users, picks = z["user_id"], z["picks"]
    if not (
        np.array_equal(users, a_arr["sel_users"])
        and np.array_equal(picks, a_arr["sel_picks"])
    ):
        raise ValueError(
            "the stored listeners or picks differ from P13's "
            "selection listeners and picks"
        )
    if not np.array_equal(z["gam_rank"], arr["ranks"][i_final]):
        raise ValueError(
            "the stored GAM ranks differ from P13b part 2's " "ranks of the final model"
        )
    if not (
        np.array_equal(z["knn_both"], k_ranks[("both", "id")])
        and np.array_equal(z["knn_same"], k_ranks[("same_artist", "id")])
        and np.array_equal(z["n_pos_both"], k_pos)
    ):
        raise ValueError(
            "the stored ItemKNN ranks or counts differ from "
            "itemknn_selection_ranks.npz"
        )
    top = top_knots(final_terms(fit))
    if {k: float(v) for k, v in doc["results"]["top_knots"].items()} != top:
        raise ValueError("the stored top knots differ from the final " "model's")
    if not (
        np.array_equal(z["pop_centile"], np.asarray(pop_centile)[picks])
        and np.array_equal(
            z["history"], np.diff(np.asarray(u_arrays[0], dtype=np.int64))[users]
        )
    ):
        raise ValueError(
            "the stored popularity centiles or history lengths "
            "differ from their sources"
        )
    if not np.array_equal(
        z["evidence"], evidence_codes(z["value_same"], z["value_other"])
    ):
        raise ValueError(
            "the stored evidence classes do not follow from " "the stored values"
        )
    if not (
        np.array_equal(z["flat_same"], on_top(z["value_same"], top["same_artist"]))
        and np.array_equal(
            z["flat_other"], on_top(z["value_other"], top["other_artists"])
        )
    ):
        raise ValueError(
            "the stored flat-top flags do not follow from the " "stored values"
        )
    has = {kind: has_rows(lists[kind][0]) for kind in knn.KINDS}
    for kind, key in (("same_artist", "value_same"), ("other_artists", "value_other")):
        if not np.array_equal(np.isnan(z[key]), ~has[kind][picks]):
            raise ValueError(
                f"the stored {key} marks other picks as "
                f"without data than the lists do"
            )
    rng = np.random.default_rng(CHECK_SEED)
    take = np.sort(rng.choice(len(users), size=min(n_check, len(users)), replace=False))
    worst = 0.0
    names = ("clamp_same", "clamp_both")
    for j in take:
        u = int(users[j])
        items, weights = knn.history(*u_arrays, u)
        total = float(weights.sum())
        if float(z["weight_total"][j]) != total:
            raise ValueError(
                f"listener {u}: the stored total weight " f"differs from the history's"
            )
        slow = knn.brute_scores(lists, items, weights, n_slots)
        v2 = {kind: slow[kind] / total for kind in knn.KINDS}
        for kind, key in (
            ("same_artist", "value_same"),
            ("other_artists", "value_other"),
        ):
            got = z[key][j]
            want = np.where(has[kind][picks[j]], v2[kind][picks[j]], np.nan)
            ok = np.isnan(got) == np.isnan(want)
            gap = np.abs(np.nan_to_num(got) - np.nan_to_num(want)) / np.maximum(
                1.0, np.abs(np.nan_to_num(want))
            )
            worst = max(worst, float(gap.max()))
            if not ok.all() or gap.max() > SCORE_RTOL:
                raise ValueError(
                    f"listener {u}: the stored {key} differs " f"from the second route"
                )
        cand = np.ones(n_slots, dtype=bool)
        cand[0] = False
        cand[items] = False
        for kind, key in (
            ("same_artist", "n_flat_same"),
            ("other_artists", "n_flat_other"),
        ):
            tol = SCORE_RTOL * max(1.0, top[kind])
            lo = int(np.count_nonzero(cand & (v2[kind] >= top[kind] + tol)))
            hi = int(np.count_nonzero(cand & (v2[kind] >= top[kind] - tol)))
            if not lo <= int(z[key][j]) <= hi:
                raise ValueError(
                    f"listener {u}: the stored {key} "
                    f"{int(z[key][j])} lies outside the second "
                    f"route's range {lo} to {hi}"
                )
        for key, score in (
            ("n_pos_same", slow["same_artist"]),
            ("n_pos_both", slow["both"]),
        ):
            if int(z[key][j]) != int(np.count_nonzero(cand & (score > 0.0))):
                raise ValueError(
                    f"listener {u}: the stored {key} differs " f"from the second route"
                )
        fast = knn.user_scores(t_mats, items, weights)
        c_fast = {kind: np.minimum(fast[kind] / total, top[kind]) for kind in knn.KINDS}
        c_slow = {kind: np.minimum(v2[kind], top[kind]) for kind in knn.KINDS}
        s_fast = {
            "clamp_same": c_fast["same_artist"],
            "clamp_both": c_fast["same_artist"] + c_fast["other_artists"],
        }
        s_slow = {
            "clamp_same": c_slow["same_artist"],
            "clamp_both": c_slow["same_artist"] + c_slow["other_artists"],
        }
        for key in names:
            sf = knn.candidate_scores(s_fast[key], items)
            ss = knn.candidate_scores(s_slow[key], items)
            for p in range(picks.shape[1]):
                t = int(picks[j, p])
                got = int(z[key][j, p])
                want = knn.sorted_rank(sf, t, "id")
                if got != want:
                    raise ValueError(
                        f"listener {u}, {key}: stored rank "
                        f"{got}, a full sort gives {want}"
                    )
                tol = SCORE_RTOL * max(1.0, abs(float(ss[t])))
                lo, hi = knn.rank_bounds(ss, t, tol)
                if not lo <= got <= hi:
                    raise ValueError(
                        f"listener {u}, {key}: stored rank "
                        f"{got} lies outside the second route's "
                        f"range {lo} to {hi}"
                    )
    again = summarize_slices(z, top)
    if json.dumps(again, sort_keys=True) != json.dumps(doc["results"], sort_keys=True):
        raise ValueError(
            "the stored results do not follow from the stored " "per-pick arrays"
        )
    say(
        f"{WORK_NAME}: check passed on {len(take)} listeners, largest "
        f"relative value gap {worst:.1e}"
    )
    return worst


def load_slice_inputs(model_dir, say):
    """(n_slots, u_arrays, lists, pop_centile) from the real files."""
    d = Path(model_dir)
    n_slots, _ = ev.load_id_slots()
    u_arrays = (
        np.load(d / "csr_user_indptr.npy"),
        np.load(d / "csr_user_indices.npy", mmap_mode="r"),
        np.load(d / "csr_user_data.npy", mmap_mode="r"),
    )
    say(
        f"{WORK_NAME}: loading P9b's co-listening lists (no progress lines "
        f"in this step)"
    )
    lists = knn.load_lists(d, n_slots)
    say(
        f"{WORK_NAME}: loading the popularity centiles with DuckDB (no "
        f"progress lines in this step)"
    )
    con = connect(4)
    try:
        pop_centile = ev.covariate_arrays(con, n_slots)[0]
    finally:
        con.close()
    return n_slots, u_arrays, lists, pop_centile


def run_slices(
    model_dir,
    n_slots,
    u_arrays,
    lists,
    pop_centile,
    workers=None,
    say=print,
    n_check=CHECK_USERS,
    inputs=None,
):
    """Stage 1 (module docstring). Returns the results and writes the
    three outputs."""
    d = Path(model_dir)
    t0 = time.time()
    doc, arr, i_final, fit = load_final(d)
    a_arr = load_npz(d / "p13a_arrays.npz")
    users = np.asarray(a_arr["sel_users"], dtype=np.int64)
    picks = np.asarray(a_arr["sel_picks"], dtype=np.int64)
    k_res, k_ranks, k_pos, k_users = iks.load_outputs(d)
    if picks.shape != (len(users), 2):
        raise ValueError("p13a_arrays.npz holds no two picks per listener")
    if not np.array_equal(arr["sel_users"], users):
        raise ValueError(
            "p13b2_arrays.npz holds other listeners than " "p13a_arrays.npz"
        )
    if not np.array_equal(k_users, users):
        raise ValueError(
            "ItemKNN's stored listeners differ from P13's " "selection listeners"
        )
    for v in (("both", "id"), ("same_artist", "id")):
        if k_ranks[v].shape != picks.shape:
            raise ValueError(
                f"ItemKNN's stored {knn.variant_name(*v)} ranks "
                f"do not match the picks"
            )
        stored = float(k_res["variants"][knn.variant_name(*v)]["score"])
        if abs(iks.score_picks(k_ranks[v])[0] - stored) > MEAN_TOL:
            raise ValueError(
                f"ItemKNN's stored {knn.variant_name(*v)} score "
                f"does not follow from its stored ranks"
            )
    g_rank = np.asarray(arr["ranks"][i_final])
    if int(g_rank.min()) < 1 or int(g_rank.max()) > s13.RANK_CUT + 1:
        raise ValueError("a stored GAM rank lies outside 1 to " f"{s13.RANK_CUT + 1}")
    if abs(score_of(g_rank) - float(fit["selection"]["mean"])) > MEAN_TOL:
        raise ValueError(
            "the final model's stored selection score does not "
            "follow from its stored ranks"
        )
    top = top_knots(final_terms(fit))
    shapes = shape_summary(fit)
    say(
        f"{WORK_NAME}: final model {fit['cell']}, {fit['arm']}, level "
        f"{fit['level']}, score {score_of(g_rank):.5f}. Top knots: "
        f"same-artist {top['same_artist']:.6g}, other-artist "
        f"{top['other_artists']:.6g}"
    )
    say(f"{WORK_NAME}: transposing the lists (no progress lines in this " f"step)")
    t_mats = knn.transposed(lists, n_slots)
    has = {kind: has_rows(lists[kind][0]) for kind in knn.KINDS}
    workers = int(workers or max(1, min(16, os.cpu_count() or 1)))
    say(f"{WORK_NAME}: {len(users):,} listeners on {workers} worker(s)")
    ranks, values, counts, totals = facts_all(
        t_mats, u_arrays, users, picks, top, workers, say
    )
    bad = {
        name: int(np.count_nonzero(ranks[i] != k_ranks[v]))
        for i, name, v in (
            (0, "same_artist/id", ("same_artist", "id")),
            (1, "both/id", ("both", "id")),
        )
    }
    if any(bad.values()) or not np.array_equal(counts[3], k_pos):
        raise ValueError(
            f"the recomputed ItemKNN ranks differ from the "
            f"stored ones in {bad} picks, or the counts of "
            f"positive candidates differ. ItemKNN ranked other "
            f"picks or other lists"
        )
    value_same = np.where(has["same_artist"][picks], values[0], np.nan)
    value_other = np.where(has["other_artists"][picks], values[1], np.nan)
    z = {
        "user_id": users,
        "picks": picks,
        "gam_rank": g_rank,
        "knn_both": ranks[1],
        "knn_same": ranks[0],
        "clamp_both": ranks[3],
        "clamp_same": ranks[2],
        "value_same": value_same,
        "value_other": value_other,
        "evidence": evidence_codes(value_same, value_other),
        "flat_same": on_top(value_same, top["same_artist"]),
        "flat_other": on_top(value_other, top["other_artists"]),
        "pop_centile": np.asarray(pop_centile)[picks].astype(np.int64),
        "history": np.diff(np.asarray(u_arrays[0], dtype=np.int64))[users],
        "n_flat_same": counts[0],
        "n_flat_other": counts[1],
        "n_pos_same": counts[2],
        "n_pos_both": counts[3],
        "weight_total": totals,
    }
    results = summarize_slices(z, top)
    out = {
        "version": VERSION,
        "stage": "slices",
        "final": {
            "index": i_final,
            "cell": fit["cell"],
            "arm": fit["arm"],
            "level": fit["level"],
            "m": fit.get("m"),
        },
        "shapes": shapes,
        "results": results,
        "inputs": dict(inputs or {}),
    }
    _atomic_savez(d / SLICES_FILES[1], **z)
    atomic_json_write(out, d / SLICES_FILES[0])
    gap = check_slices(d, t_mats, lists, u_arrays, n_slots, pop_centile, say, n_check)
    out["check"] = {
        "listeners": int(min(n_check, len(users))),
        "value_gap": gap,
        "itemknn_ranks_equal": True,
        "seconds": time.time() - t0,
    }
    atomic_json_write(out, d / SLICES_FILES[0])
    write_slices_report(out, d / "reports" / "gam_knn_slices.md")
    for name, m in results["models"].items():
        say(f"{WORK_NAME}: {name}: score {m['score']:.5f}")
    say(f"{WORK_NAME}: slices done in {_clock(time.time() - t0)}")
    return out


# Reports


def _f(x, fmt=".5f"):
    return "n/a" if x is None else format(x, fmt)


def _interval(d, fmt="+.5f"):
    if d.get("mean") is None:
        return "n/a"
    if d.get("lo") is None:
        return f"{format(d['mean'], fmt)}"
    return (
        f"{format(d['mean'], fmt)} ({format(d['lo'], fmt)} to "
        f"{format(d['hi'], fmt)})"
    )


def _reading(d):
    if d.get("hi") is not None and d["hi"] < 0.0:
        return "GAM behind"
    if d.get("lo") is not None and d["lo"] > 0.0:
        return "GAM ahead"
    return "no clear difference"


def write_slices_report(out, path):
    r = out["results"]
    lines = ["# The final GAM against ItemKNN, stage 1: slices", ""]
    add = lines.append
    fin = out["final"]
    add(
        f"- final model of P13b part 2: {fin['cell']}, {fin['arm']}, level "
        f"{fin['level']}. Listeners: {r['listeners']:,}, two val picks "
        f"each."
    )
    add(
        f"- top knots of the GAM's co-listening shapes: same-artist "
        f"{r['top_knots']['same_artist']:.6g}, other-artist "
        f"{r['top_knots']['other_artists']:.6g}. The shape is constant at "
        f"and above its top knot."
    )
    if "check" in out:
        c = out["check"]
        add(
            f"- checks: the recomputed ItemKNN ranks equal the stored ones "
            f"for every pick. {c['listeners']} listeners got their values "
            f"from a second route (largest relative gap "
            f"{c['value_gap']:.1e}) and their clamped ranks from a full "
            f"sort. The results follow from the stored per-pick arrays."
        )
    add(
        "- intervals: 95 percent, from listener clusters. NDCG@100 is the "
        "mean over picks."
    )
    add("")
    add("## 1. Models on the same picks")
    add("")
    add(
        "| model | score | val_last NDCG@100 | val_first NDCG@100 | "
        "val_last R@100 | val_first R@100 |"
    )
    add("|---|---:|---:|---:|---:|---:|")
    for name, m in r["models"].items():
        add(
            f"| {name} | {m['score']:.5f} | "
            f"{m['val_last']['ndcg_at_100']:.5f} | "
            f"{m['val_first']['ndcg_at_100']:.5f} | "
            f"{m['val_last']['recall_at_100']:.4f} | "
            f"{m['val_first']['recall_at_100']:.4f} |"
        )
    add("")
    add(
        f"- the mean over picks of the better of the GAM and ItemKNN (both "
        f"lists): {r['better_of_two']:.5f}. A switch between the two lists, "
        f"per pick or per listener, cannot score more."
    )
    for name, d in r["differences"].items():
        add(f"- {name}: {_interval(d)}")
    add("")
    add("## 2. The GAM's shapes at the top")
    add("")
    add(
        "Heights of the centered tables at the top knots. A top height "
        "below the largest height means that the shape falls toward its "
        "top."
    )
    add("")
    add(
        "| term | knots | top knots | heights at the top knots | largest "
        "height (at knot) | top minus largest | zero | missing |"
    )
    add("|---|---:|---|---|---|---:|---:|---:|")
    for name, s in out["shapes"].items():
        add(
            f"| {name} | {s['n_knots']} | "
            f"{', '.join(f'{x:.4g}' for x in s['knots'])} | "
            f"{', '.join(f'{x:.3f}' for x in s['heights'])} | "
            f"{s['max_height']:.3f} ({s['max_knot']:.4g}) | "
            f"{s['top_minus_max']:.3f} | {_f(s['zero'], '.3f')} | "
            f"{_f(s['missing'], '.3f')} |"
        )
    add("")
    add("## 3. Slices")
    add("")
    add(
        "Difference: GAM minus ItemKNN (both lists), NDCG@100 per pick. "
        "Only GAM, only ItemKNN: the picks in the top 100 of one model "
        "only. Better: the mean of the better model per pick."
    )
    for dim, rows in r["slices"].items():
        add("")
        add(f"### {dim}")
        add("")
        add(
            "| slice | picks | GAM NDCG@100 | ItemKNN NDCG@100 | difference "
            "(95% interval) | reading | GAM R@100 | ItemKNN R@100 | only "
            "GAM | only ItemKNN | better |"
        )
        add("|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|")
        for name, x in rows.items():
            if not x["picks"]:
                add(
                    f"| {name} | 0 | n/a | n/a | n/a | n/a | n/a | n/a | 0 | "
                    f"0 | n/a |"
                )
                continue
            add(
                f"| {name} | {x['picks']:,} | {_f(x['gam_ndcg'])} | "
                f"{_f(x['knn_ndcg'])} | {_interval(x['diff'])} | "
                f"{_reading(x['diff'])} | {_f(x['gam_r100'], '.4f')} | "
                f"{_f(x['knn_r100'], '.4f')} | {x['gam_only']:,} | "
                f"{x['knn_only']:,} | {_f(x['better'])} |"
            )
    add("")
    cuts = r["history_cuts"]
    add(
        f"History length: train tracks per listener. Short: up to "
        f"{cuts[0]:,.0f}, middle: up to {cuts[1]:,.0f}, long: more."
    )
    add("")
    add("## 4. The flat top")
    add("")
    f = r["flat_top"]
    add(
        f"- picks with same-artist evidence: "
        f"{f['picks_with_same_artist_evidence']:,}. On the flat top: "
        f"{f['picks_on_flat_top']:,}."
    )
    if f["n_flat_of_their_listener"] is not None:
        qs = f["n_flat_of_their_listener"]
        add(
            f"- candidates on the flat top of the listener of such a pick: "
            f"minimum {qs['q0']:,.0f}, quartiles {qs['q25']:,.0f}, "
            f"{qs['q50']:,.0f} and {qs['q75']:,.0f}, 90th percentile "
            f"{qs['q90']:,.0f}, maximum {qs['q100']:,.0f}. Share with 100 "
            f"or more: {f['share_with_100_or_more_on_top']:.3f}. The "
            f"same-artist term alone cannot order these candidates."
        )
    add(
        f"- per listener, median candidates on the same-artist flat top: "
        f"{f['listeners_median_flat_same']:,.0f}, on the other-artist flat "
        f"top: {f['listeners_median_flat_other']:,.0f}. Median candidates "
        f"with same-artist evidence: "
        f"{f['listeners_median_positive_same']:,.0f}, with any co-listening "
        f"evidence: {f['listeners_median_positive_both']:,.0f}."
    )
    add("")
    add("## 5. Reading")
    add("")
    for label, s in r["readings"]["S1"].items():
        verdict = (
            "the flat top costs co-listening accuracy"
            if s["holds"]
            else "the flat top costs less than SESOI, or the " "interval reaches 0"
        )
        add(
            f"- S1, {label}: unclamped minus clamped "
            f"{_interval(s['difference'])} against SESOI "
            f"{SESOI * s['unclamped_score']:.5f}. Reading: {verdict}."
        )
    s2 = r["readings"]["S2"]
    add(
        "- S2, slices where the GAM is behind ItemKNN: "
        + (", ".join(s2) if s2 else "none")
        + "."
    )
    atomic_text_write("\n".join(lines) + "\n", path)


# Stage 2: variants


def variant_specs(names):
    """[(variant name, kept terms)] in a fixed order: the full model, the
    model without each term, each term alone, each group of GROUPS alone
    and the model without the co-listening terms."""
    names = list(names)
    if len(set(names)) != len(names):
        raise ValueError("the final model names a term twice")
    for group, members in GROUPS.items():
        missing = [t for t in members if t not in names]
        if missing:
            raise ValueError(
                f"the final model lacks the terms {missing} of " f"the group {group}"
            )
    out = [("full", tuple(names))]
    out += [(f"without {t}", tuple(x for x in names if x != t)) for t in names]
    out += [(f"only {t}", (t,)) for t in names]
    out += [(f"only {g}", tuple(t for t in names if t in GROUPS[g])) for g in GROUPS]
    out.append(
        (
            "without co-listening",
            tuple(t for t in names if t not in GROUPS["co-listening"]),
        )
    )
    return out


def check_every_candidate(terms):
    """Raise ValueError unless every term applies to every candidate:
    no tiers, no multiplier, and a missing state or a source that is never
    missing. Only then does centering move a term by one constant for all
    candidates, so that a variant with the term's tables at 0 ranks as the
    model without the term."""
    for t in terms:
        if (
            t.tiers is not None
            or t.multiplier is not None
            or not (t.missing_state or t.source in cn.NEVER_MISSING)
        ):
            raise ValueError(
                f"the term {t.name} does not apply to every "
                f"candidate, so a variant without it would not "
                f"rank as the model without it"
            )


def variant_tables(full_tables, layout, specs):
    """(C, L) union tables, one row per variant: the final model's tables
    with the tables of every term outside the variant's kept terms set to
    0."""
    T = np.asarray(full_tables, dtype=np.float64)
    if T.shape != (layout.L,):
        raise ValueError(
            f"the final tables have shape {T.shape}, the " f"layout needs ({layout.L},)"
        )
    known = {t.name for t in layout.terms}
    out = np.empty((len(specs), layout.L), dtype=np.float64)
    for c, (name, kept) in enumerate(specs):
        unknown = set(kept) - known
        if unknown:
            raise ValueError(f"variant {name}: unknown terms " f"{sorted(unknown)}")
        row = T.copy()
        for t in layout.terms:
            if t.name not in kept:
                row[layout.term_slice(t.name)] = 0.0
        out[c] = row
    return out


def load_slices(model_dir):
    """Stage 1's per-pick arrays."""
    p = Path(model_dir) / SLICES_FILES[1]
    if not p.exists():
        raise ValueError(
            f"{SLICES_FILES[1]} is missing: run python3 "
            f"-m pipeline.runner --only gam_knn_slices first"
        )
    return load_npz(p)


def pass_key(specs, model_sha):
    """What the saved blocks depend on beyond the tables, the listeners and
    the blocks (StageP13a._selection_pass adds those): the variants, the
    model file and the code that scores them."""
    return {
        "version": VERSION,
        "stage": "variants",
        "variants": [[n, list(k)] for n, k in specs],
        "p13b2_model": model_sha,
        "p13a_params": s13._params(),
        "code": {
            Path(m.__file__).name: file_sha256(Path(m.__file__))
            for m in (ct, cn, cd, gam, s13)
        },
    }


def verify_dense(st, base, tables, ranks, say, n_verify=VERIFY_USERS):
    """Raise ValueError unless every variant's stored ranks of n_verify
    listeners lie in the range of catalog.reference_scores() and
    reference_rank(), censored at RANK_CUT + 1. The dense reference sums
    the terms in term order with np.interp, without the scorer's slot
    encoding, item part and corrections (as P13a's verify())."""
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
        f"{WORK_NAME}: variant ranks against the dense " f"reference",
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
                        f"listener {u}, variant {c}, pick {p}: the rank "
                        f"{got} lies outside the dense reference's range "
                        f"{lo} to {hi}, censored at {cut1}"
                    )
        prog.advance(1)
    prog.finish()
    return {"listeners": int(len(take)), "near_ties": int(near)}


def summarize_variants(specs, ranks, s1):
    """The results of stage 2 from the variants' ranks and stage 1's
    per-pick arrays. A pure function, so the check recomputes it."""
    names = [n for n, _ in specs]
    idx = {n: i for i, n in enumerate(names)}
    scores = ga.per_listener_scores(ranks)
    full_score = float(scores[0].mean())
    rows = {}
    for c, (name, kept) in enumerate(specs):
        d = ga.paired_summary(scores[c], scores[0])
        rows[name] = {
            "terms": list(kept),
            "score": float(scores[c].mean()),
            **type_blocks(ranks[c]),
            "diff_to_full": {
                k: (int(v) if k == "n" else _num(v)) for k, v in d.items()
            },
        }
    same = ranks[idx["only colisten_same_artist"]]
    col = ranks[idx["only co-listening"]]
    comps = {
        "only colisten_same_artist minus ItemKNN, same-artist list": paired(
            same, s1["knn_same"]
        ),
        "only colisten_same_artist minus ItemKNN, same-artist list, "
        "clamped": paired(same, s1["clamp_same"]),
        "only co-listening minus ItemKNN, both lists": paired(col, s1["knn_both"]),
        "only co-listening minus ItemKNN, both lists, clamped": paired(
            col, s1["clamp_both"]
        ),
        "full minus only co-listening": paired(ranks[0], col),
        "full minus ItemKNN, both lists": paired(ranks[0], s1["knn_both"]),
    }
    ev_code = s1["evidence"]
    flat = s1["flat_same"]
    band = pop_bands(s1["pop_centile"])
    masks = (
        [
            ("co-listening evidence", name, ev_code == c)
            for c, name in enumerate(EVIDENCE)
        ]
        + [
            ("same-artist value", TOP_LABELS[0], (ev_code == 0) & ~flat),
            ("same-artist value", TOP_LABELS[1], (ev_code == 0) & flat),
        ]
        + [("popularity band", name, band == c) for c, name in enumerate(POP_BANDS)]
    )
    slice_picks = {
        f"{dim}: {label}": int(np.count_nonzero(m)) for dim, label, m in masks
    }
    slices = {
        name: {
            f"{dim}: {label}": cluster_mean(pick_dcg(ranks[c]), m)["mean"]
            for dim, label, m in masks
        }
        for c, name in enumerate(names)
    }
    clamp_score = score_of(s1["clamp_same"])
    d1 = comps["only colisten_same_artist minus ItemKNN, same-artist list, " "clamped"]
    s = SESOI * clamp_score
    if d1["mean"] is None:
        v1 = "unclear"
    elif abs(d1["mean"]) <= s:
        v1 = "within"
    elif d1["mean"] < -s and d1["hi"] is not None and d1["hi"] < 0.0:
        v1 = "below"
    elif d1["mean"] > s and d1["lo"] is not None and d1["lo"] > 0.0:
        v1 = "above"
    else:
        v1 = "unclear"
    d2 = paired(col, ranks[0])
    v2 = bool(
        d2["mean"] is not None
        and d2["lo"] is not None
        and d2["mean"] > SESOI * full_score
        and d2["lo"] > 0.0
    )
    hurts = []
    for name in names:
        if name.startswith("without ") and name != "without co-listening":
            d = rows[name]["diff_to_full"]
            if d["lo"] is not None and d["lo"] > 0.0:
                hurts.append(
                    {
                        "term": name[len("without ") :],
                        "gain": d["mean"],
                        "above_sesoi": bool(d["mean"] > SESOI * full_score),
                    }
                )
    return {
        "variants": rows,
        "comparisons": comps,
        "slice_picks": slice_picks,
        "slices": slices,
        "readings": {
            "V1": {"outcome": v1, "difference": d1, "sesoi": s},
            "V2": {"holds": v2, "difference": d2, "sesoi": SESOI * full_score},
            "V3": hurts,
        },
    }


def check_variants(model_dir, specs, s1, stored_final):
    """Raise ValueError unless the stored outputs of stage 2 follow from
    the stored ranks, and the full variant's ranks equal P13b part 2's."""
    d = Path(model_dir)
    doc = load_json(d / VARIANTS_FILES[0])
    if doc is None:
        raise ValueError(f"{VARIANTS_FILES[0]} is missing")
    z = load_npz(d / VARIANTS_FILES[1])
    ranks = z["ranks"]
    if ranks.shape != (len(specs),) + tuple(stored_final.shape):
        raise ValueError("the stored ranks do not match the variants")
    if not np.array_equal(ranks[0], stored_final):
        raise ValueError(
            "the stored full variant differs from P13b part "
            "2's ranks of the final model"
        )
    if doc["variants"] != [[n, list(k)] for n, k in specs]:
        raise ValueError(
            "the stored variants differ from the variants of " "the final model"
        )
    again = summarize_variants(specs, ranks, s1)
    if json.dumps(again, sort_keys=True) != json.dumps(doc["results"], sort_keys=True):
        raise ValueError("the stored results do not follow from the stored " "ranks")


def reference_rows(model_dir):
    """(name, score) of the other models on the same picks: stage 1's
    ItemKNN rankings, and popularity and ALS where present."""
    d = Path(model_dir)
    rows = []
    doc = load_json(d / SLICES_FILES[0]) or {}
    models = (doc.get("results") or {}).get("models") or {}
    for key, name in MODELS:
        if key != "gam_rank" and name in models:
            rows.append([name, float(models[name]["score"])])
    for name, score, _, _ in iks.references(d):
        if not name.startswith("GAM"):
            rows.append([name, float(score)])
    return rows


def run_variants(st, model_dir, work, say=print, n_verify=VERIFY_USERS, inputs=None):
    """Stage 2 (module docstring). st: a StageP13b2, whose loaders give the
    item side, the CSR and the slots. Returns the output doc and writes the
    three outputs."""
    d = Path(model_dir)
    t0 = time.time()
    doc, arr, i_final, fit = load_final(d)
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
    s1 = load_slices(d)
    if not (
        np.array_equal(s1["user_id"], users)
        and np.array_equal(s1["picks"], picks)
        and np.array_equal(s1["gam_rank"], stored_final)
    ):
        raise ValueError(
            f"{SLICES_FILES[1]} belongs to other listeners, "
            f"picks or another final model: run python3 "
            f"-m pipeline.runner --only gam_knn_slices --force"
        )
    names = [t["name"] for t in fit["terms"]]
    extra = set(names) - {t.name for t in base["union"]}
    if extra:
        raise ValueError(
            f"the final model has terms outside the union: " f"{sorted(extra)}"
        )
    check_every_candidate(final_terms(fit))
    specs = variant_specs(names)
    tables = variant_tables(st._union_tables(base, fit), base["union_layout"], specs)
    key = pass_key(specs, file_sha256(d / "p13b2_model.json"))
    refs = reference_rows(d)  # before the pass: a fault stops early
    say(
        f"{WORK_NAME}: {len(specs)} variants of the final model "
        f"{fit['cell']} ({fit['arm']}, level {fit['level']}), "
        f"{len(users):,} listeners. Saved blocks: {Path(work)}"
    )
    t_pass = time.time()
    with _relabel_logs():
        ranks, scored = st._selection_pass(
            base["union"], tables, users, picks, base["m_sel"], Path(work), key
        )
    secs = time.time() - t_pass
    if not np.array_equal(ranks[0], stored_final):
        n_bad = int(np.count_nonzero(ranks[0] != stored_final))
        raise ValueError(
            f"the full variant's ranks differ from P13b part "
            f"2's stored ranks of the final model in {n_bad} "
            f"picks"
        )
    dense = verify_dense(st, base, tables, ranks, say, n_verify)
    n_slots = int(st._id_slots()[0])
    st._cache.pop("space", None)
    st._cache.pop("sources", None)
    results = summarize_variants(specs, ranks, s1)
    out = {
        "version": VERSION,
        "stage": "variants",
        "final": {
            "index": i_final,
            "cell": fit["cell"],
            "arm": fit["arm"],
            "level": fit["level"],
            "m": fit.get("m"),
        },
        "variants": [[n, list(k)] for n, k in specs],
        "results": results,
        "pass": {
            "configurations": len(specs),
            "listeners": int(len(users)),
            "full_pass_share": float(np.mean(scored == n_slots)),
            "mean_scored_share": float(np.mean(scored) / n_slots),
            "seconds": float(secs),
        },
        "check": {"full_equals_stored": True, "dense": dense},
        "references": refs,
        "inputs": dict(inputs or {}),
    }
    _atomic_savez(
        d / VARIANTS_FILES[1],
        user_id=users,
        ranks=ranks.astype(np.int32),
        scored=scored.astype(np.int64),
    )
    atomic_json_write(out, d / VARIANTS_FILES[0])
    check_variants(d, specs, s1, stored_final)
    out["seconds"] = time.time() - t0
    atomic_json_write(out, d / VARIANTS_FILES[0])
    write_variants_report(out, d / "reports" / "gam_knn_variants.md")
    r = results["variants"]
    for name in ("full", "only colisten_same_artist", "only co-listening"):
        say(f"{WORK_NAME}: {name}: score {r[name]['score']:.5f}")
    say(f"{WORK_NAME}: variants done in {_clock(time.time() - t0)}")
    return out


def write_variants_report(out, path):
    r = out["results"]
    lines = [
        "# The final GAM against ItemKNN, stage 2: variants of the " "final model",
        "",
    ]
    add = lines.append
    fin, ps, ck = out["final"], out["pass"], out["check"]
    add(
        f"- final model of P13b part 2: {fin['cell']}, {fin['arm']}, level "
        f"{fin['level']}. {ps['configurations']} variants, "
        f"{ps['listeners']:,} listeners."
    )
    add(
        "- a variant keeps the final model's tables of its terms and sets "
        "the others to 0. No variant is refit."
    )
    add(
        f"- selection pass: {100 * ps['full_pass_share']:.1f} percent of the "
        f"listeners over the whole catalog, "
        f"{100 * ps['mean_scored_share']:.1f} percent of all slots scored, "
        f"{_clock(ps['seconds'])}. Ranks are exact up to 100 and stored as "
        f"101 beyond."
    )
    add(
        f"- checks: the full variant's ranks equal P13b part 2's stored "
        f"ranks for every pick. {ck['dense']['listeners']} listeners got "
        f"every variant's ranks again from the dense reference "
        f"({ck['dense']['near_ties']} checked ranks with a near tie). The "
        f"results follow from the stored ranks."
    )
    add("- intervals: 95 percent, paired over listeners.")
    add("")
    add("## 1. Variants")
    add("")
    add(
        "| variant | score | difference to the full model (95% interval) "
        "| val_last NDCG@100 | val_first NDCG@100 | val_last R@100 | "
        "val_first R@100 |"
    )
    add("|---|---:|---|---:|---:|---:|---:|")
    for name, v in r["variants"].items():
        diff = "" if name == "full" else _interval(v["diff_to_full"])
        add(
            f"| {name} | {v['score']:.5f} | {diff} | "
            f"{v['val_last']['ndcg_at_100']:.5f} | "
            f"{v['val_first']['ndcg_at_100']:.5f} | "
            f"{v['val_last']['recall_at_100']:.4f} | "
            f"{v['val_first']['recall_at_100']:.4f} |"
        )
    add("")
    add("## 2. Other models on the same picks")
    add("")
    add("| model | score |")
    add("|---|---:|")
    for name, score in out["references"]:
        add(f"| {name} | {score:.5f} |")
    add("")
    add("## 3. Paired comparisons")
    add("")
    for name, d in r["comparisons"].items():
        add(f"- {name}: {_interval(d)}")
    add("")
    add("## 4. Variants by slice")
    add("")
    add(
        "Mean NDCG@100 per pick in each slice of stage 1. The first row "
        "gives the picks in the slice."
    )
    keys = list(r["slice_picks"])
    for dims in (("co-listening evidence", "same-artist value"), ("popularity band",)):
        cols = [k for k in keys if k.split(": ")[0] in dims]
        add("")
        add("| variant | " + " | ".join(k.split(": ", 1)[1] for k in cols) + " |")
        add("|---|" + "---:|" * len(cols))
        add("| picks | " + " | ".join(f"{r['slice_picks'][k]:,}" for k in cols) + " |")
        for name, row in r["slices"].items():
            add(f"| {name} | " + " | ".join(_f(row[k]) for k in cols) + " |")
    add("")
    add("## 5. Reading")
    add("")
    v1 = r["readings"]["V1"]
    text = {
        "within": "the term ranks as well as its clamped values. The "
        "gap to ItemKNN's same-artist list comes from the flat "
        "top (stage 1, S1)",
        "below": "the shape or the zero and missing constants lose "
        "more than the flat top",
        "above": "the term ranks better than its clamped values",
        "unclear": "the difference exceeds SESOI, but the interval " "reaches 0",
    }[v1["outcome"]]
    add(
        f"- V1, the same-artist term alone minus the clamped same-artist "
        f"list: {_interval(v1['difference'])} against SESOI "
        f"{v1['sesoi']:.5f}. Reading: {text}."
    )
    v2 = r["readings"]["V2"]
    add(
        f"- V2, the co-listening terms alone minus the full model: "
        f"{_interval(v2['difference'])} against SESOI {v2['sesoi']:.5f}. "
        f"Reading: "
        + (
            "the other terms push co-listening's good " "candidates down."
            if v2["holds"]
            else "the other terms do not lower co-listening by more " "than SESOI."
        )
    )
    hurts = r["readings"]["V3"]
    if hurts:
        add(
            "- V3, terms whose removal raises the score (interval above 0): "
            + ", ".join(
                f"{h['term']} ({h['gain']:+.5f}"
                + (", above SESOI)" if h["above_sesoi"] else ")")
                for h in hurts
            )
            + ". A refit decides whether to drop them."
        )
    else:
        add("- V3: no term's removal raises the score with an interval " "above 0.")
    atomic_text_write("\n".join(lines) + "\n", path)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="The final GAM against ItemKNN, in two stages "
        "(protocol in the module docstring)"
    )
    sub = parser.add_subparsers(dest="stage", required=True)
    p1 = sub.add_parser(
        "slices", help="stage 1: the stored ranks by slice " "(minutes)"
    )
    p1.add_argument(
        "--workers", type=int, default=None, help="worker processes (default: up to 16)"
    )
    sub.add_parser(
        "variants",
        help="stage 2: a selection pass over "
        "variants of the final model (about 1 "
        "to 1.5 hours)",
    )
    args = parser.parse_args(argv)
    setup_logging(LOG_PATH)
    say = log.info
    try:
        say(
            f"{WORK_NAME}: stage {args.stage} started, process id "
            f"{os.getpid()}, log file {LOG_PATH}"
        )
        inputs = provenance(MODEL_DIR)
        if args.stage == "slices":
            n_slots, u_arrays, lists, pop_c = load_slice_inputs(MODEL_DIR, say)
            run_slices(
                MODEL_DIR,
                n_slots,
                u_arrays,
                lists,
                pop_c,
                workers=args.workers,
                say=say,
                inputs=inputs,
            )
        else:
            run_variants(
                s13b2.StageP13b2(),
                MODEL_DIR,
                SCRATCH_DIR / WORK_NAME,
                say=say,
                inputs=inputs,
            )
    except KeyboardInterrupt:
        advice = (
            "Run it again to start over (the stage takes minutes)."
            if args.stage == "slices"
            else "Run it again: the pass continues with the blocks still " "missing."
        )
        log.info(f"{WORK_NAME}: stopped by Ctrl+C. {advice}")
        raise SystemExit(130)
    except Exception:
        log.exception(f"{WORK_NAME}: stopped by an error")
        raise SystemExit(1)


# Runner stage


class StageGamKnnSlices(ScriptStage):
    """Stage 1 of this module (the slices) as a runner stage. Stage 2
    (the variants) stays a diagnostic: python3 -m pipeline.gam_knn_diag
    variants."""

    name = "gam_knn_slices"
    upstream = ["p6_als", "p9b_colisten", "p13b2_loss", "itemknn_select"]

    def outputs(self):
        d = self.model_dir
        return {
            "results": d / SLICES_FILES[0],
            "arrays": d / SLICES_FILES[1],
            "report": d / "reports" / "gam_knn_slices.md",
        }

    def provenance(self):
        return provenance(self.model_dir)

    def recorded_inputs(self):
        doc = load_json(self.outputs()["results"])
        return None if doc is None else doc.get("inputs")

    def _load(self):
        """(n_slots, u_arrays, lists, pop_centile)."""
        return load_slice_inputs(self.model_dir, log.info)

    def run(self, params, files):
        with log_file(self.model_dir / "logs" / LOG_PATH.name):
            n_slots, u_arrays, lists, pop_c = self._load()
            run_slices(
                self.model_dir,
                n_slots,
                u_arrays,
                lists,
                pop_c,
                say=log.info,
                inputs=self.provenance(),
            )
        return self.output_info()

    def check(self):
        n_slots, u_arrays, lists, pop_c = self._load()
        t_mats = knn.transposed(lists, n_slots)
        check_slices(self.model_dir, t_mats, lists, u_arrays, n_slots, pop_c, log.info)


if __name__ == "__main__":
    main()
