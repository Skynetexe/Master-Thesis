"""P14, part 1: the cold-track GAM.
Declared before any code or result.

The final model (p13c_final.json, the model record of p13d.json) reads
colisten_same_artist. A track without a same-artist co-listening
neighbour in the history then gets that term's missing or zero state, and
on the selection listeners none of those picks reached the top 100.
The cold-track GAM is the final model without that term. It is an extra
arm of P14, and the ListenBrainz demo ranks tracks that MLHD lacks with
it, because those tracks have no co-listening at all. It is not
a candidate model, and no rule chooses or changes a model here. It never
reads a test pick.

Model, one fit:
- terms: the final model's terms without colisten_same_artist, in the
  model's order. On the real data: tag_affinity, artist_tag_affinity,
  genre_affinity, artist_affinity and audio_fit.
- bounds: the final model's bounded terms among them (P13b's monotone
  bounds: non-decreasing knot heights, and the zero state at or below the
  first knot). Missing states and the other terms stay free.
- the final model's level of the wide file, loss arm and penalty weight,
  and P13a's knots and layout constants (p13c.layout_of).
- start: the final model's parameters of these terms, by name
  (gam_refit.warm_start). The objective is strictly convex (ridge), so the
  start changes the time, not the result.
- solver: p13c.solve (gam_arms.fit_bounded with stage_p13b.SOLVER), to
  KKT_TOL. A bounded fit must keep its monotone order.
- tables: centered on the level's p_pop negatives (gam.center), as every
  model of P13.

Selection, descriptive only: one pass of P13 (StageP13a._selection_pass)
over this model. Score: full-catalog NDCG@100 of the val_last and
val_first picks of the 20,000 selection listeners, ranks exact up to 100.
The paired difference to the final model's stored ranks comes with
gam_audit's 95 percent interval. P14 reports the drop from this score to
the eval score.

Checks:
- the final model and its pointer pass p13c's checks (check_pointer,
  load_p13d_model or load_model), and its stored centered tables equal a
  centering of its parameters on the level's means (CENTER_TOL), so the
  means are the level's own.
- the fit reaches KKT_TOL and keeps its monotone order (p13c.solve), and
  a saved fit is used only with the layout's fingerprint.
- VERIFY_USERS listeners get the ranks again from the dense reference of
  catalog.py (gam_refit.verify_dense).
- the read-back: the layout rebuilt from the final model has the stored
  terms and fingerprint, the raw tables equal M theta, the centered tables
  equal the raw tables minus the stored constants, and every result
  follows from the stored ranks.

Outputs: MODEL_DIR/p14_cold_gam.json (the protocol, the model record in the
fields of gam_refit's fit record, the selection results, the shapes),
MODEL_DIR/p14_cold_gam.npz (the parameters, the tables, the selection ranks
of this model and of the final model) and MODEL_DIR/reports/p14_cold_gam.md.
Every line also goes to MODEL_DIR/logs/p14_cold_gam.log. The fit prints its
Newton steps at most every 30 seconds, and the pass its share done with the
time left.

Resume: the fit and the level's background means are saved in
SCRATCH_DIR/p14_cold_gam/fits.json, the pass in SCRATCH_DIR/p14_cold_gam/pass,
keyed by their inputs. After Ctrl+C, the same command continues.

Expected run on the real data: loading the wide level and its design a few
minutes (peak memory near P13c's 18 GB), the bounded fit about 5 to 10
minutes, the pass 10 to 30 minutes (this model reads no co-listening
list), the dense check about 2 minutes.

Usage, from the repository root, through the runner:
    python3 -m pipeline.runner --only p14_cold_gam
The pass uses P13's worker setting. To use fewer workers:
    P13A_WORKERS=8 python3 -m pipeline.runner --only p14_cold_gam
"""

import contextlib
import json
import logging
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
from pipeline.als import _clock
from pipeline.common import (
    atomic_json_write,
    atomic_text_write,
    file_sha256,
    load_json,
    log,
)

from pipeline import gam_knn_diag as gkd
from pipeline import gam_refit as gr
from pipeline import p13c as pc
from pipeline.script_stage import ScriptStage, log_file

VERSION = "p14-cold-v1"
WORK_NAME = "p14_cold_gam"
DECLARED = "before any code or result"
DROP = "colisten_same_artist"
FIT_ID = "cold"
OUT_FILES = ("p14_cold_gam.json", "p14_cold_gam.npz")
REPORT_NAME = "p14_cold_gam.md"
LOG_NAME = "p14_cold_gam.log"
VERIFY_USERS = 6
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
    "monotone_violation",
)

PROTOCOL = {
    "version": VERSION,
    "declared": DECLARED,
    "model": "the final model of p13c_final.json without "
    "colisten_same_artist: the same bounds on the terms it keeps, "
    "the same level, loss arm and penalty weight",
    "start": "the final model's parameters of these terms, by name",
    "selection": "descriptive: NDCG@100 of the val_last and val_first picks "
    "of the selection listeners, the paired difference to the "
    "final model",
    "rules": "none: an extra arm for P14 and the ListenBrainz demo, not a "
    "candidate model",
}


# Helpers


@contextlib.contextmanager
def relabel_logs(name, command):
    """P13's pass lines carry the name of the stage that runs them, and
    their advice names its command."""

    class Relabel(logging.Filter):
        def filter(self, record):
            if isinstance(record.msg, str):
                record.msg = gr._STAGE_PREFIX.sub(name, record.msg, count=1).replace(
                    "python3 -m pipeline.runner --only p13a_fit", command
                )
            return True

    flt = Relabel()
    log.addFilter(flt)
    try:
        yield
    finally:
        log.removeFilter(flt)


def provenance(model_dir):
    """The input files' SHA-256: P13c's provenance of the final model and
    the files that p13c_final.json names (as tree_audit.provenance)."""
    d = Path(model_dir)
    out = pc.provenance(d, "p13d")
    ptr = load_json(d / pc.FINAL_FILE) or {}
    names = [pc.FINAL_FILE] + sorted((ptr.get("files") or {}).values())
    for name in names:
        p = d / name
        out[name] = file_sha256(p) if p.exists() else None
    return out


def load_final(d, base):
    """The final model of p13c_final.json, checked, its bounded terms and
    its stored selection ranks. Returns (M, record, pointer, ranks,
    bounded)."""
    d = Path(d)
    doc = load_json(d / pc.P13D_FILES[0])
    if doc is None:
        raise ValueError(
            "p13d.json is missing: run python3 -m "
            "pipeline.runner --only p13d_bounds first"
        )
    rec = doc["model"]
    ptr = pc.check_pointer(d, rec)
    if rec["changed"]:
        M = pc.load_p13d_model(d, base)[0]
    else:
        M = pc.load_model(d, base)[0]
    z = gkd.load_npz(d / ptr["files"]["arrays"])
    if not np.array_equal(np.asarray(z["user_id"]), base["sel_users"]):
        raise ValueError(
            f"{ptr['files']['arrays']} holds other listeners "
            f"than P13a's selection sample"
        )
    if not np.array_equal(np.asarray(z["theta_model"], dtype=np.float64), M["theta"]):
        raise ValueError(
            f"{ptr['files']['arrays']} belongs to another " f"model than p13d.json's"
        )
    ranks = np.asarray(z["ranks"][0], dtype=np.int64)
    return M, rec, ptr, ranks, list(rec.get("bounded") or [])


def cold_layout(base, M, bounded):
    """(kept terms, bounded terms, layout, union index) of the cold-track
    GAM: the final model's terms without DROP."""
    kept = [n for n in M["kept"] if n != DROP]
    if len(kept) != len(M["kept"]) - 1:
        raise ValueError(f"the final model has no {DROP} term")
    b = [n for n in bounded if n in kept]
    lay, idx = pc.layout_of(base, kept, b)
    return kept, b, lay, idx


def centered(lay, theta, means):
    """(centered tables, constants) on the level's background."""
    Tc, consts = gam.center(
        lay.tables(theta), lay, {t.name: means[t.name] for t in lay.terms}
    )
    return Tc, {k: float(v) for k, v in consts.items()}


def done_line(label, res):
    return (
        f"{WORK_NAME}: {label}: objective {res['objective']:.8f}, largest "
        f"projected gradient {res['proj_grad_max']:.1e}, "
        f"{gr._text(res['message'])}, {_clock(res['seconds'])}"
    )


# Fit


def fit_part(st, base, M, kept, bounded, lay, idx, store, say):
    """The fit at the final model's level, saved in the store, and the
    level's background means. Returns (fit record, means)."""
    level, lam, arm = M["level"], M["lambda"], M["arm"]
    level_id = f"wide:{level}"
    if FIT_ID in store and store.means(level_id) is not None:
        say(f"{WORK_NAME}: the fit is saved, so the level's rows stay on " f"disk")
        return store.get(FIT_ID), store.means(level_id)
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
    with st._design_for(ctx, store, [FIT_ID]) as pool:
        if pool is not None:
            say(f"{WORK_NAME}: design ready in {_clock(time.time() - t1)}")
            start = gr.warm_start(M["lay"], M["theta"], lay)
            obj = st._objective(pool, ctx, lay, idx, arm, lam)
            label = (
                f"the cold-track fit, {len(kept)} terms, "
                f"{len(bounded)} bounded (m = {pc.m_of(arm, level)}, "
                f"lambda {lam:.3g})"
            )
            res = pc.solve(obj, start, label, say)
            res.update(
                fingerprint=gam_arms.layout_fingerprint(lay),
                start="the final model's parameters of these terms",
            )
            if bounded:
                res["flat_share"] = gam_arms.active_share(res["theta"], lay)
            store.put(FIT_ID, res)
            say(done_line(label, res))
    st._ensure_means(ctx, store)
    means = ctx["means_ref"]
    del ctx
    st._drop_inputs()
    return store.get(FIT_ID), means


def check_fit(rec, lay, bounded):
    """A saved fit: this layout, KKT_TOL, and the monotone order."""
    if rec.get("fingerprint") != gam_arms.layout_fingerprint(lay):
        raise ValueError("the saved cold-track fit belongs to another layout")
    if float(rec["proj_grad_max"]) > s13.KKT_TOL:
        raise ValueError(
            f"the saved cold-track fit stopped at a largest "
            f"projected gradient of {rec['proj_grad_max']:.2e}, "
            f"above KKT_TOL"
        )
    theta = np.asarray(rec["theta"], dtype=np.float64)
    viol = gam_arms.monotone_violation(lay.tables(theta), lay)
    if bounded and viol > s13b.MONOTONE_TOL:
        raise ValueError(
            f"the saved cold-track fit breaks its monotone order " f"by {viol:.2e}"
        )
    return theta, float(viol)


# Results


def summarize(ranks, final_ranks):
    """The selection results from the stored ranks: a pure function, so
    check_outputs() recomputes it."""
    s_c = ga.per_listener_scores(ranks)
    s_f = ga.per_listener_scores(final_ranks)
    return {
        "score": float(s_c.mean()),
        "final_score": float(s_f.mean()),
        "cold_minus_final": gr.paired_scores(s_c, s_f),
        "types": gkd.type_blocks(ranks),
        "final_types": gkd.type_blocks(final_ranks),
        "listeners": int(len(s_c)),
    }


def points(lay, theta, means, shape_q):
    """{term: the 7 centered values at P13c's points, None where a term
    has none}."""
    v = pc.value_array({"lay": lay}, [theta], means, shape_q)[0]
    return {
        t.name: [None if not np.isfinite(x) else float(x) for x in v[j]]
        for j, t in enumerate(lay.terms)
    }


def model_record(kept, bounded, lay, idx, M, rec, theta, Tc, consts):
    T = lay.tables(theta)
    return {
        "name": "cold-track GAM",
        "dropped": DROP,
        "kept": list(kept),
        "bounded": list(bounded),
        "level": int(M["level"]),
        "arm": s13b.arm_label(M["arm"]),
        "m": int(pc.m_of(M["arm"], M["level"])),
        "lambda": float(M["lambda"]),
        "terms": [t.to_json() for t in lay.terms],
        "fingerprint": gam_arms.layout_fingerprint(lay),
        "idx": [int(x) for x in idx],
        "theta": theta.tolist(),
        "tables_raw": T.tolist(),
        "tables": Tc.tolist(),
        "center_consts": consts,
        "objective": float(rec["objective"]),
        "fit": {k: rec.get(k) for k in FIT_KEYS},
        "flat_share": rec.get("flat_share"),
        "background": "the level's p_pop negatives, full-fit weights",
        "table_dict": gam.table_dict(Tc, lay),
    }


def check_outputs(d, base, M, bounded, final_ranks):
    """Read the outputs back (module docstring, checks)."""
    d = Path(d)
    doc = load_json(d / OUT_FILES[0])
    if doc is None:
        raise ValueError(f"{OUT_FILES[0]} is missing")
    z = gkd.load_npz(d / OUT_FILES[1])
    m = doc["model"]
    kept, b, lay, idx = cold_layout(base, M, bounded)
    if (
        list(m["kept"]) != kept
        or list(m["bounded"]) != b
        or [t.to_json() for t in lay.terms] != m["terms"]
        or gam_arms.layout_fingerprint(lay) != m["fingerprint"]
        or [int(x) for x in idx] != m["idx"]
    ):
        raise ValueError(
            "the stored cold-track layout differs from the one "
            "rebuilt from the final model"
        )
    theta = np.asarray(m["theta"], dtype=np.float64)
    T = lay.tables(theta)
    Tc = np.asarray(m["tables"], dtype=np.float64)
    if not (
        np.array_equal(np.asarray(m["tables_raw"]), T)
        and np.array_equal(z["theta"], theta)
        and np.array_equal(z["tables"], Tc)
    ):
        raise ValueError(
            "the stored cold-track tables do not follow from its " "parameters"
        )
    want = np.array(T, dtype=np.float64)
    for t, off in zip(lay.terms, lay.offsets):
        want[off : off + t.K + 2] -= m["center_consts"][t.name]
    if not np.array_equal(want, Tc):
        raise ValueError(
            "the stored centered tables are not the raw tables "
            "minus the stored constants"
        )
    if bounded and gam_arms.monotone_violation(T, lay) > s13b.MONOTONE_TOL:
        raise ValueError("the stored cold-track model breaks its monotone " "order")
    if not (
        np.array_equal(z["user_id"], base["sel_users"])
        and np.array_equal(z["final_ranks"], final_ranks)
    ):
        raise ValueError(
            "the stored selection listeners or final ranks " "differ from P13's"
        )
    again = summarize(np.asarray(z["ranks"][0]), final_ranks)
    if norm(again) != norm(doc["selection"]):
        raise ValueError(
            "the stored selection results do not follow from " "the stored ranks"
        )


def norm(x):
    """x as the json file holds it, for a comparison after a read."""
    return json.loads(json.dumps(x, sort_keys=True, default=str))


def run(st, model_dir, work, say=print, n_verify=VERIFY_USERS, inputs=None):
    """The cold-track GAM (module docstring). st: a StageP13b2 whose loaders
    give the level, the item side, the CSR and the slots. Returns the output
    doc."""
    prev = gr.WORK_NAME
    gr.WORK_NAME = WORK_NAME  # labels of the fit and dense lines
    try:
        return _run(st, Path(model_dir), Path(work), say, n_verify, inputs)
    finally:
        gr.WORK_NAME = prev


def _run(st, d, work, say, n_verify, inputs):
    t0 = time.time()
    base = st._base_context()
    M, rec, ptr, final_ranks, bounded = load_final(d, base)
    kept, b, lay, idx = cold_layout(base, M, bounded)
    work.mkdir(parents=True, exist_ok=True)
    key = {
        "version": VERSION,
        "p13d": file_sha256(d / pc.P13D_FILES[0]),
        "final": file_sha256(d / pc.FINAL_FILE),
        "kept": kept,
        "bounded": b,
        "p13b2_key": st._key(),
    }
    store = s13b2._Store(work / "fits.json", s13._sha(key))
    say(
        f"{WORK_NAME}: final model {rec['fit_name']} "
        f"({'chosen by P13d' if rec['changed'] else 'gam_refit model'}), "
        f"{len(M['kept'])} terms. Cold-track GAM: {', '.join(kept)}, bounded: "
        f"{', '.join(b) or 'none'}, arm {s13b.arm_label(M['arm'])}, level "
        f"{M['level']}, lambda {M['lambda']:.3g}. Saved work: {work}"
    )
    t1 = time.time()
    fit, means = fit_part(st, base, M, kept, b, lay, idx, store, say)
    fit_secs = time.time() - t1
    theta, viol = check_fit(fit, lay, b)
    Tc_final, _ = centered(M["lay"], M["theta"], means)
    gap = float(np.max(np.abs(Tc_final - M["tables"])))
    if gap > s13b2.CENTER_TOL:
        raise ValueError(
            f"the final model's stored centered tables differ "
            f"from a centering on the level's p_pop negatives by "
            f"{gap:.2e}, so these are not the level's means"
        )
    Tc, consts = centered(lay, theta, means)

    users, picks = base["sel_users"], base["sel_picks"]
    say(
        f"{WORK_NAME}: the fit is ready. Selection pass over the cold-track "
        f"GAM ({len(users):,} listeners)"
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
    with relabel_logs(WORK_NAME, "python3 -m pipeline.runner --only " "p14_cold_gam"):
        ranks, scored = st._selection_pass(
            list(lay.terms),
            Tc[None, :],
            users,
            picks,
            base["m_sel"],
            work / "pass",
            pkey,
        )
    pass_secs = time.time() - t2
    dense = gr.verify_dense(
        st, dict(base, union=list(lay.terms)), Tc[None, :], ranks, say, n_verify
    )
    st._cache.pop("space", None)
    st._cache.pop("sources", None)
    sel = summarize(np.asarray(ranks[0]), final_ranks)
    shapes = {
        "points": list(pc.POINTS),
        "cold": points(lay, theta, means, base["shape_q"]),
        "final": points(M["lay"], M["theta"], means, base["shape_q"]),
    }
    out = {
        "version": VERSION,
        "declared": DECLARED,
        "protocol": PROTOCOL,
        "model": model_record(kept, b, lay, idx, M, fit, theta, Tc, consts),
        "final": {
            "fit_name": rec["fit_name"],
            "tag": rec["tag"],
            "changed": bool(rec["changed"]),
            "kept": list(M["kept"]),
            "bounded": list(bounded),
            "results": ptr["files"]["results"],
            "arrays": ptr["files"]["arrays"],
        },
        "selection": sel,
        "shapes": shapes,
        "check": {
            "kkt": float(fit["proj_grad_max"]),
            "monotone_violation": viol,
            "final_centering_gap": gap,
            "dense": dense,
        },
        "pass_info": {
            "listeners": int(len(users)),
            "mean_scored_share": float(np.mean(scored) / int(st._id_slots()[0])),
        },
        "inputs": dict(inputs or {}),
        "seconds": {"fit_this_run": fit_secs, "pass_this_run": pass_secs},
    }
    gr._atomic_savez(
        d / OUT_FILES[1],
        user_id=users,
        ranks=np.asarray(ranks, dtype=np.int32),
        scored=np.asarray(scored, dtype=np.int64),
        theta=theta,
        tables=Tc,
        final_ranks=final_ranks,
    )
    atomic_json_write(out, d / OUT_FILES[0])
    check_outputs(d, base, M, bounded, final_ranks)
    out["seconds"]["total_this_run"] = time.time() - t0
    atomic_json_write(out, d / OUT_FILES[0])
    write_report(out, d / "reports" / REPORT_NAME)
    dd = sel["cold_minus_final"]
    say(
        f"{WORK_NAME}: selection NDCG@100 {sel['score']:.5f} against the "
        f"final model's {sel['final_score']:.5f} (difference "
        f"{gr._interval(dd)}). Done in {_clock(time.time() - t0)}"
    )
    return out


# Report


def write_report(out, path):
    m, sel, ch = out["model"], out["selection"], out["check"]
    fit = m["fit"]
    L = [
        "# P14, part 1: the cold-track GAM",
        "",
        f"- Model: the final model ({out['final']['fit_name']}) without "
        f"{m['dropped']}. Terms: {', '.join(m['kept'])}. Bounded: "
        f"{', '.join(m['bounded']) or 'none'}. Arm {m['arm']}, level "
        f"{m['level']} (m = {m['m']}), lambda {m['lambda']:.3g}. Protocol: "
        f"the docstring of p14_cold.py, declared {out['declared']}.",
        f"- Fit: objective {fit['objective']:.8f}, largest projected "
        f"gradient {fit['proj_grad_max']:.1e} (KKT_TOL {s13.KKT_TOL:g}), "
        f"{gr._text(fit['message'])}, {_clock(fit['seconds'] or 0.0)}. "
        f"Largest break of the monotone order {ch['monotone_violation']:.1e}."
        + (
            ""
            if not m.get("flat_share")
            else " Flat share of the bounded "
            "increments: "
            + ", ".join(f"{k} {v:.0%}" for k, v in m["flat_share"].items())
            + "."
        ),
        f"- Checks: the final model's centered tables equal a centering on "
        f"the level's means (gap {ch['final_centering_gap']:.1e}), "
        f"{ch['dense']['listeners']} listeners equal the dense reference "
        f"({ch['dense']['near_ties']} near ties), and the outputs read back.",
        "",
        "## Selection (descriptive)",
        "",
        f"NDCG@100 over val_last and val_first of {sel['listeners']:,} "
        f"selection listeners: cold-track GAM {sel['score']:.5f}, final "
        f"model {sel['final_score']:.5f}. Difference "
        f"{gr._interval(sel['cold_minus_final'])}. This model is no "
        f"candidate, so no rule reads the difference.",
        "",
        "| pick type | cold NDCG@100 | cold R@10 | cold R@100 | final "
        "NDCG@100 | final R@100 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for k in sel["types"]:
        c, f = sel["types"][k], sel["final_types"][k]
        L.append(
            f"| {k} | {c['ndcg_at_100']:.5f} | {c['recall_at_10']:.4f} | "
            f"{c['recall_at_100']:.4f} | {f['ndcg_at_100']:.5f} | "
            f"{f['recall_at_100']:.4f} |"
        )
    sh = out["shapes"]
    L += [
        "",
        "## Shapes",
        "",
        "Centered values in log-odds points at the positives' percentiles, "
        "with the zero and missing states, against the final model on the "
        "same background.",
        "",
        "| term | model | " + " | ".join(sh["points"]) + " |",
        "|---|---|" + "---:|" * len(sh["points"]),
    ]

    def cell(x):
        return "" if x is None else f"{x:+.3f}"

    for name in sh["cold"]:
        for which in ("cold", "final"):
            L.append(
                f"| {name if which == 'cold' else ''} | {which} | "
                + " | ".join(cell(x) for x in sh[which][name])
                + " |"
            )
    atomic_text_write("\n".join(L).replace(";", ",") + "\n", path)


# Runner stage


class StageP14Cold(ScriptStage):
    """The cold-track GAM as a runner stage."""

    name = "p14_cold_gam"
    upstream = ["p13c_final"]

    def __init__(self, model_dir=None, scratch_dir=None):
        super().__init__(model_dir, scratch_dir)
        self.n_verify = VERIFY_USERS

    def _stage(self):
        return s13b2.StageP13b2()

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

    def run(self, params, files):
        with log_file(self.model_dir / "logs" / LOG_NAME):
            run(
                self._stage(),
                self.model_dir,
                self.scratch_dir / WORK_NAME,
                say=log.info,
                n_verify=self.n_verify,
                inputs=self.provenance(),
            )
        return self.output_info()

    def check(self):
        st = self._stage()
        base = st._base_context()
        M, _, _, final_ranks, bounded = load_final(self.model_dir, base)
        check_outputs(self.model_dir, base, M, bounded, final_ranks)
