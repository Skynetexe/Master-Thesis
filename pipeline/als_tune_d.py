"""ALS phase D: more factors, and the frequency-scaled regularizer.

Protocol, fixed before any phase D result exists:

- Users, target and metric of phases A to C (als_tune.py): the 20,000
  selection users of pipeline/selection.py, their val_last and val_first
  picks under the P5 harness rules, and the mean NDCG@100 over the two.
  This script never reads a test pick.
- Fixed: alpha and seed of the phase A to C winner (alpha 1, seed
  20260907), up to 12 iterations with patience 2, as in phases A to C.
- Regularizer: pipeline/als_freq.py, reg * sum over rows of f^nu |v|^2,
  after Rendle, Krichene, Zhang & Koren (2022), section 2.4. nu = 0 is the
  regularizer of phases A to C.
- Cells, in this order. Each cell searches reg on the grid
  start * 10^(k/2) for integer k:
    D1: 128 factors, nu = 0, start = the phase A to C winner's reg (100).
    D2: 128 factors, nu = 1, start = reg*(D1) / sqrt(F_u F_i). F_u and F_i
        are the median frequencies f of the users and of the items that
        hold data. At this start, a user and an item of median frequency
        get the effective regularizations reg*(D1) sqrt(F_u / F_i) and
        reg*(D1) sqrt(F_i / F_u), whose geometric mean is D1's best reg.
    D3: 256 factors, only when the best 128-factor config beats the phase
        A to C winner by more than SESOI (2 percent of that winner's
        score). nu of the better 128-factor cell, start = that cell's
        best reg.
- Search within a cell: run k = 0, then k = +1. If k = +1 beats k = 0,
  go on upward while each step beats the cell's best so far. Otherwise go
  down from k = -1 in the same way. At most 3 steps in a direction with
  128 factors, 2 with 256.
- Winner: the highest score over all phase D configs, a tie to the
  earlier one. Its factors at its best iteration go to MODEL_DIR/als_d/,
  described by best.json. P14 takes the better of this winner and the
  phase A to C winner (P6's factors).
- Unchanged: pipeline/als.py, CURATED/als_params.json and P6. P11 to
  P13b2 hold P6's marker as an input, so a change there would make the
  runner rebuild every stage from P11 on.

Why nu: Rendle et al. found that "even though the quality plateaued with
nu = 0, with nu = 1 further increasing the embedding dimension gave
additional improvements" (section 3.2). Every phase A to C config peaked
at iteration 2 to 4, which also points at the regularizer.

Outputs:
- MODEL_DIR/als_tuning_d.json, updated after every iteration.
- MODEL_DIR/als_d/: the winner's factors and best.json.
- MODEL_DIR/reports/als_tuning_d.md, written at the end.

A stopped run continues where it stopped: run the same command again. A
config resumes from its last finished iteration.

Every line goes to the console and to MODEL_DIR/logs/als_tune_d.log.
Long steps print a progress line at most every 30 seconds, with the time
left.

Usage, from the repository root, through the runner:
    python3 -m pipeline.runner --only als_phase_d
The module's own entry runs it without a marker (the runner then asks for
--adopt):
    python3 -m pipeline.als_tune_d
"""

import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np

from pipeline import als_tune
from pipeline import als
from pipeline import als_freq
from pipeline import selection
from pipeline.common import atomic_json_write, atomic_text_write, connect, log
from pipeline.common import file_sha256, load_json
from pipeline.script_stage import ScriptStage, log_file
from pipeline.config import MODEL_DIR, SCRATCH_DIR

LOG_PATH = MODEL_DIR / "logs" / "als_tune_d.log"
RESULTS_PATH = MODEL_DIR / "als_tuning_d.json"
BEST_DIR = MODEL_DIR / "als_d"
WORK_DIR = SCRATCH_DIR / "als_tune_d"

FACTORS_128, FACTORS_256 = 128, 256
MAX_STEPS = {FACTORS_128: 3, FACTORS_256: 2}
MAX_ITER = 12
PATIENCE = 2
SESOI = 0.02  # relative to the phase A to C winner's score
MAX_CONFIGS = (1 + 1 + 3) * 2 + (1 + 1 + 2)  # upper bound, for the log
DISK_MARGIN = 1.2

PROTOCOL = {
    "declared": "before any phase D result",
    "users": f"selection sample, {selection.SIZE} users, seed "
    f"{selection.SEED}, disjoint from the eval sample",
    "target": "val_last and val_first picks, P5 harness rules",
    "metric": "mean NDCG@100 over val_last and val_first",
    "fixed": "alpha and seed of the phase A to C winner",
    "iterations": {"max": MAX_ITER, "patience": PATIENCE},
    "regularizer": "reg * sum over rows of f^nu |v|^2, f = sum of (c - 1) "
    "over the row's observed pairs + n_other "
    "(pipeline/als_freq.py)",
    "cells": {
        "D1": "128 factors, nu = 0, start = the phase A to C winner's reg",
        "D2": "128 factors, nu = 1, start = reg*(D1) / sqrt(F_u F_i)",
        "D3": "256 factors, only if the best 128-factor config beats the "
        "phase A to C winner by more than SESOI, nu of the better "
        "128-factor cell, start = its best reg",
    },
    "search": "grid start * 10^(k/2). k = 0, then k = +1. Upward while "
    "each step beats the cell's best, else downward from k = -1 "
    "in the same way. At most 3 steps per direction with 128 "
    "factors, 2 with 256.",
    "sesoi": SESOI,
    "winner": "highest score over phase D, ties to the earlier config",
}


def reg_at(start, k):
    return float(start) * 10.0 ** (k / 2.0)


def config_name(factors, nu, alpha, reg):
    return f"f{factors}_nu{nu:g}_alpha{alpha:g}_reg{reg:.6g}"


def best_of(entries):
    """The entry with the highest score. A tie goes to the earliest."""
    best = None
    for e in entries:
        if best is None or e["score"] > best["score"]:
            best = e
    return best


# The search


def search_cell(run_config, cell, factors, nu, start, max_steps):
    """The search within one cell. run_config(cell, factors, nu, reg, k)
    returns an entry with "score". Returns the entries in run order."""
    entries = [run_config(cell, factors, nu, reg_at(start, 0), 0)]
    best = entries[0]
    up = run_config(cell, factors, nu, reg_at(start, 1), 1)
    entries.append(up)
    if up["score"] > best["score"]:
        best = up
        steps = range(2, max_steps + 1)
    else:
        steps = range(-1, -max_steps - 1, -1)
    for k in steps:
        e = run_config(cell, factors, nu, reg_at(start, k), k)
        entries.append(e)
        if e["score"] > best["score"]:
            best = e
        else:
            break
    return entries


def run_cells(run_config, ac_winner, d2_start, sesoi=SESOI):
    """The cells and the gate. ac_winner: the phase A to C winner (reg and
    score). d2_start(reg_d1) gives D2's start. Returns (entries in run
    order, gate record)."""
    d1 = search_cell(
        run_config, "D1", FACTORS_128, 0.0, ac_winner["reg"], MAX_STEPS[FACTORS_128]
    )
    d2 = search_cell(
        run_config,
        "D2",
        FACTORS_128,
        1.0,
        d2_start(best_of(d1)["reg"]),
        MAX_STEPS[FACTORS_128],
    )
    best128 = best_of(d1 + d2)
    threshold = ac_winner["score"] * (1.0 + sesoi)
    gate = {
        "best_128": best128["name"],
        "best_128_score": best128["score"],
        "threshold": threshold,
        "passed": bool(best128["score"] > threshold),
    }
    entries = d1 + d2
    if gate["passed"]:
        entries += search_cell(
            run_config,
            "D3",
            FACTORS_256,
            best128["nu"],
            best128["reg"],
            MAX_STEPS[FACTORS_256],
        )
    return entries, gate


# One config


def _need_bytes(inter, factors):
    """Disk for one config: the factor files, their work copies during an
    iteration and a copy for the winner."""
    return int(3 * (inter.n_users + inter.n_items) * factors * 4 * DISK_MARGIN)


def make_run_config(
    inter,
    users,
    picks,
    masses,
    results,
    save,
    work_dir,
    best_dir,
    alpha,
    seed,
    max_iter=MAX_ITER,
    patience=PATIENCE,
    workers=None,
    say=print,
    rank=als.rank_items,
):
    """run_config() for search_cell(): fits one config with als_freq.fit()
    and early stopping on the val picks, keeps the phase-wide best
    factors in best_dir, and records everything in results."""
    work_dir, best_dir = Path(work_dir), Path(best_dir)
    counter = {"n": 0}

    def run_config(cell, factors, nu, reg, k):
        counter["n"] += 1
        n = counter["n"]
        name = config_name(factors, nu, alpha, reg)
        cfg_dir = work_dir / name
        x_path = cfg_dir / "user_factors.npy"
        y_path = cfg_dir / "item_factors.npy"
        done = results["configs"].get(name)
        if done and done.get("finished"):
            # A stop between the last save and the removal of the factor
            # files leaves them behind. They are 7 GB per pair at 256 factors.
            for p in (x_path, y_path):
                p.unlink(missing_ok=True)
            say(
                f"als_tune_d: config {n} of at most {MAX_CONFIGS}: {name} "
                f"already finished, score {done['score']:.5f}"
            )
            return done
        work_dir.mkdir(parents=True, exist_ok=True)
        need = _need_bytes(inter, factors)
        free = shutil.disk_usage(work_dir).free
        if free < need:
            raise OSError(
                f"config {name} needs about {need / 2**30:.1f} GiB "
                f"of free disk in {work_dir}, "
                f"{free / 2**30:.1f} GiB is free"
            )
        say(
            f"als_tune_d: config {n} of at most {MAX_CONFIGS}: {name} (cell "
            f"{cell}, k = {k:+d}) started, at most {max_iter} iterations"
        )
        params = als_freq.FreqParams(
            factors=factors, reg=reg, alpha=alpha, iterations=max_iter, seed=seed, nu=nu
        )
        seconds = []

        def on_iteration(iteration, X, Y, history):
            t0 = time.time()
            ranks = rank(
                X,
                Y,
                inter,
                users,
                picks,
                log=say,
                label=f"als_tune_d: {name} iteration {iteration}, "
                f"ranking val picks",
            )
            score, last, first = als_tune.score_picks(ranks)
            scores = [h["score"] for h in history[:-1]] + [score]
            best = als_tune._best_index(scores)
            since = len(scores) - 1 - best
            stop = since >= patience
            results["configs"][name] = {
                "cell": cell,
                "k": k,
                "factors": factors,
                "nu": nu,
                "alpha": alpha,
                "reg": reg,
                "seed": seed,
                "finished": False,
                "position": n,
                "iterations_run": iteration,
                "scores": scores,
                "name": name,
            }
            if score > results["best"]["score"]:
                keep_best(
                    best_dir,
                    results,
                    name,
                    cell,
                    k,
                    params,
                    iteration,
                    score,
                    last,
                    first,
                    x_path,
                    y_path,
                    inter,
                )
            save()
            seconds.append(history[-1]["seconds"] + (time.time() - t0))
            left = 0 if stop else (max_iter - iteration)
            say(
                f"als_tune_d: {name}, iteration {iteration}: score "
                f"{score:.5f}, best {scores[best]:.5f} at iteration "
                f"{best + 1}, no gain for {since} of {patience}. val_last "
                f"R@100 {last['recall_at_100']:.4f}, val_first R@100 "
                f"{first['recall_at_100']:.4f}. This config: at most "
                f"{als._clock(left * sum(seconds) / len(seconds))} left "
                f"(early stopping usually ends sooner)."
            )
            return {"score": score, "val_last": last, "val_first": first, "stop": stop}

        _, _, history = als_freq.fit(
            inter,
            params,
            x_path,
            y_path,
            cfg_dir / "fit_state.json",
            masses[0],
            masses[1],
            workers=workers,
            on_iteration=on_iteration,
            log=say,
        )
        scores = [h["score"] for h in history]
        best = als_tune._best_index(scores)
        entry = {
            "cell": cell,
            "k": k,
            "factors": factors,
            "nu": nu,
            "alpha": alpha,
            "reg": reg,
            "seed": seed,
            "finished": True,
            "position": n,
            "name": name,
            "iterations_run": len(history),
            "best_iteration": best + 1,
            "score": scores[best],
            "scores": scores,
            "best_val_last": history[best]["val_last"],
            "best_val_first": history[best]["val_first"],
            "seconds_per_iteration": [h["seconds"] for h in history],
            "max_relative_rise": max(
                (
                    h["max_relative_rise"]
                    for h in history
                    if h["max_relative_rise"] is not None
                ),
                default=None,
            ),
        }
        results["configs"][name] = entry
        save()
        for p in (x_path, y_path):
            p.unlink(missing_ok=True)
        say(
            f"als_tune_d: config {n}: {name} finished, best iteration "
            f"{best + 1}, score {scores[best]:.5f}"
        )
        return entry

    return run_config


FACTOR_SUFFIXES = ("_factors.npy", "_factors.npy.tmp")


def keep_best(
    best_dir,
    results,
    name,
    cell,
    k,
    params,
    iteration,
    score,
    last,
    first,
    x_path,
    y_path,
    inter,
):
    """Copy the committed factor files of this iteration to best_dir under
    new names, then point best.json at them, then remove every other
    factor file in best_dir. A stop at any point leaves best.json
    describing files that exist, and the next call removes what the stop
    left behind. Files with other names in best_dir stay."""
    best_dir = Path(best_dir)
    best_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{name}_it{iteration}"
    files = {}
    for tag, src in (("user", x_path), ("item", y_path)):
        dst = best_dir / f"{stem}_{tag}_factors.npy"
        tmp = dst.with_name(dst.name + ".tmp")
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
        files[tag] = {"file": dst.name, "sha256": als._file_sha(dst)}
    record = {
        "config": name,
        "cell": cell,
        "k": k,
        "factors": params.factors,
        "nu": params.nu,
        "alpha": params.alpha,
        "reg": params.reg,
        "seed": params.seed,
        "iteration": iteration,
        "score": score,
        "val_last": last,
        "val_first": first,
        "files": files,
        "fit_key": als_freq.fit_key(inter, params),
    }
    atomic_json_write(record, best_dir / "best.json")
    results["best"] = record
    keep = {files["user"]["file"], files["item"]["file"]}
    for p in best_dir.iterdir():
        if p.name not in keep and p.name.endswith(FACTOR_SUFFIXES):
            p.unlink(missing_ok=True)


# The phase


def run_phase(
    inter,
    users,
    picks,
    masses,
    ac_results,
    results_path,
    work_dir,
    best_dir,
    workers=None,
    say=print,
    max_iter=MAX_ITER,
    patience=PATIENCE,
    rank=als.rank_items,
    sesoi=SESOI,
):
    """The protocol above. ac_results: phases A to C (als_tuning.json).
    Returns the results dict."""
    results_path = Path(results_path)
    results = (
        json.loads(results_path.read_text(encoding="utf-8"))
        if results_path.exists()
        else {}
    )
    results.setdefault("protocol", PROTOCOL)
    results.setdefault("configs", {})
    results.setdefault("best", {"score": -math.inf})
    ac = ac_results["winner"]
    ac_winner = {
        "reg": float(ac["reg"]),
        "score": float(ac["score"]),
        "alpha": float(ac["alpha"]),
        "seed": int(ac["seed"]),
        "factors": int(ac["factors"]),
        "iterations": int(ac["iterations"]),
    }
    results["phase_ac_winner"] = ac_winner
    if ac_results.get("reference"):
        results["reference"] = ac_results["reference"]
    alpha, seed = ac_winner["alpha"], ac_winner["seed"]
    F_u = als_freq.median_frequency(masses[0], inter.u[0], alpha, inter.n_items - 1)
    F_i = als_freq.median_frequency(masses[1], inter.i[0], alpha, inter.n_users - 1)
    results["median_frequency"] = {"users": F_u, "items": F_i}
    results["frequency_quantiles"] = {
        "users": als_freq.frequency_quantiles(
            masses[0], inter.u[0], alpha, inter.n_items - 1
        ),
        "items": als_freq.frequency_quantiles(
            masses[1], inter.i[0], alpha, inter.n_users - 1
        ),
    }

    def save():
        doc = dict(results)
        if not math.isfinite(doc["best"]["score"]):
            doc["best"] = {"score": None}
        atomic_json_write(doc, results_path)

    if results["best"].get("score") is None:
        results["best"] = {"score": -math.inf}
    save()
    fq = results["frequency_quantiles"]
    say(
        f"als_tune_d: median frequency f: users {F_u:,.0f}, items "
        f"{F_i:,.0f}. The largest f is {fq['users']['q1'] / fq['users']['q0']:.3g} "
        f"times the smallest for the users and "
        f"{fq['items']['q1'] / fq['items']['q0']:.3g} times for the items. "
        f"Phase A to C winner: alpha {alpha:g}, reg {ac_winner['reg']:g}, "
        f"score {ac_winner['score']:.5f}."
    )
    run_config = make_run_config(
        inter,
        users,
        picks,
        masses,
        results,
        save,
        work_dir,
        best_dir,
        alpha,
        seed,
        max_iter=max_iter,
        patience=patience,
        workers=workers,
        say=say,
        rank=rank,
    )
    entries, gate = run_cells(
        run_config, ac_winner, lambda reg: reg / math.sqrt(F_u * F_i), sesoi
    )
    results["gate"] = gate
    if not gate["passed"]:
        say(
            f"als_tune_d: the best 128-factor config ({gate['best_128']}, "
            f"{gate['best_128_score']:.5f}) does not beat the phase A to C "
            f"winner by more than SESOI (threshold "
            f"{gate['threshold']:.5f}), so the 256-factor cell does not run."
        )
    win = best_of(entries)
    results["order"] = [e["name"] for e in entries]
    results["winner"] = {
        "config": win["name"],
        "cell": win["cell"],
        "factors": win["factors"],
        "nu": win["nu"],
        "alpha": win["alpha"],
        "reg": win["reg"],
        "iterations": win["best_iteration"],
        "score": win["score"],
        "val_last": win["best_val_last"],
        "val_first": win["best_val_first"],
    }
    results["for_p14"] = (
        "phase D winner"
        if win["score"] > ac_winner["score"]
        else "phase A to C winner (P6)"
    )
    if (
        results["best"].get("config") != win["name"]
        or results["best"].get("iteration") != win["best_iteration"]
    ):
        raise RuntimeError(
            "the kept factors do not belong to the winner's " "best iteration"
        )
    save()
    return results


def write_report(results, path):
    lines = ["# ALS phase D: more factors and the frequency-scaled " "regularizer", ""]
    add = lines.append
    p = results["protocol"]
    add(f"- declared: {p['declared']}")
    add(f"- users: {p['users']}")
    add(f"- metric: {p['metric']}")
    add(
        f"- fixed: {p['fixed']}, iterations up to {p['iterations']['max']} "
        f"with patience {p['iterations']['patience']}"
    )
    add(f"- regularizer: {p['regularizer']}")
    for cell, text in p["cells"].items():
        add(f"- {cell}: {text}")
    add(f"- search: {p['search']}")
    mf = results.get("median_frequency") or {}
    if mf:
        add(
            f"- median frequency f: users {mf['users']:,.0f}, items "
            f"{mf['items']:,.0f}"
        )
    fq = results.get("frequency_quantiles") or {}
    for side in ("users", "items"):
        q = fq.get(side)
        if q:
            add(
                f"- f of the {side} with data: minimum {q['q0']:,.0f}, 1st "
                f"percentile {q['q0.01']:,.0f}, median {q['q0.5']:,.0f}, "
                f"99th percentile {q['q0.99']:,.0f}, maximum {q['q1']:,.0f}. "
                f"With nu = 1 the maximum gets {q['q1'] / q['q0']:.3g} times "
                f"the regularization of the minimum."
            )
    add("")
    add(
        "| cell | factors | nu | k | reg | best iteration | score | "
        "val_last R@100 | val_first R@100 | seconds per iteration | "
        "largest objective rise |"
    )
    add("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name in results.get("order", []):
        e = results["configs"][name]
        rise = e["max_relative_rise"]
        add(
            f"| {e['cell']} | {e['factors']} | {e['nu']:g} | {e['k']:+d} | "
            f"{e['reg']:.4g} | {e['best_iteration']} | {e['score']:.5f} | "
            f"{e['best_val_last']['recall_at_100']:.4f} | "
            f"{e['best_val_first']['recall_at_100']:.4f} | "
            f"{np.mean(e['seconds_per_iteration']):.0f} | "
            f"{'n/a' if rise is None else f'{rise:.1e}'} |"
        )
    ac = results["phase_ac_winner"]
    add("")
    add(
        f"Phase A to C winner: {ac['factors']} factors, alpha "
        f"{ac['alpha']:g}, reg {ac['reg']:g}, {ac['iterations']} iterations, "
        f"score {ac['score']:.5f}."
    )
    ref = results.get("reference")
    if ref:
        add(f"Popularity on the same picks: score {ref['score']:.5f}.")
    gate = results["gate"]
    add("")
    add(
        f"Gate for 256 factors: the best 128-factor config "
        f"({gate['best_128']}) scores {gate['best_128_score']:.5f} against "
        f"the threshold {gate['threshold']:.5f}. "
        + (
            "The 256-factor cell ran."
            if gate["passed"]
            else "The 256-factor cell did not run."
        )
    )
    w = results["winner"]
    add("")
    add(
        f"Phase D winner: {w['config']} (cell {w['cell']}), "
        f"{w['iterations']} iterations, score {w['score']:.5f}. Its factors "
        f"are in als_d/ (best.json)."
    )
    add(f"For P14: the {results['for_p14']}.")
    add("")
    add(
        "A negative objective rise means the objective fell at every "
        "half-step, as exact ALS requires."
    )
    atomic_text_write("\n".join(lines) + "\n", path)


def _stop(message):
    log.error(message)
    raise SystemExit(1)


def main(argv=None):
    als_tune.setup_logging(LOG_PATH)
    try:
        _run()
    except KeyboardInterrupt:
        log.info(
            "als_tune_d: stopped by Ctrl+C. Run python3 -m pipeline.runner --only als_phase_d "
            "again to continue from the last finished iteration."
        )
        raise SystemExit(130)
    except Exception:
        log.exception("als_tune_d: stopped by an error")
        raise SystemExit(1)


def _run():
    say = log.info
    t0 = time.time()
    say(f"als_tune_d: started, process id {os.getpid()}, log file " f"{LOG_PATH}")
    say(als_tune.blas_setup())
    ac_path = MODEL_DIR / "als_tuning.json"
    if not ac_path.exists():
        _stop(
            f"als_tune_d: {ac_path} is missing. Phases A to C must finish "
            f"first (python3 -m pipeline.als_tune)."
        )
    ac_results = json.loads(ac_path.read_text(encoding="utf-8"))
    if not ac_results.get("winner"):
        _stop(
            "als_tune_d: als_tuning.json holds no winner. Phases A to C "
            "must finish first."
        )
    p6 = als.Interactions.from_p6(MODEL_DIR)
    say(
        f"als_tune_d: {p6.n_users:,} user slots, {p6.n_items:,} item slots, "
        f"{p6.nnz:,} train pairs"
    )
    say("als_tune_d: fingerprinting the user-major CSR (reads it once in " "full)")
    p6.fingerprint(log=say)
    inter = p6.with_transposed_item_major(SCRATCH_DIR / "csr_item_major", log=say)
    real = als.orientation_real_fraction(inter, rows=500, per_row=200, seed=1)
    if real < 1.0:
        _stop(
            f"als_tune_d: the transposed item-major CSR scored {real:.2%}, "
            f"not 1.0. Stopping."
        )
    say(
        "als_tune_d: loading the listen mass of every row (cached after the "
        "first run)"
    )
    masses = als_freq.mass_of(inter, WORK_DIR / "listen_mass.npz", log=say)
    say(
        "als_tune_d: loading the selection users and their val picks with "
        "DuckDB (no progress lines in this step)"
    )
    con = connect(4)
    try:
        users = selection.selection_user_ids(con)
        picks = selection.val_picks(con, users)
    finally:
        con.close()
    say(f"als_tune_d: {len(users):,} selection users, {picks.size:,} val " f"picks")
    results = run_phase(
        inter,
        users,
        picks,
        masses,
        ac_results,
        RESULTS_PATH,
        WORK_DIR,
        BEST_DIR,
        say=say,
    )
    report = MODEL_DIR / "reports" / "als_tuning_d.md"
    write_report(results, report)
    w = results["winner"]
    say(
        f"als_tune_d: done in {(time.time() - t0) / 3600:.1f} h. Phase D "
        f"winner: {w['config']}, score {w['score']:.5f}. For P14: the "
        f"{results['for_p14']}. Report: {report}"
    )


# Read-back


def check_outputs(model_dir):
    """Raise ValueError unless the stored outputs agree with each other and
    with the protocol: the winner, the gate and P14's choice follow from
    the stored configs, best.json is the winner's best iteration, and the
    factor files match their SHA-256 in best.json. Returns the winner."""
    d = Path(model_dir)
    results = load_json(d / RESULTS_PATH.name)
    best = load_json(d / BEST_DIR.name / "best.json")
    if results is None or best is None:
        raise ValueError("als_tuning_d.json or als_d/best.json is missing")
    order = results.get("order") or []
    configs = results.get("configs") or {}
    if not order or any(n not in configs for n in order):
        raise ValueError("the stored order names configs that are missing")
    entries = [configs[n] for n in order]
    if not all(e.get("finished") for e in entries):
        raise ValueError("a config of the stored order did not finish")
    win = best_of(entries)
    want = {
        "config": win["name"],
        "cell": win["cell"],
        "factors": win["factors"],
        "nu": win["nu"],
        "alpha": win["alpha"],
        "reg": win["reg"],
        "iterations": win["best_iteration"],
        "score": win["score"],
        "val_last": win["best_val_last"],
        "val_first": win["best_val_first"],
    }
    if results.get("winner") != want:
        raise ValueError("the stored winner does not follow from the stored " "configs")
    ac = results.get("phase_ac_winner") or {}
    if "score" not in ac:
        raise ValueError("als_tuning_d.json holds no phase A to C winner")
    p14 = (
        "phase D winner"
        if win["score"] > float(ac["score"])
        else "phase A to C winner (P6)"
    )
    if results.get("for_p14") != p14:
        raise ValueError(
            "the stored choice for P14 does not follow from the " "winner's score"
        )
    gate = results.get("gate") or {}
    b128 = best_of([e for e in entries if e["cell"] in ("D1", "D2")])
    passed = bool(b128["score"] > float(ac["score"]) * (1.0 + SESOI))
    if (
        gate.get("best_128") != b128["name"]
        or gate.get("passed") != passed
        or passed != any(e["cell"] == "D3" for e in entries)
    ):
        raise ValueError("the stored gate does not follow from the stored " "configs")
    if best != results.get("best"):
        raise ValueError(
            "als_d/best.json differs from the record in " "als_tuning_d.json"
        )
    if (
        best.get("config") != win["name"]
        or best.get("iteration") != win["best_iteration"]
        or best.get("score") != win["score"]
    ):
        raise ValueError("als_d/best.json is not the winner's best iteration")
    for tag, f in sorted((best.get("files") or {}).items()):
        p = d / BEST_DIR.name / f["file"]
        if not p.exists() or als._file_sha(p) != f["sha256"]:
            raise ValueError(
                f"the {tag} factors differ from their SHA-256 " f"in als_d/best.json"
            )
    return win


# Runner stage


class StageALSD(ScriptStage):
    """ALS phase D as a runner stage."""

    name = "als_phase_d"
    upstream = ["p6_als"]

    def outputs(self):
        d = self.model_dir
        out = {
            "results": d / RESULTS_PATH.name,
            "best": d / BEST_DIR.name / "best.json",
            "report": d / "reports" / "als_tuning_d.md",
        }
        best = load_json(out["best"])
        for tag, f in sorted(((best or {}).get("files") or {}).items()):
            out[f"factors_{tag}"] = d / BEST_DIR.name / f["file"]
        return out

    def provenance(self):
        p = self.model_dir / "als_tuning.json"
        return {"als_tuning.json": file_sha256(p) if p.exists() else None}

    def _p6(self):
        return als.Interactions.from_p6(self.model_dir)

    def stale_input(self):
        """The phase A to C winner and P6's data, against what the outputs
        recorded: phase_ac_winner, and the fit key's shape and data
        fingerprint."""
        results = load_json(self.model_dir / RESULTS_PATH.name)
        best = load_json(self.model_dir / BEST_DIR.name / "best.json")
        ac = load_json(self.model_dir / "als_tuning.json")
        if results is None or best is None or ac is None:
            return "(als_tuning_d.json, als_d/best.json or als_tuning.json)"
        win = ac.get("winner") or {}
        rec = results.get("phase_ac_winner") or {}
        for key, cast in (
            ("reg", float),
            ("score", float),
            ("alpha", float),
            ("seed", int),
            ("factors", int),
            ("iterations", int),
        ):
            if key not in win or cast(win[key]) != rec.get(key):
                return f"als_tuning.json (the phase A to C winner's {key})"
        p6 = self._p6()
        fit_key = best.get("fit_key") or {}
        if fit_key.get("shape") != [p6.n_users, p6.n_items, p6.nnz]:
            return "P6's interactions (their shape)"
        if fit_key.get("data") != p6.fingerprint(log=log.info):
            return "P6's interactions (their fingerprint)"
        return None

    def run(self, params, files):
        with log_file(self.model_dir / "logs" / LOG_PATH.name):
            _run()
        return self.output_info()

    def check(self):
        check_outputs(self.model_dir)


if __name__ == "__main__":
    main()
